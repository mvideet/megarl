// policy_versions.cuh -- the table of published snapshots of the trained layer (mrl::Versions).
//
//   pin     thread 0 of R_BEGIN, once per admitted group
//   unpin   thread 0 of R_ADVANCE, or of the cancel path
//   publish all threads of T_FINISH
//
// Why a pinned snapshot is never overwritten. A reader counts itself into ref[s] and then re-reads cur; it keeps s
// only if cur still names s. The writer picks a slot v that is not current and whose count is zero, and it is the
// only thread that ever changes cur. Take a reader that kept s and a writer that later picked the same slot. The
// writer picked it while cur != s, the reader saw cur == s after its increment, so the reader's re-check ran before
// the writer's earlier switch away from s, and its increment before that: the writer's later look at ref[s] comes
// after the increment and finds it. Both "store, then load the other's word" sequences are separated by a
// sequentially consistent fence (__threadfence() is fence.sc on sm_70+).
#pragma once
#include "rl_state.cuh"

namespace pv {
using mrl::bf16;

// Thread 0. Pin the current snapshot. stats (optional): cnt.stats, for the retry statistic.
__device__ __forceinline__ void pin(const mrl::Versions& V, int& slot, int& id, int& gen, int* stats = nullptr) {
  int s;
  for (;;) {
    s = atomicAdd(V.cur, 0);
    atomicAdd(V.ref + s, 1);
    __threadfence();
    if (atomicAdd(V.cur, 0) == s) break;
    atomicAdd(V.ref + s, -1);                                // the writer may have chosen s before it saw us: let go
    if (stats) rls::stat_add(stats, rls::PIN_RETRIES);
  }
  __threadfence();                                           // acquire side of the writer's release (fence, then the cur switch)
  slot = s; gen = rls::ldi(V.gen + s); id = rls::ldi(V.id + s);
}

// Thread 0. Give the pin back. A generation counter that moved since the pin, or that was odd at the pin, means the
// slot was written while this reader used it: counted in errors[ERR_SNAPSHOT] (cannot happen with pv::publish).
__device__ __forceinline__ void unpin(const mrl::Versions& V, int slot, int gen, int* errors) {
  if (atomicAdd(V.gen + slot, 0) != gen || (gen & 1)) atomicAdd(errors + rls::ERR_SNAPSHOT, 1);
  atomicAdd(V.ref + slot, -1);
}

// Copy helpers, all threads of the block. Loads bypass L1 (the sources are rewritten by other SMs every update).
__device__ __forceinline__ void copy16(void* dst, const void* src) { *reinterpret_cast<int4*>(dst) = __ldcg(reinterpret_cast<const int4*>(src)); }
__device__ __forceinline__ void copy_snapshot(bf16* dst, const bf16* src) {
  static_assert(mrl::MID_NW % 8 == 0, "16-byte chunks");
  for (size_t q = threadIdx.x; q < mrl::MID_NW / 8; q += 256) copy16(dst + q * 8, src + q * 8);
}

// All threads of the learner's block. Publish W as version `id` (= train_ver + 1) and make it current.
// Returns, to every thread, the slot it was written to, or -1 if *abort was set while no slot was free (then
// nothing was changed). smem: at least 8 ints of the block's dynamic shared memory (thread 0 tells the others which
// slot to fill); polls (optional, thread 0): how many times the scan found no free slot.
__device__ __forceinline__ int publish(const mrl::Versions& V, const bf16* W, int id, int* stats, const int* abort, char* smem, int* polls = nullptr) {
  const int tid = threadIdx.x; int* sm = reinterpret_cast<int*>(smem);
  if (tid == 0) {
    int v = -1, np = 0;
    const int c = atomicAdd(V.cur, 0);
    for (;;) {
      for (int k = 1; k < V.nslots && v < 0; k++) { const int cand = (c + k) % V.nslots; if (atomicAdd(V.ref + cand, 0) == 0) v = cand; }
      if (v >= 0) break;
      rls::stat_add(stats, rls::PUB_WAITS); np++;            // every other slot is pinned: only by stale groups that
      if (rls::ldi(abort)) break;                            // the next R_BEGIN cancels (nslots >= lag + 4)
      __nanosleep(200);
    }
    if (v >= 0) { atomicAdd(V.gen + v, 1); __threadfence(); }   // odd: being written
    sm[0] = v; sm[1] = np;
  }
  __syncthreads();
  const int v = sm[0];
  if (v >= 0) copy_snapshot(V.W + (size_t)v * mrl::MID_NW, W);
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    if (v >= 0) {
      *reinterpret_cast<volatile int*>(V.id + v) = id;       // stored before the slot turns even
      atomicAdd(V.gen + v, 1); __threadfence();              // even again
      atomicExch(V.cur, v); __threadfence();                 // from here on new groups pin version `id`
      mk::st_rel(V.train_ver, id);
    }
    if (polls) *polls = sm[1];
  }
  __syncthreads();                                           // sm[] may be reused by the caller
  return v;
}

}  // namespace pv
