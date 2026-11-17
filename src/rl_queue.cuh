// rl_queue.cuh -- bounded lock-free queue of ints with several producers and several consumers (D. Vyukov's bounded
// MPMC queue) on mrl::Queue. Used for the NTASKS ready queues (group buffers waiting for the learner) and for the list
// of free group buffers.
//
// Layout: capacity mask + 1 (a power of two, at least 2); cell i has a sequence number seq[i] and a value val[i];
// *tail is the next position to push, *head the next position to pop. Positions are 32-bit and wrap; all comparisons
// are differences, so the queue works across the wrap.
//   cell of position p is free for the push of p    <=>  seq == p
//   cell of position p holds the value pushed at p  <=>  seq == p + 1
//   after the pop of p the cell waits for p + cap   <=>  seq == p + cap
// An empty queue with both counters at position b therefore has seq[i] = the position in [b, b + cap) that maps to
// cell i (rlc::set_queue writes this layout). With capacity 1 "holds p" and "waits for p + 1" would be the same
// number, which is why the smallest queue has two cells (rlc::alloc_queue enforces it).
//
// Memory ordering: a producer claims its position with a compare-and-swap on *tail, writes the value, and only then
// stores the cell's sequence number with a release operation. A consumer loads the sequence number with an acquire
// operation before it claims with a compare-and-swap on *head and reads the value. Hence everything the pushing
// thread wrote, and everything that was ordered before the push (by a barrier of its block or by the scheduler), is
// visible to the popping thread after the pop, and to whatever that thread orders after it.
//
// All functions are for one thread per block (thread 0 of a control task). None of them waits: push returns false
// if the queue is full, pop / front return false if it is empty. A retry happens only when another thread's
// operation got in between, so a handful of iterations is the rule; the loops are still bounded (SPIN_MAX), so that a
// queue whose memory was overwritten by something else ends in a counted error (push: the caller's ERR_QUEUE_FULL;
// pop: "empty", and the run starves) instead of a kernel that never returns.
//
// "Full" and "empty" are what the one cell the operation needs shows, not what the counters say:
//   * pop is refused while the push of the oldest position has claimed its cell and not yet published it;
//   * push is refused while the cell of its position has not been released by the pop of one lap ago, which has
//     moved *head already. With one consumer that pop is the oldest outstanding one, so capacity - 1 values are
//     still queued behind it and the queue really is full. With several consumers (the free list: every rollout job
//     pops it) the positions behind a slow pop can have been popped by the others meanwhile, and a push is refused
//     although the queue is nearly empty. That lasts for the few instructions between the slow consumer's
//     compare-and-swap and its release; a caller that must not lose its value repeats the push (ctl::release_buf).
#pragma once
#include "mrl_common.cuh"

namespace rlq {
constexpr unsigned SPIN_MAX = 1u << 22;

__device__ __forceinline__ unsigned ld_rlx(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ int ld_rlx_i(const int* p) { int v; asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

// Append v. False if the queue is full: the cell of the next position still holds the value of one lap ago, or the
// pop of that value has not released the cell yet (see above: with several consumers that is a refusal for a moment).
__device__ __forceinline__ bool push(const mrl::Queue& q, int v) {
  unsigned pos = ld_rlx(q.tail);
  for (unsigned spin = 0;; spin++) {
    const unsigned s = ld_acq(q.seq + (pos & q.mask));
    const int dif = (int)(s - pos);
    if (dif == 0) {                                          // free for this position: claim it
      const unsigned old = atomicCAS(q.tail, pos, pos + 1u);
      if (old == pos) break;
      pos = old;                                             // somebody else took it: the returned tail is fresher
    } else if (dif < 0 || spin >= SPIN_MAX) return false;    // full
    else pos = ld_rlx(q.tail);                               // our tail was stale
  }
  q.val[pos & q.mask] = v;                                   // exclusive: the claim gave this thread the cell
  st_rel(q.seq + (pos & q.mask), pos + 1u);                  // publish: orders the value (and all earlier writes) before the pop
  return true;
}

// Remove the oldest value. False if the queue is empty (or its oldest push has claimed a cell but not published yet).
__device__ __forceinline__ bool pop(const mrl::Queue& q, int& v) {
  unsigned pos = ld_rlx(q.head);
  for (unsigned spin = 0;; spin++) {
    const unsigned s = ld_acq(q.seq + (pos & q.mask));
    const int dif = (int)(s - (pos + 1u));
    if (dif == 0) {
      const unsigned old = atomicCAS(q.head, pos, pos + 1u);
      if (old == pos) break;
      pos = old;
    } else if (dif < 0 || spin >= SPIN_MAX) return false;    // empty
    else pos = ld_rlx(q.head);
  }
  v = ld_rlx_i(q.val + (pos & q.mask));                      // the acquire load above made the producer's store visible
  st_rel(q.seq + (pos & q.mask), pos + q.mask + 1u);         // the cell is free for the push one lap later
  return true;
}

// The oldest value without removing it. Valid only for a queue with a single consumer (nobody else moves *head), such
// as a ready queue: then a following pop returns the same value.
__device__ __forceinline__ bool front(const mrl::Queue& q, int& v) {
  const unsigned pos = ld_rlx(q.head);
  if (ld_acq(q.seq + (pos & q.mask)) != pos + 1u) return false;
  v = ld_rlx_i(q.val + (pos & q.mask));
  return true;
}

// Number of queued values. Exact when nobody is pushing or popping; otherwise a value the queue had or nearly had
// (the two counters are read one after the other), clamped to [0, capacity].
__device__ __forceinline__ int size(const mrl::Queue& q) {
  const unsigned h = ld_rlx(q.head), t = ld_rlx(q.tail);
  const int n = (int)(t - h);
  return n < 0 ? 0 : (n > (int)q.mask + 1 ? (int)q.mask + 1 : n);
}

}  // namespace rlq
