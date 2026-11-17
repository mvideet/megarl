// trace.cuh -- the event trace (mrl::Trace, trace.bin) and the group log (mrl::GroupLog, groups.bin with
// group_tokens.i32 and group_logp.f32).
// Both are append-only arrays whose index is claimed with one atomicAdd; entries beyond the capacity are not stored
// but still counted, so the host can tell how much is missing. They are read by the host after the kernel exits
// (rlc::host_download); nothing on the device reads them, except log_copy_guard reading back a terminal state.
// Event types and their fields are listed at rls::Ev (rl_state.cuh).
#pragma once
#include "rl_state.cuh"

namespace tr {
using mrl::u64;

// Nanoseconds since kernel start. t0 = *Ctx.t0 (every block did mk::mark_start before its first task).
__device__ __forceinline__ u64 since(u64 t0) { const u64 t = mk::gtimer(); return t > t0 ? t - t0 : 0ull; }

// Thread 0. Append one event (fields a..e: rls::Ev). t0 is the value of *Ctx.t0.
__device__ __forceinline__ void emit(const mrl::Trace& T, u64 t0, int type, int a, int b, int c, int d, int e) {
  const u64 i = atomicAdd(T.n, 1ull);
  if (i >= T.cap) return;
  mrl::TraceEvent* ev = T.ev + i;
  ev->t = since(t0); ev->type = type; ev->a = a; ev->b = b; ev->c = c; ev->d = d; ev->e = e;
}

// Thread 0. Append the terminal record of the group that holds buffer record `rec` (still owned by the caller: the
// buffer has not been released). Returns the log index; if it is < L.tok_cap the caller must have all threads run
// log_copy_tokens(L, S, buf, index) before the buffer is released.
// train_step and lag are stored for CONSUMED only (-1 otherwise); a CANCELLED group has no rewards, advantages or
// completion time (stored as 0).
__device__ __forceinline__ u64 log_group(const mrl::GroupLog& L, const mrl::GroupRec* rec, int terminal, int train_step, int lag, u64 t_end) {
  const u64 n = atomicAdd(L.n, 1ull);
  if (n >= L.cap) return n;
  mrl::GroupLogEntry* e = L.e + n;
  const bool cancelled = terminal == rls::CANCELLED, consumed = terminal == rls::CONSUMED;
  e->uid = rls::ldi(&rec->uid); e->task = rls::ldi(&rec->task); e->prompt = rls::ldi(&rec->prompt); e->key = rls::ldi(&rec->key);
  e->ver_id = rls::ldi(&rec->ver_id); e->terminal = terminal;
  e->train_step = consumed ? train_step : -1; e->lag = consumed ? lag : -1;
  for (int j = 0; j < mrl::GSZ; j++) {
    e->len[j] = rls::ldi(&rec->len[j]);
    e->reward[j] = cancelled ? 0.f : rls::ldf(&rec->reward[j]);
    e->adv[j] = cancelled ? 0.f : rls::ldf(&rec->adv[j]);
  }
  e->t_admit = rls::ld64(&rec->t_admit); e->t_done = cancelled ? 0ull : rls::ld64(&rec->t_done); e->t_end = t_end;
  return n;
}

// Thread 0, when the tokens of group buffer `buf` are copied into log slot n: the group must still hold the buffer,
// i.e. its record is still in the terminal state that log entry n was written with (by this thread, in log_group).
// A buffer that went back to the free list before the copy can already belong to a new group, whose tokens would be
// logged under the old group's uid, and nothing else would notice: the entry itself is right, and the tokens are
// wrong only if another job happened to take the buffer in between. Counted in errors[ERR_STATE].
__device__ __forceinline__ void log_copy_guard(const mrl::GroupLog& L, const mrl::Store& S, int buf, u64 n, int* errors) {
  if (rls::ldi(&S.rec[buf].state) != rls::ldi(&L.e[n].terminal)) atomicAdd(errors + rls::ERR_STATE, 1);
}

// All threads. Tokens and generation-time log-probabilities of group buffer `buf` into log slot n (< L.tok_cap):
// GSZ * MAXGEN = 256 elements each, one per thread. errors = cnt.errors: thread 0 runs log_copy_guard first.
// ctl::roll_advance passes no errors pointer and runs the guard itself.
__device__ __forceinline__ void log_copy_tokens(const mrl::GroupLog& L, const mrl::Store& S, int buf, u64 n, int* errors = nullptr) {
  static_assert(mrl::GSZ * mrl::MAXGEN == 256, "one element per thread");
  const int tid = threadIdx.x; const size_t src = (size_t)buf * mrl::GSZ * mrl::MAXGEN + tid, dst = (size_t)n * mrl::GSZ * mrl::MAXGEN + tid;
  if (tid == 0 && errors) log_copy_guard(L, S, buf, n, errors);
  L.tokens[dst] = rls::ldi(S.tokens + src);
  L.logp[dst] = rls::ldf(S.logp + src);
}

}  // namespace tr
