// rl_state.cuh -- the shared vocabulary of the RL core and its host side.
//
//   namespace rls  group states, error / stop / statistic / trace-event numbers, the compare-and-swap state transition
//   namespace rlc  host: allocation and reset of mrl::Shared, download of what a run left behind, and the audit
// The device side of the core is in rl_queue.cuh, policy_versions.cuh, trace.cuh and rl_control.cuh.
//
// Memory ordering in the whole core:
//   * every word that two blocks may touch without the scheduler ordering them (counters, queue cells, the version
//     table, group states, stop flags) is read and written with atomics, acquire / release operations or volatile
//     ("strong") accesses, never with plain loads: a plain load may be served from an SM-local L1 line that predates
//     another SM's store;
//   * data handed from one block to another travels behind a release operation of the producer's thread 0 (queue
//     push, snapshot switch, the scheduler's completion bump) and is picked up after an acquire operation of the
//     consumer's thread 0; inside a block the hand-over from and to thread 0 is a __syncthreads().
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "mrl_common.cuh"

namespace rls {
using mrl::u64;

// Group states. store.rec[buf].state is the state of the group that holds the buffer; GroupSlot.state is only ever
// FREE or GENERATING. The values are part of the groups.bin format (field `terminal`).
//   FREE -> GENERATING (R_BEGIN admits) -> SCORING (R_ADVANCE) -> READY (queued) -> TRAINING (T_SELECT) -> CONSUMED
//   terminal otherwise: DROPPED_ZEROVAR (R_ADVANCE), DROPPED_STALE (T_SELECT), CANCELLED (R_BEGIN, while generating)
// A terminal state lasts until the group is logged and its buffer released (FREE again).
enum State { FREE = 0, GENERATING = 1, SCORING = 2, READY = 3, TRAINING = 4, CONSUMED = 5, DROPPED_ZEROVAR = 6, DROPPED_STALE = 7, CANCELLED = 8, NSTATES = 9 };

// Indices into cnt.errors[16]. All must stay 0 in a correct run.
enum Err {
  ERR_STATE = 0,        // a state transition found the group in a state other than the one it had to be in, or a
                        // group's tokens were copied to the log when its buffer was no longer the group's (tr::log_copy_guard)
  ERR_QUEUE_FULL = 1,   // a push did not get its value into the queue. Every queue has room for every buffer, so no queue
                        // is ever full; a push on the free list can be refused for a moment (rl_queue.cuh), which
                        // ctl::release_buf repeats. Counted: a push that was not repeated, or never succeeded
  ERR_SNAPSHOT = 2,     // a pinned snapshot was written while it was pinned (generation counter moved or odd)
  ERR_POP = 3,          // the learner's pop did not return the group its front() had shown
  ERR_LAG = 4,          // the learner took a group whose version is newer than train_ver
  ERR_BUF = 5,          // a queue delivered something that is not a group buffer id
  ERR_PROMPT = 6,       // reserved, never raised: an environment without prompts is simply not admitted
  ERR_NONFINITE = 7,    // T_FINISH found a gradient norm that is not finite: the update was not published, the run
                        // stopped with reason ERROR, and the last good version stays current and archived
  // Two slots belong to other modules, which count into them without including this header (qw::ERR_ROW in
  // qwen_fwd.cuh, hd::ERR_NO_WINNER in head_sample.cuh). They are named here so that the audit can say what a non-zero
  // count means. megarl.cu, which sees all three headers, asserts that the modules' slots are distinct and above
  // ERR_PROMPT; that they equal the two numbers below is kept by hand.
  ERR_TRUNK_ROW = 14,   // trunk: a fed row outside the contract (counted per attention task: 48 per row and step), or a token id outside the vocabulary
  ERR_HEAD_NO_WINNER = 15,   // head: a row for which no token could be sampled (every score was NaN)
  NERR = 16
};
enum Stop { STOP_NONE = 0, DONE = 1, DEADLINE = 2, STARVED = 3, ERROR = 4 };   // cnt.stop_reason, "stop_reason" in run.json

// Indices into cnt.stats[16]: cumulative, informational. Counted with rls::stat_add, which stops at STAT_MAX: the
// poll counters grow by one per 2 us of waiting (the publisher's per 200 ns), summed over the waiting tasks, and would
// otherwise overflow in a long run. A value >= STAT_MAX in run.json means "at least that many".
enum Stat {
  PUB_WAITS = 0,        // polls of the publisher that found every other snapshot slot pinned
  IDLE_POLLS = 1,       // polls of R_BEGIN while its job did not step: nothing to generate, or (alternate) the learner's turn
  SELECT_POLLS = 2,     // polls of T_SELECT that found nothing to take
  BACKPRESSURE = 3,     // admissions given up because no group buffer was free
  PIN_RETRIES = 4,      // pin attempts repeated because the current snapshot changed under the reader
  NSTAT = 16
};
constexpr int STAT_MAX = 0x7fff0000;   // 65,535 below INT_MAX: every block of the kernel may pass stat_add's test at once

// Trace event types. A trace.bin record (mrl::TraceEvent, 32 bytes) is {u64 t (ns since kernel start), type, a..e}:
//   ADMIT         uid, task, prompt, version, pipeline * 256 + slot
//   GROUP_DONE    uid, task, rewards equal to 1, longest answer, pipeline * 256 + slot
//   ENQUEUE       uid, task, buffer, ready-queue size after the push, version
//   DROP_ZEROVAR  uid, task, buffer, rewards equal to 1, version
//   CANCEL        uid, task, buffer, train_ver, version
//   DROP_STALE    uid, task, buffer, train_ver, version
//   SELECT        step, uid, task, lag, index in the batch
//   PUBLISH       version, snapshot slot, step, publisher polls, 0
//   STEP          step, tokens, tokens without gradient, packed tiles, 0
//   IDLE          job, 1 = idle from here / 0 = steps again, 0, 0, 0
//   SELECT_WAIT   step, us until the batch was full, stale drops, 1 if a group beyond its environment's share was
//                 taken, us waiting for the turn (alternate)
//   STOP          reason (rls::Stop), 0, 0, 0, 0
enum Ev { EV_ADMIT = 1, EV_GROUP_DONE = 2, EV_ENQUEUE = 3, EV_DROP_ZEROVAR = 4, EV_CANCEL = 5, EV_DROP_STALE = 6, EV_SELECT = 7,
          EV_PUBLISH = 8, EV_STEP = 9, EV_IDLE = 10, EV_SELECT_WAIT = 11, EV_STOP = 12, NEV = 13 };

// cnt.training, one word (written with atomics only):
//   bit 0 (TRAINING_BATCH)   the learner has a complete batch: set by T_SELECT when it has taken the last group of the
//                            batch, cleared by T_FINISH when it has published the update and released the groups.
//                            Maintained always; with hp.alternate == 0 the word is 0 or 1 and only the audit reads it.
//   bit TURN_SHIFT + j       hp.alternate only: rollout job j holds the generation turn, from before the admissions of an
//                            R_BEGIN that starts a step until the R_BEGIN of the next step (ctl::turn_take / turn_give).
// The two kinds of bits exclude each other in time once the learner's wait is over: a job sets its bit only with a
// compare-and-swap that finds bit 0 clear, and the learner, having set bit 0, waits until no job bit is left.
constexpr int TRAINING_BATCH = 1, TURN_SHIFT = 8;
static_assert(TURN_SHIFT + mrl::MAXPIPE <= 30, "one bit of cnt.training per rollout job");

// Strong loads of words that other blocks write.
__device__ __forceinline__ int ldi(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ float ldf(const float* p) { return *reinterpret_cast<const volatile float*>(p); }
__device__ __forceinline__ u64 ld64(const u64* p) { return *reinterpret_cast<const volatile u64*>(p); }

// The only way a group state changes: a compare-and-swap that must succeed. A failure means two owners of one group
// buffer (or a lost update of the bookkeeping) and is counted; the caller carries on, the audit reports it.
__device__ __forceinline__ bool transition(int* state, int from, int to, int* errors) {
  if (atomicCAS(state, from, to) == from) return true;
  atomicAdd(errors + ERR_STATE, 1);
  return false;
}
// One more of statistic k (cnt.stats). It saturates: see enum Stat. The test and the add are two steps, so the value
// can end a few counts (at most one per block) above STAT_MAX; it can never wrap.
__device__ __forceinline__ void stat_add(int* stats, int k) { if (ldi(stats + k) < STAT_MAX) atomicAdd(stats + k, 1); }
__host__ __device__ inline bool is_terminal(int s) { return s == CONSUMED || s == DROPPED_ZEROVAR || s == DROPPED_STALE || s == CANCELLED; }
}  // namespace rls

// ================================================================================================ host side
namespace rlc {
using mrl::u64; using mrl::bf16;

struct Sizes { int nbuf, nslots, narch; mk::u64 trace_cap, log_cap, tok_cap; };

inline void die(const std::string& m) { fprintf(stderr, "rlc: %s\n", m.c_str()); exit(1); }
inline unsigned pow2ceil(unsigned n) { unsigned c = 1; while (c < n) c <<= 1; return c; }

inline mrl::Queue alloc_queue(unsigned cap) {
  if (cap < 2 || (cap & (cap - 1))) die("queue capacity must be a power of two, at least 2 (rl_queue.cuh)");
  mrl::Queue q; q.seq = mk::dalloc<unsigned>(cap); q.val = mk::dalloc<int>(cap); q.head = mk::dalloc<unsigned>(1); q.tail = mk::dalloc<unsigned>(1); q.mask = cap - 1;
  return q;
}
// Set a queue to "holds `items` in this order, counters at position `base`" (see rl_queue.cuh for the cell encoding).
inline void set_queue(const mrl::Queue& q, const std::vector<int>& items, unsigned base = 0) {
  const unsigned cap = q.mask + 1, n = (unsigned)items.size();
  if (n > cap) die("set_queue: more items than the queue holds");
  std::vector<unsigned> seq(cap); std::vector<int> val(cap, -1);
  for (unsigned k = 0; k < cap; k++) {
    const unsigned pos = base + k, cell = pos & q.mask;
    if (k < n) { seq[cell] = pos + 1u; val[cell] = items[k]; } else seq[cell] = pos;
  }
  mk::up(q.seq, seq); mk::up(q.val, val);
  const std::vector<unsigned> h(1, base), t(1, base + n); mk::up(q.head, h); mk::up(q.tail, t);
}

// Allocate everything of mrl::Shared. Every queue (the NTASKS ready queues and the free list) holds pow2ceil(nbuf)
// entries, so none is ever full. (A push can still be refused for a moment on the free list, which has several
// consumers: rl_queue.cuh, ctl::release_buf.)
// narch = number of snapshots the archive keeps (0 = no archive); which versions go where is decided by T_FINISH from
// Versions::arch_stride, which the caller sets.
// The first tok_cap logged groups also keep their tokens and log-probabilities (tok_cap <= log_cap).
inline mrl::Shared host_alloc(const Sizes& z) {
  if (z.nbuf < 1 || z.nslots < 2 || z.narch < 0) die("host_alloc: need nbuf >= 1, nslots >= 2, narch >= 0");
  if (z.tok_cap > z.log_cap || z.tok_cap > (1ull << 22)) die("host_alloc: tok_cap must be <= log_cap and <= 2^22");
  mrl::Shared sh; memset(&sh, 0, sizeof sh);
  const unsigned cap = std::max(2u, pow2ceil((unsigned)z.nbuf));
  for (int t = 0; t < mrl::NTASKS; t++) sh.ready[t] = alloc_queue(cap);
  sh.freebuf = alloc_queue(cap);
  mrl::Versions& V = sh.ver; V.nslots = z.nslots; V.narch = z.narch;
  V.W = mk::dalloc<bf16>((size_t)z.nslots * mrl::MID_NW);
  V.cur = mk::dalloc<int>(1); V.ref = mk::dalloc<int>(z.nslots); V.gen = mk::dalloc<int>(z.nslots); V.id = mk::dalloc<int>(z.nslots); V.train_ver = mk::dalloc<int>(1);
  V.arch = z.narch ? mk::dalloc<bf16>((size_t)z.narch * mrl::MID_NW) : nullptr;
  V.arch_id = z.narch ? mk::dalloc<int>(z.narch) : nullptr;
  sh.trace.ev = mk::dalloc<mrl::TraceEvent>(z.trace_cap); sh.trace.n = mk::dalloc<u64>(1); sh.trace.cap = z.trace_cap;
  mrl::Counters& N = sh.cnt;
  N.admitted = mk::dalloc<int>(mrl::NTASKS); N.finished = mk::dalloc<int>(mrl::NTASKS); N.drop_zerovar = mk::dalloc<int>(mrl::NTASKS);
  N.drop_stale_inflight = mk::dalloc<int>(mrl::NTASKS); N.drop_stale_queued = mk::dalloc<int>(mrl::NTASKS); N.consumed = mk::dalloc<int>(mrl::NTASKS);
  N.inflight = mk::dalloc<int>(mrl::NTASKS); N.queued = mk::dalloc<int>(mrl::NTASKS); N.prompt_next = mk::dalloc<int>(mrl::NTASKS);
  N.errors = mk::dalloc<int>(rls::NERR); N.uid_next = mk::dalloc<int>(1); N.tokens_generated = mk::dalloc<u64>(1); N.stop_reason = mk::dalloc<int>(1); N.stats = mk::dalloc<int>(rls::NSTAT);
  N.held = mk::dalloc<int>(mrl::NTASKS); N.training = mk::dalloc<int>(1);
  mrl::GroupLog& L = sh.glog;
  L.e = mk::dalloc<mrl::GroupLogEntry>(z.log_cap); L.n = mk::dalloc<u64>(1); L.cap = z.log_cap; L.tok_cap = z.tok_cap;
  L.tokens = mk::dalloc<int>((size_t)z.tok_cap * mrl::GSZ * mrl::MAXGEN); L.logp = mk::dalloc<float>((size_t)z.tok_cap * mrl::GSZ * mrl::MAXGEN);
  return sh;
}

template <class X> inline void dzero(X* p, size_t n) { if (p && n) MK_CK(cudaMemset(p, 0, n * sizeof(X))); }

// State at launch: ready queues empty, every group buffer free (in index order), snapshot slot 0 = the learner's
// working copy tr.W as version 0 and current, archive entry 0 the same, train_ver 0, every counter zero (cnt.held and
// cnt.training included), the trace and the group log empty. Call it after tr.W holds the initial weights
// (mt::alloc_mid_state copies them from layer LMID).
// Not touched (not owned by the core): Pipe::grp and Pipe::row (must be all zero = every slot FREE, every row
// inactive), Train::batch, Versions::arch_stride, the job tickets, *finished, *abort, *t0, sm_count.
inline void host_reset(const mrl::Shared& sh, const mrl::Store& st, const mrl::Train& tr) {
  if ((unsigned)st.nbuf > sh.freebuf.mask + 1u) die("host_reset: the store has more buffers than the queues hold");
  for (int t = 0; t < mrl::NTASKS; t++) set_queue(sh.ready[t], {});
  std::vector<int> all(st.nbuf); for (int i = 0; i < st.nbuf; i++) all[i] = i;
  set_queue(sh.freebuf, all);
  dzero(st.rec, st.nbuf);
  const mrl::Versions& V = sh.ver;
  dzero(V.cur, 1); dzero(V.ref, V.nslots); dzero(V.gen, V.nslots); dzero(V.train_ver, 1);
  std::vector<int> ids(V.nslots, -1); ids[0] = 0; mk::up(V.id, ids);
  dzero(V.W, (size_t)V.nslots * mrl::MID_NW);
  MK_CK(cudaMemcpy(V.W, tr.W, (size_t)mrl::MID_NW * sizeof(bf16), cudaMemcpyDeviceToDevice));
  if (V.narch > 0) {
    dzero(V.arch, (size_t)V.narch * mrl::MID_NW);
    MK_CK(cudaMemcpy(V.arch, tr.W, (size_t)mrl::MID_NW * sizeof(bf16), cudaMemcpyDeviceToDevice));
    std::vector<int> aids(V.narch, -1); aids[0] = 0; mk::up(V.arch_id, aids);
  }
  dzero(sh.trace.n, 1); dzero(sh.trace.ev, sh.trace.cap);
  dzero(sh.glog.n, 1); dzero(sh.glog.e, sh.glog.cap);
  dzero(sh.glog.tokens, (size_t)sh.glog.tok_cap * mrl::GSZ * mrl::MAXGEN); dzero(sh.glog.logp, (size_t)sh.glog.tok_cap * mrl::GSZ * mrl::MAXGEN);
  const mrl::Counters& N = sh.cnt;
  for (int* p : {N.admitted, N.finished, N.drop_zerovar, N.drop_stale_inflight, N.drop_stale_queued, N.consumed, N.inflight, N.queued, N.prompt_next}) dzero(p, mrl::NTASKS);
  dzero(N.errors, rls::NERR); dzero(N.stats, rls::NSTAT); dzero(N.uid_next, 1); dzero(N.tokens_generated, 1); dzero(N.stop_reason, 1);
  dzero(N.held, mrl::NTASKS); dzero(N.training, 1);
}

// ---------------------------------------------------------------- what a run left behind
struct QueueCopy {
  unsigned head = 0, tail = 0, mask = 0; std::vector<unsigned> seq; std::vector<int> val;
  int size() const { return (int)(tail - head); }
  std::vector<int> items() const { std::vector<int> v; if (size() >= 0 && (unsigned)size() <= mask + 1u) for (unsigned p = head; p != tail; p++) v.push_back(val[p & mask]); return v; }
};
inline QueueCopy download_queue(const mrl::Queue& q) {
  QueueCopy c; c.mask = q.mask; c.seq = mk::down(q.seq, (size_t)q.mask + 1); c.val = mk::down(q.val, (size_t)q.mask + 1);
  c.head = mk::down(q.head, 1)[0]; c.tail = mk::down(q.tail, 1)[0];
  return c;
}

// Host copy of everything in mrl::Shared after the kernel has exited. `log` and `trace` are exactly the records of
// groups.bin (mrl::GroupLogEntry, 104 bytes) and trace.bin (mrl::TraceEvent, 32 bytes), `log_tokens` / `log_logp`
// the contents of group_tokens.i32 / group_logp.f32, `arch` / `cur_w` those of mid_arch.bf16 / mid_cur.bf16.
struct Snapshot {
  std::vector<int> admitted, finished, drop_zerovar, drop_stale_inflight, drop_stale_queued, consumed, inflight, queued, prompt_next;   // [NTASKS]
  std::vector<int> errors, stats;                  // [16] each
  int uid_next = 0, train_ver = 0; u64 tokens_generated = 0;
  int stop_reason = 0;                             // why the run ended (rls::Stop), as megarl.cu prints it and writes it to run.json:
                                                   // the reason the device recorded or, if the launch ended without one, DEADLINE
                                                   // (resolve_stop_reason says why that is the only possibility)
  int stop_reason_dev = 0;                         // cnt.stop_reason as the device left it (0: none). The trace holds a STOP event
                                                   // exactly if this is not 0, and that event carries this value
  int held = 0;                                    // groups in state TRAINING at exit, from the counters: sum over tasks of
                                                   // finished - drop_zerovar - drop_stale_queued - consumed - queued (the audit checks it against the buffers)
  std::vector<int> held_t;                         // [NTASKS] cnt.held as the device left it: groups the learner took and has not released
                                                   // (the audit checks it against the buffers in state TRAINING and against `held`)
  int training = 0;                                // cnt.training as the device left it: bit 0 = the learner holds a complete batch;
                                                   // bit rls::TURN_SHIFT + j = rollout job j held the generation turn (alternate only)
  u64 log_total = 0, trace_total = 0;              // appended in all (the vectors hold min(total, capacity))
  std::vector<mrl::GroupLogEntry> log; int groups_with_tokens = 0; std::vector<int> log_tokens; std::vector<float> log_logp;
  std::vector<mrl::TraceEvent> trace;
  int nslots = 0, cur = 0, narch = 0, versions = 0; std::vector<int> ref, gen, id;
  std::vector<bf16> arch;                          // [versions][MID_NW] archived snapshots (unused entries are zero)
  std::vector<bf16> cur_w;                         // [MID_NW] the current snapshot
  QueueCopy ready[mrl::NTASKS], freebuf;
};

// The stop reason of a launch that has ended, from what the device left (dev = cnt.stop_reason, events = how many
// trace events were emitted). The device records a reason when a control task ends the run (DONE, STARVED, DEADLINE,
// ERROR) or notices that someone else has (ctl::note_abort). One way of ending leaves none: the scheduler's watchdog.
// A block of mk::wave_loop that waits for work raises *abort once the kernel has run past the job deadline
// (hp.deadline_ns + WATCHDOG_NS, megarl.cu), and it knows nothing of the counter; after *abort no block claims another
// task, so unless a control task happens to be polling at that moment, no code of the core runs again.
// Nothing else ends a launch without a reason:
//   * a block leaves mk::wave_loop only on *abort or when every job has ended (the rollout jobs on *finished, the
//     learner after its last update);
//   * *finished is set by the last T_FINISH, which records DONE unless a reason is there already or its step did not
//     publish, and a step does not publish only after a recorded stop or *abort;
//   * *abort is raised by ctl::stop_run after it has recorded its reason, by the watchdog, and by the scheduler's guard
//     against its 29-bit step field wrapping (out of reach in practice).
// Hence "the launch ended and no reason was recorded" = "the watchdog ended it": DEADLINE.
// That the launch ran at all is read off the trace: a control task that gets as far as its first decision emits an
// event (ADMIT, IDLE or STOP). With no event the answer stays STOP_NONE (nothing ran).
inline int resolve_stop_reason(int dev, u64 events) { return dev != rls::STOP_NONE ? dev : (events > 0 ? (int)rls::DEADLINE : (int)rls::STOP_NONE); }

// Call it after the launch has ended (cudaDeviceSynchronize): stop_reason is resolved on that premise.
inline Snapshot host_download(const mrl::Shared& sh) {
  Snapshot s; const mrl::Counters& N = sh.cnt;
  s.admitted = mk::down(N.admitted, mrl::NTASKS); s.finished = mk::down(N.finished, mrl::NTASKS); s.drop_zerovar = mk::down(N.drop_zerovar, mrl::NTASKS);
  s.drop_stale_inflight = mk::down(N.drop_stale_inflight, mrl::NTASKS); s.drop_stale_queued = mk::down(N.drop_stale_queued, mrl::NTASKS); s.consumed = mk::down(N.consumed, mrl::NTASKS);
  s.inflight = mk::down(N.inflight, mrl::NTASKS); s.queued = mk::down(N.queued, mrl::NTASKS); s.prompt_next = mk::down(N.prompt_next, mrl::NTASKS);
  s.errors = mk::down(N.errors, rls::NERR); s.stats = mk::down(N.stats, rls::NSTAT);
  s.uid_next = mk::down(N.uid_next, 1)[0]; s.stop_reason_dev = mk::down(N.stop_reason, 1)[0]; s.tokens_generated = mk::down(N.tokens_generated, 1)[0];
  s.held_t = mk::down(N.held, mrl::NTASKS); s.training = mk::down(N.training, 1)[0];
  for (int t = 0; t < mrl::NTASKS; t++) s.held += s.finished[t] - s.drop_zerovar[t] - s.drop_stale_queued[t] - s.consumed[t] - s.queued[t];
  const mrl::Versions& V = sh.ver;
  s.train_ver = mk::down(V.train_ver, 1)[0]; s.nslots = V.nslots; s.cur = mk::down(V.cur, 1)[0]; s.narch = V.narch;
  s.ref = mk::down(V.ref, V.nslots); s.gen = mk::down(V.gen, V.nslots); s.id = mk::down(V.id, V.nslots);
  s.versions = V.narch;
  if (V.narch > 0 && V.arch) s.arch = mk::down(V.arch, (size_t)V.narch * mrl::MID_NW);
  if (s.cur >= 0 && s.cur < V.nslots) s.cur_w = mk::down(V.W + (size_t)s.cur * mrl::MID_NW, mrl::MID_NW);
  s.log_total = mk::down(sh.glog.n, 1)[0]; s.trace_total = mk::down(sh.trace.n, 1)[0];
  s.stop_reason = resolve_stop_reason(s.stop_reason_dev, s.trace_total);
  s.log = mk::down(sh.glog.e, (size_t)std::min<u64>(s.log_total, sh.glog.cap));
  s.groups_with_tokens = (int)std::min<u64>(s.log.size(), sh.glog.tok_cap);
  s.log_tokens = mk::down(sh.glog.tokens, (size_t)s.groups_with_tokens * mrl::GSZ * mrl::MAXGEN); s.log_logp = mk::down(sh.glog.logp, (size_t)s.groups_with_tokens * mrl::GSZ * mrl::MAXGEN);
  s.trace = mk::down(sh.trace.ev, (size_t)std::min<u64>(s.trace_total, sh.trace.cap));
  for (int t = 0; t < mrl::NTASKS; t++) s.ready[t] = download_queue(sh.ready[t]);
  s.freebuf = download_queue(sh.freebuf);
  return s;
}
}  // namespace rlc

// ================================================================================================ the audit
namespace rlc {

// What host_audit found, by kind. Each number counts violated checks.
struct AuditReport {
  int errors = 0;        // an entry of cnt.errors is not zero
  int counters = 0;      // the counter equations (admitted, finished, held, cnt.training), totals per update and per terminal state
  int conservation = 0;  // free + in flight + queued + held = nbuf, and the census of buffer states at exit
  int queues = 0;        // queue structure and contents against the buffer states at exit
  int exactly_once = 0;  // every admitted uid ended exactly once, or is alive in exactly one buffer
  int terminal = 0;      // every terminal record is a legal one
  int lag = 0;           // lag value against train_step, lag bound, nothing dropped as stale that was not
  int fifo = 0;          // first-in first-out per environment among the groups the learner took
  int versions = 0;      // the snapshot table at exit
  int trace = 0;         // event sequence of every group and of every update
  int incomplete = 0;    // the group log or the trace overflowed: part of the audit could not be done
  std::vector<std::string> msgs;
  int total() const { return errors + counters + conservation + queues + exactly_once + terminal + lag + fifo + versions + trace + incomplete; }
};

// Audit of a finished run: every invariant of the core that can be checked from what the kernel left in device
// memory (counters, buffer conservation, queues, exactly-once, terminal records, lag, FIFO, snapshot table, trace).
// Returns the number of violated checks (0 = clean); *out gets them by kind.
// The log and the trace must not have overflowed (that alone counts as a violation). Enough is log_cap >= groups
// admitted and trace_cap >= 4 events per group + 3 per update + 2 per idle period + 1.
inline int host_audit(const mrl::Shared& sh, const mrl::Store& st, const Snapshot& s, const mrl::Hyper& hp, bool verbose, AuditReport* out = nullptr) {
  using namespace rls;
  AuditReport R; const int NT = mrl::NTASKS, G = mrl::GSZ, nbuf = st.nbuf, U = s.uid_next;
  auto fail = [&](int& cat, const char* name, const char* fmt, auto... a) {
    char b[512]; snprintf(b, sizeof b, fmt, a...);
    cat++; if (R.msgs.size() < 400) R.msgs.push_back(std::string(name) + ": " + b);
    if (verbose && cat <= 6) printf("    audit violation [%s] %s%s\n", name, b, cat == 6 ? "  (further ones of this kind are only counted)" : "");
  };
  const std::vector<mrl::GroupRec> rec = mk::down(st.rec, nbuf);
  auto task_ok = [&](int t) { return t >= 0 && t < NT; };
  auto alive = [](int stt) { return stt == GENERATING || stt == READY || stt == TRAINING; };

  // ---- census of buffer states at exit
  int nstate[NSTATES] = {0}; std::vector<int> gen_t(NT, 0), ready_t(NT, 0), train_t(NT, 0);
  for (int b = 0; b < nbuf; b++) {
    const int stt = rec[b].state;
    if (stt < 0 || stt >= NSTATES) { fail(R.conservation, "state", "buffer %d has state %d", b, stt); continue; }
    nstate[stt]++;
    if (stt == SCORING || is_terminal(stt)) fail(R.conservation, "state", "buffer %d was left in the transient state %d", b, stt);
    if (alive(stt)) {
      if (!task_ok(rec[b].task)) { fail(R.conservation, "state", "buffer %d (state %d) has task %d", b, stt, rec[b].task); continue; }
      (stt == GENERATING ? gen_t : stt == READY ? ready_t : train_t)[rec[b].task]++;
    }
  }
  // ---- (1) no error counter set
  static const char* ename[NERR] = {"ERR_STATE", "ERR_QUEUE_FULL", "ERR_SNAPSHOT", "ERR_POP", "ERR_LAG", "ERR_BUF", "ERR_PROMPT",
                                    "ERR_NONFINITE: an update with a gradient norm that is not finite was not published", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                    "ERR_TRUNK_ROW (qw::ERR_ROW): the trunk was fed a row outside its contract", "ERR_HEAD_NO_WINNER (hd::ERR_NO_WINNER): the head had a row with no token to sample"};
  for (int i = 0; i < NERR; i++) if (s.errors[i]) fail(R.errors, "errors", "errors[%d] (%s) = %d", i, ename[i] ? ename[i] : "?", s.errors[i]);
  // ---- (1b) the stop reason: the device's word is one of rls::Stop, and what is reported is that word or, if the
  //      device left none, what resolve_stop_reason makes of it (the STOP event is checked with the trace below)
  if (s.stop_reason_dev < STOP_NONE || s.stop_reason_dev > ERROR) fail(R.counters, "stop", "cnt.stop_reason = %d is not a stop reason", s.stop_reason_dev);
  if (s.stop_reason != resolve_stop_reason(s.stop_reason_dev, s.trace_total))
    fail(R.counters, "stop", "stop reason %d is reported, but the device recorded %d and emitted %llu events: that makes %d", s.stop_reason, s.stop_reason_dev, s.trace_total, resolve_stop_reason(s.stop_reason_dev, s.trace_total));
  // ---- (2) counter equations
  long long adm = 0, inf = 0, que = 0; const int held = nstate[TRAINING];
  for (int t = 0; t < NT; t++) {
    if (s.admitted[t] != s.finished[t] + s.drop_stale_inflight[t] + s.inflight[t])
      fail(R.counters, "admitted", "task %d: admitted %d != finished %d + drop_stale_inflight %d + inflight %d", t, s.admitted[t], s.finished[t], s.drop_stale_inflight[t], s.inflight[t]);
    if (s.finished[t] != s.drop_zerovar[t] + s.drop_stale_queued[t] + s.consumed[t] + s.queued[t] + train_t[t])
      fail(R.counters, "finished", "task %d: finished %d != drop_zerovar %d + drop_stale_queued %d + consumed %d + queued %d + held %d", t, s.finished[t], s.drop_zerovar[t], s.drop_stale_queued[t], s.consumed[t], s.queued[t], train_t[t]);
    if (s.inflight[t] < 0 || s.queued[t] < 0) fail(R.counters, "negative", "task %d: inflight %d, queued %d", t, s.inflight[t], s.queued[t]);
    adm += s.admitted[t]; inf += s.inflight[t]; que += s.queued[t];
  }
  if (adm != U) fail(R.counters, "uids", "admitted %lld groups but handed out %d uids", adm, U);
  // ---- (2b) the learner's own counters. cnt.held[t] = groups of environment t in state TRAINING (T_SELECT raises it for a
  //      group it keeps, T_FINISH lowers it when it has released the group's buffer). The batch bit of cnt.training is set
  //      exactly while a complete batch is held: T_SELECT sets it with the last group of a batch and T_FINISH clears it
  //      after the last release, both within one task, so at exit "set" and "TGRP groups held" are the same thing. The
  //      other bits of the word are the generation turns of hp.alternate: at most bits TURN_SHIFT .. + MAXPIPE - 1 (a
  //      job that was inside a step when the run ended leaves its bit set), and none without the option.
  if ((int)s.held_t.size() != NT) fail(R.counters, "held", "the snapshot has %zu values of cnt.held, not %d", s.held_t.size(), NT);
  else {
    long long hsum = 0;
    for (int t = 0; t < NT; t++) {
      hsum += s.held_t[t];
      if (s.held_t[t] != train_t[t]) fail(R.counters, "held", "task %d: cnt.held = %d but %d buffers are in state TRAINING", t, s.held_t[t], train_t[t]);
    }
    if (hsum != s.held) fail(R.counters, "held", "cnt.held sums to %lld but finished - (dropped + consumed + queued) = %d over the tasks", hsum, s.held);
  }
  { const int batch = s.training & TRAINING_BATCH, turns = s.training >> TURN_SHIFT;
    if (batch != (held == mrl::TGRP ? 1 : 0)) fail(R.counters, "training", "cnt.training says the learner %s a complete batch, but %d of %d groups are in state TRAINING", batch ? "holds" : "does not hold", held, mrl::TGRP);
    if ((s.training & ~(TRAINING_BATCH | (((1 << mrl::MAXPIPE) - 1) << TURN_SHIFT))) != 0 || s.training < 0) fail(R.counters, "training", "cnt.training = %#x has bits that nobody sets", s.training);
    else if (turns && !hp.alternate) fail(R.counters, "training", "cnt.training = %#x carries generation turns although hp.alternate is 0", s.training); }
  // ---- (3) buffer conservation
  const int nfree = s.freebuf.size();
  if (nfree + inf + que + held != nbuf) fail(R.conservation, "buffers", "free %d + inflight %lld + queued %lld + held %d != nbuf %d", nfree, inf, que, held, nbuf);
  if (nstate[FREE] != nfree) fail(R.conservation, "buffers", "%d buffers are in state FREE but the free list holds %d", nstate[FREE], nfree);
  for (int t = 0; t < NT; t++) {
    if (gen_t[t] != s.inflight[t]) fail(R.conservation, "buffers", "task %d: %d buffers GENERATING but inflight = %d", t, gen_t[t], s.inflight[t]);
    if (ready_t[t] != s.queued[t]) fail(R.conservation, "buffers", "task %d: %d buffers READY but queued = %d", t, ready_t[t], s.queued[t]);
  }
  // ---- (4) queues at exit
  std::vector<int> inq(nbuf, 0);
  auto check_queue = [&](const QueueCopy& q, const char* name, int want_state, int want_task) {
    const unsigned cap = q.mask + 1u; const int n = q.size();
    if (n < 0 || (unsigned)n > cap) { fail(R.queues, name, "head %u, tail %u, capacity %u", q.head, q.tail, cap); return; }
    for (unsigned k = 0; k < cap; k++) {                     // rl_queue.cuh: a held cell has seq = position + 1, a free one its next position
      const unsigned pos = q.head + k, cell = pos & q.mask, want = k < (unsigned)n ? pos + 1u : pos;
      if (q.seq[cell] != want) fail(R.queues, name, "cell %u has sequence number %u, expected %u", cell, q.seq[cell], want);
    }
    for (int b : q.items()) {
      if (b < 0 || b >= nbuf) { fail(R.queues, name, "holds %d, which is not a buffer", b); continue; }
      if (++inq[b] > 1) fail(R.queues, name, "buffer %d is queued more than once", b);
      if (rec[b].state != want_state) fail(R.queues, name, "holds buffer %d, which is in state %d", b, rec[b].state);
      else if (want_task >= 0 && rec[b].task != want_task) fail(R.queues, name, "holds buffer %d of task %d", b, rec[b].task);
    }
  };
  check_queue(s.freebuf, "freebuf", FREE, -1);
  for (int t = 0; t < NT; t++) {
    const std::string nm = "ready[" + std::to_string(t) + "]"; check_queue(s.ready[t], nm.c_str(), READY, t);
    if (s.ready[t].size() != s.queued[t]) fail(R.queues, nm.c_str(), "holds %d groups but queued = %d", s.ready[t].size(), s.queued[t]);
  }
  for (int b = 0; b < nbuf; b++) if ((rec[b].state == FREE || rec[b].state == READY) && inq[b] == 0) fail(R.queues, "lost", "buffer %d is in state %d but in no queue", b, rec[b].state);

  // ---- (5) the group log: legal terminal records, exactly once per uid, lag
  const bool log_complete = s.log_total == (u64)s.log.size();
  if (!log_complete) fail(R.incomplete, "log", "%llu groups ended but the log holds %zu: enlarge log_cap", s.log_total, s.log.size());
  std::vector<char> seen(std::max(U, 0), 0); std::vector<long long> nterm((size_t)NT * NSTATES, 0);
  std::vector<int> step_cnt((size_t)std::max(s.train_ver, 0) * NT, 0); long long tok_sum = 0;
  for (size_t n = 0; n < s.log.size(); n++) {
    const mrl::GroupLogEntry& e = s.log[n];
    if (e.uid < 0 || e.uid >= U) { fail(R.exactly_once, "uid", "log record %zu has uid %d (uids handed out: %d)", n, e.uid, U); continue; }
    if (seen[e.uid]++) fail(R.exactly_once, "uid", "uid %d ended more than once (log record %zu)", e.uid, n);
    if (!task_ok(e.task) || !is_terminal(e.terminal)) { fail(R.terminal, "record", "uid %d: task %d, terminal %d", e.uid, e.task, e.terminal); continue; }
    nterm[(size_t)e.task * NSTATES + e.terminal]++;
    if (e.ver_id < 0 || e.ver_id > s.train_ver) fail(R.terminal, "record", "uid %d was generated with version %d (train_ver at exit %d)", e.uid, e.ver_id, s.train_ver);
    bool same = true; float mean = 0.f;
    for (int j = 0; j < G; j++) { same = same && e.reward[j] == e.reward[0]; mean += e.reward[j] / G; tok_sum += e.len[j]; }
    const bool can = e.terminal == CANCELLED;
    for (int j = 0; j < G; j++) if (e.len[j] < (can ? 0 : 1) || e.len[j] > mrl::MAXGEN) fail(R.terminal, "len", "uid %d (terminal %d): len[%d] = %d", e.uid, e.terminal, j, e.len[j]);
    if (e.terminal != CONSUMED && (e.train_step != -1 || e.lag != -1)) fail(R.terminal, "record", "uid %d (terminal %d) carries train_step %d, lag %d", e.uid, e.terminal, e.train_step, e.lag);
    if (can) {
      for (int j = 0; j < G; j++) if (e.reward[j] != 0.f || e.adv[j] != 0.f) fail(R.terminal, "cancelled", "uid %d has reward %g, advantage %g", e.uid, e.reward[j], e.adv[j]);
      if (e.t_done != 0 || e.t_admit > e.t_end) fail(R.terminal, "times", "cancelled uid %d: admit %llu, done %llu, end %llu", e.uid, e.t_admit, e.t_done, e.t_end);
    } else {
      if (e.t_admit > e.t_done || e.t_done > e.t_end) fail(R.terminal, "times", "uid %d: admit %llu, done %llu, end %llu", e.uid, e.t_admit, e.t_done, e.t_end);
      if ((e.terminal == DROPPED_ZEROVAR) != same) fail(R.terminal, "rewards", "uid %d ended with terminal %d but its rewards are %g %g %g %g", e.uid, e.terminal, e.reward[0], e.reward[1], e.reward[2], e.reward[3]);
      // advantages = (r - mean) / (unbiased std + eps), zero for equal rewards. The device evaluates this in float64 and
      // rounds once to float32 (gr::advantages), so it is within about 6e-8 relative of the value here: 1e-5 of the
      // largest advantage is a wide margin and still far below what a wrong formula gives (biased std: 15%; missing
      // eps: 1e-4 / std >= 1.7e-4).
      double ss = 0; for (int j = 0; j < G; j++) ss += ((double)e.reward[j] - mean) * ((double)e.reward[j] - mean);
      const double sd = sqrt(ss / (G - 1)); double amax = 0, err = 0;
      for (int j = 0; j < G; j++) { const double want = same ? 0.0 : ((double)e.reward[j] - mean) / (sd + (double)hp.adv_eps); amax = std::max(amax, fabs(want)); err = std::max(err, fabs((double)e.adv[j] - want)); }
      if (!(err <= 1e-5 * amax)) fail(R.terminal, "advantage", "uid %d: advantages %g %g %g %g are off by %.3g from (r - mean) / (std + eps)", e.uid, e.adv[0], e.adv[1], e.adv[2], e.adv[3], err);
    }
    if (e.terminal == CONSUMED) {
      if (e.train_step < 0 || e.train_step >= s.train_ver) fail(R.lag, "train_step", "uid %d was consumed by update %d but %d updates were made", e.uid, e.train_step, s.train_ver);
      else step_cnt[(size_t)e.train_step * NT + e.task]++;
      if (e.lag != e.train_step - e.ver_id) fail(R.lag, "lag", "uid %d: lag %d but update %d - version %d = %d", e.uid, e.lag, e.train_step, e.ver_id, e.train_step - e.ver_id);
      if (e.lag < 0 || e.lag > hp.lag) fail(R.lag, "lag", "uid %d was trained on with lag %d (limit %d)", e.uid, e.lag, hp.lag);
    }
  }
  for (int b = 0; b < nbuf; b++) if (alive(rec[b].state)) {
    const int u = rec[b].uid;
    if (u < 0 || u >= U) { fail(R.exactly_once, "uid", "buffer %d holds uid %d (uids handed out: %d)", b, u, U); continue; }
    if (seen[u]++) fail(R.exactly_once, "uid", "uid %d is alive in buffer %d but has also ended or lives in a second buffer", u, b);
    if (rec[b].state != GENERATING) for (int j = 0; j < G; j++) tok_sum += rec[b].len[j];
  }
  if (log_complete) {
    for (int u = 0; u < U; u++) if (!seen[u]) fail(R.exactly_once, "uid", "uid %d was admitted but has neither ended nor is it alive", u);
    for (int t = 0; t < NT; t++) {
      const int want[4] = {s.consumed[t], s.drop_zerovar[t], s.drop_stale_queued[t], s.drop_stale_inflight[t]}; const int term[4] = {CONSUMED, DROPPED_ZEROVAR, DROPPED_STALE, CANCELLED};
      for (int k = 0; k < 4; k++) if (nterm[(size_t)t * NSTATES + term[k]] != want[k]) fail(R.counters, "log", "task %d: %lld log records with terminal %d but the counter says %d", t, nterm[(size_t)t * NSTATES + term[k]], term[k], want[k]);
    }
    for (int k = 0; k < s.train_ver; k++) {
      int tot = 0; for (int t = 0; t < NT; t++) tot += step_cnt[(size_t)k * NT + t];
      if (tot != mrl::TGRP) fail(R.counters, "update", "update %d consumed %d groups, not %d", k, tot, mrl::TGRP);
      else if (hp.pair_mode == 0) for (int t = 0; t < NT; t++) if (step_cnt[(size_t)k * NT + t] != mrl::TGRP / NT) fail(R.counters, "update", "update %d took %d groups of task %d (pair_mode 0 takes %d of each)", k, step_cnt[(size_t)k * NT + t], t, mrl::TGRP / NT);
    }
    // tokens: all of ended groups and of finished live ones are known; a generating group adds at most GSZ * MAXGEN
    if ((long long)s.tokens_generated < tok_sum || (long long)s.tokens_generated > tok_sum + (long long)nstate[GENERATING] * G * mrl::MAXGEN)
      fail(R.counters, "tokens", "tokens_generated = %llu, but ended and finished groups hold %lld tokens and %d groups are still generating", s.tokens_generated, tok_sum, nstate[GENERATING]);
  }

  // ---- (6) the snapshot table at exit
  if (s.cur < 0 || s.cur >= s.nslots) fail(R.versions, "cur", "current slot %d of %d", s.cur, s.nslots);
  else {
    if (s.id[s.cur] != s.train_ver) fail(R.versions, "cur", "the current slot %d holds version %d but train_ver = %d", s.cur, s.id[s.cur], s.train_ver);
    long long refs = 0;
    for (int v = 0; v < s.nslots; v++) {
      if (s.gen[v] & 1) fail(R.versions, "gen", "slot %d was left half written (generation %d)", v, s.gen[v]);
      int want = 0; for (int b = 0; b < nbuf; b++) if (rec[b].state == GENERATING && s.id[v] >= 0 && rec[b].ver_id == s.id[v]) want++;
      if (s.ref[v] != want) fail(R.versions, "ref", "slot %d (version %d) has reference count %d, but %d generating groups use that version", v, s.id[v], s.ref[v], want);
      refs += s.ref[v];
      for (int w = 0; w < v; w++) if (s.id[v] >= 0 && s.id[v] == s.id[w]) fail(R.versions, "id", "slots %d and %d both hold version %d", w, v, s.id[v]);
    }
    if (refs != nstate[GENERATING]) fail(R.versions, "ref", "%lld pins are held but %d groups are generating", refs, nstate[GENERATING]);
  }

  // ---- (7) the trace: life of every group, sequence of updates, stale drops, first-in first-out
  const bool trace_complete = s.trace_total == (u64)s.trace.size();
  if (!trace_complete) fail(R.incomplete, "trace", "%llu events were emitted but the trace holds %zu: enlarge trace_cap", s.trace_total, s.trace.size());
  else {
    const std::vector<mrl::TraceEvent>& T = s.trace; const size_t Un = (size_t)std::max(U, 0);
    std::vector<int> e_adm(Un, -1), e_done(Un, -1), e_enq(Un, -1), e_zv(Un, -1), e_can(Un, -1), e_stale(Un, -1), e_sel(Un, -1);
    std::vector<std::vector<int>> takes(NT); int npub = 0, nstep = 0, nstop = 0;
    auto put = [&](std::vector<int>& v, int uid, int i, const char* what) {
      if (uid < 0 || uid >= U) { fail(R.trace, what, "event %d names uid %d", i, uid); return false; }
      if (v[uid] >= 0) { fail(R.trace, what, "uid %d has two such events (%d and %d)", uid, v[uid], i); return false; }
      v[uid] = i; return true;
    };
    for (int i = 0; i < (int)T.size(); i++) {
      const mrl::TraceEvent& ev = T[i];
      switch (ev.type) {
        case EV_ADMIT: put(e_adm, ev.a, i, "ADMIT"); break;
        case EV_GROUP_DONE: put(e_done, ev.a, i, "GROUP_DONE"); break;
        case EV_ENQUEUE: put(e_enq, ev.a, i, "ENQUEUE"); break;
        case EV_DROP_ZEROVAR: put(e_zv, ev.a, i, "DROP_ZEROVAR"); break;
        case EV_CANCEL: if (put(e_can, ev.a, i, "CANCEL") && ev.d - ev.e <= hp.lag) fail(R.lag, "CANCEL", "uid %d (version %d) was cancelled at train_ver %d: lag %d is within the limit %d", ev.a, ev.e, ev.d, ev.d - ev.e, hp.lag); break;
        case EV_DROP_STALE:
          if (put(e_stale, ev.a, i, "DROP_STALE")) {
            if (task_ok(ev.b)) takes[ev.b].push_back(ev.a);
            if (ev.d - ev.e <= hp.lag) fail(R.lag, "DROP_STALE", "uid %d (version %d) was dropped at train_ver %d: lag %d is within the limit %d", ev.a, ev.e, ev.d, ev.d - ev.e, hp.lag);
          }
          break;
        case EV_SELECT: if (put(e_sel, ev.b, i, "SELECT") && task_ok(ev.c)) takes[ev.c].push_back(ev.b); break;
        case EV_PUBLISH: npub++; if (ev.a != npub || ev.c != npub - 1) fail(R.trace, "PUBLISH", "publication number %d is version %d of update %d", npub, ev.a, ev.c); break;
        case EV_STEP: if (ev.a != nstep) fail(R.trace, "STEP", "step record number %d is of update %d", nstep, ev.a); nstep++; break;
        case EV_STOP: nstop++; if (ev.a != s.stop_reason_dev) fail(R.trace, "STOP", "STOP event with reason %d but cnt.stop_reason = %d", ev.a, s.stop_reason_dev); break;
        case EV_IDLE: case EV_SELECT_WAIT: break;
        default: fail(R.trace, "type", "event %d has type %d", i, ev.type);
      }
    }
    if (npub != s.train_ver || nstep != s.train_ver) fail(R.trace, "updates", "%d PUBLISH and %d STEP events but train_ver = %d", npub, nstep, s.train_ver);
    // whoever records the reason on the device emits the one STOP event; a reason that only the host could fill in
    // (resolve_stop_reason) has none
    if (nstop != (s.stop_reason_dev != STOP_NONE ? 1 : 0)) fail(R.trace, "STOP", "%d STOP events, cnt.stop_reason = %d", nstop, s.stop_reason_dev);
    for (int u = 0; u < U; u++) {
      const int a = e_adm[u], d = e_done[u], q = e_enq[u], z = e_zv[u], c = e_can[u], x = e_stale[u], y = e_sel[u];
      if (a < 0) { fail(R.trace, "life", "uid %d has no ADMIT event", u); continue; }
      bool ok = true;
      if (c >= 0) ok = d < 0 && q < 0 && z < 0 && x < 0 && y < 0 && T[c].t >= T[a].t;                // cancelled while generating: nothing else
      else {
        if (d < 0) ok = q < 0 && z < 0 && x < 0 && y < 0;                                             // still generating
        else ok = T[d].t >= T[a].t && ((q >= 0) != (z >= 0)) && (q < 0 || T[q].t >= T[d].t) && (z < 0 || T[z].t >= T[d].t);   // finished: enqueued or dropped
        // Taken by the learner: once, and not before the group had finished. Not necessarily after its ENQUEUE event:
        // R_ADVANCE traces ENQUEUE after the push (the event carries the queue length after it), so the learner, which
        // is another block, can pop the group and trace its SELECT / DROP_STALE first, with an earlier time stamp.
        // What the code orders is GROUP_DONE (traced before the push) before the take (traced after the pop).
        if (ok && (x >= 0 || y >= 0)) ok = q >= 0 && !(x >= 0 && y >= 0) && T[x >= 0 ? x : y].t >= T[d].t;
      }
      if (!ok) fail(R.trace, "life", "uid %d: illegal event sequence (ADMIT %d, GROUP_DONE %d, ENQUEUE %d, DROP_ZEROVAR %d, CANCEL %d, DROP_STALE %d, SELECT %d)", u, a, d, q, z, c, x, y);
    }
    for (const mrl::GroupLogEntry& e : s.log) {            // the log and the trace tell the same story
      if (e.uid < 0 || e.uid >= U || e_adm[e.uid] < 0) continue;
      const mrl::TraceEvent& A = T[e_adm[e.uid]];
      if (A.b != e.task || A.c != e.prompt || A.d != e.ver_id) fail(R.trace, "ADMIT", "uid %d: admitted as task %d prompt %d version %d, logged as task %d prompt %d version %d", e.uid, A.b, A.c, A.d, e.task, e.prompt, e.ver_id);
      const int want = e.terminal == CONSUMED ? e_sel[e.uid] : e.terminal == DROPPED_ZEROVAR ? e_zv[e.uid] : e.terminal == DROPPED_STALE ? e_stale[e.uid] : e_can[e.uid];
      if (want < 0) fail(R.trace, "terminal", "uid %d ended with terminal %d but the matching event is missing", e.uid, e.terminal);
      else if (e.terminal == CONSUMED && (T[want].a != e.train_step || T[want].d != e.lag)) fail(R.trace, "SELECT", "uid %d: selected for update %d with lag %d, logged with update %d, lag %d", e.uid, T[want].a, T[want].d, e.train_step, e.lag);
    }
    // First-in first-out per environment. The order the learner took groups in is exact (its SELECT / DROP_STALE
    // events come from one thread at a time), followed by what is still queued, in queue order. The order of the
    // pushes is known exactly for two groups of the same pipeline (one thread pushes them: order of their ENQUEUE
    // events) and otherwise only when the pushes did not overlap: a push happens between the group's GROUP_DONE
    // and ENQUEUE events. So: no group may be taken after one whose push certainly came later.
    for (int t = 0; t < NT; t++) {
      std::vector<int> seq = takes[t];
      for (int b : s.ready[t].items()) if (b >= 0 && b < nbuf && rec[b].uid >= 0 && rec[b].uid < U) seq.push_back(rec[b].uid);
      std::vector<int> ok_seq; for (int u : seq) if (e_done[u] >= 0 && e_enq[u] >= 0) ok_seq.push_back(u);      // the others are reported above
      const int n = (int)ok_seq.size(); std::vector<u64> smin(n + 1, ~0ull); std::vector<int> sarg(n + 1, -1);
      for (int i = n - 1; i >= 0; i--) { const u64 te = T[e_enq[ok_seq[i]]].t; smin[i] = smin[i + 1]; sarg[i] = sarg[i + 1]; if (te <= smin[i]) { smin[i] = te; sarg[i] = ok_seq[i]; } }
      std::map<int, int> last;                               // pipeline -> ENQUEUE event of its last group in the sequence
      for (int i = 0; i < n; i++) {
        const int u = ok_seq[i];
        if (smin[i + 1] < T[e_done[u]].t) fail(R.fifo, "FIFO", "task %d: uid %d was taken before uid %d, which had been enqueued (t = %llu) before uid %d even finished (t = %llu)", t, u, sarg[i + 1], smin[i + 1], u, T[e_done[u]].t);
        const int pipe = T[e_done[u]].e >> 8; auto it = last.find(pipe);
        if (it != last.end() && it->second > e_enq[u]) fail(R.fifo, "FIFO", "task %d: uid %d was taken after a group that the same pipeline %d enqueued later", t, u, pipe);
        last[pipe] = e_enq[u];
      }
    }
  }
  if (verbose && s.stop_reason != s.stop_reason_dev)
    printf("    audit: the device recorded no stop reason: the scheduler's deadline wait raised *abort and no control task ran after it. Reported as stop reason %d; the trace has no STOP event\n", s.stop_reason);
  if (verbose) printf("    audit: %d buffers, %d uids, %zu ended groups, %zu events, %d updates: %d violations (errors %d, counters %d, conservation %d, queues %d, exactly-once %d, terminal %d, lag %d, fifo %d, versions %d, trace %d, incomplete %d)\n",
                      nbuf, U, s.log.size(), s.trace.size(), s.train_ver, R.total(), R.errors, R.counters, R.conservation, R.queues, R.exactly_once, R.terminal, R.lag, R.fifo, R.versions, R.trace, R.incomplete);
  if (out) *out = R;
  return R.total();
}

}  // namespace rlc
