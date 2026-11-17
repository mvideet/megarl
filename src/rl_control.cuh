// rl_control.cuh -- the four control tasks of the RL core.
//   ctl::roll_begin    R_BEGIN    start of a rollout step: cancel stale groups, admit new ones, wait if nothing can run
//   ctl::roll_advance  R_ADVANCE  after sampling, beside the lower trunk: score finished groups, queue or drop them
//   ctl::train_select  T_SELECT   collect TGRP eligible groups for one update, drop stale ones
//   ctl::train_finish  T_FINISH   check that the update is finite, publish and archive it, log and release the groups
//
// Each is called by all 256 threads of the block that claimed the task. Thread 0 does the bookkeeping: it is the only
// thread that touches queues, counters, the version table and group states. All threads take part in the parallel
// scan of the job's rows and slots and in the bulk work (prompt-state copies, snapshot copies, token copies for the
// group log, scoring). The pattern is always the same: thread 0 writes a small command into the block's dynamic shared
// memory, a barrier, everybody executes it, a barrier. While thread 0 waits (the idle poll of R_BEGIN, the poll of
// T_SELECT, a publisher without a free slot) the other threads sit in that barrier.
// A command is valid from the barrier after thread 0 wrote it until the next barrier. Nothing may overwrite it in
// between, because the threads of a block do not leave a barrier at the same instant; bulk work that overwrites shared
// memory (the score callable) therefore starts behind a second barrier.
// No __shared__ variables and no blockIdx. `smem` is scratch (under 10 KB) that holds nothing between tasks; the score
// callable of roll_advance may overwrite all of it.
//
// Why thread 0's view of its job is exclusive: R_BEGIN depends on the last trunk phase and on R_ADVANCE of the previous
// step, every other phase of a step depends on R_BEGIN, and R_ADVANCE depends on H_SAMPLE. So while R_BEGIN runs
// nothing else of its job runs, and while R_ADVANCE runs only the lower trunk (layers 0..LMID-1) does, which neither
// reads GroupSlot nor writes Row.active.
//
// Two control options, both off by default (cnt.held and cnt.training are maintained either way):
//   hp.count_held   cnt.held[t] counts the groups of environment t that the learner has taken and not released (state
//                   TRAINING). T_SELECT raises it for a group it keeps before it lowers queued[t]; T_FINISH lowers it
//                   after releasing the group's buffer. With the option a group is admitted while
//                   inflight + queued + held < target, and the limit is exact: a job re-reads the counters after its
//                   reservation and gives the reservation back if another job got in first (admit_group). Without it
//                   held does not count and concurrent admissions can exceed the limit by up to nroll - 1.
//   hp.alternate    generation and training take turns on cnt.training. Bit 0 (rls::TRAINING_BATCH) means the learner
//                   has a complete batch: T_SELECT sets it, T_FINISH clears it after the last release. A rollout job
//                   needs the generation turn (bit rls::TURN_SHIFT + job) for the admissions and the step of an
//                   R_BEGIN; it can take the turn only while bit 0 is clear, and holds it until the R_BEGIN of its next
//                   step. The learner, having set bit 0, waits in T_SELECT until no job holds a turn and only then
//                   marks the batch valid, so no rollout step overlaps an update. A job without the turn waits in
//                   R_BEGIN's idle loop and keeps cancelling groups that go stale, which is what frees snapshot slots
//                   for the publisher of the update it is waiting for.
#pragma once
#include "rl_state.cuh"
#include "rl_queue.cuh"
#include "policy_versions.cuh"
#include "trace.cuh"
#include "group_reduce.cuh"

namespace ctl {
using mrl::u64; using mrl::bf16;

// ------------------------------------------------------------------------------------------------ shared-memory layout (ints)
constexpr int NSLOT = mrl::MAXPIPE * mrl::MAXGRP, NROW = mrl::MAXPIPE * mrl::MAXROWS;   // a job has at most MAXPIPE pipelines
constexpr int SM_CMD = 0;                      // [64]     command of the current round
constexpr int SM_ST = 64;                      // [NSLOT]  scan: GroupSlot.state, index sp * MAXGRP + g (sp = pipeline inside the job)
constexpr int SM_VER = SM_ST + NSLOT;          // [NSLOT]  scan: GroupSlot.ver_id
constexpr int SM_ACT = SM_VER + NSLOT;         // [NROW]   scan: Row.active, index sp * MAXROWS + row
constexpr int SM_GEN = SM_ACT + NROW;          // [NROW]   scan: Row.gen
constexpr int SM_JOB = SM_GEN + NROW;          // [2 * NSLOT][4] bulk jobs of the round: kind, a, b, c (a pass makes at most
                                               //   one cancel and one admission per slot)
constexpr int SM_REL = SM_JOB + 8 * NSLOT;     // [NSLOT]  buffers to release after the round's token copies
constexpr int SM_INTS = SM_REL + NSLOT;
static_assert(SM_INTS * 4 <= mrl::SMEM, "control scratch must fit in the block's dynamic shared memory");
enum { OP_DONE = 0, OP_SCORE = 1, OP_COPY = 2 };
enum { JOB_TOKENS = 1, JOB_PROMPT = 2 };

// ------------------------------------------------------------------------------------------------ small helpers (thread 0)
__device__ __forceinline__ bool past_deadline(const mrl::Ctx& C, u64 t0) { return C.hp.deadline_ns != 0ull && tr::since(t0) > C.hp.deadline_ns; }

// End the run: the first reason wins, then *abort stops every job.
__device__ __noinline__ void stop_run(const mrl::Ctx& C, u64 t0, int reason) {
  if (atomicCAS(C.sh.cnt.stop_reason, 0, reason) == 0) tr::emit(C.sh.trace, t0, rls::EV_STOP, reason, 0, 0, 0, 0);
  __threadfence();
  mk::st_rel(C.abort, 1);
}
// *abort was raised by someone who left no reason: the scheduler's watchdog (a block in mk::wave_loop that waits for
// work past the job deadline, hp.deadline_ns + WATCHDOG_NS in megarl.cu), which knows nothing of cnt.stop_reason.
__device__ __forceinline__ void note_abort(const mrl::Ctx& C, u64 t0) {
  if (rls::ldi(C.sh.cnt.stop_reason) == 0 && past_deadline(C, t0) && atomicCAS(C.sh.cnt.stop_reason, 0, (int)rls::DEADLINE) == 0)
    tr::emit(C.sh.trace, t0, rls::EV_STOP, rls::DEADLINE, 0, 0, 0, 0);
}
// The check at the top of a control task and inside every wait. True: return now.
// If no control task reaches such a check after the scheduler raised *abort, no reason is recorded on the device;
// the host then fills it in (rlc::resolve_stop_reason).
__device__ __forceinline__ bool must_stop(const mrl::Ctx& C, u64 t0, bool also_finished) {
  if (rls::ldi(C.abort)) { note_abort(C, t0); return true; }
  if (also_finished && rls::ldi(C.finished)) return true;
  if (past_deadline(C, t0)) { stop_run(C, t0, rls::DEADLINE); return true; }
  return false;
}
// False for NaN and for +-inf.
__host__ __device__ __forceinline__ bool finite_f(float x) { return fabsf(x) <= 3.4028234664e38f; }

// hp.alternate: take the generation turn of rollout job `job` (bit rls::TURN_SHIFT + job of cnt.training). The
// compare-and-swap succeeds only while the learner's batch bit is clear, so no turn begins after the learner has a
// complete batch; false means it has one. A failed swap means another job or the learner changed the word in between,
// so the loop only repeats on someone else's progress.
__device__ __forceinline__ bool turn_take(const mrl::Ctx& C, int job) {
  int* w = C.sh.cnt.training; const int mine = 1 << (rls::TURN_SHIFT + job);
  for (;;) {
    const int old = rls::ldi(w);
    if (old & rls::TRAINING_BATCH) return false;
    if (atomicCAS(w, old, old | mine) == old) return true;
  }
}
__device__ __forceinline__ void turn_give(const mrl::Ctx& C, int job) { atomicAnd(C.sh.cnt.training, ~(1 << (rls::TURN_SHIFT + job))); }

// After a terminal state and its log entry: state FREE first, then the buffer goes back to the free list.
// The free list has room for every buffer, so it is never full. But it has several consumers (every rollout job
// admits), and rlq::push is refused while another job's pop of the position one lap back has moved *head and not yet
// released its cell (rl_queue.cuh). That lasts a few instructions of another block, so the push is repeated: giving up
// at once would lose the buffer for the rest of the run. Only a push that never succeeds is an error.
constexpr unsigned RELEASE_TRIES = 4096;         // with 100 ns sleeps: far longer than the refusal it waits out
__device__ __forceinline__ void release_buf(const mrl::Ctx& C, int buf, int terminal) {
  rls::transition(&C.store.rec[buf].state, terminal, rls::FREE, C.sh.cnt.errors);
  for (unsigned tries = 0; !rlq::push(C.sh.freebuf, buf); tries++) {
    if (tries >= RELEASE_TRIES) { atomicAdd(C.sh.cnt.errors + rls::ERR_QUEUE_FULL, 1); break; }
    __nanosleep(100);
  }
}

// ------------------------------------------------------------------------------------------------ all threads
// Copy Row.active / Row.gen and GroupSlot.state / ver_id of the job's pipelines into shared memory (SM_ACT, SM_GEN,
// SM_ST, SM_VER): one load chain per thread instead of hundreds of serial loads by thread 0. The caller issues the
// barrier.
__device__ __forceinline__ void scan_job(const mrl::Ctx& C, int job, int* sm) {
  const int tid = threadIdx.x, ppj = C.ppj, ngrp = C.ngrp, nrows = ngrp * mrl::GSZ;
  for (int i = tid; i < ppj * mrl::MAXROWS; i += 256) {
    const int sp = i / mrl::MAXROWS, r = i % mrl::MAXROWS; int act = 0, gen = 0;
    if (r < nrows) { const mrl::Row* row = C.pipe[job * ppj + sp].row + r; act = rls::ldi(&row->active); gen = rls::ldi(&row->gen); }
    sm[SM_ACT + i] = act; sm[SM_GEN + i] = gen;
  }
  for (int i = tid; i < ppj * mrl::MAXGRP; i += 256) {
    const int sp = i / mrl::MAXGRP, g = i % mrl::MAXGRP; int st = rls::FREE, ver = 0;
    if (g < ngrp) { const mrl::GroupSlot* s = C.pipe[job * ppj + sp].grp + g; st = rls::ldi(&s->state); ver = rls::ldi(&s->ver_id); }
    sm[SM_ST + i] = st; sm[SM_VER + i] = ver;
  }
}

// One bulk job of R_BEGIN: JOB_TOKENS {buf, log index} or JOB_PROMPT {prompt, pipeline, slot}.
__device__ __forceinline__ void run_job(const mrl::Ctx& C, const int* jb) {
  const int tid = threadIdx.x;
  if (jb[0] == JOB_TOKENS) { tr::log_copy_tokens(C.sh.glog, C.store, jb[1], (u64)jb[2], C.sh.cnt.errors); return; }
  // JOB_PROMPT: the residual stream entering layer LMID at the prompt's last position (prompts.hlast) into the GSZ
  // rows of slot g. A step starts at layer LMID, so for a new group this takes the place of the lower trunk's output.
  static_assert(mrl::FF % 8 == 0 && mrl::D % 4 == 0, "16-byte chunks");
  const int prompt = jb[1], row0 = mrl::GSZ * jb[3]; const mrl::Pipe* P = C.pipe + jb[2];
  float* hres = P->hres;
  const float* hs = C.prompts.hlast + (size_t)prompt * mrl::D;
  for (int q = tid; q < mrl::D / 4; q += 256) {              // columns < D only
    const int4 x = __ldcg(reinterpret_cast<const int4*>(hs + (size_t)q * 4));
    for (int j = 0; j < mrl::GSZ; j++) *reinterpret_cast<int4*>(hres + (size_t)(row0 + j) * mrl::DP + (size_t)q * 4) = x;
  }
}

// ------------------------------------------------------------------------------------------------ R_BEGIN pieces (thread 0)
// Cancel the group in slot (p, g), which at train_ver = tv is too old to be trained on. gens: Row.gen of its GSZ rows
// from the scan. If its tokens go to the log, the copy becomes a bulk job and the buffer is released after it (SM_REL).
__device__ __forceinline__ void cancel_group(const mrl::Ctx& C, u64 t0, int p, int g, int tv, const int* gens, int* sm, int& njob, int& nrel) {
  const mrl::Counters& N = C.sh.cnt; mrl::GroupSlot* S = C.pipe[p].grp + g; mrl::Row* row = C.pipe[p].row + mrl::GSZ * g;
  const int buf = rls::ldi(&S->buf), uid = rls::ldi(&S->uid), task = rls::ldi(&S->task), ver = rls::ldi(&S->ver_id);
  for (int j = 0; j < mrl::GSZ; j++) { row[j].active = 0; row[j].feed = 0; }
  pv::unpin(C.sh.ver, rls::ldi(&S->ver_slot), rls::ldi(&S->ver_gen), N.errors);
  atomicAdd(N.inflight + task, -1);
  atomicAdd(N.drop_stale_inflight + task, 1);
  mrl::GroupRec* rec = C.store.rec + buf;
  for (int j = 0; j < mrl::GSZ; j++) rec->len[j] = gens[j];  // tokens generated so far
  rls::transition(&rec->state, rls::GENERATING, rls::CANCELLED, N.errors);
  const u64 n = tr::log_group(C.sh.glog, rec, rls::CANCELLED, -1, -1, tr::since(t0));
  tr::emit(C.sh.trace, t0, rls::EV_CANCEL, uid, task, buf, tv, ver);
  if (n < C.sh.glog.tok_cap) {                               // its tokens go into the log first: release after the copy
    int* jb = sm + SM_JOB + 4 * njob++; jb[0] = JOB_TOKENS; jb[1] = buf; jb[2] = (int)n; jb[3] = 0;
    sm[SM_REL + nrel++] = buf;
  } else release_buf(C, buf, rls::CANCELLED);
  S->state = rls::FREE;
}

// Try to admit a new group into the free slot (p, g). `blocked` (bit per environment) and `nobuf` remember, for the
// rest of this pass over the slots, what already failed: an environment at its target, an empty free list.
__device__ __forceinline__ bool admit_group(const mrl::Ctx& C, u64 t0, int p, int g, unsigned& blocked, bool& nobuf, int* sm, int& njob, int& ver_out) {
  const mrl::Counters& N = C.sh.cnt; const int pt = C.pipe_task[p];
  if (pt >= mrl::NTASKS) return false;
  const bool ch = C.hp.count_held != 0;                      // the groups the learner holds count against the limit
  int t = -1;
  for (;;) {
    // Admissible: inflight + queued (+ held) < target. Choice: smallest queued (pair_mode 2: consumed + queued +
    // inflight), then smallest inflight, then lowest index. The three counters are read in the order in which a group
    // passes through them (it is counted where it goes before it is discounted where it was), so a group on its way is
    // seen twice, never not at all.
    int bq = 0, binf = 0; t = -1;
    for (int c = (pt >= 0 ? pt : 0); c < (pt >= 0 ? pt + 1 : mrl::NTASKS); c++) {
      if ((blocked >> c) & 1u) continue;
      const int inf = rls::ldi(N.inflight + c), q = rls::ldi(N.queued + c), h = ch ? rls::ldi(N.held + c) : 0;
      if (inf + q + h >= C.hp.target_per_task || C.prompts.ntask[c] <= 0) { blocked |= 1u << c; continue; }
      const int score_q = (C.hp.pair_mode == 2) ? (rls::ldi(N.consumed + c) + q + inf) : q;
      if (t < 0 || score_q < bq || (score_q == bq && inf < binf)) { t = c; bq = score_q; binf = inf; }
    }
    if (t < 0) return false;
    const int inf0 = atomicAdd(N.inflight + t, 1);           // reserve
    if (!ch) break;                                          // (test and increment are not one step: pacing, not safety)
    // count_held: the limit is exact. Every job that reserves gets its own rank in inflight[t] from the atomic add and
    // then reads what is queued and held. A group that was in the system when this job reserved is in inf0, or has
    // moved on to queued or held, where it is read later than it arrived (the fences keep the three reads in order,
    // and with the option on the writers fence between "count it there" and "discount it here"). So of several jobs
    // that pass the test above at the same moment only as many keep their reservation as there is room for.
    __threadfence();
    const int q1 = rls::ldi(N.queued + t);
    __threadfence();
    const int h1 = rls::ldi(N.held + t);
    if (inf0 + q1 + h1 < C.hp.target_per_task) break;
    atomicAdd(N.inflight + t, -1); blocked |= 1u << t;       // another job was first: not in this pass
  }
  int buf = -1;
  if (!rlq::pop(C.sh.freebuf, buf)) {                        // back-pressure, not an error
    atomicAdd(N.inflight + t, -1); rls::stat_add(N.stats, rls::BACKPRESSURE); nobuf = true; return false;
  }
  if (buf < 0 || buf >= C.store.nbuf) { atomicAdd(N.errors + rls::ERR_BUF, 1); atomicAdd(N.inflight + t, -1); return false; }
  const unsigned k = (unsigned)atomicAdd(N.prompt_next + t, 1);
  const int prompt = C.prompts.by_task[t][k % (unsigned)C.prompts.ntask[t]];
  const int plen = C.prompts.len[prompt], key = C.prompts.key[prompt];
  int vslot, vid, vgen;
  pv::pin(C.sh.ver, vslot, vid, vgen, N.stats);
  const int uid = atomicAdd(N.uid_next, 1);
  mrl::GroupRec* rec = C.store.rec + buf;
  rls::transition(&rec->state, rls::FREE, rls::GENERATING, N.errors);
  rec->uid = uid; rec->task = t; rec->prompt = prompt; rec->key = key; rec->ver_id = vid;
  for (int j = 0; j < mrl::GSZ; j++) { rec->len[j] = 0; rec->reward[j] = 0.f; rec->adv[j] = 0.f; }
  rec->t_admit = tr::since(t0); rec->t_done = 0ull;
  mrl::GroupSlot* S = C.pipe[p].grp + g;
  S->buf = buf; S->uid = uid; S->task = t; S->prompt = prompt; S->key = key; S->ver_slot = vslot; S->ver_id = vid; S->ver_gen = vgen; S->pad = 0;
  S->state = rls::GENERATING;
  mrl::Row* row = C.pipe[p].row + mrl::GSZ * g;
  for (int j = 0; j < mrl::GSZ; j++) {
    row[j].active = 1; row[j].feed = 1; row[j].prompt = prompt; row[j].plen = plen; row[j].gen = 0;
    row[j].tok = 0; row[j].pos = plen - 1; row[j].buf = buf; row[j].seq = j;
  }
  if (C.prompts.hlast) { int* jb = sm + SM_JOB + 4 * njob++; jb[0] = JOB_PROMPT; jb[1] = prompt; jb[2] = p; jb[3] = g; }
  atomicAdd(N.admitted + t, 1);
  tr::emit(C.sh.trace, t0, rls::EV_ADMIT, uid, t, prompt, vid, p * 256 + g);
  ver_out = vid;
  return true;
}

// ================================================================================================ R_BEGIN
__device__ __noinline__ void roll_begin(const mrl::Ctx& C, int job, mk::u64 step, char* smem) {
  const int tid = threadIdx.x; int* sm = reinterpret_cast<int*>(smem);
  const int ppj = C.ppj, ngrp = C.ngrp, p0 = job * ppj;
  scan_job(C, job, sm);
  __syncthreads();
  u64 t0 = 0; int tv = 0, nrel = 0; bool first = true, idle = false;      // thread 0's state across rounds
  const bool alt = C.hp.alternate != 0;                      // generation and training take turns (see turn_take)
  for (;;) {
    if (tid == 0) {
      int njob = 0, done = 0;
      if (first) {
        first = false; t0 = rls::ld64(C.t0);
        if (alt) turn_give(C, job);                          // the step this job started last has ended: R_BEGIN depends on all of it
        if (must_stop(C, t0, true)) done = 1; else tv = mk::ld_acq(C.sh.ver.train_ver);
      }
      while (!done) {
        // alternate: the admissions of this pass and the start of a step need the generation turn. Without it (the
        // learner has a complete batch) the pass only cancels. The flags are checked again after a turn is taken: the
        // last T_FINISH of a run sets *finished before it clears its batch bit, so a job that got the turn because the
        // run is over does not admit into the end of it.
        bool go = true;
        if (alt) {
          go = turn_take(C, job);
          if (go && must_stop(C, t0, true)) { turn_give(C, job); done = 1; break; }
        }
        // one pass over the job's group slots (sm[SM_ST], sm[SM_VER] are thread 0's working copy of them)
        int nact = 0; bool nobuf = false; unsigned blocked = 0;
        for (int sp = 0; sp < ppj; sp++) for (int g = 0; g < ngrp; g++) {
          const int i = sp * mrl::MAXGRP + g;
          if (sm[SM_ST + i] == rls::GENERATING) {
            // a version newer than tv (admitted between the cur switch and the train_ver store) counts as lag 0
            if (tv - sm[SM_VER + i] > C.hp.lag) { cancel_group(C, t0, p0 + sp, g, tv, sm + SM_GEN + sp * mrl::MAXROWS + mrl::GSZ * g, sm, njob, nrel); sm[SM_ST + i] = rls::FREE; }
            else nact++;
          }
          if (sm[SM_ST + i] == rls::FREE && !nobuf && go) {
            int ver = 0;
            if (admit_group(C, t0, p0 + sp, g, blocked, nobuf, sm, njob, ver)) { sm[SM_ST + i] = rls::GENERATING; sm[SM_VER + i] = ver; nact++; }
          }
        }
        if (nact > 0 && go) { if (idle) tr::emit(C.sh.trace, t0, rls::EV_IDLE, job, 0, 0, 0, 0); done = 1; break; }   // a row is active: the step starts (alternate: with the turn, which the job keeps until its next R_BEGIN)
        if (alt && go) turn_give(C, job);                    // nothing to generate after all: the learner must not wait for this job
        if (njob > 0) break;                                 // token copies of cancelled groups first, then look again
        if (must_stop(C, t0, true)) { done = 1; break; }
        if (!idle) { idle = true; tr::emit(C.sh.trace, t0, rls::EV_IDLE, job, 1, 0, 0, 0); }
        rls::stat_add(C.sh.cnt.stats, rls::IDLE_POLLS);
        __nanosleep(2000);
        tv = mk::ld_acq(C.sh.ver.train_ver);
      }
      sm[SM_CMD] = njob; sm[SM_CMD + 1] = done;
    }
    __syncthreads();
    const int njob = sm[SM_CMD], done = sm[SM_CMD + 1];
    for (int k = 0; k < njob; k++) run_job(C, sm + SM_JOB + 4 * k);
    if (njob) __syncthreads();
    if (tid == 0) { for (int i = 0; i < nrel; i++) release_buf(C, sm[SM_REL + i], rls::CANCELLED); nrel = 0; }
    if (done) break;
  }
}

// ================================================================================================ R_ADVANCE
// score(rec, tokens, smem): called by all threads for one finished group. rec is the group's mrl::GroupRec in the
// store, tokens its [GSZ][MAXGEN] generated tokens (rec.len[j] valid ones per sequence). It must leave
// rec.reward[0..GSZ) written and may use all of smem. megarl.cu passes a Scorer, which wraps rw::score_group.
// It is entered behind a block barrier and followed by a fence and a block barrier (both issued here, by every
// thread), so any thread may write the rewards without a barrier of its own; any barrier it does contain must be
// reached by all 256 threads.
template <class Score>
__device__ __noinline__ void roll_advance(const mrl::Ctx& C, int job, mk::u64 step, char* smem, Score score) {
  const int tid = threadIdx.x; int* sm = reinterpret_cast<int*>(smem);
  const int ppj = C.ppj, ngrp = C.ngrp, p0 = job * ppj; const mrl::Counters& N = C.sh.cnt;
  scan_job(C, job, sm);
  __syncthreads();
  // Thread 0's state: the slots still to finish (bit sp * MAXGRP + g) and the group in progress. Kept in registers,
  // because the score callable owns the shared memory while it runs.
  u64 todo[(NSLOT + 63) / 64]; u64 t0 = 0; int stage = 0, cp = 0, cg = 0, cbuf = 0, cuid = 0, ctask = 0, cver = 0, cn1 = 0;
  if (tid == 0) {
    for (int w = 0; w < (NSLOT + 63) / 64; w++) todo[w] = 0ull;
    t0 = rls::ld64(C.t0);
    if (!(rls::ldi(C.abort) || rls::ldi(C.finished))) {
      for (int sp = 0; sp < ppj; sp++) for (int g = 0; g < ngrp; g++) {
        const int i = sp * mrl::MAXGRP + g; const int* act = sm + SM_ACT + sp * mrl::MAXROWS + mrl::GSZ * g;
        int any = 0; for (int j = 0; j < mrl::GSZ; j++) any |= act[j];
        if (sm[SM_ST + i] == rls::GENERATING && !any) todo[i >> 6] |= 1ull << (i & 63);     // all GSZ sequences have ended
      }
    } else if (rls::ldi(C.abort)) note_abort(C, t0);
  }
  for (;;) {
    if (tid == 0) {
      int op = OP_DONE, a = 0, b = 0;
      if (stage == 1) {                                      // the group in (cp, cg) has just been scored
        mrl::GroupRec* rec = C.store.rec + cbuf;
        float r[mrl::GSZ], adv[mrl::GSZ]; int lmax = 0; cn1 = 0;
        for (int j = 0; j < mrl::GSZ; j++) { r[j] = rls::ldf(&rec->reward[j]); cn1 += (r[j] == 1.f); lmax = max(lmax, rls::ldi(&rec->len[j])); }
        const bool usable = gr::advantages(r, adv, C.hp.adv_eps);
        for (int j = 0; j < mrl::GSZ; j++) rec->adv[j] = adv[j];
        rec->t_done = tr::since(t0);
        atomicAdd(N.finished + ctask, 1);
        tr::emit(C.sh.trace, t0, rls::EV_GROUP_DONE, cuid, ctask, cn1, lmax, cp * 256 + cg);
        if (!usable) {                                       // equal (or non-finite) rewards: nothing to learn from
          atomicAdd(N.drop_zerovar + ctask, 1);
          rls::transition(&rec->state, rls::SCORING, rls::DROPPED_ZEROVAR, N.errors);
          const u64 n = tr::log_group(C.sh.glog, rec, rls::DROPPED_ZEROVAR, -1, -1, tr::since(t0));
          tr::emit(C.sh.trace, t0, rls::EV_DROP_ZEROVAR, cuid, ctask, cbuf, cn1, cver);
          if (n < C.sh.glog.tok_cap) { op = OP_COPY; a = cbuf; b = (int)n; }
          stage = 2;
          // The guard of the token copy runs here rather than inside tr::log_copy_tokens as in the other three tasks:
          // the copy below is called without the errors pointer. Thread 0 does nothing that could free the buffer
          // between this line and the copy, so the guard sees what the copy will see. It stays inside this branch;
          // moved to the command write below it changes register allocation elsewhere in the kernel.
          if (op == OP_COPY) tr::log_copy_guard(C.sh.glog, C.store, cbuf, n, N.errors);
        } else {
          rls::transition(&rec->state, rls::SCORING, rls::READY, N.errors);
          atomicAdd(N.queued + ctask, 1); if (C.hp.count_held) __threadfence(); atomicAdd(N.inflight + ctask, -1);   // queued before it leaves inflight (count_held: also in memory order)
          __threadfence();
          if (!rlq::push(C.sh.ready[ctask], cbuf)) atomicAdd(N.errors + rls::ERR_QUEUE_FULL, 1);   // from here on the learner owns the buffer
          tr::emit(C.sh.trace, t0, rls::EV_ENQUEUE, cuid, ctask, cbuf, rlq::size(C.sh.ready[ctask]), cver);
          C.pipe[cp].grp[cg].state = rls::FREE;
          stage = 0;
        }
      }
      if (stage == 2 && op == OP_DONE) {                     // dropped, and its tokens (if wanted) are in the log
        release_buf(C, cbuf, rls::DROPPED_ZEROVAR);
        atomicAdd(N.inflight + ctask, -1);
        C.pipe[cp].grp[cg].state = rls::FREE;
        stage = 0;
      }
      if (stage == 0 && op == OP_DONE) {                     // next finished group
        int i = -1;
        for (int w = 0; w < (NSLOT + 63) / 64 && i < 0; w++) if (todo[w]) { const int bit = __ffsll((long long)todo[w]) - 1; todo[w] &= todo[w] - 1ull; i = w * 64 + bit; }
        if (i >= 0) {
          cp = p0 + i / mrl::MAXGRP; cg = i % mrl::MAXGRP;
          const mrl::GroupSlot* S = C.pipe[cp].grp + cg;
          cbuf = rls::ldi(&S->buf); cuid = rls::ldi(&S->uid); ctask = rls::ldi(&S->task); cver = rls::ldi(&S->ver_id);
          pv::unpin(C.sh.ver, rls::ldi(&S->ver_slot), rls::ldi(&S->ver_gen), N.errors);   // the group stops using its snapshot
          rls::transition(&C.store.rec[cbuf].state, rls::GENERATING, rls::SCORING, N.errors);
          op = OP_SCORE; a = cbuf; stage = 1;
        }
      }
      sm[SM_CMD] = op; sm[SM_CMD + 1] = a; sm[SM_CMD + 2] = b;
    }
    __syncthreads();
    const int op = sm[SM_CMD], a = sm[SM_CMD + 1], b = sm[SM_CMD + 2];
    if (op == OP_DONE) break;
    // Second barrier: every thread holds the command in registers before the callable starts on any thread. The
    // callable may overwrite all of the shared memory, the command words included (rw::score_group writes its masks
    // exactly there); without this barrier a thread that is late out of the first one could read a command that an
    // early thread had already overwritten.
    __syncthreads();
    if (op == OP_SCORE) score(C.store.rec[a], static_cast<const int*>(C.store.tokens + (size_t)a * mrl::GSZ * mrl::MAXGEN), smem);
    else tr::log_copy_tokens(C.sh.glog, C.store, a, (u64)b);
    __threadfence();                                         // the rewards were written by threads other than the one that pushes the group
    __syncthreads();
  }
}

// ================================================================================================ T_SELECT
// While the learner waits here train_ver does not move, so a group it already holds cannot go stale.
// Take rule: a group of environment t is taken while the batch is not full and (need[t] > 0 or the fallback is on).
//   pair_mode 0  need[t] = TGRP / NTASKS, no fallback
//   pair_mode 1  the same until hp.select_wait_ns after the start of the task, then any eligible group
//   pair_mode 2  always the non-empty ready queue with the smallest consumed[t] + groups of t already in the batch
// Starvation (hp.starve_ns) is measured from the last group this task took or dropped (from the start if none), so a
// learner that fills a large batch slowly but steadily is not declared starved.
// The local `held` counts the groups of the batch so far; cnt.held[t] counts the same groups per environment and stays
// up until T_FINISH has released them.
__device__ __noinline__ void train_select(const mrl::Ctx& C, mk::u64 step, char* smem) {
  const int tid = threadIdx.x; int* sm = reinterpret_cast<int*>(smem);
  const mrl::Counters& N = C.sh.cnt; mrl::Batch* B = C.tr.batch;
  // thread 0's state, kept across the rounds that copy the tokens of a stale group to the log
  u64 t0 = 0, t_start = 0, t_prog = 0; int held = 0, need[mrl::NTASKS], hbuf[mrl::TGRP], t = 0, ndrop = 0, pend = -1;
  bool fallback = false, used_fb = false, progress = false, fresh = true, fin = false;
  if (tid == 0) {
    t0 = rls::ld64(C.t0); t_start = tr::since(t0); t_prog = t_start; B->valid = 0;
    for (int e = 0; e < mrl::NTASKS; e++) need[e] = (C.hp.pair_mode == 2) ? 0 : (mrl::TGRP / mrl::NTASKS);
    if (C.hp.pair_mode == 2) fallback = true;
  }
  for (;;) {
    if (tid == 0) {
      int op = OP_DONE, a = 0, b = 0;
      if (pend >= 0) { release_buf(C, pend, rls::DROPPED_STALE); pend = -1; }
      while (!fin && op == OP_DONE) {
        if (fresh) {                                         // start of a sweep over the environments
          fresh = false; progress = false;
          if (rls::ldi(C.abort)) { note_abort(C, t0); fin = true; break; }
          if (past_deadline(C, t0)) { stop_run(C, t0, rls::DEADLINE); fin = true; break; }
        }
        while (t < mrl::NTASKS && op == OP_DONE) {
          if (C.hp.pair_mode == 2 && held < mrl::TGRP) {
            // deficit pick: the non-empty ready queue with the smallest consumed[e] + taken in this batch (= -need[e])
            int best_e = -1, best_tot = 0x7fffffff, dummy = -1;
            for (int e = 0; e < mrl::NTASKS; e++) {
              if (!rlq::front(C.sh.ready[e], dummy)) continue;
              const int tot_e = rls::ldi(N.consumed + e) - need[e];
              if (best_e < 0 || tot_e < best_tot) { best_e = e; best_tot = tot_e; }
            }
            if (best_e < 0) { t = mrl::NTASKS; break; }
            t = best_e;
          }
          if (!(held < mrl::TGRP && (need[t] > 0 || fallback))) { t++; continue; }
          const mrl::Queue& Q = C.sh.ready[t];
          int buf = -1, b2 = -1;
          if (!rlq::front(Q, buf)) { t++; continue; }        // nothing (more) of this environment
          if (!rlq::pop(Q, b2)) { atomicAdd(N.errors + rls::ERR_POP, 1); t++; continue; }
          if (b2 != buf) { atomicAdd(N.errors + rls::ERR_POP, 1); buf = b2; }      // one front, one decision
          if (buf < 0 || buf >= C.store.nbuf) { atomicAdd(N.errors + rls::ERR_BUF, 1); continue; }
          progress = true; t_prog = tr::since(t0);           // a group was taken or dropped: the starvation clock starts again
          mrl::GroupRec* rec = C.store.rec + buf;
          const int tv = mk::ld_acq(C.sh.ver.train_ver), ver = rls::ldi(&rec->ver_id), uid = rls::ldi(&rec->uid), lag = tv - ver;
          if (lag > C.hp.lag || lag < 0) {     // too old to train on (lag < 0 is an error: ERR_LAG)
            atomicAdd(N.queued + t, -1);                     // it leaves the system: never counted as held
            if (lag < 0) atomicAdd(N.errors + rls::ERR_LAG, 1);
            atomicAdd(N.drop_stale_queued + t, 1);
            rls::transition(&rec->state, rls::READY, rls::DROPPED_STALE, N.errors);
            const u64 n = tr::log_group(C.sh.glog, rec, rls::DROPPED_STALE, -1, -1, tr::since(t0));
            tr::emit(C.sh.trace, t0, rls::EV_DROP_STALE, uid, t, buf, tv, ver);
            ndrop++;
            if (n < C.sh.glog.tok_cap) { op = OP_COPY; a = buf; b = (int)n; pend = buf; }   // t stays: resume here after the copy
            else release_buf(C, buf, rls::DROPPED_STALE);
          } else {
            // kept: counted as held before it is discounted as queued, so that inflight + queued + held never dips
            // below the groups that exist (with hp.count_held an engine that saw the dip would admit one group too many)
            atomicAdd(N.held + t, 1); if (C.hp.count_held) __threadfence();
            atomicAdd(N.queued + t, -1);
            rls::transition(&rec->state, rls::READY, rls::TRAINING, N.errors);
            hbuf[held] = buf; B->buf[held] = buf; B->uid[held] = uid; B->task[held] = t; B->lag[held] = lag;
            tr::emit(C.sh.trace, t0, rls::EV_SELECT, (int)step, uid, t, lag, held);
            if (need[t] <= 0) used_fb = true;
            held++; need[t]--;
          }
        }
        if (op != OP_DONE) break;
        t = 0; fresh = true;                                 // one sweep over the environments is done
        if (held == mrl::TGRP) {
          int off = 0;
          for (int g = 0; g < mrl::TGRP; g++) {
            const mrl::GroupRec* rec = C.store.rec + hbuf[g];
            for (int j = 0; j < mrl::GSZ; j++) {
              const int s = g * mrl::GSZ + j, len = rls::ldi(&rec->len[j]);
              B->len[s] = len; B->off[s] = off; B->adv[s] = rls::ldf(&rec->adv[j]); off += len;
            }
          }
          B->ntok = off; B->ntile = (off + mrl::TM - 1) / mrl::TM;
          const u64 t_full = tr::since(t0); int turn_us = 0;
          atomicOr(N.training, rls::TRAINING_BATCH);         // the learner has a complete batch, until T_FINISH has released it
          if (C.hp.alternate) {
            // The learner's turn begins when the rollout steps in flight have ended: every job gives its turn back in
            // its next R_BEGIN, and none can take one while the batch bit is set. Only then is the batch valid. If the
            // run ends first, the batch stays held and is not trained on (valid == 0: the learner's tasks return).
            bool over = false;
            while ((rls::ldi(N.training) >> rls::TURN_SHIFT) != 0) {
              if (rls::ldi(C.abort)) { note_abort(C, t0); over = true; break; }
              if (past_deadline(C, t0)) { stop_run(C, t0, rls::DEADLINE); over = true; break; }
              __nanosleep(2000);
            }
            if (over) { fin = true; break; }
            turn_us = (int)((tr::since(t0) - t_full) / 1000ull);
          }
          B->valid = 1;
          tr::emit(C.sh.trace, t0, rls::EV_SELECT_WAIT, (int)step, (int)((t_full - t_start) / 1000ull), ndrop, used_fb ? 1 : 0, turn_us);
          fin = true; break;
        }
        if (!progress) {
          const u64 now = tr::since(t0);
          if (C.hp.pair_mode == 1 && now - t_start > C.hp.select_wait_ns) fallback = true;     // from now on any eligible group
          if (now - t_prog > C.hp.starve_ns) { stop_run(C, t0, rls::STARVED); fin = true; break; }   // no progress for too long; held groups stay in TRAINING
          rls::stat_add(N.stats, rls::SELECT_POLLS);
          __nanosleep(2000);
        }
      }
      sm[SM_CMD] = op; sm[SM_CMD + 1] = a; sm[SM_CMD + 2] = b;
    }
    __syncthreads();
    const int op = sm[SM_CMD], a = sm[SM_CMD + 1], b = sm[SM_CMD + 2];
    if (op == OP_DONE) break;
    tr::log_copy_tokens(C.sh.glog, C.store, a, (u64)b, N.errors);
    __syncthreads();
  }
}

// ================================================================================================ T_FINISH
// After the update of step `step`: publish C.tr.W as version train_ver + 1, archive it if its id is due, mark the
// batch's groups CONSUMED, log and release them, and write the step record (loss_hist, step_done, step_info,
// step_stat; steps.bin on the host). If the batch is not valid, the gradient is not finite or the publication was
// aborted, only the end-of-run flags are handled.
__device__ __noinline__ void train_finish(const mrl::Ctx& C, mk::u64 step, char* smem) {
  const int tid = threadIdx.x; int* sm = reinterpret_cast<int*>(smem); float* sf = reinterpret_cast<float*>(smem);
  const mrl::Counters& N = C.sh.cnt; const mrl::Versions& V = C.sh.ver; mrl::Batch* B = C.tr.batch;
  constexpr int SM_ARG = 8, SM_NUM = 32;                     // sm[0..8) belongs to pv::publish, then to the group loop
  constexpr int NNUM = mrl::NGNORM + 3 * mrl::TTILE;
  static_assert((SM_NUM + NNUM) * 4 <= mrl::SMEM, "step numbers must fit");
  u64 t0 = 0; int polls = 0, gtask = 0; float g2 = 0.f; bool released = false;   // thread 0's state
  if (tid == 0) {
    t0 = rls::ld64(C.t0);
    if (rls::ldi(C.abort)) note_abort(C, t0);
    sm[SM_ARG] = rls::ldi(&B->valid); sm[SM_ARG + 1] = mk::ld_acq(V.train_ver) + 1;
  }
  __syncthreads();
  const int valid = sm[SM_ARG], id = sm[SM_ARG + 1];
  int slot = -1, sound = 0;
  if (valid) {
    // The step's numbers, fetched by all threads and added up by thread 0 in index order. They are fetched before the
    // publication because the gradient-norm partials decide whether there is one (nothing touches SM_NUM in between).
    for (int i = tid; i < NNUM; i += 256) {
      int k = i;
      if (k < mrl::NGNORM) { sf[SM_NUM + i] = rls::ldf(&B->gnorm2[k]); continue; } k -= mrl::NGNORM;
      if (k < mrl::TTILE) { sf[SM_NUM + i] = rls::ldf(&B->loss_part[k]); continue; } k -= mrl::TTILE;
      if (k < mrl::TTILE) { sf[SM_NUM + i] = rls::ldf(&B->maxdlogp_part[k]); continue; } k -= mrl::TTILE;
      sm[SM_NUM + i] = rls::ldi(&B->nclip_part[k]);
    }
    __syncthreads();
    if (tid == 0) {
      // An update whose gradient is not finite is not published: the weights the learner has just written are then not
      // finite either (or will not stay so), every group admitted after the publication would sample from them, and
      // nothing downstream could tell. The sum of squares of the gradient shows it: NaN if an entry was NaN, infinite
      // if one was infinite or the squares overflowed. The run ends with reason ERROR and a count in
      // errors[ERR_NONFINITE]; the version table and the archive keep the last good snapshot, the batch stays held
      // (state TRAINING), and the step leaves no record: no PUBLISH or STEP event, no entry in the step arrays.
      const float* fg = sf + SM_NUM;
      for (int k = 0; k < mrl::NGNORM; k++) g2 = __fadd_rn(g2, fg[k]);
      const bool ok = finite_f(g2);
      if (!ok) { atomicAdd(N.errors + rls::ERR_NONFINITE, 1); stop_run(C, t0, rls::ERROR); }
      sm[SM_ARG + 2] = ok ? 1 : 0;
    }
    __syncthreads();
    sound = sm[SM_ARG + 2];
  }
  if (sound) {
    slot = pv::publish(V, C.tr.W, id, N.stats, C.abort, smem, &polls);   // -1: aborted while every slot was pinned
    if (slot >= 0) {
      if (V.narch > 0 && V.arch_stride > 0) {
        int aidx = -1;
        if (id % V.arch_stride == 0 && (id / V.arch_stride) < V.narch - 1) aidx = id / V.arch_stride;
        else if (id == C.updates) aidx = V.narch - 1;
        if (aidx >= 0 && aidx < V.narch) {
          bf16* dst = V.arch + (size_t)aidx * mrl::MID_NW;
          for (size_t q = tid; q < mrl::MID_NW / 8; q += 256) pv::copy16(dst + q * 8, C.tr.W + q * 8);
          if (tid == 0 && V.arch_id) V.arch_id[aidx] = id;
        }
      }
      __syncthreads();                                       // the archive entry is complete before PUBLISH is traced
      int uids[mrl::TGRP];
      for (int g = 0; g < mrl::TGRP; g++) {                  // TRAINING -> CONSUMED, log, release
        if (tid == 0) {
          const int buf = rls::ldi(&B->buf[g]), task = rls::ldi(&B->task[g]), lag = rls::ldi(&B->lag[g]);
          mrl::GroupRec* rec = C.store.rec + buf; uids[g] = rls::ldi(&B->uid[g]); gtask = task;
          rls::transition(&rec->state, rls::TRAINING, rls::CONSUMED, N.errors);
          atomicAdd(N.consumed + task, 1);
          const u64 n = tr::log_group(C.sh.glog, rec, rls::CONSUMED, (int)step, lag, tr::since(t0));
          sm[0] = n < C.sh.glog.tok_cap ? 1 : 0; sm[1] = buf; sm[2] = (int)n;
        }
        __syncthreads();
        const int cp = sm[0], buf = sm[1], n = sm[2];
        if (cp) tr::log_copy_tokens(C.sh.glog, C.store, buf, (u64)n, N.errors);
        __syncthreads();
        if (tid == 0) {                                      // the buffer first, then the unit of the admission limit it stood for
          release_buf(C, buf, rls::CONSUMED);
          atomicAdd(N.held + gtask, -1);
        }
      }
      if (tid == 0) {                                        // the step record
        released = true;
        const int ntok = rls::ldi(&B->ntok), ntile = min(rls::ldi(&B->ntile), mrl::TTILE);
        const float* fl = sf + SM_NUM + mrl::NGNORM; const float* fd = fl + mrl::TTILE;
        const int* ic = sm + SM_NUM + mrl::NGNORM + 2 * mrl::TTILE;
        float loss = 0.f, mdl = 0.f; int nclip = 0;
        for (int m = 0; m < ntile; m++) { loss = __fadd_rn(loss, fl[m]); mdl = fmaxf(mdl, fd[m]); nclip += ic[m]; }   // tiles >= ntile were not written this step
        const u64 now = tr::since(t0);
        if (step < (u64)C.updates) {
          C.tr.loss_hist[step] = loss; C.tr.step_done[step] = now;
          int* info = C.tr.step_info + step * 16; float* stat = C.tr.step_stat + step * 4;
          info[0] = ntok; info[1] = nclip; info[2] = id; info[3] = ntile;
          for (int g = 0; g < 12; g++) info[4 + g] = g < mrl::TGRP ? uids[g] : -1;
          stat[0] = loss; stat[1] = sqrtf(g2); stat[2] = 0.f; stat[3] = mdl;
        }
        tr::emit(C.sh.trace, t0, rls::EV_PUBLISH, id, slot, (int)step, polls, 0);
        tr::emit(C.sh.trace, t0, rls::EV_STEP, (int)step, ntok, nclip, ntile, 0);
      }
    }
  }
  if (tid == 0 && step + 1ull == (u64)C.updates) {           // the last update of the run
    if (valid && slot >= 0 && atomicCAS(N.stop_reason, 0, (int)rls::DONE) == 0) tr::emit(C.sh.trace, t0, rls::EV_STOP, rls::DONE, 0, 0, 0, 0);
    __threadfence();
    mk::st_rel(C.finished, 1);
  }
  // The batch is published and released: the learner no longer has one. Cleared after *finished on purpose (a rollout
  // job that takes its turn after the last update finds the flag and does not start again, see R_BEGIN), and after a
  // fence when jobs wait for this bit, so that every release above is in memory before a job can read the counters.
  if (tid == 0 && released) {
    if (C.hp.alternate) __threadfence();
    atomicAnd(N.training, ~rls::TRAINING_BATCH);
  }
}

}  // namespace ctl
