// mk_sched.cuh -- the task scheduler of the MegaRL persistent kernel.
//
// The model comes from open-flapping-airplanes (llm/megakernel/ofa_llm.cu):
//   * A JOB is an endless or bounded sequence of STEPS. A step is a fixed table of PHASES; a phase is `ntasks`
//     independent tasks of one kind. One task is executed by all 256 threads of one block.
//   * Dependencies name a phase and a step delta <= 0, and a same-step dependency must name an earlier phase. Phases
//     with zero tasks are allowed and stand for their own dependencies.
//   * A job has R ring slots: step s uses slot s % R, and completion counters are kept per (phase, slot).
//   * Unbounded jobs run until *finished is set. Every job stops when *abort is set. A block waiting for work sets
//     *abort itself once the kernel has run longer than the job's deadline, so a lost dependency ends the launch
//     instead of hanging it.
//
// Host side:    mk::Builder -> finalize(R) -> mk::upload(...) per job, then mk::wave_upload(...) for all of them.
// Device side:  mk::wave_loop(...), below.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define MK_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

namespace mk {
typedef unsigned long long u64;

struct PhaseDesc { int kind, a, b, c; };          // what a task of this phase does; interpreted by the job's run function

struct Job {
  int nph, R, id, unbounded;                     // phases per step, ring slots, job id, 1 = no step limit (runs until *finished)
  u64 tps, limit;                                // tasks per step, number of steps (bounded jobs)
  const PhaseDesc* ph;                           // [nph]
  const int *prefix, *ntasks, *dep_off, *dep_cnt;
  const int2* deps;                              // (phase, step delta)
  u64* done;                                     // [nph * R] tasks completed per (phase, slot), cumulative
  int* finished;                                 // set -> unbounded jobs stop
  int* abort;                                    // set -> every job stops (error / deadline)
  u64* phase_busy;                               // [nph] ns spent inside tasks, or nullptr
  u64* blk_busy;                                 // [1] total, or nullptr
  u64* t0;                                       // [1] device time at kernel start: initialise to ~0, the kernel entry does atomicMin(t0, gtimer())
  u64 deadline_ns;                               // 0 = none; otherwise waiting blocks set *abort when gtimer() - *t0 exceeds it
};

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ __forceinline__ int ld_vol(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(int* p, int v) { asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ u64 ld_vol64(const u64* p) { return *reinterpret_cast<const volatile u64*>(p); }
__device__ __forceinline__ u64 ld_rlx64(const u64* p) { u64 v; asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void fence_acq() { asm volatile("fence.acquire.gpu;" ::: "memory"); }

// call once per block at kernel entry (thread 0)
__device__ __forceinline__ void mark_start(u64* t0) { atomicMin(t0, gtimer()); }

// ------------------------------------------------------------------------------------------------ host: job graphs
// Zero-filled device memory. While a scope is open (dscope() points to a list), every allocation is noted in it and
// dscope_free returns all of it: megarl opens one scope per run, so that a process can make several runs and each
// starts from freshly allocated, zeroed state. Memory allocated inside an open scope must not be freed in any other way.
inline std::vector<void*>*& dscope() { static std::vector<void*>* s = nullptr; return s; }
template <class X> inline X* dalloc(size_t n) {
  X* p; MK_CK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(X))); MK_CK(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(X)));
  if (dscope()) dscope()->push_back(p);
  return p;
}
inline void dscope_free(std::vector<void*>& s) { for (size_t i = s.size(); i-- > 0;) MK_CK(cudaFree(s[i])); s.clear(); }
template <class X> inline void up(X* d, const std::vector<X>& h) { if (!h.empty()) MK_CK(cudaMemcpy(d, h.data(), h.size() * sizeof(X), cudaMemcpyHostToDevice)); }
template <class X> inline std::vector<X> down(const X* d, size_t n) { std::vector<X> h(n); if (n) MK_CK(cudaMemcpy(h.data(), d, n * sizeof(X), cudaMemcpyDeviceToHost)); return h; }

struct Dep { int phase, delta; };                 // depends on `phase` of step (this step + delta), delta <= 0

struct Builder {
  std::vector<PhaseDesc> ph; std::vector<int> ntasks; std::vector<std::string> names;
  std::vector<std::vector<Dep>> raw;              // as written by the caller
  std::vector<std::vector<Dep>> eff;              // after finalize: zero-task phases expanded, duplicates removed
  int R = 0;

  // Add a phase; returns its index. Dependencies may name phases that are added later only with delta < 0;
  // use dep() to attach such an edge once the later phase's index is known.
  void dep(int p, Dep d) { raw[p].push_back(d); }
  int add(const std::string& name, int kind, int n, std::vector<Dep> d, int a = 0, int b = 0, int c = 0) {
    PhaseDesc p; p.kind = kind; p.a = a; p.b = b; p.c = c;
    ph.push_back(p); ntasks.push_back(n); names.push_back(name); raw.push_back(d);
    return (int)ph.size() - 1;
  }
  int nph() const { return (int)ph.size(); }
  long long tps() const { long long t = 0; for (int n : ntasks) t += n; return t; }

  // Resolve and validate for R ring slots. Aborts the program with a message on an invalid graph.
  void finalize(int R_) {
    R = R_; const int n = nph(); eff.assign(n, {});
    for (int p = 0; p < n; p++) for (const Dep& e : raw[p]) {
      if (e.phase < 0 || e.phase >= n || e.delta > 0) die(p, "dependency names an unknown phase or a future step");
      if (e.delta == 0 && e.phase >= p) die(p, "a same-step dependency must name an earlier phase (lower ticket)");
    }
    // a dependency on a zero-task phase is replaced by that phase's own dependencies (transitively)
    // An edge that returns to a zero-task phase already on the path constrains nothing and is dropped (what it would
    // imply, the same dependencies one ring period earlier, follows from the same-slot ordering checked below).
    for (int p = 0; p < n; p++) {
      std::set<std::pair<int, int>> out; std::set<int> path;
      std::function<void(Dep, int)> push = [&](Dep e, int depth) {
        if (depth > 4 * n + 64) die(p, "zero-task dependency chain does not terminate");
        if (ntasks[e.phase] > 0) { out.insert({e.phase, e.delta}); return; }
        if (path.count(e.phase)) return;
        path.insert(e.phase);
        for (const Dep& f : raw[e.phase]) push(Dep{f.phase, e.delta + f.delta}, depth + 1);
        path.erase(e.phase);
      };
      if (ntasks[p] == 0) path.insert(p);
      for (const Dep& e : raw[p]) push(e, 0);
      for (auto& q : out) eff[p].push_back(Dep{q.first, q.second});
    }
    // same-slot ordering: phase p of step s must transitively depend on phase p of step s - R, otherwise the
    // per-(phase, slot) counter could be satisfied by tasks of the wrong step
    for (int p = 0; p < n; p++) {
      if (ntasks[p] == 0) continue;
      std::set<std::pair<int, int>> seen; std::vector<std::pair<int, int>> st; st.push_back({p, 0}); bool ok = false;
      while (!st.empty() && !ok) {
        auto cur = st.back(); st.pop_back();
        for (const Dep& e : eff[cur.first]) {
          const int d = cur.second + e.delta;
          if (d < -R) continue;
          if (e.phase == p && d == -R) { ok = true; break; }
          if (seen.insert({e.phase, d}).second) st.push_back({e.phase, d});
        }
      }
      if (!ok) die(p, "steps of the same ring slot are not ordered for this phase: it must depend, directly or transitively, on itself R steps earlier");
    }
  }
  void die(int p, const char* msg) const { fprintf(stderr, "mk::Builder: phase %d (%s): %s\n", p, names[p].c_str(), msg); exit(1); }
};

// Upload a finalized graph. `finished` and `abort` are device ints shared by all jobs of a kernel.
inline Job upload(const Builder& b, u64 limit, int id, int unbounded, int* finished, int* abort, u64* t0 = nullptr, u64 deadline_ns = 0) {
  if (b.R <= 0 || b.eff.size() != b.ph.size()) { fprintf(stderr, "mk::upload: Builder::finalize was not called after the last add\n"); exit(1); }
  const int n = b.nph();
  std::vector<int> pre(n + 1, 0), doff(n), dcnt(n); std::vector<int2> deps;
  for (int i = 0; i < n; i++) {
    pre[i + 1] = pre[i] + b.ntasks[i]; doff[i] = (int)deps.size(); dcnt[i] = (int)b.eff[i].size();
    for (const Dep& e : b.eff[i]) deps.push_back(make_int2(e.phase, e.delta));
  }
  Job J; memset(&J, 0, sizeof J);
  PhaseDesc* dph = dalloc<PhaseDesc>(n); up(dph, b.ph);
  int *dpre = dalloc<int>(n + 1), *dnt = dalloc<int>(n), *ddoff = dalloc<int>(n), *ddcnt = dalloc<int>(n); int2* dd = dalloc<int2>(deps.size());
  up(dpre, pre); up(dnt, b.ntasks); up(ddoff, doff); up(ddcnt, dcnt); up(dd, deps);
  J.nph = n; J.R = b.R; J.id = id; J.unbounded = unbounded; J.tps = (u64)pre[n]; J.limit = limit;
  J.ph = dph; J.prefix = dpre; J.ntasks = dnt; J.dep_off = ddoff; J.dep_cnt = ddcnt; J.deps = dd;
  J.done = dalloc<u64>((size_t)n * b.R); J.finished = finished; J.abort = abort;
  J.phase_busy = dalloc<u64>(n); J.blk_busy = dalloc<u64>(1);
  if (!t0) { t0 = dalloc<u64>(1); std::vector<u64> big(1, ~0ull); up(t0, big); }
  J.t0 = t0; J.deadline_ns = deadline_ns;
  if (J.tps == 0) { fprintf(stderr, "mk::upload: job %d has no tasks\n", id); exit(1); }
  return J;
}

// Number of steps whose last phase is complete (for R > 1 these need not be consecutive steps).
inline u64 steps_done(const Job& J, int last_phase, int ntasks_last) {
  const std::vector<u64> d = down(J.done + (size_t)last_phase * J.R, J.R); u64 s = 0;
  for (u64 x : d) s += x / (u64)std::max(1, ntasks_last);
  return s;
}

inline void print_busy(const Job& J, const Builder& b, const char* title, double steps) {
  const std::vector<u64> pb = down(J.phase_busy, J.nph);
  printf("%s: busy time by phase (block-seconds; us per task)\n", title);
  for (int i = 0; i < J.nph; i++) if (b.ntasks[i] > 0)
    printf("  %-22s %5d tasks/step %9.4f  (%6.1f us/task)\n", b.names[i].c_str(), b.ntasks[i], pb[i] * 1e-9, steps > 0 ? pb[i] * 1e-3 / ((double)b.ntasks[i] * steps) : 0.0);
}

// ------------------------------------------------------------------------------------------------ wave loop
// mk::wave_loop: every block serves every job, and a block never holds a ticket it cannot run.
//
// Windows. The phases of a step that have tasks are handed out one after another, in ticket order. The WINDOW of a
// job is the one (step, phase) whose tasks can be claimed now. The next window opens when every task of the open one
// has been claimed and the phases the next one depends on are complete. Ticket order is a topological order of the
// dependencies, so the next window can always open once the tasks handed out so far have run: no deadlock. A block
// that finds no open window in one job takes work from another instead of waiting.
//
// Claim word. K[j] = step (29 bits) | phase (12 bits) | END (bit 22) | v (22 bits). A window of m tasks is opened by a
// compare-and-swap that stores v = WK_X - m (WK_X = 2^21). Blocks claim with atomicAdd(K[j], 1): the returned value is
// a ticket iff its WK_X bit is clear, and then names step, phase and local = WK_X - 1 - v. A returned value with WK_X
// set is no ticket and changed nothing that matters. (step, phase) never repeats in a launch and v only grows inside
// a window, so an opener's compare-and-swap cannot succeed on a stale view.
//
// One strike. Adds on an exhausted word must not carry v into END, and each one makes a pending swap fail. A block
// adds (a) after it saw the word claimable, (b) once per event that named the job, (c) once after every task it
// completes, on the job it prefers next, without looking first. Once an add of kind (c) on job j has returned no
// ticket, the block makes no more of them on j until an add of kind (a) or (b) on j returns one. That keeps the adds
// on one exhausted window to a handful per block, far inside the 2^21 of headroom.
//
// Who opens the window N that follows window (s, q):
//   - the block that completes the last task of q (its add to `done` returned need - 1). If N depends on q alone
//     (WF_FAST) it swaps right away; otherwise it checks the claim word and N's dependencies (wave_try_open);
//   - the block that claims the last ticket of q, unless N depends on q (WF_NEEDS_ME). This is what lets independent
//     phases overlap: N opens while the tasks of q still run.
// Nothing is missed: N is openable once its last condition becomes true, each condition is made true by one atomic
// add of one block, and each such block executes fence.sc and then looks. The block whose add comes last in the
// order of the fences sees every condition true and swaps.
//
// Event word. A block with nothing to do polls one word, E. Whoever opens a window stores a value nobody stored before
// ((opens << 32) | (block << 8) | job); a waiting block that sees E change tries the job it names. E is a hint only:
// the block that opens a window takes a ticket of it itself, and completions never consult E.
//
// Watching. A waiting block also polls the completion counter of the phase it last worked in, if the next window
// depends on that phase alone, and opens that window itself when the phase completes.
//
// Rescue. A block that has waited for 256 polls runs wave_try_open on every job. If it finds the same window openable
// twice, RESCUE_NS apart, it opens the window and counts a rescue. By the argument above this never happens; a count
// that is not zero (megarl prints it) points at a defect of the protocol or of the device, not at wrong results.
//
// Priorities. A block looks at its home job first, then at the jobs of the mask `first` starting from `start`, then
// at the rest. After a task it adds, without looking, on the job it just served if that is its home job or both are
// in `first`, else on its home job. A job that is no block's home job can starve: the caller must give every job at
// least one home block (megarl checks this).
//
// Memory ordering (PTX memory consistency model). The words of the protocol (K, E, done, the flags) are only accessed
// with strong operations. The stores of a task reach a task that depends on it through this chain:
//   1. the writer's threads store and pass the block barrier; thread 0 executes fence.sc and atomicAdd(done):
//      a release pattern on `done`;
//   2. the opener reads `done` with a strong operation and executes fence.sc: an acquire pattern on `done`, and,
//      with its atomicCAS(K[j]), a release pattern on K[j];
//   3. the claiming block's atomicAdd(K[j]) is followed by fence.acquire and the block barrier: an acquire pattern
//      on K[j].
// A stale read can only delay a block: a task starts only on a value returned by the add on K[j], and K[j] only gets
// a new window from a block that has seen the window's dependencies complete.
//
// The memory operations go through a policy class (WaveDev, below), so that the protocol reads in one place.
constexpr unsigned WAVE_SLEEP_NS = 60;            // a block with nothing to do sleeps this long between two polls of E
constexpr unsigned RESCUE_NS = 2000000;           // how long a window must stay openable before a waiting block opens it
constexpr int WAVE_MAXJOBS = 64;
constexpr int WK_XBIT = 21, WK_ENDBIT = WK_XBIT + 1;
constexpr u64 WK_X = 1ull << WK_XBIT, WK_END = 1ull << WK_ENDBIT, WK_STEP_MAX = (1ull << 29) - 1ull;
constexpr int WK_PH_SHIFT = WK_ENDBIT + 1, WK_STEP_SHIFT = WK_PH_SHIFT + 12, WK_PH_MAX = 4095;
static_assert(WK_ENDBIT < 32 && WK_STEP_SHIFT + 29 <= 64, "claim word: both flag bits in the low half, 12 bits of phase and 29 bits of step above them");
constexpr int WF_FAST = 1, WF_NEEDS_ME = 2, WF_WRAP = 4, WF_MANY = 8;
// One record per phase q of a job (one 64-byte line): its task count, and what the window after it (phase nxt, in the
// next step if WF_WRAP) depends on.
struct WaveRec {
  int m, flags, nxt, nxt_m, ndep;                 // ndep: dependencies of nxt (the first 3 inline; WF_MANY: read the Job's tables)
  int dph[3], ddl[3], dnt[3];                     // phase, step delta relative to the step of nxt, task count of that phase
  int pad[2];
};
static_assert(sizeof(WaveRec) == 64, "one record is one 64-byte line");
struct Wave {
  u64* K;                                         // [njobs rounded up to 16] claim words; the padding is WK_END | WK_X
  u64* E;                                         // [1] event word
  const WaveRec* rec;                             // [njobs * stride], record of phase q of job j at j * stride + q
  u64* rescued;                                   // [1] windows opened by a rescue pass, summed over blocks at kernel exit
  int stride, njobs;
};
// Block state in the shared scratch. Slots 0..5 are the selection, 6 and 7 belong to the caller.
// WS_NEXT: the completion of the last task already claimed the next one (the selection slots hold it).
// WS_NB: jobs on which this block has a strike. WS_SEEN + j: the window of job j that a rescue pass of this block
// found openable, and WS_SEEN_T + j the device time of that pass (low 32 bits of the ns).
enum { WS_RES = 0, WS_PH = 1, WS_LOCAL = 2, WS_STEP_LO = 3, WS_STEP_HI = 4, WS_JOB = 5, WS_HINT = 8, WS_SEQ = 9, WS_UNB_LO = 10, WS_UNB_HI = 11,
       WS_HOME = 12, WS_START = 13, WS_FIRST_LO = 14, WS_FIRST_HI = 15, WS_LAST = 16, WS_NEXT = 17, WS_NB_LO = 18, WS_NB_HI = 19,
       WS_RESCUED = 20 };
constexpr int WS_SEEN = WS_RESCUED + 1, WS_SEEN_T = WS_SEEN + WAVE_MAXJOBS;
constexpr int WAVE_SEL = WS_SEEN_T + WAVE_MAXJOBS; // ints of block-shared scratch the caller provides

__host__ __device__ __forceinline__ u64 wave_word(u64 step, int ph, int m) { return (step << WK_STEP_SHIFT) | ((u64)ph << WK_PH_SHIFT) | (WK_X - (u64)m); }


// The job a block takes when the jobs of mask c (not zero) have claimable windows.
template <class MEM>
__device__ __forceinline__ int wave_pick(u64 c, int home, u64 first, int start) {
  if ((c >> home) & 1ull) return home;
  u64 x = c & first; if (!x) x = c;
  const u64 hi = (x >> start) << start;
  return MEM::ctz(hi ? hi : x);
}
// Tell the blocks that wait: a value no block stored before, naming job j (255: none).
template <class MEM>
__device__ __forceinline__ void wave_publish(const Wave& W, volatile int* st, int j) {
  const int s = st[WS_SEQ] + 1; st[WS_SEQ] = s;
  MEM::st(W.E, ((u64)(unsigned)s << 32) | ((u64)MEM::block() << 8) | (u64)(j & 255));
}

// This block has just swapped job j's claim word to nk (window (nstep, nxt), or the end of the job): tell the others.
template <class MEM>
__device__ __forceinline__ void wave_opened(const Wave& W, volatile int* st, int j, u64 nk, u64 nstep, int nxt, bool mine) {
  const bool ended = (nk & WK_END) != 0;
  wave_publish<MEM>(W, st, ended ? 255 : j);
  if (mine && !ended) st[WS_HINT] = j;
}

// Open the window after the open one of job j if the open one is exhausted and the next one's dependencies are
// complete. Safe to call at any time by any block. mine: the calling block will look for a task next (it then takes
// a ticket of the new window itself). True: this block opened a window (or ended the job).
// rescue: the call is a pass of a waiting block; see "Rescue" at the top of this section for what it does differently.
// The reads of the claim word and of the counters are MEM::ld_sc: after the SC fence they must return what every add
// ordered before the fence wrote ("Why nothing is missed" above rests on them). On the device they are ld.relaxed.
template <class MEM, class JF>
__device__ __forceinline__ bool wave_try_open(JF job, const Wave& W, volatile int* st, int j, bool mine, bool rescue = false) {
  const Job& J = job(j);
  MEM::fence_sc();                                              // the caller's add (to K[j] or to a counter) is ordered before the reads below, for every block
  u64 k = MEM::ld_sc(W.K + j);
  for (;;) {
    if (!(k & WK_X) || (k & WK_END)) return false;              // tickets left: the block that takes the last one will look. Or the job has ended.
    const u64 step = (k >> WK_STEP_SHIFT) & WK_STEP_MAX; const int ph = (int)((k >> WK_PH_SHIFT) & (u64)WK_PH_MAX);
    const WaveRec* r = W.rec + (size_t)j * W.stride + ph;
    const int flags = MEM::ldc(&r->flags), nxt = MEM::ldc(&r->nxt), nxt_m = MEM::ldc(&r->nxt_m), ndep = MEM::ldc(&r->ndep);
    int dph[3], ddl[3], dnt[3];
#pragma unroll
    for (int q = 0; q < 3; q++) { dph[q] = MEM::ldc(&r->dph[q]); ddl[q] = MEM::ldc(&r->ddl[q]); dnt[q] = MEM::ldc(&r->dnt[q]); }
    const u64 nstep = step + ((flags & WF_WRAP) ? 1ull : 0ull);
    u64 nk = WK_END | WK_X;                                     // a bounded job after its last step: nothing is handed out any more
    if (J.unbounded || nstep < J.limit) {
      if (nstep >= WK_STEP_MAX) { MEM::set_flag(J.abort); return false; } // the step field would wrap: end the launch instead
      bool ready = true;
      if (flags & WF_MANY) {
        const int off = MEM::ldc(J.dep_off + nxt), cnt = MEM::ldc(J.dep_cnt + nxt);
        for (int q = 0; q < cnt && ready; q++) {
          const int ex = MEM::ldc(&J.deps[off + q].x), ey = MEM::ldc(&J.deps[off + q].y);
          const long long t = (long long)nstep + ey;
          if (t < 0) continue;
          const u64 need = (u64)MEM::ldc(J.ntasks + ex) * ((u64)t / (u64)J.R + 1ull);
          if (MEM::ld_sc(J.done + (size_t)ex * J.R + (u64)t % (u64)J.R) < need) ready = false;
        }
      } else {
#pragma unroll
        for (int q = 0; q < 3; q++) if (q < ndep) {
          const long long t = (long long)nstep + ddl[q];
          if (t >= 0) {                                         // before the first step: nothing to wait for
            const u64 need = (u64)dnt[q] * ((u64)t / (u64)J.R + 1ull);
            if (MEM::ld_sc(J.done + (size_t)dph[q] * J.R + (u64)t % (u64)J.R) < need) ready = false;
          }
        }
      }
      if (!ready) return false;                                 // the block that completes the missing phase will look
      nk = wave_word(nstep, nxt, nxt_m);
    }
    if (rescue) {                                               // the window is openable, and this block is not the one whose add made it so
      if (nk & WK_END) {
        // The end of a bounded job has no dependency: the block that takes the last ticket of the last window ends
        // the job, or, if the record says WF_NEEDS_ME, the block that completes that phase does. For a pass the end
        // is due only when that phase is complete; otherwise the last tasks, however long they run, would count as
        // a missed hand-off.
        const u64 need = (u64)MEM::ldc(&r->m) * (step / (u64)J.R + 1ull);
        if (MEM::ld_sc(J.done + (size_t)ph * J.R + step % (u64)J.R) < need) return false;
      }
      const int tag = (int)((((unsigned)(k >> WK_PH_SHIFT)) << 1) | 1u), was = st[WS_SEEN + j];     // (step, phase) of the exhausted window; never 0
      const unsigned tnow = (unsigned)MEM::now();
      if (was != tag && was != tag + 1) { st[WS_SEEN + j] = tag; st[WS_SEEN_T + j] = (int)tnow; return false; }   // first sighting: note it; the block whose add it was is about to swap
      if (tnow - (unsigned)st[WS_SEEN_T + j] < RESCUE_NS) return false;
    }
    MEM::fence_sc();                                            // acquire for the strong reads of the counters, release for the swap
    const u64 old = MEM::cas(W.K + j, k, nk);
    if (old == k) { wave_opened<MEM>(W, st, j, nk, nstep, nxt, mine); return true; }
    // The word changed. If it is still the same exhausted window, adds of other blocks reached it; its successor is
    // still openable (counters only grow, and the fence above is still the release for the swap), so the swap is
    // repeated from the word as it is. Reading the dependencies again would let the adds of the other blocks win the
    // race again and again.
    u64 was = old;
    while ((was & WK_X) && !(was & WK_END) && (was >> WK_PH_SHIFT) == (k >> WK_PH_SHIFT)) {
      const u64 w2 = MEM::cas(W.K + j, was, nk);
      if (w2 == was) { wave_opened<MEM>(W, st, j, nk, nstep, nxt, mine); return true; }
      was = w2;
    }
    k = was;                                                    // another block opened the window (or ended the job): look again
  }
}

// Open the window after (step, ph) of job j, given that the phase is complete and that the next window depends on
// nothing else (WF_FAST; flags, nxt, nxt_m are those of the record of ph). k: a value read from K[j] at any time. The
// caller has executed a fence after the strong read that showed it the phase complete, and no acquire since: that
// fence is the release for the swap. A complete phase has an exhausted window, so the swap succeeds only on the
// exhausted word of this window; if another block opened the next window the word has left it.
// True: this block opened it.
template <class MEM, class JF>
__device__ __forceinline__ bool wave_open_fast(JF job, const Wave& W, volatile int* st, int j, u64 step, int ph, int flags, int nxt, int nxt_m, u64 k, bool mine) {
  const Job& J = job(j);
  const u64 here = (step << (WK_STEP_SHIFT - WK_PH_SHIFT)) | (u64)ph;
  const u64 nstep = step + ((flags & WF_WRAP) ? 1ull : 0ull);
  u64 nk = WK_END | WK_X;
  if (J.unbounded || nstep < J.limit) {
    if (nstep >= WK_STEP_MAX) { MEM::set_flag(J.abort); return false; }
    nk = wave_word(nstep, nxt, nxt_m);
  }
  for (;;) {
    if ((k & WK_END) || (k >> WK_PH_SHIFT) != here) return false;
    const u64 was = MEM::cas(W.K + j, k, nk);
    if (was == k) { wave_opened<MEM>(W, st, j, nk, nstep, nxt, mine); return true; }
    k = was;
  }
}

// One strike (see the claim word at the top of this section): no add of kind (c) on job j until an add on it returns a ticket.
__device__ __forceinline__ void wave_strike(volatile int* st, int j) {
  const u64 nb = (u64)(unsigned)st[WS_NB_LO] | ((u64)(unsigned)st[WS_NB_HI] << 32), n2 = nb | (1ull << j);
  st[WS_NB_LO] = (int)(unsigned)(n2 & 0xffffffffull); st[WS_NB_HI] = (int)(unsigned)(n2 >> 32);
}
// k: the value an add on K[j] returned; the caller has executed fence.acquire after the add. True: a ticket, written
// to the selection slots. dead: the job hands out nothing any more (*finished): a ticket of it is not run.
template <class MEM, class JF>
__device__ __forceinline__ bool wave_got(JF job, const Wave& W, volatile int* st, int j, u64 k, bool dead) {
  if (k & (WK_X | WK_END)) return false;
  if (st[WS_NB_LO] | st[WS_NB_HI]) {                            // an add on this job returned a ticket: adds without a look are allowed again
    const u64 nb = (u64)(unsigned)st[WS_NB_LO] | ((u64)(unsigned)st[WS_NB_HI] << 32), n2 = nb & ~(1ull << j);
    st[WS_NB_LO] = (int)(unsigned)(n2 & 0xffffffffull); st[WS_NB_HI] = (int)(unsigned)(n2 >> 32);
  }
  if (dead) return false;
  const u64 step = (k >> WK_STEP_SHIFT) & WK_STEP_MAX; const int ph = (int)((k >> WK_PH_SHIFT) & (u64)WK_PH_MAX);
  const int local = (int)(WK_X - 1ull - (k & (WK_X - 1ull)));
  st[WS_PH] = ph; st[WS_LOCAL] = local; st[WS_STEP_LO] = (int)(unsigned)(step & 0xffffffffull); st[WS_STEP_HI] = (int)(unsigned)(step >> 32); st[WS_JOB] = j;
  st[WS_LAST] = j;
  if (local == 0) {                                             // the last ticket of the window: it is exhausted now, by this add
    const WaveRec* r = W.rec + (size_t)j * W.stride + ph;
    if (!(MEM::ldc(&r->flags) & WF_NEEDS_ME)) wave_try_open<MEM>(job, W, st, j, false);
  }
  return true;
}

// Look at the claim word of every job: c = jobs with a claimable window, ended = jobs that have ended, kw = the word
// of job wj. The loads are issued first, all of them, before any of their values is used (16 bytes per load).
template <class MEM>
__device__ __forceinline__ void wave_look(const Wave& W, int wj, u64& c, u64& ended, u64& kw) {
  const int nj = W.njobs;
  c = 0; ended = 0; kw = WK_END;
  for (int b = 0; b < nj; b += 16) {
    u64 k[16];
#pragma unroll
    for (int i = 0; i < 8; i++) { k[2 * i] = WK_END | WK_X; k[2 * i + 1] = WK_END | WK_X; if (b + 2 * i < nj) MEM::ld2_rlx(W.K + b + 2 * i, k[2 * i], k[2 * i + 1]); }
#pragma unroll
    for (int i = 0; i < 16; i++) if (b + i < nj) {
      const unsigned lo = (unsigned)k[i];                       // both flag bits are in the low half
      c |= (u64)((lo & (unsigned)(WK_X | WK_END)) == 0u) << (b + i); ended |= (u64)((lo >> WK_ENDBIT) & 1u) << (b + i);
      if (b + i == wj) kw = k[i];
    }
  }
}

// Thread 0: find a task. Returns 0 with the selection slots written, or -2: the launch is over for this block
// (abort, or every job has ended: unbounded jobs end with *finished, bounded jobs with their last window).
// unb: mask of the unbounded jobs.
template <class MEM, class JF>
__device__ __forceinline__ int wave_claim(JF job, const Wave& W, volatile int* st, int home, u64 first, int start, u64 unb) {
  const Job& H = job(home);                                     // abort, finished, t0 and the deadline are shared by the jobs of a kernel
  const int nj = W.njobs; const u64 valid = nj >= 64 ? ~0ull : (1ull << nj) - 1ull;
  int hint = st[WS_HINT]; st[WS_HINT] = -1;
  unsigned spin = 0, polls = 0;
  for (;;) {
    // One burst: the flags, an add on the job this block opened a window of or that the event word named, the event
    // word, and the claim word of every job. Nothing below is issued before all of it has been.
    const int ab = MEM::flag(H.abort), fin = MEM::flag(H.finished);
    const bool named = hint >= 0 && hint < nj;
    u64 kh = WK_END | WK_X; if (named) kh = MEM::add(W.K + hint, 1ull);
    const u64 e = MEM::ld_rlx(W.E);
    u64 c = 0, ended = 0, kw = WK_END;
    const int wj = st[WS_LAST];                                 // the job this block served last (its home job before its first task)
    wave_look<MEM>(W, wj, c, ended, kw);
    MEM::fence_acq();                                           // acquire pattern on K[hint]: strong read by the add, then the fence
    if (ab) return -2;
    const u64 off = fin ? unb : 0ull;                           // jobs that hand out nothing any more
    if (named) {
      const int j = hint; hint = -1;
      if (wave_got<MEM>(job, W, st, j, kh, ((off >> j) & 1ull) != 0)) return 0;
    }
    hint = -1;
    c &= ~off;
    if (c) {
      int j = 0;
      j = wave_pick<MEM>(c, home, first, start);
      const u64 k = MEM::add(W.K + j, 1ull);
      MEM::fence_acq();
      if (wave_got<MEM>(job, W, st, j, k, false)) return 0;
      continue;                                                 // another block took the last ticket in between: look again
    }
    if (((ended | off) & valid) == valid) return -2;
    // Watch. If the window of job wj is exhausted and the next one waits for that phase alone (WF_FAST), this block
    // polls the phase's counter beside the event word and opens the next window itself when the phase is complete.
    // The block that completes the phase does the same; it is ONE block, on an SM that may be busy, and the blocks
    // that wait are many: the first of them to see the counter is faster than that one block on average.
    bool watch = false; u64 wstep = 0, wneed = 0; int wph = 0, wflags = 0, wnxt = 0, wnxt_m = 0; const u64* wcnt = nullptr;
    if (!(kw & WK_END) && (kw & WK_X) && !((off >> wj) & 1ull)) {
      wstep = (kw >> WK_STEP_SHIFT) & WK_STEP_MAX; wph = (int)((kw >> WK_PH_SHIFT) & (u64)WK_PH_MAX);
      const WaveRec* r = W.rec + (size_t)wj * W.stride + wph;
      wflags = MEM::ldc(&r->flags); wnxt = MEM::ldc(&r->nxt); wnxt_m = MEM::ldc(&r->nxt_m);
      const Job& J = job(wj);
      const u64 slot = J.R == 1 ? 0ull : wstep % (u64)J.R, round = J.R == 1 ? wstep : wstep / (u64)J.R;
      wneed = (u64)MEM::ldc(&r->m) * (round + 1ull); wcnt = J.done + (size_t)wph * J.R + slot;
      watch = (wflags & WF_FAST) != 0;
    }
    for (;; spin++) {                                           // wait for an event after the read of e
      const u64 e2 = MEM::ld_rlx(W.E);
      u64 d = 0; if (watch) d = MEM::ld_rlx(wcnt);              // issued with the load of E: neither waits for the other
      polls++;
      if (e2 != e) { hint = (int)(e2 & 255ull); break; }
      if (watch && d >= wneed) {
        MEM::fence_sc();                                        // acquire for the strong read of the counter, release for the swap
        wave_open_fast<MEM>(job, W, st, wj, wstep, wph, wflags, wnxt, wnxt_m, kw, true);
        hint = wj;                                              // opened by this block or by another: try it
        break;
      }
      if ((spin & 7u) == 7u) {
        if (MEM::flag(H.abort)) return -2;
        if (MEM::flag(H.finished) != fin) break;
        if (H.deadline_ns && (spin & 1023u) == 1023u && MEM::now() - MEM::ld_t0(H.t0) > H.deadline_ns) MEM::set_flag(H.abort);
        if ((spin & 255u) == 255u) {                            // a pass over the jobs, then look again (see "Rescue" at the top of this section)
          const u64 idle = valid & ~(ended | off);
          for (int j = 0; j < nj; j++) if ((idle >> j) & 1ull) { if (wave_try_open<MEM>(job, W, st, j, true, true)) st[WS_RESCUED] = st[WS_RESCUED] + 1; }
          hint = st[WS_HINT]; st[WS_HINT] = -1;
          spin++;
          break;
        }
      }
      MEM::pause();
    }
    polls = 0;
  }
}

// Thread 0, after the block barrier that follows the task (ph, step) of job j: count it; if it was the last task of
// its phase, see to the next window; and try to claim the next task. target: the job on whose claim word this block
// adds without looking first (-1: none). unb: mask of the unbounded jobs. If a ticket comes back the selection slots
// hold it and WS_NEXT is set; otherwise the caller goes on with wave_claim.
template <class MEM, class JF>
__device__ __forceinline__ void wave_complete(JF job, const Wave& W, volatile int* st, int j, int ph, u64 step, int target, u64 unb) {
  const Job& J = job(j);
  const WaveRec* r = W.rec + (size_t)j * W.stride + ph;
  const u64 slot = J.R == 1 ? 0ull : step % (u64)J.R, round = J.R == 1 ? step : step / (u64)J.R;
  u64* const cnt = J.done + (size_t)ph * J.R + slot;
  bool blind = target >= 0;
  if (blind && (st[WS_NB_LO] | st[WS_NB_HI])) blind = !((((u64)(unsigned)st[WS_NB_LO] | ((u64)(unsigned)st[WS_NB_HI] << 32)) >> target) & 1ull);
  // Release: with the block barrier before it, every store of the task precedes the strong write to the counter.
  // Nothing of this thread is in flight when the fence is issued (the accounting comes after the bump).
  MEM::fence_sc();
  // One burst: the bump, the add that claims the next task, the claim word of this job if that add is elsewhere, the
  // record of the phase and the flags.
  const u64 old = MEM::add(cnt, 1ull);
  u64 kt = WK_END | WK_X; if (blind) kt = MEM::add(W.K + target, 1ull);
  u64 kj = WK_END; if (!(blind && target == j)) kj = MEM::ld_rlx(W.K + j);
  const int m = MEM::ldc(&r->m), flags = MEM::ldc(&r->flags), nxt = MEM::ldc(&r->nxt), nxt_m = MEM::ldc(&r->nxt_m);
  const int ab = MEM::flag(J.abort), fin = MEM::flag(J.finished);
  bool have = false;
  if (blind) {
    MEM::fence_acq();                                           // acquire pattern on K[target]: strong read by the add, then the fence
    have = wave_got<MEM>(job, W, st, target, kt, ab || (fin && ((unb >> target) & 1ull)));
    if (have) st[WS_NEXT] = 1;
    else if (kt & (WK_X | WK_END)) wave_strike(st, target);     // no ticket: one strike (see the claim word at the top of this section)
    if (target == j) kj = kt + 1ull;                            // the claim word right after this block's own add
  }
  if (old + 1ull != (u64)m * (round + 1ull)) return;            // not the last task of this phase instance
  if (!(flags & WF_FAST)) { wave_try_open<MEM>(job, W, st, j, !have); return; }
  // Fast path: the next window depends on this phase instance and on nothing else.
  MEM::fence_sc();                                              // acquire for the read of the counter by the add, release for the swap
  wave_open_fast<MEM>(job, W, st, j, step, ph, flags, nxt, nxt_m, kj, !have);
}

// The memory operations of the wave loop on the device.
struct WaveDev {
  static __device__ __forceinline__ u64 ld_rlx(const u64* p) { return ld_rlx64(p); }
  static __device__ __forceinline__ u64 ld_sc(const u64* p) { return ld_rlx64(p); }     // a strong load after fence.sc (see wave_try_open)
  static __device__ __forceinline__ void ld2_rlx(const u64* p, u64& a, u64& b) { asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory"); }
  static __device__ __forceinline__ u64 add(u64* p, u64 v) { return atomicAdd(p, v); }
  static __device__ __forceinline__ u64 cas(u64* p, u64 c, u64 v) { return atomicCAS(p, c, v); }
  static __device__ __forceinline__ void st(u64* p, u64 v) { asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }
  static __device__ __forceinline__ void fence_sc() { asm volatile("fence.sc.gpu;" ::: "memory"); }
  static __device__ __forceinline__ void fence_acq() { mk::fence_acq(); }
  static __device__ __forceinline__ int flag(const int* p) { return ld_vol(p); }
  static __device__ __forceinline__ void set_flag(int* p) { st_rel(p, 1); }
  static __device__ __forceinline__ int ldc(const int* p) { return __ldg(p); }          // tables that nothing writes during the launch (rule 4)
  static __device__ __forceinline__ u64 now() { return gtimer(); }
  static __device__ __forceinline__ u64 ld_t0(const u64* p) { return ld_vol64(p); }
  static __device__ __forceinline__ void pause() { __nanosleep(WAVE_SLEEP_NS); }
  static __device__ __forceinline__ int ctz(u64 x) { return __ffsll((long long)x) - 1; }
  static __device__ __forceinline__ unsigned block() { return blockIdx.x; }
};

// job(j): const Job& of job j (0 .. W.njobs - 1). home: the job this block looks at first; first, start: see
// "Priorities" above. run(j, pd, phase, local, step): executed by ALL threads of the block for the claimed task.
// sw, sj: block-shared memory for one Wave and W.njobs Jobs (static __shared__ objects of the calling kernel).
//
// Two rules for everything thread 0 does between two tasks:
//   * Nothing is read from the kernel's parameters or from a job descriptor. They live behind the L1, which every
//     fence and acquire of the hand-off invalidates, so each read would be a round trip to the L2. The loop copies the
//     Wave and the Jobs into shared memory once and the scheduler reads only the copies.
//   * Nothing of the scheduler stays in a register across the call of the task: what thread 0 needs afterwards is
//     read back from the shared scratch. A value kept in a register across the call takes that register from every
//     task function of the kernel.
template <class JF, class F>
__device__ void wave_loop(JF job_param, const Wave& W_param, Wave* sw, Job* sj, int home, u64 first, int start, volatile int* s_sel, F run) {
  if (threadIdx.x == 0) {
    *sw = W_param;
    for (int j = 0; j < W_param.njobs; j++) sj[j] = job_param(j);
    for (int i = WS_HINT; i < WAVE_SEL; i++) s_sel[i] = 0;
    s_sel[WS_HINT] = -1;
    u64 unb = 0; for (int j = 0; j < W_param.njobs; j++) if (sj[j].unbounded) unb |= 1ull << j;
    s_sel[WS_UNB_LO] = (int)(unsigned)(unb & 0xffffffffull); s_sel[WS_UNB_HI] = (int)(unsigned)(unb >> 32);
    s_sel[WS_HOME] = home; s_sel[WS_START] = start; s_sel[WS_LAST] = home;
    s_sel[WS_FIRST_LO] = (int)(unsigned)(first & 0xffffffffull); s_sel[WS_FIRST_HI] = (int)(unsigned)(first >> 32);
  }
  const Wave& W = *sw;                                          // from here on only the copies in shared memory are used
  auto job = [sj](int j) -> const Job& { return sj[j]; };
  for (;;) {
    if (threadIdx.x == 0) {
      if (s_sel[WS_NEXT]) s_sel[WS_NEXT] = 0;                   // the completion of the last task claimed this one: WS_RES is still 0
      else {
        const u64 unb = (u64)(unsigned)s_sel[WS_UNB_LO] | ((u64)(unsigned)s_sel[WS_UNB_HI] << 32);
        const u64 fst = (u64)(unsigned)s_sel[WS_FIRST_LO] | ((u64)(unsigned)s_sel[WS_FIRST_HI] << 32);
        s_sel[WS_RES] = wave_claim<WaveDev>(job, W, s_sel, s_sel[WS_HOME], fst, s_sel[WS_START], unb);
      }
    }
    __syncthreads();
    if (s_sel[WS_RES] < 0) break;
    __syncthreads();
    u64 t0 = 0;
    if (threadIdx.x == 0) t0 = gtimer();
    {
      const int ph = s_sel[WS_PH], local = s_sel[WS_LOCAL], jsel = s_sel[WS_JOB];
      const u64 step = (u64)(unsigned)s_sel[WS_STEP_LO] | ((u64)(unsigned)s_sel[WS_STEP_HI] << 32);
      run(jsel, job(jsel).ph[ph], ph, local, step);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const u64 t1 = gtimer(), dt = t1 - t0;
      const int ph = s_sel[WS_PH], jsel = s_sel[WS_JOB], hm = s_sel[WS_HOME];
      const u64 step = (u64)(unsigned)s_sel[WS_STEP_LO] | ((u64)(unsigned)s_sel[WS_STEP_HI] << 32);
      const u64 fst = (u64)(unsigned)s_sel[WS_FIRST_LO] | ((u64)(unsigned)s_sel[WS_FIRST_HI] << 32);
      // The job this block adds on without looking: the one it just served if that is its home job or if both are in
      // `first` (megarl: an engine's block stays with another engine's window), else its home job (an engine's block
      // that ran a learner task goes back to its engine, a training block that helped an engine goes back to the learner).
      const int target = (jsel == hm || (((fst >> jsel) & 1ull) && ((fst >> hm) & 1ull))) ? jsel : hm;
      const u64 unb = (u64)(unsigned)s_sel[WS_UNB_LO] | ((u64)(unsigned)s_sel[WS_UNB_HI] << 32);
      wave_complete<WaveDev>(job, W, s_sel, jsel, ph, step, target, unb);
      // Accounting comes after the bump, so that the bump's fence does not wait for it. phase_busy: time inside the
      // tasks of the phase, whichever block ran them. blk_busy of the block's home job: its time inside tasks of any
      // job, so that blk_busy over the number of home blocks is the busy share of those blocks.
      const Job& J = job(jsel);
      if (J.phase_busy) atomicAdd(J.phase_busy + ph, dt);
      { const Job& H = job(hm); if (H.blk_busy) atomicAdd(H.blk_busy, dt); }
    }
  }
  if (threadIdx.x == 0 && W.rescued && s_sel[WS_RESCUED]) atomicAdd(W.rescued, (u64)(unsigned)s_sel[WS_RESCUED]);
}

// Host: the claim words, the event word and the records of the jobs of one kernel. bs[j] is the finalized graph that
// jobs[j] was uploaded from. Every job starts with the window of its first phase with tasks open.
inline Wave wave_upload(const std::vector<const Builder*>& bs, const std::vector<Job>& jobs) {
  const int n = (int)bs.size();
  if (n < 1 || n > WAVE_MAXJOBS || jobs.size() != bs.size()) { fprintf(stderr, "mk::wave_upload: between 1 and %d jobs, one Builder per Job\n", WAVE_MAXJOBS); exit(1); }
  int stride = 1; for (const Builder* b : bs) stride = std::max(stride, b->nph());
  std::vector<WaveRec> rec((size_t)n * stride); memset(rec.data(), 0, rec.size() * sizeof(WaveRec));
  std::vector<u64> k0((size_t)((n + 16) / 16) * 16, WK_END | WK_X);
  for (int j = 0; j < n; j++) {
    const Builder& b = *bs[j];
    if (b.R <= 0 || b.eff.size() != b.ph.size() || b.R != jobs[j].R || b.nph() != jobs[j].nph) { fprintf(stderr, "mk::wave_upload: job %d: the Builder is not the finalized graph of the Job\n", j); exit(1); }
    if (b.nph() > WK_PH_MAX + 1) { fprintf(stderr, "mk::wave_upload: job %d has %d phases, the claim word holds %d\n", j, b.nph(), WK_PH_MAX + 1); exit(1); }
    std::vector<int> ne; for (int p = 0; p < b.nph(); p++) if (b.ntasks[p] > 0) ne.push_back(p);
    if (ne.empty()) { fprintf(stderr, "mk::wave_upload: job %d has no tasks\n", j); exit(1); }
    std::vector<int> pos(b.nph(), -1); for (size_t i = 0; i < ne.size(); i++) pos[ne[i]] = (int)i;
    for (size_t i = 0; i < ne.size(); i++) {
      const int q = ne[i], nxt = ne[(i + 1) % ne.size()]; const bool wrap = i + 1 == ne.size();
      // v counts from WK_X - m up to WK_X inside a window; the adds that find it exhausted come on top and must
      // stay below 2 * WK_X (see "One strike")
      if ((u64)b.ntasks[q] > WK_X - 1ull) { fprintf(stderr, "mk::wave_upload: job %d phase %d (%s) has %d tasks, a window holds %llu\n", j, q, b.names[q].c_str(), b.ntasks[q], WK_X - 1ull); exit(1); }
      WaveRec& r = rec[(size_t)j * stride + q];
      r.m = b.ntasks[q]; r.nxt = nxt; r.nxt_m = b.ntasks[nxt]; r.flags = wrap ? WF_WRAP : 0;
      const std::vector<Dep>& d = b.eff[nxt];
      r.ndep = (int)d.size(); bool needs = false;
      for (size_t x = 0; x < d.size(); x++) {
        if (b.ntasks[d[x].phase] <= 0 || d[x].delta > 0 || (d[x].delta == 0 && (wrap || pos[d[x].phase] > (int)i))) { fprintf(stderr, "mk::wave_upload: job %d phase %d: a dependency that ticket order cannot serve\n", j, nxt); exit(1); }
        if (d[x].phase == q && d[x].delta == (wrap ? -1 : 0)) needs = true;
        if (x < 3) { r.dph[x] = d[x].phase; r.ddl[x] = d[x].delta; r.dnt[x] = b.ntasks[d[x].phase]; }
      }
      if (d.size() > 3) r.flags |= WF_MANY;
      if (needs) r.flags |= WF_NEEDS_ME;
      if (needs && d.size() == 1) r.flags |= WF_FAST;
    }
    for (const Dep& e : b.eff[ne[0]]) if (e.delta == 0) { fprintf(stderr, "mk::wave_upload: job %d: the first phase with tasks has a same-step dependency\n", j); exit(1); }
    if (jobs[j].unbounded || jobs[j].limit > 0) k0[j] = wave_word(0, ne[0], b.ntasks[ne[0]]);
  }
  Wave W; memset(&W, 0, sizeof W);
  u64* dK = dalloc<u64>(k0.size()); up(dK, k0);
  WaveRec* dr = dalloc<WaveRec>(rec.size()); up(dr, rec);
  W.K = dK; W.E = dalloc<u64>(n); W.rec = dr; W.rescued = dalloc<u64>(1); W.stride = stride; W.njobs = n;
  return W;
}
}  // namespace mk
