// megarl.cu -- the MegaRL persistent kernel: rollout engines, rewards, queues and a GRPO learner in one launch.
//
// Host: load the model and the prompt store and build the prompt caches once per process. Then, for every run:
// allocate its device state, build one job graph per rollout engine and one for the learner, launch the kernel once,
// wait, download the logs, audit them, and write the run directory.
// Device: every block serves every job through mk::wave_loop, starting with its home job from the role table.
// Options and the run.json writer live in megarl_opts.cuh.
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include "mrl_host.cuh"
#include "mrl_lmhead.cuh"
#include "rl_queue.cuh"
#include "policy_versions.cuh"
#include "rl_state.cuh"
#include "trace.cuh"
#include "reward.cuh"
#include "group_reduce.cuh"
#include "rl_control.cuh"
#include "qwen_fwd.cuh"
#include "head_sample.cuh"
#include "grpo_loss.cuh"
#include "mid_train.cuh"
#include "megarl_opts.cuh"

using namespace mrl;
using mgo::Opt;

static_assert(sizeof(GroupLogEntry) == 104 && sizeof(TraceEvent) == 32, "record sizes of groups.bin and trace.bin");
// cnt.errors has 16 slots: rls::Err from slot 0 up, and one slot each that the trunk (qw::ERR_ROW) and the head
// (hd::ERR_NO_WINNER) took from the top without including rl_state.cuh. This is the only file that includes all three
// headers, so the check lives here.
static_assert(qw::ERR_ROW != hd::ERR_NO_WINNER && qw::ERR_ROW > rls::ERR_PROMPT && qw::ERR_ROW < rls::NERR && hd::ERR_NO_WINNER > rls::ERR_PROMPT && hd::ERR_NO_WINNER < rls::NERR,
              "the trunk's and the head's error slots must be free, distinct slots of cnt.errors");

// The pipelines' buffer tables as a kernel parameter: a task reads its Pipe from the parameter space instead of
// loading C.pipe[p] from global memory.
struct PipeTable {
  Pipe p[MAXPIPE];
  mk::Wave wave;                                  // claim words, event word and records of mk::wave_loop
};

// ------------------------------------------------------------------------------------------------ device
// Reward of one finished group, for ctl::roll_advance. Called by all threads of the block.
struct Scorer {
  const Prompts* P;
  int require_eos;
  __device__ void operator()(GroupRec& rec, const int* tokens, char* smem) const {
    float* out = reinterpret_cast<float*>(smem + SMEM - 64);          // the top of the block's scratch; score_group works from the bottom
    rw::score_group(*P, tokens, rec.len, rec.key, out, smem);
    __syncthreads();
    if (threadIdx.x < GSZ) {
      float r = out[threadIdx.x];
      if (require_eos && r != 0.f) {
        const int n = rec.len[threadIdx.x];
        const int last = (n >= 1 && n <= MAXGEN) ? tokens[threadIdx.x * MAXGEN + n - 1] : -1;
        if (last != EOS_IM_END && last != EOS_ENDOFTEXT) r = 0.f;
      }
      rec.reward[threadIdx.x] = r;
    }
    __syncthreads();
  }
};

// A body task of engine `job`, routed by kind range (mrl::Kind). In a phase with lockstep multiplier pd.c > 1,
// consecutive local indices go to the engine's pipelines in turn.
__device__ __forceinline__ void dispatch_roll_body(const Ctx& C, const PipeTable& PT, int job, const mk::PhaseDesc& pd, int local, mk::u64 step, char* smem) {
  const int mult = pd.c > 1 ? pd.c : 1;
  const int pipe_idx = job * C.ppj + local % mult;
  const Pipe& P = PT.p[pipe_idx];
  const int l2 = local / mult;
  if (pd.kind >= K_H_Y0P) hd::run(C, P, pd, l2, step, smem);
  else if (pd.kind >= K_Q_MID_COHORT) mt::run_roll(C, P, pipe_idx, pd, l2, smem);
  else qw::run(C, P, pd, l2, smem);
}

__global__ void __launch_bounds__(256, 2) k_megarl(Ctx C, PipeTable PT) {
  extern __shared__ __align__(1024) unsigned char dsm[];
  __shared__ int s_sel[mk::WAVE_SEL];              // the selection, the role (slot 6) and the block state of mk::wave_loop
  __shared__ mk::Wave s_wave;                      // copies of the scheduler's tables for thread 0 (see mk::wave_loop)
  __shared__ mk::Job s_job[MAXPIPE + 1];
  char* smem = reinterpret_cast<char*>(dsm);
  wg::check_smem_base(dsm);
  if (threadIdx.x == 0) {
    mk::mark_start(C.t0);
    const int sm = (int)mk::smid();
    int role = -1;
    if (sm < C.nsm) { const int r = atomicAdd(C.sm_count + sm, 1); if (r < 2) role = C.role[2 * sm + r]; }
    s_sel[6] = role;
  }
  __syncthreads();
  const int role = s_sel[6];
  __syncthreads();
  if (role < 0) return;
  // A block's role is its home job: it looks there first, then at the other engines starting with the next one (a
  // training block: starting with an engine picked by its block index), then at the learner.
  mk::wave_loop([&](int j) -> const mk::Job& { return j < C.nroll ? C.roll[j] : C.train; }, PT.wave, &s_wave, s_job, role, (1ull << C.nroll) - 1ull,
                role < C.nroll ? role : (int)(blockIdx.x % (unsigned)C.nroll), s_sel,
                [&](int job, const mk::PhaseDesc& pd, int ph, int local, mk::u64 step) {
    if (job == C.nroll) {
      if (pd.kind == K_T_SELECT) ctl::train_select(C, step, smem);
      else if (pd.kind == K_T_FINISH) ctl::train_finish(C, step, smem);
      else if (!C.hp.no_train) mt::run_train(C, pd, local, step, smem);
      return;
    }
    if (pd.kind == K_R_BEGIN) { ctl::roll_begin(C, job, step, smem); return; }
    if (pd.kind == K_R_ADVANCE) { Scorer sc; sc.P = &C.prompts; sc.require_eos = C.hp.require_eos; ctl::roll_advance(C, job, step, smem, sc); return; }
    dispatch_roll_body(C, PT, job, pd, local, step, smem);
  });
}

// ------------------------------------------------------------------------------------------------ host: helpers
// Job of each (SM, arrival rank). share: rank 0 generates everywhere, rank 1 trains on train_sms SMs spread evenly.
// split: the last train_sms SMs train with both blocks. SMs are dealt to the engines in contiguous ranges.
static std::vector<int> make_roles(const Opt& o, int nsm, int& train_blocks, std::vector<int>& roll_blocks) {
  std::vector<int> role(2 * (size_t)nsm, -1); train_blocks = 0; roll_blocks.assign(o.nroll, 0);
  if (o.layout == "split") {
    const int gen = nsm - o.train_sms;
    for (int s = 0; s < nsm; s++) for (int r = 0; r < 2; r++) role[2 * s + r] = s < gen ? (int)((long long)s * o.nroll / gen) : o.nroll;
  } else {
    for (int s = 0; s < nsm; s++) {
      const int job = (int)((long long)s * o.nroll / nsm);
      const bool trains = ((long long)(s + 1) * o.train_sms / nsm) != ((long long)s * o.train_sms / nsm);
      role[2 * s] = job; role[2 * s + 1] = trains ? o.nroll : job;
    }
  }
  for (int x : role) { if (x == o.nroll) train_blocks++; else if (x >= 0) roll_blocks[x]++; }
  return role;
}

static void print_graph(const mk::Builder& b, const char* title) {
  printf("%s: %d phases, %lld tasks per step\n", title, b.nph(), b.tps());
  for (int p = 0; p < b.nph(); p++) {
    printf("  %3d %-14s kind %3d layer %2d tasks %5d  deps:", p, b.names[p].c_str(), b.ph[p].kind, b.ph[p].a, b.ntasks[p]);
    for (const mk::Dep& d : b.eff[p]) printf(" %s%s", b.names[d.phase].c_str(), d.delta ? ("(" + std::to_string(d.delta) + ")").c_str() : "");
    printf("\n");
  }
}

// The deadline (hp.deadline_ns, which is also the time budget of --budget-s) is acted on by the RL core's control
// tasks: they record the stop reason, trace the STOP event and raise *abort. The scheduler's own deadline, the one a
// block waiting on a dependency applies, is set this much later: it is the watchdog for a kernel in which no control
// task runs any more, and it knows nothing of the stop reason. With equal deadlines the scheduler would often notice
// first, and the run would end without a recorded reason or STOP event (see rlc::resolve_stop_reason).
constexpr u64 WATCHDOG_NS = 1000000000ull;

// The job graph of a rollout engine and that of the learner (host only). Returns the index of the sampling phase,
// whose completion counter counts the engine's steps.
static int roll_graph(mk::Builder& b, const Opt& o) {
  int ph_sample = -1;
  return mt::add_roll_mid(b, o.ngrp, o.hp, o.ppj, ph_sample);
}
static void train_graph(mk::Builder& tb, const Opt& o) {
  const int sel = tb.add("T_SELECT", K_T_SELECT, 1, {}); int la = -1, lb = -1;
  mt::add_train_mid(tb, sel, o.hp, la, lb);
  const int ph_fin = tb.add("T_FINISH", K_T_FINISH, 1, {{la, 0}});
  tb.dep(sel, mk::Dep{ph_fin, -1}); tb.finalize(1);
}

// ------------------------------------------------------------------------------------------------ host: the process
// What the runs of a process share: the device, the model, the prompt store with its caches, and the times of setting
// them up. Times are seconds on one clock that starts at the entry of main.
struct Process {
  const mgo::Proc* pr = nullptr; cudaDeviceProp prop; int nsm = 0;
  Model model; Prompts prompts; host::PromptsHost ph; std::vector<long long> prompt_ids; std::string prompt_sha;
  std::vector<std::string> env_names;
  int cache_ks[4] = {0, 0, 0, 0};               // the trunk's chunk counts (ksQ, ksO, ksG, ksD) the prompt caches were built with
  double pre_load_s = 0, load_s = 0, cache_s = 0;   // creation of the process to the start of loading; loading; building the caches now in use
  double pre_main_s = 0;                        // the age of the process at the entry of main (part of pre_load_s)
  int from_creation = 0;                        // 0: /proc could not be read and pre_load_s starts at the entry of main
  std::chrono::steady_clock::time_point clock0;
  double now() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock0).count(); }
};

// The prompt caches depend on the trunk's chunk counts (the order of the partial sums) and on nothing else of a run.
static void build_caches(Process& P, const Hyper& hp, bool again) {
  // the cache build frees its own working memory directly: inside an open allocation scope that would be freed twice
  if (mk::dscope()) host::die("megarl: the prompt caches must not be built inside a run's allocation scope");
  if (again) {                                  // as a fresh process has them: zero where the build does not write
    MK_CK(cudaMemset(P.prompts.kv, 0, (size_t)P.prompts.n * NL * MAXPROMPT * KVW * sizeof(float)));
    MK_CK(cudaMemset(P.prompts.hlast, 0, (size_t)P.prompts.n * D * sizeof(float)));
  }
  qw::build_prompt_cache(P.model, P.prompts, hp);
  MK_CK(cudaDeviceSynchronize());
  P.cache_ks[0] = hp.ksQ; P.cache_ks[1] = hp.ksO; P.cache_ks[2] = hp.ksG; P.cache_ks[3] = hp.ksD;
}

// ------------------------------------------------------------------------------------------------ host: one run
// The device memory of one run: everything a launch reads or writes except what the runs of a process share. The
// caller has an allocation scope open. megarl also calls this once per shape of run before the first run of a list,
// and frees the result at once: a run whose state does not fit the device then stops the process at its start.
struct RunState { Ctx C; PipeTable PT; std::vector<Pipe> pipes; int nbuf = 0, nslots = 0; };
static void alloc_run_state(const Opt& o, int nsm, const std::vector<int>& role, RunState& st) {
  const Hyper& hp = o.hp; const int npipe = o.nroll * o.ppj, nrows = GSZ * o.ngrp;
  Ctx& C = st.C; memset(&C, 0, sizeof C); memset(&st.PT, 0, sizeof st.PT);
  st.pipes.assign(npipe, Pipe());
  for (int p = 0; p < npipe; p++) { st.pipes[p] = host::alloc_pipe(nrows); st.PT.p[p] = st.pipes[p]; }
  Pipe* dpipe = mk::dalloc<Pipe>(npipe); mk::up(dpipe, st.pipes);
  C.pipe = dpipe; C.npipe = npipe; C.ngrp = o.ngrp; C.nroll = o.nroll; C.ppj = o.ppj; C.nsm = nsm; C.hp = hp; C.updates = o.updates;
  for (int p = 0; p < MAXPIPE; p++) C.pipe_task[p] = o.bind ? p % NTASKS : -1;
  st.nbuf = npipe * o.ngrp + NTASKS * hp.target_per_task + TGRP + o.nroll;
  st.nslots = std::max(4, std::min(hp.lag > (1 << 20) - 8 ? (1 << 20) : hp.lag + 4, npipe * o.ngrp + 3));
  C.store = host::alloc_store(st.nbuf);
  C.tr = host::alloc_train(o.updates, hp.tksV);
  rlc::Sizes z; z.nbuf = st.nbuf; z.nslots = st.nslots;
  const int arch_stride = std::max(1, (o.updates + 25) / 26);
  z.narch = (o.updates / arch_stride) + 2;
  z.trace_cap = (mk::u64)o.trace_cap; z.log_cap = (mk::u64)o.log_cap; z.tok_cap = (mk::u64)o.tok_cap;
  C.sh = rlc::host_alloc(z);
  C.sh.ver.arch_stride = arch_stride;
  C.finished = mk::dalloc<int>(1); C.abort = mk::dalloc<int>(1);
  C.t0 = mk::dalloc<u64>(1); { std::vector<u64> big(1, ~0ull); mk::up(C.t0, big); }
  int* drole = mk::dalloc<int>(role.size()); mk::up(drole, role); C.role = drole; C.sm_count = mk::dalloc<int>(nsm);
}
// What decides how much device memory a run needs: two runs that agree in these need the same.
static std::vector<long long> shape_of(const Opt& o) {
  return {o.nroll, o.ppj, o.ngrp, o.hp.target_per_task, o.hp.lag, o.updates, o.hp.tksV, o.trace_cap, o.log_cap, o.tok_cap};
}

// Run number r of the process (t_start: when its preparation began, on the process clock). All device state of the run
// is allocated here, zero-filled, and freed before the function returns, so a run does not see what an earlier run of
// the process left. What it shares with the other runs it only reads: the model, the prompt tokens and tables, and the
// prompt caches (the hand-out order of the prompts is uploaded anew). Returns true if the run is complete and consistent.
static bool run_one(Process& P, size_t r, double t_start) {
  const mgo::Proc& pr = *P.pr; const Opt& o = pr.runs[r]; const Hyper& hp = o.hp; const int nsm = P.nsm;
  host::die_context() = pr.listed ? "run " + std::to_string(r + 1) + " (--out " + o.out + "): " : "";
  if (pr.listed) { printf("\n==== run %zu of %zu of process %s:", r + 1, pr.runs.size(), pr.id.c_str()); for (const std::string& w : o.run_argv) printf(" %s", w.c_str()); printf("\n"); }
  host::check_hyper(hp);                        // checked when the command line was read; its warnings once more, in this run's console
  const int npipe = o.nroll * o.ppj;
  printf("device: %s, %d SMs. engines %d x %d pipeline(s) x %d groups x %d answers; batch %d groups; lag %d; target %d per environment; layout %s (%d training SMs)\n",
         P.prop.name, nsm, o.nroll, o.ppj, o.ngrp, GSZ, TGRP, hp.lag, hp.target_per_task, o.layout.c_str(), o.train_sms);
  if (hp.count_held || hp.alternate || hp.no_train)
    printf("control options:%s%s%s\n", hp.count_held ? " held groups count against the target;" : "", hp.alternate ? " engines wait while a batch is trained;" : "", hp.no_train ? " NO TRAINING (the learner's phases are skipped);" : "");
  // what the lag rule allows: a group admitted behind n usable groups of its environment sees about n / per_update updates
  { const int per_update = std::max(1, TGRP / NTASKS); const int slots = npipe * o.ngrp / (o.bind ? NTASKS : 1);
    if (o.bind && npipe % NTASKS != 0) fprintf(stderr, "warning: --bind 1 with %d pipelines gives the environments unequal capacity\n", npipe);
    if (hp.lag < (1 << 20) && std::min(slots, hp.target_per_task) > (hp.lag + 1) * per_update)
      fprintf(stderr, "warning: up to %d groups per environment are in flight but lag %d admits only about %d per generation time: expect stale drops\n", std::min(slots, hp.target_per_task), hp.lag, (hp.lag + 1) * per_update); }
  if (r > 0) {                                  // the first run's console has these lines from the process itself
    printf("loaded model and %d prompts (", P.prompts.n);
    for (int t = 0; t < NTASKS; t++) printf("%s%d %s", t ? ", " : "", P.prompts.ntask[t], P.env_names[t].c_str());
    printf(") in %.1f s; prompt store sha256 %.16s (once, before the first run of this process)\n", P.load_s, P.prompt_sha.c_str());
  }

  // ---- prompt caches and prompt order of this run
  double rebuild_s = 0; bool rebuilt = false;
  if (hp.ksQ != P.cache_ks[0] || hp.ksO != P.cache_ks[1] || hp.ksG != P.cache_ks[2] || hp.ksD != P.cache_ks[3]) {
    const double t = P.now();
    build_caches(P, hp, true);
    rebuilt = true; rebuild_s = P.now() - t; P.cache_s = rebuild_s;
    printf("prompt caches built in %.1f s (again, for this run's chunk counts)\n", rebuild_s);
  }
  const int cache_shared = r > 0 && !rebuilt ? 1 : 0;
  if (cache_shared) printf("prompt caches built in %.1f s (before this run, by an earlier run of this process with the same chunk counts)\n", P.cache_s);
  std::vector<int> prompt_order[NTASKS];                      // per environment: the prompt indices in the order they are handed out
  host::set_prompt_order(P.prompts, P.ph, pr.base.use_per_task, (u64)o.prompt_seed, prompt_order);

  // ---- device state: everything from here to the job graphs belongs to this run
  int train_blocks = 0; std::vector<int> roll_blocks;
  const std::vector<int> role = make_roles(o, nsm, train_blocks, roll_blocks);
  std::vector<void*> scope; mk::dscope() = &scope;
  RunState st; alloc_run_state(o, nsm, role, st);
  Ctx& C = st.C; PipeTable& PT = st.PT; const int nbuf = st.nbuf, nslots = st.nslots;
  C.model = P.model; C.prompts = P.prompts;
  C.mid = mt::alloc_mid_state(P.model, C.tr);
  rlc::host_reset(C.sh, C.store, C.tr);

  // ---- job graphs: one per engine (each needs its own counters), one for the learner
  std::vector<mk::Builder> rb(o.nroll); int ph_sample = -1;
  for (int j = 0; j < o.nroll; j++) {
    ph_sample = roll_graph(rb[j], o);
    C.roll[j] = mk::upload(rb[j], 0, j, 1, C.finished, C.abort, C.t0, hp.deadline_ns + WATCHDOG_NS);
  }
  mk::Builder tb; train_graph(tb, o);
  C.train = mk::upload(tb, (mk::u64)o.updates, o.nroll, 0, C.finished, C.abort, C.t0, hp.deadline_ns + WATCHDOG_NS);
  // A job that is no block's home job could wait for ever: the wave loop only looks at other jobs once its home is idle.
  { bool none = train_blocks < 1; for (int x : roll_blocks) none = none || x < 1; if (none) host::die("every job needs at least one home block: change the layout"); }
  { std::vector<const mk::Builder*> wb; std::vector<mk::Job> wj;
    for (int j = 0; j < o.nroll; j++) { wb.push_back(&rb[j]); wj.push_back(C.roll[j]); }
    wb.push_back(&tb); wj.push_back(C.train);
    PT.wave = mk::wave_upload(wb, wj); }
  mk::dscope() = nullptr;
  printf("rollout step: %d phases, %lld tasks; update: %d phases, %lld tasks; blocks: training %d", rb[0].nph(), rb[0].tps(), tb.nph(), tb.tps(), train_blocks);
  for (int j = 0; j < o.nroll; j++) printf(", engine %d %d", j, roll_blocks[j]);
  printf("; buffers %d, snapshot slots %d\n", nbuf, nslots);
  if (o.dry) { print_graph(rb[0], "rollout job"); print_graph(tb, "training job"); mk::dscope_free(scope); host::die_context() = ""; return true; }

  // ---- the one launch
  MK_CK(cudaFuncSetAttribute(k_megarl, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  const double t_launch = P.now();
  k_megarl<<<2 * nsm, 256, SMEM>>>(C, PT);
  MK_CK(cudaGetLastError());
  MK_CK(cudaDeviceSynchronize());
  const double wall = P.now() - t_launch;
  // Set-up of this run as if it were the only run of its process: what the process did before loading, loading, the
  // prompt caches this run used, and the run's own preparation. For the first run of a process this is the time from
  // the creation of the process (or the entry of main) to the launch, less the trial allocations of a list of runs.
  const double prep_s = (t_launch - t_start) - rebuild_s, setup_s = P.pre_load_s + P.load_s + P.cache_s + prep_s;

  // ---- results
  const std::vector<int> smc = mk::down(C.sm_count, nsm); int bad_sm = 0; for (int x : smc) if (x != 2) bad_sm++;
  if (bad_sm) fprintf(stderr, "warning: %d SMs did not receive exactly two blocks; the role table assumed they would\n", bad_sm);
  const rlc::Snapshot S = rlc::host_download(C.sh);
  const int violations = rlc::host_audit(C.sh, C.store, S, hp, o.audit_verbose != 0);
  const int steps_done = S.train_ver;
  const std::vector<float> loss = mk::down(C.tr.loss_hist, o.updates), stat = mk::down(C.tr.step_stat, (size_t)o.updates * 4);
  const std::vector<u64> tdone = mk::down(C.tr.step_done, o.updates); const std::vector<int> info = mk::down(C.tr.step_info, (size_t)o.updates * 16);
  const u64 t_end = steps_done > 0 ? tdone[steps_done - 1] : 0;
  const double kern_s = steps_done > 0 ? (double)t_end * 1e-9 : wall;
  // Device time of the last logged event: when the kernel stopped. The counters and the step counts are those at
  // exit, so rates are taken over this time; it equals the time of the last update (kern_s) to within microseconds
  // only for a run that ends with its last update.
  u64 t_exit = t_end; for (const TraceEvent& e : S.trace) t_exit = std::max(t_exit, e.t);
  const double exit_s = t_exit > 0 ? (double)t_exit * 1e-9 : wall;
  long long consumed = 0, finished = 0, admitted = 0;
  for (int t = 0; t < NTASKS; t++) { consumed += S.consumed[t]; finished += S.finished[t]; admitted += S.admitted[t]; }
  printf("\nrun: %d of %d updates in %.3f s (kernel wall %.3f s), stop reason %d, audit violations %d\n", steps_done, o.updates, kern_s, wall, S.stop_reason, violations);
  printf("  kernel stopped at %.3f s; the token, group, step and busy figures below are taken over that time\n", exit_s);
  if (S.stop_reason_dev != S.stop_reason)
    printf("  the device recorded stop reason %d; %d is what the host resolved: the scheduler's watchdog ended the launch and no control task recorded why\n", S.stop_reason_dev, S.stop_reason);
  printf("  set-up %.2f s from %s to the launch, counted as for the only run of a process: %.2f before loading, %.2f loading, %.2f prompt caches%s, %.2f this run's state and job graphs\n",
         setup_s, P.from_creation ? "the creation of the process" : "the entry of main", P.pre_load_s, P.load_s, P.cache_s, cache_shared ? " (built before this run)" : "", prep_s);
  if (o.budget_s > 0) printf("  time budget %.3f s: %s\n", o.budget_s, S.stop_reason == 2 && S.stop_reason_dev == 2 ? "reached" : S.stop_reason == 2 ? "passed without the control tasks noticing: the watchdog ended the launch" : S.stop_reason == 1 ? "NOT reached, the run ended at --updates first" : "not reached");
  printf("  tokens generated %llu (%.0f per s), groups admitted %lld, finished %lld (%.1f per s), trained on %lld, updates per s %.1f\n",
         (unsigned long long)S.tokens_generated, S.tokens_generated / exit_s, admitted, finished, finished / exit_s, consumed, steps_done / kern_s);
  for (int t = 0; t < NTASKS; t++)
    printf("  %-8s admitted %d finished %d trained %d dropped: equal rewards %d, stale in queue %d, cancelled %d; at exit generating %d queued %d\n", P.env_names[t].c_str(),
           S.admitted[t], S.finished[t], S.consumed[t], S.drop_zerovar[t], S.drop_stale_queued[t], S.drop_stale_inflight[t], S.inflight[t], S.queued[t]);
  if (steps_done > 0) printf("  loss first %.5f last %.5f; largest |logp_new - logp_beh| over the run %.3e\n", loss[0], loss[steps_done - 1],
                             [&]() { float m = 0; for (int i = 0; i < steps_done; i++) m = std::max(m, stat[(size_t)i * 4 + 3]); return m; }());
  // per job: blocks, steps completed, total busy time of its blocks, and per phase the task count and the time spent inside tasks
  std::vector<mgo::JobRec> jobs;
  for (int j = 0; j <= o.nroll; j++) {
    const bool tr = j == o.nroll; const mk::Job& J = tr ? C.train : C.roll[j]; const mk::Builder& b = tr ? tb : rb[j];
    mgo::JobRec jr; jr.name = tr ? std::string("learner") : "engine " + std::to_string(j); jr.blocks = tr ? train_blocks : roll_blocks[j];
    jr.steps = tr ? (double)steps_done : (double)mk::steps_done(J, ph_sample, b.ntasks[ph_sample]);
    const std::vector<u64> pb = mk::down(J.phase_busy, J.nph), bb = mk::down(J.blk_busy, 1); jr.busy_ns = bb[0];
    for (int i = 0; i < J.nph; i++) { jr.ph_name.push_back(b.names[i]); jr.ph_kind.push_back(b.ph[i].kind); jr.ph_tasks.push_back(b.ntasks[i]); jr.ph_busy.push_back(pb[i]); }
    jobs.push_back(jr);
  }
  for (int j = 0; j < o.nroll; j++)
    printf("  engine %d: %.0f steps (%.3f ms per step), blocks busy %.1f%%\n", j, jobs[j].steps, jobs[j].steps > 0 ? exit_s * 1e3 / jobs[j].steps : 0.0, 100.0 * jobs[j].busy_ns * 1e-9 / (exit_s * std::max(1, roll_blocks[j])));
  printf("  learner: blocks busy %.1f%%\n", 100.0 * jobs[o.nroll].busy_ns * 1e-9 / (exit_s * std::max(1, train_blocks)));
  // Every block serves every job: busy_ns of a job is the time its home blocks spent inside tasks of any job, while the
  // per-phase times are the time inside that phase's tasks by whichever block ran them.
  { const u64 rescued = mk::down(PT.wave.rescued, 1)[0];
    double all_busy = 0; for (int j = 0; j <= o.nroll; j++) all_busy += jobs[j].busy_ns * 1e-9;
    int allb = train_blocks; for (int x : roll_blocks) allb += x;
    printf("  all %d blocks busy %.1f%% of the time\n", allb, 100.0 * all_busy / (exit_s * std::max(1, allb)));
    printf("  time inside the tasks of each job, by any block, in blocks kept busy on average:");
    for (int j = 0; j <= o.nroll; j++) { u64 s = 0; for (u64 x : jobs[j].ph_busy) s += x; printf(" %s %.1f%s", jobs[j].name.c_str(), s * 1e-9 / exit_s, j < o.nroll ? "," : "\n"); }
    printf("  scheduler rescues: %llu\n", (unsigned long long)rescued);
    if (rescued) fprintf(stderr, "warning: %llu window hand-offs were missed and repaired by a rescue pass (see \"Rescue\" in mk_sched.cuh). The results are not affected, but the scheduler has a defect\n", (unsigned long long)rescued);
    // A job that stopped while every block stayed busy is seen by no rescue pass; engines this far apart in steps are the sign of it.
    double smax = 0; for (int j = 0; j < o.nroll; j++) smax = std::max(smax, jobs[j].steps);
    for (int j = 0; j < o.nroll; j++) if (jobs[j].steps < 0.5 * smax)
      fprintf(stderr, "warning: engine %d made %.0f steps and the engine with the most made %.0f. It spent %.3f s inside R_BEGIN (waiting for work counts there); if that does not explain the difference, the scheduler left the engine standing\n",
              j, jobs[j].steps, smax, jobs[j].ph_busy.empty() ? 0.0 : jobs[j].ph_busy[0] * 1e-9); }
  mk::print_busy(C.roll[0], rb[0], "engine 0", jobs[0].steps);
  mk::print_busy(C.train, tb, "learner", (double)steps_done);

  // ---- run directory
  host::write_file(o.out + "/groups.bin", S.log);
  host::write_file(o.out + "/group_tokens.i32", S.log_tokens);
  host::write_file(o.out + "/group_logp.f32", S.log_logp);
  host::write_file(o.out + "/trace.bin", S.trace);
  host::write_file(o.out + "/mid_arch.bf16", S.arch);
  host::write_file(o.out + "/mid_cur.bf16", S.cur_w);
  if (C.sh.ver.arch_id && C.sh.ver.narch > 0) {
    const std::vector<int> aid = mk::down(C.sh.ver.arch_id, C.sh.ver.narch);
    host::write_file(o.out + "/mid_arch_ids.i32", aid);
  }
  { FILE* mf = fopen((o.out + "/mid_layer.json").c_str(), "w");
    if (mf) { fprintf(mf, "{\"mid_layer\": %d, \"mid_nw\": %zu, \"arch_stride\": %d, \"narch\": %d, \"from_last\": 1}\n",
                      LMID, MID_NW, C.sh.ver.arch_stride, C.sh.ver.narch); fclose(mf); } }
  { struct __attribute__((packed)) StepRec { u64 t_done; float loss, gnorm, reserved, max_dlogp; int info[16]; };
    static_assert(sizeof(StepRec) == 88, "steps.bin record");
    std::vector<StepRec> sr(steps_done);
    for (int i = 0; i < steps_done; i++) { sr[i].t_done = tdone[i]; sr[i].loss = stat[(size_t)i * 4]; sr[i].gnorm = stat[(size_t)i * 4 + 1]; sr[i].reserved = 0.f; sr[i].max_dlogp = stat[(size_t)i * 4 + 3];
      memcpy(sr[i].info, &info[(size_t)i * 16], sizeof sr[i].info); }
    host::write_file(o.out + "/steps.bin", sr); }
  { mgo::RunRecord R; R.o = o; R.npipe = npipe; R.nbuf = nbuf; R.nslots = nslots;
    for (int p = 0; p < npipe; p++) R.pipe_task.push_back(C.pipe_task[p]);
    R.prompt_ids = P.prompt_ids; for (int t = 0; t < NTASKS; t++) R.prompt_order[t] = mgo::ll(prompt_order[t]);
    R.prompt_sha = P.prompt_sha; R.process_id = pr.id; R.argv = pr.argv; R.run_index = (int)r; R.run_count = (int)pr.runs.size();
    R.S = &S; R.violations = violations; R.steps_done = steps_done; R.t_end = t_end; R.t_exit = t_exit; R.wall = wall;
    R.pre_main_s = P.pre_main_s; R.pre_load_s = P.pre_load_s; R.load_s = P.load_s; R.cache_s = P.cache_s; R.prep_s = prep_s; R.setup_s = setup_s; R.from_creation = P.from_creation; R.cache_shared = cache_shared;
    R.train_blocks = train_blocks; R.bad_sm = bad_sm; R.roll_blocks = mgo::ll(roll_blocks); R.jobs = jobs;
    mgo::write_run_json(o.out, R); }
  printf("wrote %s\n", o.out.c_str());
  // Complete: every update done, or stopped at its own time budget by the RL core (the device recorded the deadline).
  // A launch that only the scheduler's watchdog ended ran a second past its budget without any control task noticing:
  // that is a stuck kernel, not a finished run, whatever reason the host resolves for it.
  const bool ok = violations == 0 && ((S.stop_reason == 1 && steps_done == o.updates) || (o.budget_s > 0 && S.stop_reason == 2 && S.stop_reason_dev == 2));
  printf("megarl: %s\n", ok ? "RUN COMPLETE" : "RUN INCOMPLETE OR INCONSISTENT");
  mk::dscope_free(scope);
  host::die_context() = "";
  return ok;
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char** argv) {
  Process P; P.clock0 = std::chrono::steady_clock::now();
  const double age = mgo::process_age_s();                      // how old the process is as main is entered
  // A program reaches main a few milliseconds after its process was created. A process that is older has done other
  // work before: a shell replaces itself by the last command of a `bash -c` line, and megarl then inherits the shell's
  // age with everything the line did before. Such an age is not megarl's start: the set-up is then counted from the
  // entry of main (and the age is kept in timing.pre_main_s for the record).
  const bool inherited = age > 1.0;
  P.from_creation = age >= 0 && !inherited ? 1 : 0;
  setvbuf(stdout, nullptr, _IOLBF, 1 << 16);                    // whole lines, whatever standard output is: the order with standard error is kept
  mgo::Proc pr = mgo::parse_process(argc, argv); P.pr = &pr;
  MK_CK(cudaGetDeviceProperties(&P.prop, 0));
  const int nsm = P.nsm = P.prop.multiProcessorCount;
  std::vector<std::pair<dev_t, ino_t>> seen_dirs;
  // Every run is checked before anything is loaded: --train-sms (its default and range need the device), the limits
  // of the graph builders, and the output directory.
  for (size_t r = 0; r < pr.runs.size(); r++) {
    Opt& o = pr.runs[r];
    host::die_context() = pr.listed ? "--run number " + std::to_string(r + 1) + ": " : "";
    if (o.train_sms < 0) o.train_sms = o.layout == "share" ? nsm / 2 : nsm / 4;
    if (o.train_sms < 1 || o.train_sms > nsm - (o.layout == "split" ? o.nroll : 0)) host::die("train-sms out of range for this layout");
    // every limit the graph builders check, now and not in the middle of a list of runs
    if (pr.listed) fprintf(stderr, "checking --run number %zu: %s\n", r + 1, o.out.c_str());
    { mk::Builder b, tb; roll_graph(b, o); train_graph(tb, o); }
    if (pr.base.dry) continue;                                  // --dry writes nothing
    if (mgo::file_exists(o.out + "/run.json") && !pr.base.overwrite)
      host::die(o.out + " already holds a run (run.json). megarl does not write over a run: name another --out, or give --overwrite 1 to replace it");
    mgo::mkdir_p(o.out);                                        // a directory that cannot be made stops the process now
    if (access(o.out.c_str(), W_OK | X_OK) != 0) host::die("cannot write into " + o.out);
    // two runs into one directory, by whatever names: decided by what the directory is
    struct stat sd; if (stat(o.out.c_str(), &sd) != 0) host::die("cannot examine " + o.out);
    for (size_t q = 0; q < seen_dirs.size(); q++)
      if (seen_dirs[q].first == sd.st_dev && seen_dirs[q].second == sd.st_ino)
        host::die("it writes to the same directory as --run number " + std::to_string(q + 1) + " (" + pr.runs[q].out + " and " + o.out + ")");
    seen_dirs.push_back({sd.st_dev, sd.st_ino});
  }
  host::die_context() = "";
  // Every run directory keeps the text of its own run in console.log. The first run's file is opened here, so that it
  // also holds the set-up of the process.
  mgo::Console con;
  auto open_console = [&](size_t r) {
    const std::string& out = pr.runs[r].out;
    host::die_context() = pr.listed ? "run " + std::to_string(r + 1) + " (--out " + out + "): " : "";
    mgo::mkdir_p(out);
    // an older run in this directory (--overwrite 1) ends here: without its run.json no reader takes the files of two
    // runs for one, whatever happens to this run
    if (mgo::file_exists(out + "/run.json") && remove((out + "/run.json").c_str()) != 0) host::die("cannot remove the run.json of the run that " + out + " held");
    if (!con.open(out + "/console.log")) fprintf(stderr, "warning: cannot write %s/console.log: the text of this run goes to the terminal only\n", out.c_str());
    host::die_context() = "";
  };
  if (!pr.base.dry) open_console(0);
  if (age < 0) fprintf(stderr, "warning: /proc/self/stat cannot be read: set-up times start at the entry of main, not at the creation of the process\n");
  if (inherited)
    fprintf(stderr, "warning: this process was %.2f s old when megarl's main was entered: megarl was started by a shell that replaced itself with it (the last command of a "
                    "`bash -c` line) after other work. Set-up times of this process start at the entry of main, not at the creation of the process; "
                    "start megarl as a process of its own\n", age);
  if (pr.listed) printf("megarl: process %s with %zu runs\n", pr.id.c_str(), pr.runs.size());

  // ---- shared by the runs: model, prompt store, prompt caches
  const double t_a = P.now();
  P.model = host::load_model(pr.base.weights);
  P.ph = host::read_prompts(pr.base.prompts);
  P.env_names = host::read_env_names(pr.base.prompts);
  P.prompts = host::upload_prompts(P.ph, pr.base.use_per_task, true);          // store order; every run sets its own order
  { int cnt[NTASKS] = {0}; for (int i = 0; i < P.ph.n; i++) { if (pr.base.use_per_task && cnt[P.ph.task[i]] >= pr.base.use_per_task) continue; cnt[P.ph.task[i]]++; P.prompt_ids.push_back(P.ph.id[i]); } }
  for (int t = 0; t < NTASKS; t++) if (P.prompts.ntask[t] == 0) host::die("the prompt store has no prompt of environment " + std::to_string(t));
  P.prompt_sha = mgo::store_sha256(pr.base.prompts);
  const double t_b = P.now();
  printf("loaded model and %d prompts (", P.prompts.n);
  for (int t = 0; t < NTASKS; t++) printf("%s%d %s", t ? ", " : "", P.prompts.ntask[t], P.env_names[t].c_str());
  printf(") in %.1f s; prompt store sha256 %.16s\n", t_b - t_a, P.prompt_sha.c_str());
  build_caches(P, pr.runs[0].hp, false);
  const double t_c = P.now();
  printf("prompt caches built in %.1f s\n", t_c - t_b);
  P.pre_main_s = age >= 0 ? age : 0.0; P.pre_load_s = (P.from_creation ? age : 0.0) + t_a; P.load_s = t_b - t_a; P.cache_s = t_c - t_b;

  // ---- a list of runs: does the device hold the state of every one of them? One trial allocation per shape of run,
  // freed at once (the runs follow one another, so each needs to fit alone). A lone run needs no trial: it finds out
  // at its own start. The time of the trials belongs to no run's set-up.
  double t_first = t_c;
  if (pr.listed && !pr.base.dry) {
    std::vector<std::vector<long long>> seen;
    for (size_t r = 0; r < pr.runs.size(); r++) {
      const Opt& o = pr.runs[r]; const std::vector<long long> shape = shape_of(o);
      bool had = false; for (const auto& x : seen) had = had || x == shape;
      if (had) continue;
      seen.push_back(shape);
      fprintf(stderr, "checking the device memory of --run number %zu: %s\n", r + 1, o.out.c_str());
      int tb_ = 0; std::vector<int> rb_; const std::vector<int> role = make_roles(o, nsm, tb_, rb_);
      std::vector<void*> scope; mk::dscope() = &scope;
      { RunState st; alloc_run_state(o, nsm, role, st); }
      mk::dscope() = nullptr; mk::dscope_free(scope);
    }
    t_first = P.now();
  }

  // ---- the runs
  int complete = 0;
  for (size_t r = 0; r < pr.runs.size(); r++) {
    const double t_start = r == 0 ? t_first : P.now();
    if (r > 0) open_console(r);
    complete += run_one(P, r, t_start) ? 1 : 0;
    con.close();
    if (pr.base.dry) return 0;
  }
  if (pr.listed) printf("\nmegarl: process %s: %d of %zu runs complete\n", pr.id.c_str(), complete, pr.runs.size());
  return complete == (int)pr.runs.size() ? 0 : 2;
}
