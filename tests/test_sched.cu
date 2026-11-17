// test_sched.cu -- randomised GPU test of the scheduler (mk::wave_loop in src/mk_sched.cuh).
// Three jobs with random phase graphs run in one persistent kernel: job 0 is bounded, jobs 1 and 2 run until job 0
// finishes. Every block serves every job (home job by SM, random priorities). Every task records device timestamps,
// and the host then checks that
//   * every task of every completed step ran exactly once,
//   * no task started before all phases it depends on (zero-task phases are transparent) had completed,
//   * data hand-off works: every thread of a task writes one word after a random delay, and every thread of a
//     dependent task reads the words of all tasks it depends on with plain loads and must find them all,
//   * the unbounded jobs stop, and an abort request stops everything,
//   * no window hand-off was missed (the rescue count stays 0).
// Build and run: make test     or by hand: build/test_sched [trials] [seed]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>
#include "mk_sched.cuh"

using mk::u64;

constexpr int NJ = 3;
struct Ctx {
  mk::Job job[NJ]; mk::Wave wave;
  u64 *t_start[NJ], *t_end[NJ]; int* count[NJ]; u64 cap_steps[NJ];
  int abort_at;                         // job 0 requests abort when it reaches this step (-1: never)
  u64 seed, first;                      // first: the mask of preferred jobs of mk::wave_loop
  int* data[NJ]; u64 cap_data[NJ]; int* data_bad;   // one word per (task, thread) for the first cap_data steps
};

__device__ __forceinline__ u64 mix(u64 z) { z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }

__device__ __forceinline__ int word(u64 seed, u64 task, int tid, int j) { return (int)((mix(seed ^ (task * 512 + tid * 2 + j)) & 0x7fffffffull) | 1ull); }

__device__ void run_task(const Ctx& C, int j, const mk::PhaseDesc& pd, int ph, int local, u64 step) {
  const mk::Job& J = C.job[j]; const int tid = threadIdx.x;
  const u64 idx = step * J.tps + (u64)__ldg(J.prefix + ph) + local;
  const bool rec = step < C.cap_steps[j], recd = step < C.cap_data[j];
  if (tid == 0 && rec) C.t_start[j][idx] = mk::gtimer();
  if (recd) {                                              // every thread checks its word of every task this phase depends on
    const int off = __ldg(J.dep_off + ph), cnt = __ldg(J.dep_cnt + ph); int bad = 0;
    for (int k = 0; k < cnt; k++) {
      const int2 e = __ldg(J.deps + off + k); const long long t = (long long)step + e.y;
      if (t < 0) continue;
      const int nq = __ldg(J.ntasks + e.x); const u64 base = (u64)t * J.tps + (u64)__ldg(J.prefix + e.x);
      for (int i = 0; i < nq; i++) if (C.data[j][(base + i) * 256 + tid] != word(C.seed, base + i, tid, j)) bad++;
    }
    if (bad) atomicAdd(C.data_bad, bad);
  }
  const unsigned ns = (unsigned)(mix(C.seed ^ (idx * 2 + j) ^ ((u64)tid << 40)) % 3000u);     // 0..3 us of "work", different per thread
  if (ns) __nanosleep(ns);
  if (recd) C.data[j][idx * 256 + tid] = word(C.seed, idx, tid, j);
  if (tid != 0) return;
  if (rec) { atomicAdd(C.count[j] + idx, 1); C.t_end[j][idx] = mk::gtimer(); }
  if (j == 0) {
    if (C.abort_at >= 0 && (long long)step == C.abort_at) mk::st_rel(J.abort, 1);
    if (pd.kind == 1 && step == J.limit - 1) { __threadfence(); mk::st_rel(J.finished, 1); }   // last phase of the last step (it has 1 task)
  }
}

__global__ void __launch_bounds__(256, 2) kernel(const Ctx C) {
  __shared__ volatile int s_sel[mk::WAVE_SEL];
  __shared__ mk::Wave s_wave;
  __shared__ mk::Job s_job[NJ];
  const int home = (int)(mk::smid() % (unsigned)NJ);
  if (threadIdx.x == 0) mk::mark_start(C.job[0].t0);
  mk::wave_loop([&](int j) -> const mk::Job& { return C.job[j]; }, C.wave, &s_wave, s_job, home, C.first, (int)(blockIdx.x % (unsigned)NJ), s_sel,
                [&](int j, const mk::PhaseDesc& pd, int ph, int local, u64 step) { run_task(C, j, pd, ph, local, step); });
}

struct Graph { mk::Builder b; int R; };

static Graph make_graph(std::mt19937& rng, int R) {
  Graph g; g.R = R; auto rnd = [&](int n) { return (int)(rng() % (unsigned)n); };
  const int P = 6 + rnd(30);
  std::vector<std::vector<mk::Dep>> deps(P); std::vector<int> nt(P);
  for (int p = 0; p < P; p++) nt[p] = (p == 0 || p == P - 1) ? 1 + rnd(4) : (rnd(5) == 0 ? 0 : 1 + rnd(12));
  nt[P - 1] = 1;                                                            // the test uses "kind 1, one task" as the end marker
  deps[0].push_back(mk::Dep{P - 1, -R});
  for (int p = 1; p < P; p++) {
    const int k = 1 + rnd(3);
    for (int i = 0; i < k; i++) deps[p].push_back(mk::Dep{rnd(p), 0});
    if (rnd(3) == 0) { const int q = rnd(P); if (nt[q] > 0) deps[p].push_back(mk::Dep{q, -(1 + rnd(R))}); }   // extra cross-step dependency (always a lower ticket); a zero-task target could form an endless chain, which the Builder rejects
  }
  // "computed ahead" pattern: phase p depends only on its consumer of the previous same-slot step. Every other phase
  // keeps a same-step path to phase 0 (and through it to the previous step's end phase), which orders its instances.
  std::vector<char> ahead(P, 0);
  for (int p = 1; p + 2 < P; p++) if (rnd(7) == 0 && nt[p] > 0 && nt[p + 1] > 0 && !ahead[p - 1]) {
    deps[p].clear(); deps[p].push_back(mk::Dep{p + 1, -R}); ahead[p] = 1;
    deps[p + 1].push_back(mk::Dep{p, 0});
  }
  for (int p = 1; p < P; p++) if (!ahead[p]) deps[p].push_back(mk::Dep{0, 0});
  for (int p = 0; p + 1 < P; p++) deps[P - 1].push_back(mk::Dep{p, 0});       // the end phase closes the step
  for (int p = 0; p < P; p++) g.b.add("p" + std::to_string(p), p == P - 1 ? 1 : 0, nt[p], deps[p]);
  g.b.finalize(R);
  return g;
}

// completion time of (step, phase) with zero-task phases transparent; 0 if before the first step
static u64 comp(const Graph& g, const std::vector<u64>& tend, const std::vector<int>& prefix, long long tps, long long step, int ph, std::map<std::pair<long long, int>, u64>& memo, int depth = 0) {
  if (step < 0) return 0;
  auto key = std::make_pair(step, ph); auto it = memo.find(key); if (it != memo.end()) return it->second;
  u64 t = 0;
  if (g.b.ntasks[ph] > 0) { for (int i = 0; i < g.b.ntasks[ph]; i++) t = std::max(t, tend[(size_t)(step * tps + prefix[ph] + i)]); }
  else if (depth < 10000) for (const mk::Dep& e : g.b.raw[ph]) t = std::max(t, comp(g, tend, prefix, tps, step + e.delta, e.phase, memo, depth + 1));
  memo[key] = t; return t;
}

int main(int argc, char** argv) {
  int trials = argc > 1 ? atoi(argv[1]) : 20; unsigned seed0 = argc > 2 ? atoi(argv[2]) : 1;
  cudaDeviceProp prop; MK_CK(cudaGetDeviceProperties(&prop, 0));
  const int nsm = prop.multiProcessorCount, nblocks = nsm * 2;
  long long total_tasks = 0, bad = 0, data_bad = 0, rescues = 0; int trial_fail = 0;
  for (int tr = 0; tr < trials; tr++) {
    std::mt19937 rng(seed0 * 7919u + tr);
    const bool abort_run = (tr % 5 == 4);
    int* d_fin = mk::dalloc<int>(1); int* d_abort = mk::dalloc<int>(1);
    Ctx C; memset(&C, 0, sizeof C);
    Graph g[NJ]; std::vector<int> prefix[NJ]; u64 steps0 = 40 + rng() % 200;
    u64* d_t0 = mk::dalloc<u64>(1); { std::vector<u64> big(1, ~0ull); mk::up(d_t0, big); }
    for (int j = 0; j < NJ; j++) {
      g[j] = make_graph(rng, 1 + (int)(rng() % 4));
      C.job[j] = mk::upload(g[j].b, j == 0 ? steps0 : 0, j, j >= 1 ? 1 : 0, d_fin, d_abort, d_t0, 60000000000ull);   // a trial takes about a second: after 60 s a waiting block raises *abort and the trial fails
      C.cap_steps[j] = j == 0 ? steps0 : 100000;
      const size_t n = (size_t)C.cap_steps[j] * C.job[j].tps;
      C.t_start[j] = mk::dalloc<u64>(n); C.t_end[j] = mk::dalloc<u64>(n); C.count[j] = mk::dalloc<int>(n);
      C.cap_data[j] = std::min<u64>(C.cap_steps[j], 250); C.data[j] = mk::dalloc<int>((size_t)C.cap_data[j] * C.job[j].tps * 256);
      prefix[j].assign(g[j].b.nph() + 1, 0); for (int i = 0; i < g[j].b.nph(); i++) prefix[j][i + 1] = prefix[j][i] + g[j].b.ntasks[i];
    }
    C.abort_at = abort_run ? (int)(steps0 / 2) : -1; C.seed = rng(); C.data_bad = mk::dalloc<int>(1);
    C.first = (u64)(rng() % (1u << NJ));
    { std::vector<const mk::Builder*> bs; std::vector<mk::Job> js;
      for (int j = 0; j < NJ; j++) { bs.push_back(&g[j].b); js.push_back(C.job[j]); }
      C.wave = mk::wave_upload(bs, js); }
    kernel<<<nblocks, 256>>>(C);
    MK_CK(cudaDeviceSynchronize()); MK_CK(cudaGetLastError());
    long long tb = 0, tt = 0; bool ok = true;
    for (int j = 0; j < NJ; j++) {
      const long long tps = (long long)C.job[j].tps; const int n = g[j].b.nph();
      // steps that are completely done: the end phase (one task) of slot-ordered steps
      const std::vector<u64> done = mk::down(C.job[j].done, (size_t)n * g[j].R);
      u64 nsteps = 0; for (int s = 0; s < g[j].R; s++) nsteps += done[(size_t)(n - 1) * g[j].R + s];
      // steps complete in order? count the prefix of steps whose end task ran
      const u64 cap = std::min<u64>(C.cap_steps[j], nsteps + 8);
      const std::vector<u64> ts = mk::down(C.t_start[j], (size_t)cap * tps), te = mk::down(C.t_end[j], (size_t)cap * tps); const std::vector<int> cnt = mk::down(C.count[j], (size_t)cap * tps);
      u64 full = 0; while (full < cap && cnt[(size_t)(full * tps + prefix[j][n - 1])] == 1) full++;
      if (!abort_run && j == 0 && full != steps0) { printf("trial %d job 0: %llu of %llu steps complete\n", tr, full, steps0); ok = false; }
      std::map<std::pair<long long, int>, u64> memo;
      for (u64 s = 0; s < full; s++) for (int p = 0; p < n; p++) for (int i = 0; i < g[j].b.ntasks[p]; i++) {
        const size_t idx = (size_t)(s * tps + prefix[j][p] + i); tt++;
        if (cnt[idx] != 1) { tb++; if (tb < 5) printf("trial %d job %d step %llu phase %d task %d ran %d times\n", tr, j, s, p, i, cnt[idx]); continue; }
        for (const mk::Dep& e : g[j].b.raw[p]) {
          const u64 c = comp(g[j], te, prefix[j], tps, (long long)s + e.delta, e.phase, memo);
          if (ts[idx] < c) { tb++; if (tb < 5) printf("trial %d job %d step %llu phase %d task %d started %llu ns before dependency (phase %d, delta %d) completed\n", tr, j, s, p, i, c - ts[idx], e.phase, e.delta); }
        }
      }
      // no task beyond the completed steps may have run twice
      for (size_t i = 0; i < cnt.size(); i++) if (cnt[i] > 1) { tb++; }
      if (j >= 1 && full == 0 && !abort_run) { printf("trial %d: unbounded job %d made no progress\n", tr, j); ok = false; }
      if (tr < 3 || !ok || tb) printf("trial %d job %d: %d phases, %lld tasks/step, R=%d, %llu steps checked%s\n", tr, j, n, tps, g[j].R, full, abort_run ? " (abort run)" : "");
    }
    const int fin = mk::down(d_fin, 1)[0], ab = mk::down(d_abort, 1)[0]; const int db = mk::down(C.data_bad, 1)[0];
    if (db) { printf("trial %d: %d data words of dependencies were not visible to a dependent task\n", tr, db); ok = false; }
    data_bad += db;
    if (!abort_run && (fin != 1 || ab != 0)) { printf("trial %d: finished=%d abort=%d\n", tr, fin, ab); ok = false; }
    if (abort_run && ab != 1) { printf("trial %d: abort flag not set\n", tr); ok = false; }
    total_tasks += tt; bad += tb; if (tb || !ok) trial_fail++;
    rescues += (long long)mk::down(C.wave.rescued, 1)[0];
    for (int j = 0; j < NJ; j++) { cudaFree(C.t_start[j]); cudaFree(C.t_end[j]); cudaFree(C.count[j]); cudaFree(C.data[j]); }
  }
  const bool pass = bad == 0 && trial_fail == 0 && rescues == 0;
  printf("test_sched: %d trials, %lld tasks checked, %lld ordering violations, %lld stale data words, %d failed trials, %lld rescued hand-offs -> %s\n",
         trials, total_tasks, bad, data_bad, trial_fail, rescues, pass ? "PASSED" : "FAILED");
  return pass ? 0 : 2;
}
