// mrl_common.cuh -- constants and device-side data layout shared by every module of the kernel.
// Every buffer is allocated by mrl_host.cuh, rl_state.cuh or mid_train.cuh and is zero-filled at allocation.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "lgemm.cuh"
#include "mk_sched.cuh"

namespace mrl {
typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;

// ------------------------------------------------------------------------------------------------ model: Qwen2.5-0.5B-Instruct
constexpr int D = 896, NL = 24, NH = 14, NKV = 2, HD = 64, FF = 4864, V = 151936;
constexpr int DQKV = D + 2 * NKV * HD;          // 1152 = q (896) | k (128) | v (128)
constexpr int KQ = 928;                         // reduction length of the q/k/v GEMM: 896 inputs, 1 bias column, zero padding to 32
constexpr int MAXPOS = 1024;                    // rows of the RoPE tables
constexpr float RMS_EPS = 1e-6f;
constexpr int EOS_IM_END = 151645, EOS_ENDOFTEXT = 151643;

// ------------------------------------------------------------------------------------------------ GEMM tile geometry
constexpr int TM = lg::BM, TN = lg::BN;
static_assert(TM == 64 && TN == 256, "MegaRL is laid out for 64 x 256 GEMM tiles (build with the lgemm.cuh defaults)");
constexpr int DP = 1024;                        // D padded to N tiles
constexpr int QKVP = 1280;                      // DQKV padded
constexpr int GUP = 2 * FF;                     // 9728 = gate | up
constexpr int VP = 152064;                      // V padded. Vocabulary ids >= V do not exist: every reduction over logits skips them.
constexpr int NT_D = DP / TN, NT_QKV = QKVP / TN, NT_GU = GUP / TN, NT_V = VP / TN, NT_FF = FF / TN;   // 4, 5, 38, 594, 19
constexpr int SMEM = lg::SMEM_BYTES;            // dynamic shared memory of every kernel (81,920 B)

// ------------------------------------------------------------------------------------------------ trained layer and rollout geometry
constexpr int LT = NL - 1;                      // the last layer (its down_proj runs in the head)
#ifndef MRL_MID_LAYER
#define MRL_MID_LAYER 13
#endif
constexpr int LMID = MRL_MID_LAYER;             // the layer whose weights are trained (all of attention and MLP)
constexpr size_t MID_OFF_QKV = 0;
constexpr size_t MID_OFF_O   = MID_OFF_QKV + (size_t)DQKV * KQ;
constexpr size_t MID_OFF_GU  = MID_OFF_O   + (size_t)D * D;
constexpr size_t MID_OFF_DN  = MID_OFF_GU  + (size_t)GUP * D;
constexpr size_t MID_NW      = MID_OFF_DN  + (size_t)D * FF;   // 14,946,304 bf16 = 29.89 MB per full-layer snapshot
constexpr int GSZ = 4;                          // completions per prompt group
constexpr int MAXGEN = 64;                      // generated tokens per response (= rows of one sequence = TM)
constexpr int MAXROWS = 64;                     // sequence rows of one rollout pipeline (= TM)
constexpr int MAXGRP = MAXROWS / GSZ;           // group slots of one pipeline (16)
constexpr int MAXPROMPT = 128;                  // longest prompt
#ifndef MRL_MAXPIPE
#define MRL_MAXPIPE 8
#endif
constexpr int MAXPIPE = MRL_MAXPIPE;            // most rollout pipelines a run can have
constexpr int KVW = 2 * NKV * HD;               // 256 floats per (layer, position): rotated keys of the 2 kv heads, then values
#ifndef MRL_NTASKS
#define MRL_NTASKS 4
#endif
constexpr int NTASKS = MRL_NTASKS;              // environments (tasks) of the prompt store
static_assert(NTASKS >= 1 && NTASKS <= 32, "NTASKS must be in 1..32");
#ifndef MRL_TGRP
#define MRL_TGRP 16
#endif
constexpr int TGRP = MRL_TGRP;                  // groups consumed by one optimiser update
constexpr int TSEQ = TGRP * GSZ;                // sequences of one training batch
constexpr int TROWS = TSEQ * MAXGEN;            // rows of one training batch
constexpr int TTILE = TROWS / TM;               // M tiles of one training batch (= TSEQ)
constexpr int KS_MAX = 64;                      // largest split-K chunk count of the partial buffers
constexpr int NGNORM = 64;                      // per-update gradient-norm partials handed to T_FINISH
static_assert(MAXGEN == TM && MAXROWS == TM && TGRP >= 1, "one sequence and one pipeline are exactly one M tile");

// ------------------------------------------------------------------------------------------------ small vector types
struct __align__(16) V8 { bf16 v[8]; };
struct __align__(8)  V4 { bf16 v[4]; };
struct __align__(16) F4 { float v[4]; };
struct __align__(16) I4 { int v[4]; };

// ------------------------------------------------------------------------------------------------ frozen model weights
struct Model {
  const bf16* emb;                              // [V, D]            embedding and LM head (tied)
  const bf16* qkv[NL];                          // [DQKV, KQ]        q | k | v rows; column D is the bias
  const bf16* o[NL];                            // [D, D]
  const bf16* gu[NL];                           // [2*FF, D]         gate rows, then up rows
  const bf16* down[NL];                         // [D, FF]
  const float* ln1[NL];                         // [D]
  const float* ln2[NL];                         // [D]
  const float* lnf;                             // [D]
  const float* rope_cos;                        // [MAXPOS, HD/2]
  const float* rope_sin;                        // [MAXPOS, HD/2]
};

// ------------------------------------------------------------------------------------------------ rollout pipeline
// One sequence row. Written by the control tasks and the sample task of its own pipeline only.
struct Row {
  int active;                                   // 1: the head samples a token for this row in the current step
  int feed;                                     // 1: the trunk feeds `tok` at position `pos` in the current step
  int prompt, plen;                             // prompt id; number of prompt positions (they come from the prompt cache)
  int gen;                                      // tokens sampled so far
  int tok;                                      // token to feed (the last sampled token; in the prompt-cache build: the prompt token)
  int pos;                                      // absolute position at which `tok` is fed
  int buf, seq;                                 // group buffer, index of this sequence inside its group
};

// A group slot = rows [GSZ*g, GSZ*g + GSZ) of a pipeline. A slot is free iff state == rls::FREE (= 0).
// buf, uid, task, prompt, key and the ver_* fields are valid only while the slot is not free; the snapshot is pinned
// (and may be dereferenced by the head) only while state == rls::GENERATING.
struct GroupSlot {
  int state;
  int buf;
  int uid, task, prompt, key;
  int ver_slot, ver_id, ver_gen;                // pinned snapshot: table slot, version id, generation counter at pin time
  int pad;
};

struct SampPart { float gmax; int gidx; float glogit; float lmax; float lsum; int pad[3]; };   // per (LM-head tile, row): Gumbel-max and log-sum-exp partials (32 bytes)
struct LsePart { float lmax, lsum; };                                                          // per (LM-head tile, row): log-sum-exp partial

struct Pipe {                                   // working buffers of one rollout pipeline; "rows" means MAXROWS
  float* hres;                                  // [rows, DP]        residual stream
  bf16*  xn;                                    // [rows, KQ]        normalised GEMM input; column D is 1.0, columns D+1.. are 0
  float* qkv;                                   // [rows, QKVP]      q | k | v including bias, before RoPE
  float* qkvp;                                  // [KS_MAX/4][rows, QKVP] optional split-K partials of q/k/v (trunk-internal)
  bf16*  ao;                                    // [rows, D]         attention output, heads concatenated
  float* gu;                                    // [rows, GUP]       gate | up pre-activations
  bf16*  xm;                                    // [rows, FF]        silu(gate) * up
  float* part;                                  // [KS_MAX][rows, DP] split-K partials (trunk o_proj / down_proj, head Y0; with ksG > 1 also the
                                                //   trunk's gate/up chunks 1.., with row pitch MAXROWS * GUP)
  bf16*  xf;                                    // [rows, D]         head: final-norm output (LM-head operand)
  SampPart* lmpart;                             // [NT_V][rows]      head: per-tile sampling partials
  float* kv;                                    // [rows][NL][MAXGEN][KVW]  key/value cache of the generated positions
  Row* row;                                     // [rows]
  GroupSlot* grp;                               // [MAXGRP]
  int* nrows;                                   // [1] rows in use; written once when the pipe is created, constant during a launch
};

// ------------------------------------------------------------------------------------------------ prompt store
// Read-only during the persistent launch. kv and hlast are written by the prompt-cache build at setup.
struct Prompts {
  int n;                                        // prompts
  const int* len;                               // [n]
  const int* task;                              // [n]  environment 0 .. NTASKS-1
  const int* key;                               // [n]  correct option 0..3
  const int* tok;                               // [n][MAXPROMPT] token ids (zero padded)
  float* kv;                                    // [n][NL][MAXPROMPT][KVW]  prompt key/value cache
  float* hlast;                                 // [n][D]            residual stream entering layer LMID at the last prompt position
  const signed char* tok_letter;                // [V]  -1 or 0..3
  const unsigned char* tok_colon;               // [V]
  const unsigned char* tok_answer;              // [V]
  int verifier_mode;                            // 0: "Answer: X" pattern, 1: first token is the letter
  int ntask[NTASKS];                            // prompts per task
  const int* by_task[NTASKS];                   // [ntask[t]] prompt ids of task t
};

// ------------------------------------------------------------------------------------------------ rollout store: group buffers
struct GroupRec {                               // one per group buffer
  int state;                                    // rls::State (the authoritative state of the group that holds the buffer)
  int uid, task, prompt, key, ver_id;
  int len[GSZ];                                 // generated tokens per sequence, end token included
  float reward[GSZ], adv[GSZ];
  u64 t_admit, t_done;
};

struct Store {
  int nbuf;
  GroupRec* rec;                                // [nbuf]
  int* tokens;                                  // [nbuf][GSZ][MAXGEN]
  float* logp;                                  // [nbuf][GSZ][MAXGEN] generation-time log-probabilities
  float* hbase;                                 // [nbuf][GSZ*MAXGEN][D]    residual stream entering layer LMID
};

// ------------------------------------------------------------------------------------------------ concurrency primitives (RL core)
struct Queue {                                  // bounded lock-free queue of ints, capacity a power of two (mask + 1)
  unsigned* seq; int* val; unsigned* head; unsigned* tail; unsigned mask;
};

struct Versions {                               // published snapshots of the trained layer
  int nslots;
  bf16* W;                                      // [nslots][MID_NW]
  int* cur;                                     // [1] slot of the current snapshot
  int* ref;                                     // [nslots] readers
  int* gen;                                     // [nslots] generation counter, odd while the slot is being written
  int* id;                                      // [nslots] version id of the snapshot in the slot
  int* train_ver;                               // [1] number of optimiser updates applied so far
  bf16* arch;                                   // [narch][MID_NW] archived versions: 0, every arch_stride-th, and the last
  int narch;
  int arch_stride;
  int* arch_id;                                 // [narch] version id stored in each archive slot (-1 if empty)
};

struct TraceEvent { u64 t; int type; int a, b, c, d, e; };   // 32 bytes
struct Trace { TraceEvent* ev; u64* n; u64 cap; };            // events with index >= cap are not stored (n keeps counting)

struct Counters {                               // device atomics, all cumulative unless noted
  int* admitted;                                // [NTASKS]
  int* finished;                                // [NTASKS] groups whose GSZ sequences all ended
  int* drop_zerovar;                            // [NTASKS]
  int* drop_stale_inflight;                     // [NTASKS] cancelled while generating
  int* drop_stale_queued;                       // [NTASKS] dropped by the learner at selection time
  int* consumed;                                // [NTASKS]
  int* inflight;                                // [NTASKS] currently generating (not cumulative)
  int* queued;                                  // [NTASKS] currently in the ready queue (not cumulative)
  int* prompt_next;                             // [NTASKS] round-robin prompt counters
  int* errors;                                  // [16] must all stay 0: rls::Err (slots 0..6), qw::ERR_ROW (14, the trunk), hd::ERR_NO_WINNER (15, the head)
  int* uid_next;                                // [1]
  u64* tokens_generated;                        // [1] written by the sample task
  int* stop_reason;                             // [1] rls::Stop
  int* stats;                                   // [16] rls::Stat: publish waits, idle polls and the like (cumulative, informational; counting
                                                //   stops at rls::STAT_MAX = 0x7fff0000, so a value that large means "at least")
  int* held;                                    // [NTASKS] groups the learner holds: taken for the batch it is collecting or training,
                                                //   not yet released (not cumulative)
  int* training;                                // [1] bit rls::TRAINING_BATCH: T_SELECT has a complete batch and T_FINISH has not
                                                //   released it yet; bits rls::TURN_SHIFT + job: generation turns (hp.alternate)
};

struct GroupLogEntry {                          // terminal record of a group, for the host audit
  int uid, task, prompt, key, ver_id, terminal; // terminal: rls::State (CONSUMED, DROPPED_*, CANCELLED)
  int train_step, lag;                          // for CONSUMED
  int len[GSZ]; float reward[GSZ], adv[GSZ];
  u64 t_admit, t_done, t_end;
};
struct GroupLog { GroupLogEntry* e; u64* n; u64 cap; int* tokens; float* logp; u64 tok_cap; };   // tokens/logp: [tok_cap][GSZ][MAXGEN] of the first tok_cap logged groups

struct Shared {
  Queue ready[NTASKS];                          // completed groups (group buffer ids)
  Queue freebuf;                                // free group buffers
  Versions ver;
  Trace trace;
  Counters cnt;
  GroupLog glog;
};

// ------------------------------------------------------------------------------------------------ hyper-parameters
struct Hyper {
  float temperature, inv_temperature;
  float clip_eps;                               // GRPO clip (0.2)
  float adv_eps;                                // 1e-4
  float lr, beta1, beta2, adam_eps, wd;
  int lag;                                      // largest allowed (train_ver - group version) at selection time
  int norm_mode;                                // loss weight per token: 0 = 1 / (TSEQ * MAXGEN) (constant), 1 = 1 / (TSEQ * len[s]) (per-sequence mean)
  int pair_mode;                                // batch composition: 0 = TGRP/NTASKS groups of each task, 1 = balanced if available, else any TGRP
                                                //   eligible groups, 2 = least-consumed task first, in admission and selection (rl_control.cuh)
  u64 select_wait_ns;                           // pair_mode 1: how long the learner waits for a balanced batch
  u64 starve_ns;                                // the run ends (stop_reason STARVED) if T_SELECT makes no progress (takes or drops no group) for this long
  int ksY, ksD;                                 // split-K chunks: last layer's down_proj (head and learner), other down_projs
  int ksQ, ksO, ksG;                            // trunk: chunks of q/k/v, o_proj and gate/up (1 = unsplit)
  int tksV;                                     // learner: chunks of the LM-head backward (over the vocabulary)
  int target_per_task;                          // admission: (generating + queued) groups allowed per task
  u64 seed;
  u64 deadline_ns;                              // *abort is set if the kernel runs longer than this
  int count_held;                               // 1: admission counts the groups the learner holds: a group of environment t is
                                                //   admitted while inflight[t] + queued[t] + held[t] < target_per_task (0: without held[t])
  int alternate;                                // 1: a rollout engine does not start a step while the learner has a complete batch
                                                //   (Counters::training): generation and training take turns
  int no_train;                                 // 1: the learner's own phases do nothing. Batches are selected, held and released as
                                                //   usual and every update publishes the unchanged weights
  int require_eos;                              // 1: reward is 0 unless the sequence ended with an EOS token (prevents 64-token rambling)
  float grad_clip;                              // >0: clip the global gradient L2 norm to this before AdamW
};

// ------------------------------------------------------------------------------------------------ training
// Row indexing. "Sequence rows": row = s * MAXGEN + t for sequence s = g * GSZ + j of batch group g, token t < len[s].
// "Packed rows": row = off[s] + t, with off the prefix sums of len; ntok rows in all, ntile = ceil(ntok / TM) M tiles.
// The vocabulary-sized passes work on packed rows so that short answers do not pay for 64-row tiles.
struct Batch {                                  // written by the select task; the *_part arrays by the learner's tasks
  int valid;                                    // 0: nothing to train in this step (shutdown)
  int buf[TGRP], uid[TGRP], task[TGRP], lag[TGRP];
  int len[TSEQ], off[TSEQ];
  float adv[TSEQ];
  int ntok, ntile;
  float loss_part[TTILE];                       // per packed M tile: sum of weighted per-token losses (stored, not accumulated)
  int nclip_part[TTILE];                        // per packed M tile: tokens without gradient (ratio clipped, log-ratio beyond the +-20 clamp, or not a number)
  float maxdlogp_part[TTILE];                   // per packed M tile: largest |logp_new - logp_beh|
  float gnorm2[NGNORM];                         // partial sums of squared gradient entries
};

struct Train {
  bf16* W;                                      // [MID_NW] bf16 working copy of the trained layer (what is published)
  Batch* batch;
  // sequence-row buffers ("rows" = TROWS)
  float* hfin;                                  // [rows, DP]        residual stream after the last layer
  float* rinv;                                  // [rows]            1 / rms of hfin
  // packed-row buffers
  int* y;                                       // [rows]            taken token (copied from the store by the gather task)
  float* logp_beh;                              // [rows]            generation-time log-probability
  int* rseq;                                    // [rows]            batch sequence s of the packed row (t = row - off[s])
  bf16* xf;                                     // [rows, D]         final-norm output
  LsePart* lsepart;                             // [TTILE][NT_V][TM]
  float* logit_y;                               // [rows]            z of the taken token
  float* lse;                                   // [rows]
  float* logp_new;                              // [rows]
  float* coef;                                  // [rows]            dLoss / dlogp
  bf16* dlogits;                                // [rows, VP]
  float* dxpart;                                // [tksV][rows, DP]  partials of dLoss/d(xf)
  float* dxf;                                   // [rows, DP]
  // per-step records
  float* loss_hist;                             // [steps]
  u64* step_done;                               // [steps]
  int* step_info;                               // [steps][16]: tokens, clipped tokens, version id after the update, packed tiles, then the uids of the first 12 groups (-1: none)
  float* step_stat;                             // [steps][4]: loss, |grad|, 0, max |logp_new - logp_beh|
};

// ------------------------------------------------------------------------------------------------ everything
struct Ctx {
  mk::Job roll[MAXPIPE];                        // rollout jobs ("engines"); nroll in use
  mk::Job train;
  Model model;
  Prompts prompts;
  Pipe* pipe;                                   // [npipe]
  int npipe, ngrp;                              // pipelines; group slots in use per pipeline (*pipe[p].nrows == GSZ * ngrp)
  int nroll, ppj;                               // rollout jobs; pipelines per job, advanced in lockstep (npipe == nroll * ppj):
                                                //   job j owns pipelines [j * ppj, (j + 1) * ppj)
  int pipe_task[MAXPIPE];                       // environment a pipeline admits, or -1: any (admission picks)
  const int* role;                              // [2 * nsm] job of the r-th block (r = 0, 1) to arrive on SM s, at role[2 * s + r]:
                                                //   0..nroll-1 = that rollout job, nroll = the training job, -1 = exit at once
  int* sm_count;                                // [nsm] arrival counter per SM (zero at launch)
  int nsm;
  Store store;
  Shared sh;
  Hyper hp;
  Train tr;
  int updates;                                  // optimiser updates of this run (the training job's step limit)
  int* finished; int* abort;                    // the same flags the jobs use
  u64* t0;                                      // [1] device time at kernel start (initialised to ~0; every block does mk::mark_start)
  void* mid;                                    // mt::MidState* (the learner's own buffers, mid_train.cuh)
};

// ------------------------------------------------------------------------------------------------ phase kinds
// PhaseDesc.kind; PhaseDesc.a = layer where it applies; PhaseDesc.b = 1 on trunk phases of the prompt-cache build;
// PhaseDesc.c = lockstep multiplier of rollout phases (set by the add_* functions, read only by megarl.cu's dispatcher).
// The four control kinds are added to the job graphs by megarl.cu and dispatched to ctl:: (rl_control.cuh);
// qw::run handles the K_Q_ kinds below K_Q_MID_COHORT, mt::run_roll the K_Q_MID_ kinds, hd::run the K_H_ kinds and
// mt::run_train the K_M_ kinds.
enum Kind {
  K_R_BEGIN = 1, K_R_ADVANCE = 2, K_T_SELECT = 3, K_T_FINISH = 4,
  K_Q_NORM1 = 100, K_Q_QKV, K_Q_ATTN, K_Q_OPROJ, K_Q_NORM2, K_Q_GU, K_Q_ACT, K_Q_DOWNP, K_Q_DOWNR, K_Q_NORM1F, K_Q_EMBN,
  K_Q_MID_COHORT = 120, K_Q_MID_QKV, K_Q_MID_OPROJ, K_Q_MID_GU, K_Q_MID_DOWNP,
  K_H_Y0P = 200, K_H_Y0R, K_H_NORMF, K_H_LMHEAD, K_H_SAMPLE,
  K_M_GATHER = 400, K_M_FNORM1, K_M_FQKV, K_M_FATTN, K_M_FOPROJ, K_M_FNORM2, K_M_FGU, K_M_FACT, K_M_FDOWNP,
  K_M_NORMF, K_M_LM1, K_M_LOSS, K_M_LM2, K_M_LMB, K_M_LMBR, K_M_NORMB,
  K_M_BDOWN, K_M_BACT, K_M_BGU, K_M_BNORM2, K_M_BOPROJ, K_M_BATTN_ROW, K_M_BATTN_KVR, K_M_BQKV, K_M_BNORM1,
  K_M_GDOWN, K_M_GGU, K_M_GO, K_M_GQKV, K_M_GNORM, K_M_ADAM
};

// ------------------------------------------------------------------------------------------------ shared helpers
__host__ __device__ inline u64 mix64(u64 z) { z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
// K range [k0, k0 + kc) of chunk c of n, for a reduction of nk32 steps of 32. Requires 1 <= n <= nk32 (checked by mrl::host::check_hyper).
__host__ __device__ inline void kchunk(int nk32, int n, int c, int& k0, int& kc) {
  const int a = (int)((long long)c * nk32 / n), b = (int)((long long)(c + 1) * nk32 / n); k0 = a * 32; kc = (b - a) * 32;
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// Sampling noise. The key identifies one sampling event (group uid, sequence in the group, token index = Row.gen before
// the increment); gumbel(key, v) is the Gumbel(0,1) noise of vocabulary entry v. A token is sampled as
// argmax over v < V of (logit_v * inv_temperature + gumbel(key, v)), ties to the lower v.
// The uniform has 23 bits: (n + 0.5) / 2^23 is exact in float32 for n < 2^23, so u lies strictly inside (0, 1) and the
// noise is finite, in [-2.82, 16.64]. (With 24 bits the largest n rounds to u = 1 and the noise is +inf.)
// Host and device logf differ in the last bit for some inputs, so a host replay can flip a near-tie (scores within 1e-6).
__host__ __device__ inline u64 sample_key(u64 seed, int uid, int seq, int t) {
  return mix64(mix64(seed + 0x9e3779b97f4a7c15ull * (u64)(unsigned)uid) + (((u64)(unsigned)seq << 32) | (u64)(unsigned)t));
}
__host__ __device__ inline float gumbel_from_bits(unsigned n23) {     // n23 < 2^23
  const float u = ((float)n23 + 0.5f) * (1.f / 8388608.f);
  return -logf(-logf(u));
}
__host__ __device__ inline float gumbel(u64 key, int v) { return gumbel_from_bits((unsigned)(mix64(key + (u64)(unsigned)v) >> 41)); }

static_assert(sizeof(Ctx) <= 16384, "Ctx is passed to the persistent kernel by value (kernel parameters are limited to 32,764 bytes)");

}  // namespace mrl
