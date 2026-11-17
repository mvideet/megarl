// qwen_fwd.cuh -- the frozen Qwen2.5-0.5B layers of the rollout (namespace qw, phase kinds K_Q_* below
// K_Q_MID_COHORT) and the prompt-cache build.
//
// A task moves the fed rows (Row.feed) of one pipeline one token forward through one piece of a layer. The rollout
// graph (mt::add_roll_mid) uses these tasks for every layer, except that the four GEMMs of layer LMID are the cohort
// GEMMs of mid_train.cuh. The prompt-cache build runs them as ordinary kernels (k_phase) over the graph of add_trunk.
// Phases of layer l (PhaseDesc.a = l; `rows` = nrows; r = rows_per_task of the phase):
//   Q_EMBN       rows/8       l = 0: hres = float(emb[tok]); xn = bf16(rmsnorm(hres) * ln1[0]), xn[:, D] = 1
//   Q_NORM1F[l]  rows/r       hres += sum_c part[c] (layer l-1's down_proj); xn = bf16(rmsnorm(hres) * ln1[l]), xn[:, D] = 1
//   Q_NORM1[l]   rows/8       xn = bf16(rmsnorm(hres) * ln1[l]), xn[:, D] = 1                (rollout only, l = LMID)
//   Q_QKV[l]     NT_QKV*ksQ   q | k | v = xn * Wqkv[l]^T over KQ (the bias is column D of the weight)
//   Q_ATTN[l]    2*rows       (row, kv head): RoPE, append key and value to the cache, attend, write ao
//   Q_OPROJ[l]   NT_D*ksO     part[c] = ao[:, chunk c] * Wo[l]^T
//   Q_NORM2[l]   rows/r       hres += sum_c part[c]; xn = bf16(rmsnorm(hres) * ln2[l])
//   Q_GU[l]      NT_GU*ksG    gate | up = xn * Wgu[l]^T
//   Q_ACT[l]     NT_FF*s      xm = bf16(silu(gate) * up), s = act_split(ksG)
//   Q_DOWNP[l]   NT_D*ksD     part[c] = xm[:, chunk c] * Wdown[l]^T                           (l < LT; layer LT's is H_Y0P)
//   Q_DOWNR[l]   rows/r       hres += sum_c part[c]                                            (rollout only, l = LMID - 1)
//
// Hand-off in the rollout. Row is written only by R_BEGIN and H_SAMPLE. The lower half of step s and the upper half
// of step s + 1 take the same token at the same position Row.pos = plen + gen - 1:
//   lower half, after H_SAMPLE: Q_EMBN .. Q_DOWNR[LMID-1] feed Row.tok through layers 0..LMID-1, write entry gen - 1
//     of their row caches and leave the residual entering layer LMID in P.hres;
//   upper half, after Q_MID_COH: Q_NORM1[LMID] .. Q_ACT[LT] take that residual through layers LMID..LT, write entry
//     gen of their row caches and leave P.xm = the input of layer LT's down_proj and P.hres = the residual before that
//     down_proj is added (H_Y0P and H_Y0R of the head finish the layer).
// At a group's first step (gen = 0, pos = plen - 1) there is no lower half: R_BEGIN copies Prompts.hlast into P.hres.
// The prompt cache holds every prompt position at layers < LMID and positions 0 .. plen-2 at layers >= LMID, all
// computed with the base weights of Model, layer LMID's included. The last prompt position at layers >= LMID is entry
// 0 of the row caches: the rollout computes it (layer LMID with the group's pinned snapshot), the learner recomputes it.
//
// Arithmetic. Every rounding is explicit (__fmul_rn, __fadd_rn, fmaf), so nothing depends on the compiler's
// contraction choices. Split-K partials are summed in chunk order, s = p0; s += p1; ..., and then added to the value
// (value + s). Attention depends only on the logical key index j = 0..n-1 (prompt segment, then row segment) and on n,
// not on where the segments split, so with the same weights and chunk counts a position gets the same bits whether
// the prompt-cache build or the rollout computes it. The learner's forward over layers LMID..NL-1 (mid_train.cuh) repeats this arithmetic: change both or
// neither.
// A row that is not fed is left untouched in hres, xn, ao, xm and both caches by every phase. The GEMM tiles still
// compute its rows of qkv, qkvp, gu and part, which nobody reads. For that reason o_proj goes to P.part and Q_NORM2
// adds it to the residual even when ksO = 1, instead of an accumulating tile epilogue (the value is the same single
// fp32 addition).
//
// Task rules: every task function is entered by all 256 threads of a block, uses only the dynamic shared memory it is
// given, never uses blockIdx and never waits. One __noinline__ function per task kind and per gemm_tile instantiation.
// Visibility: whatever a task reads that another block wrote during the launch (Row, hres, xn, qkv, ao, gu, xm, the
// partials, the row cache) was written by a phase it depends on, and the scheduler's release / acquire chain orders
// that write before the read. The loads bypass L1 (ld.global.cg, cp.async.cg, volatile) as a cache choice, not for
// correctness. __ldg is used only for *P.nrows and the RoPE tables, which nothing writes during a launch. The error
// counter errors[ERR_ROW] is the only word shared with unordered tasks: atomicAdd only, never read on the device.
// Every thread loads Row.feed (the attention task also pos, prompt, plen and gen) itself and branches on it before the
// next barrier or shuffle, so all threads of a block must see the same values; the dependency chain guarantees that.
//
// Registers. ptxas allocates registers for the whole call graph of the persistent kernel. A local that some path reads
// before writing it (`F4 t; if (c) t = load(); ...; if (c) use(t);`) is live on entry to its function and therefore
// around every call in the kernel's task loop; that takes registers away from every other function, the GEMM tiles
// first, which then spill inside their K loops. So every local of a task function here is defined on every path
// before it is read: zero4() and `= 0.f` below exist for that, not for the arithmetic. After a change, the tile
// functions (g_nt_f32, lmhead_tile) should show no spill stores in the output of
//   nvcc -arch=sm_90a -O3 -std=c++17 -Isrc -Xptxas -v -o build/megarl src/megarl.cu 2>&1 | grep -A1 'g_nt_\|lmhead_tile'
#pragma once
#include "mrl_common.cuh"
#include "mrl_lmhead.cuh"
#include "mrl_host.cuh"

namespace qw {
using mrl::bf16; using mrl::u64; using mrl::F4; using mrl::V4; using mrl::V8;

constexpr int HPK = mrl::NH / mrl::NKV;                       // 7 query heads read one key/value head: head a reads a / 7
constexpr int HH = mrl::HD / 2;                               // RoPE pairs element i with element i + 32
static_assert(mrl::NH * mrl::HD == mrl::D && HPK * mrl::NKV == mrl::NH && mrl::KVW == 4 * mrl::HD && mrl::NKV == 2, "head layout");
static_assert(mrl::D % 4 == 0 && mrl::FF % mrl::TN == 0 && mrl::GUP == 2 * mrl::FF && 16 * mrl::D * 4 <= mrl::SMEM, "row and tile layout; row tasks keep 8 rows and 8 weight copies in shared memory");

// Slot of C.sh.cnt.errors (16 slots that must all stay 0) in which the trunk counts a fed row whose Row fields are
// outside what R_BEGIN and H_SAMPLE produce (the two places are marked "outside the contract" in row_task and
// attn_task). rl_state.cuh names the same slot rls::ERR_TRUNK_ROW for the host audit, which fails a run on any
// non-zero slot.
// Unit: one per embedding of a token id outside [0, V), one per attention task (row, kv head, layer) whose row has
// prompt, plen, gen or pos out of range or pos != plen + gen - 1: 48 per such row and step.
// A Ctx without counters (C.sh.cnt.errors == nullptr, as in the prompt-cache build) counts nothing.
constexpr int ERR_ROW = 14;
__device__ __forceinline__ void count_row_error(const mrl::Ctx& C) { int* e = C.sh.cnt.errors; if (e) atomicAdd(e + ERR_ROW, 1); }

// Task geometry shared by the graph builders (host) and the tasks (device).
// A row task adds `ks` fp32 partial rows to the residual row (0: none): it reads (1 + ks) * 3.6 KB per row, so
// wider reductions get fewer rows per task.
__host__ __device__ inline int rows_per_task(int ks) { return ks <= 1 ? 8 : ks <= 3 ? 4 : ks <= 8 ? 2 : 1; }
__host__ __device__ inline int row_ks(int kind, const mrl::Hyper& hp) {
  return kind == mrl::K_Q_NORM2 ? hp.ksO : (kind == mrl::K_Q_DOWNR || kind == mrl::K_Q_NORM1F) ? hp.ksD : 0;
}
// The activation phase has NT_FF * act_split tasks, each TN / act_split columns of xm for all rows.
__host__ __device__ inline int act_split(int ksG) { return ksG <= 1 ? 2 : 4; }
constexpr int KSG_MAX = 1 + (mrl::KS_MAX * mrl::DP) / mrl::GUP;   // 7: chunks 1.. of gate/up are stored in P.part

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ F4 ld4(const float* p) {           // 16-byte load past L1
  const float4 v = __ldcg(reinterpret_cast<const float4*>(p)); F4 r; r.v[0] = v.x; r.v[1] = v.y; r.v[2] = v.z; r.v[3] = v.w; return r;
}
__device__ __forceinline__ float ld1(const float* p) { return __ldcg(p); }
__device__ __forceinline__ F4 zero4() { F4 r; r.v[0] = 0.f; r.v[1] = 0.f; r.v[2] = 0.f; r.v[3] = 0.f; return r; }
__device__ __forceinline__ void add4(F4& a, const F4& b) {
#pragma unroll
  for (int i = 0; i < 4; i++) a.v[i] = __fadd_rn(a.v[i], b.v[i]);
}

// the only gemm_tile instantiation of the trunk
__device__ __noinline__ void g_nt_f32(const lg::Args& a, int tm, int tn, char* smem) { lg::gemm_tile<lg::NT, lg::EPI_F32>(a, tm, tn, smem); }


// ------------------------------------------------------------------------------------------------ GEMM phases
// Q_QKV, Q_OPROJ, Q_GU, Q_DOWNP: task `local` = N tile (local / ks), K chunk (local % ks). All rows < nrows are
// computed (rows are independent in a tile). Outputs: q/k/v to P.qkv when ksQ = 1, else chunk c to P.qkvp[c];
// gate/up chunk 0 to P.gu and chunk c >= 1 to P.part (row pitch GUP); o_proj and down_proj chunk c to P.part[c].
__device__ __forceinline__ void gemm_task(const mrl::Ctx& C, const mrl::Pipe& P, int kind, int layer, int local, char* smem) {
  const int nrows = __ldg(P.nrows);
  lg::Args a = lg::args_default(); a.M = mrl::MAXROWS; a.avalid = nrows; a.rvalid = nrows;
  int ks, nk32;
  switch (kind) {
    case mrl::K_Q_QKV:   ks = C.hp.ksQ; nk32 = mrl::KQ / 32; break;
    case mrl::K_Q_OPROJ: ks = C.hp.ksO; nk32 = mrl::D / 32; break;
    case mrl::K_Q_GU:    ks = C.hp.ksG; nk32 = mrl::D / 32; break;
    default:             ks = C.hp.ksD; nk32 = mrl::FF / 32; break;      // K_Q_DOWNP
  }
  const int tn = local / ks, c = local % ks; int k0, kc; mrl::kchunk(nk32, ks, c, k0, kc); a.K = kc;
  switch (kind) {
    case mrl::K_Q_QKV:
      a.A = P.xn + k0; a.lda = mrl::KQ; a.B = C.model.qkv[layer] + k0; a.ldb = mrl::KQ; a.bvalid = mrl::DQKV;
      a.C = ks == 1 ? P.qkv : P.qkvp + (size_t)c * mrl::MAXROWS * mrl::QKVP; a.ldc = mrl::QKVP; a.cvalid = mrl::DQKV; a.N = mrl::QKVP; break;
    case mrl::K_Q_OPROJ:
      a.A = P.ao + k0; a.lda = mrl::D; a.B = C.model.o[layer] + k0; a.ldb = mrl::D; a.bvalid = mrl::D;
      a.C = P.part + (size_t)c * mrl::MAXROWS * mrl::DP; a.ldc = mrl::DP; a.cvalid = mrl::D; a.N = mrl::DP; break;
    case mrl::K_Q_GU:
      a.A = P.xn + k0; a.lda = mrl::KQ; a.B = C.model.gu[layer] + k0; a.ldb = mrl::D;
      a.C = c == 0 ? P.gu : P.part + (size_t)(c - 1) * mrl::MAXROWS * mrl::GUP; a.ldc = mrl::GUP; a.N = mrl::GUP; break;
    default:
      a.A = P.xm + k0; a.lda = mrl::FF; a.B = C.model.down[layer] + k0; a.ldb = mrl::FF; a.bvalid = mrl::D;
      a.C = P.part + (size_t)c * mrl::MAXROWS * mrl::DP; a.ldc = mrl::DP; a.cvalid = mrl::D; a.N = mrl::DP; break;
  }
  g_nt_f32(a, 0, tn, smem);
}

// ------------------------------------------------------------------------------------------------ row phases
// Q_EMBN, Q_NORM1, Q_NORM1F, Q_NORM2, Q_DOWNR. A task has rows_per_task(ks) rows and WPR = 8 / rows warps per row.
// Per fed row, in this order:
//   1. the residual row: the embedding of Row.tok (EMBN) or hres;
//   2. plus the sum of ks partial rows of P.part, chunk 0 first (NORM2: o_proj; NORM1F, DOWNR: down_proj);
//   3. stored to hres if 1 or 2 changed it;
//   4. except for DOWNR, normalised into xn by one warp (lm::warp_rmsnorm_row with ln1[layer], or ln2[layer] for
//      NORM2; all kinds but NORM2 also set xn[:, D] = 1).
// The thread that owns F4 chunk ch of a row does 1-3 for it and leaves the chunk in shared memory (the row buffer rb)
// for step 4, together with the same chunk of the norm weight, so the norm works from shared memory alone (it reads
// its operands one 16-byte piece at a time, which from global memory would cost a round trip per piece).
// Rows without Row.feed are skipped.
//
// row_chunks: steps 1-3 for the chunks of one thread, WPR warps per row. Loads are issued in rounds and used
// afterwards (a thread that alternates load and use waits one memory round trip per load). A round is the thread's NCH
// chunks with their residual, their norm weight and CB partial rows each, at most 21 loads. rows_per_task pairs ks
// with WPR so that ks <= CB (1, 3, 8 for WPR 1, 2, 4) and a task is one round; only with WPR = 8 (ks > 8) are there
// several rounds of partials.
template <int WPR>
__device__ __forceinline__ void row_chunks(float* h, const float* part, const bf16* emb_row, int ks, float* rb, const float* w, float* wb, int ti) {
  constexpr int NTHR = 32 * WPR, NCH = (mrl::D / 4 + NTHR - 1) / NTHR;          // 7, 4, 2, 1 chunks per thread
  constexpr int G = NCH, CB = WPR == 1 ? 1 : WPR == 2 ? 3 : 8;                    // all chunks of the thread in one round
#pragma unroll
  for (int k0 = 0; k0 < NCH; k0 += G) {
    F4 x[G], wv[G];
#pragma unroll
    for (int k = 0; k < G; k++) {
      const int ch = ti + (k0 + k) * NTHR;
      x[k] = zero4(); wv[k] = zero4();                                         // defined on every path (Registers, top of file)
      if (k0 + k < NCH && ch < mrl::D / 4) {
        if (w) wv[k] = *reinterpret_cast<const F4*>(w + 4 * ch);               // the norm weight rides along (read-only data)
        if (emb_row) {
          const V4 v = *reinterpret_cast<const V4*>(emb_row + 4 * ch);
#pragma unroll
          for (int i = 0; i < 4; i++) x[k].v[i] = __bfloat162float(v.v[i]);
        } else x[k] = ld4(h + 4 * ch);
      }
    }
    for (int c0 = 0; c0 < ks; c0 += CB) {                                      // partial rows c0 .. c0 + CB - 1, added in chunk order
      F4 p[CB][G];
#pragma unroll
      for (int j = 0; j < CB; j++) {
#pragma unroll
        for (int k = 0; k < G; k++) {
          const int ch = ti + (k0 + k) * NTHR;
          p[j][k] = zero4();
          if (c0 + j < ks && k0 + k < NCH && ch < mrl::D / 4) p[j][k] = ld4(part + (size_t)(c0 + j) * (mrl::MAXROWS * mrl::DP) + 4 * ch);
        }
      }
#pragma unroll
      for (int k = 0; k < G; k++) {
        const int ch = ti + (k0 + k) * NTHR;
        if (k0 + k < NCH && ch < mrl::D / 4) {
          F4 s = zero4();                                                      // s = p[0] + p[1] + ... strictly in chunk order;
          if (c0 > 0) s = *reinterpret_cast<const F4*>(rb + 4 * ch);           // between rounds (ks > CB only) it waits in the row buffer
#pragma unroll
          for (int j = 0; j < CB; j++) if (c0 + j < ks) { if (c0 + j == 0) s = p[j][k]; else add4(s, p[j][k]); }
          if (c0 + CB >= ks) add4(x[k], s); else *reinterpret_cast<F4*>(rb + 4 * ch) = s;
        }
      }
    }
#pragma unroll
    for (int k = 0; k < G; k++) {
      const int ch = ti + (k0 + k) * NTHR;
      if (k0 + k < NCH && ch < mrl::D / 4) {
        if (emb_row || ks > 0) *reinterpret_cast<F4*>(h + 4 * ch) = x[k];
        *reinterpret_cast<F4*>(rb + 4 * ch) = x[k];
        if (w) *reinterpret_cast<F4*>(wb + 4 * ch) = wv[k];
      }
    }
  }
}

// One instantiation per rows-per-task value, each its own __noinline__ function (as for the tile instantiations:
// several unrolled loop nests in one function compete for its registers).
template <int WPR>
__device__ __noinline__ void row_task(const mrl::Ctx& C, const mrl::Pipe& P, int kind, int layer, int local, char* smem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool embed = kind == mrl::K_Q_EMBN;
  const bool norm = kind != mrl::K_Q_DOWNR;
  const int ks = row_ks(kind, C.hp);
  const int slot = warp / WPR, sub = warp % WPR, row = local * (8 / WPR) + slot;
  float* rb = reinterpret_cast<float*>(smem) + slot * mrl::D;                  // this row's residual for the norm, [D] floats
  float* wb = reinterpret_cast<float*>(smem) + (8 + slot) * mrl::D;            // and the norm weight (as many copies as rows: every row's threads load their own chunks)
  const bool first = kind != mrl::K_Q_NORM2;
  const float* w = norm ? (first ? C.model.ln1[layer] : C.model.ln2[layer]) : nullptr;
  // the row count and the row's flags are loaded together: row < MAXROWS holds for every task of a phase, so the
  // Row read is in bounds whatever nrows is
  const int nrows = __ldg(P.nrows), feed = mk::ld_vol(&P.row[row].feed);
  int tok = embed ? mk::ld_vol(&P.row[row].tok) : 0;
  const bool on = row < nrows && feed != 0;
  if (on) {
    const bf16* e = nullptr;
    if (embed) {                                                               // ids >= V do not exist: outside the contract, counted once per row;
      if ((unsigned)tok >= (unsigned)mrl::V && sub == 0 && lane == 0) count_row_error(C);
      tok = tok < 0 ? 0 : tok >= mrl::VP ? mrl::VP - 1 : tok;                  // the clamp keeps the read inside the table (its rows >= V are zero)
      e = C.model.emb + (size_t)tok * mrl::D;
    }
    row_chunks<WPR>(P.hres + (size_t)row * mrl::DP, P.part + (size_t)row * mrl::DP, e, ks, rb, w, wb, sub * 32 + lane);   // partial chunk c is at + c * MAXROWS * DP
  }
  if (!norm) return;
  __syncthreads();
  if (on && sub == 0) lm::warp_rmsnorm_row(rb, wb, P.xn + (size_t)row * mrl::KQ, first);
}

// ------------------------------------------------------------------------------------------------ activation
// Q_ACT: xm = bf16(silu(gate) * up) for columns [local * W, local * W + W) of every fed row, W = TN / act_split.
// The unit of work is a group of 8 columns of one row: two 16-byte loads of gate, two of up (per gate/up chunk), one
// 16-byte store. The lanes of a warp take consecutive groups; a thread takes 8 / act_split groups, in rounds of 16
// loads (one round for ksG <= 2). With ksG > 1, gate and up are the sums of the chunk partials, chunk 0 first.
// act_split trades tasks for time: silu costs a software division per element, 64 rows x W of them per task.
template <int NG, int CB>                                     // NG groups per round, CB gate/up chunks per round: 4 * NG * CB = 16 loads in flight
__device__ __forceinline__ void act_round(const mrl::Pipe& P, int ksG, const int* feed, int q0, int W8, int c0, int nq) {
  F4 a[NG][4]; bool on[NG]; size_t o[NG], ox[NG];              // per group: gate[0..3], gate[4..7], up[0..3], up[4..7]
#pragma unroll
  for (int g = 0; g < NG; g++) {
    const int q = q0 + 256 * g, row = q < nq ? q / W8 : 0, col = c0 + (q % W8) * 8;
    on[g] = q < nq && feed[row] != 0; o[g] = (size_t)row * mrl::GUP + col; ox[g] = (size_t)row * mrl::FF + col;
#pragma unroll
    for (int i = 0; i < 4; i++) a[g][i] = zero4();             // defined on every path (Registers, top of file)
  }
  for (int cb = 0; cb < ksG; cb += CB) {
    F4 t[NG][CB][4];
#pragma unroll
    for (int g = 0; g < NG; g++) {
#pragma unroll
      for (int j = 0; j < CB; j++) {
#pragma unroll
        for (int i = 0; i < 4; i++) t[g][j][i] = zero4();
        if (on[g] && cb + j < ksG) {
          const float* src = (cb + j == 0 ? P.gu : P.part + (size_t)(cb + j - 1) * mrl::MAXROWS * mrl::GUP) + o[g];
          t[g][j][0] = ld4(src); t[g][j][1] = ld4(src + 4); t[g][j][2] = ld4(src + mrl::FF); t[g][j][3] = ld4(src + mrl::FF + 4);
        }
      }
    }
#pragma unroll
    for (int g = 0; g < NG; g++) {
#pragma unroll
      for (int j = 0; j < CB; j++) if (on[g] && cb + j < ksG) {
#pragma unroll
        for (int i = 0; i < 4; i++) { if (cb + j == 0) a[g][i] = t[g][j][i]; else add4(a[g][i], t[g][j][i]); }     // chunk order, chunk 0 first
      }
    }
  }
#pragma unroll
  for (int g = 0; g < NG; g++) if (on[g]) {
    V8 out;
#pragma unroll
    for (int i = 0; i < 8; i++) out.v[i] = __float2bfloat16(__fmul_rn(mrl::silu(a[g][i >> 2].v[i & 3]), a[g][2 + (i >> 2)].v[i & 3]));
    *reinterpret_cast<V8*>(P.xm + ox[g]) = out;
  }
}

__device__ __noinline__ void act_task(const mrl::Ctx& C, const mrl::Pipe& P, int local, char* smem) {
  const int tid = threadIdx.x, ksG = C.hp.ksG, W = mrl::TN / act_split(ksG), W8 = W / 8;
  const int c0 = local * W, nrows = __ldg(P.nrows), nq = mrl::MAXROWS * W8;
  int* feed = reinterpret_cast<int*>(smem);                    // feed flag of every row, for all threads
  if (tid < mrl::MAXROWS) { const int f = mk::ld_vol(&P.row[tid].feed); feed[tid] = tid < nrows ? f : 0; }
  __syncthreads();
  if (ksG == 1) for (int q = tid; q < nq; q += 4 * 256) act_round<4, 1>(P, ksG, feed, q, W8, c0, nq);        // 4 groups per thread: 1 round
  else if (ksG == 2) for (int q = tid; q < nq; q += 2 * 256) act_round<2, 2>(P, ksG, feed, q, W8, c0, nq);  // 2 groups: 1 round
  else for (int q = tid; q < nq; q += 256) act_round<1, 4>(P, ksG, feed, q, W8, c0, nq);                    // 2 groups, 4 chunks per round
}

// ------------------------------------------------------------------------------------------------ attention
// Q_ATTN: task `local` = (row local / 2, key/value head g = local % 2): query heads 7g .. 7g + 6 of one row.
//   context            n keys with logical index j = 0..n-1: segment A (a prompt cache) first, then segment B (the
//                      row cache); the last one, j = n - 1, is the position being fed, and this task writes it.
//   rollout, l < LMID  A = prompts.kv[prompt][l][0, plen),     B = P.kv[row][l][0, gen),     writes B[gen - 1]
//   rollout, l >= LMID A = prompts.kv[prompt][l][0, plen - 1), B = P.kv[row][l][0, gen + 1), writes B[gen]
//   prefill            A = prompts.kv[prompt][l][0, pos], no B, writes A[pos]                 (PhaseDesc.b = 1)
// One cache entry is KVW = 256 floats: rotated key of kv head 0, of head 1, value of head 0, of head 1. The two tasks
// of a row write disjoint halves of the entry.
// Steps (thread roles in brackets):
//   1. stage the n - 1 older keys of head g in shared memory with cp.async           [16-byte pieces over all threads]
//   2. q (7 heads) and k from qkv (sum of the ksQ partials, loaded at task entry), RoPE at Row.pos; v as is; key and
//      value to the cache; rotated queries and the new key to shared memory           [warp w < 7: query head 7g + w; warp 7: k, v; lane i: pair (i, i + 32)]
//   3. scores s[a][j] = 0.125 * dot(q'[a], K[j]), a 64-term fmaf chain per (a, j)     [thread j: key j, all 7 heads]
//   4. softmax over j per head: max, exp(s - max), sum, divide                        [fixed shuffle trees over the thread index]
//   5. the values replace the keys in shared memory (staged while 4 runs)
//   6. o[a][d] = sum_j p[a][j] * V[j][d]: four interleaved fmaf chains (j mod 4), added in order   [thread (j mod 4, d)]
//   7. ao[row][64 (7g + a) + d] = bf16(o[a][d])                                        [56 threads, 8 columns each]
// Shared memory: keys and values share one region (pitch 68 floats so that consecutive rows start 4 banks apart);
// with queries, probabilities and partial outputs the task needs 68,864 of the 81,920 bytes for n up to 192.
constexpr int AT_MAXN = mrl::MAXPROMPT + mrl::MAXGEN;         // longest context: 128 prompt positions + 64 generated
constexpr int AT_PITCH = mrl::HD + 4;
constexpr int AT_KV = 0;                                      // [AT_MAXN][AT_PITCH]  keys, then values
constexpr int AT_Q = AT_KV + AT_MAXN * AT_PITCH;              // [HPK][HD]            rotated queries
constexpr int AT_P = AT_Q + HPK * mrl::HD;                    // [HPK][256]           probabilities, thread index = key index
constexpr int AT_O = AT_P + HPK * 256;                        // [4][HPK][HD]         partial outputs of the four chains
constexpr int AT_R = AT_O + 4 * HPK * mrl::HD;                // [2][8][8]            per-warp maxima, per-warp sums
constexpr int AT_END = AT_R + 2 * 8 * 8;
static_assert(AT_END * 4 <= mrl::SMEM && AT_MAXN <= 256 && HPK <= 8, "attention scratch must fit in the block's shared memory");

__device__ __noinline__ void attn_task(const mrl::Ctx& C, const mrl::Pipe& P, int layer, int prefill, int local, char* smem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int row = local >> 1, g = local & 1;
  // The row count, the row's bookkeeping and its q, k, v are loaded together, before anything is decided: the loads
  // of a row that turns out not to be fed are wasted, but a fed row saves two memory round trips. (row < MAXROWS
  // for every task of the phase, so the reads are in bounds whatever nrows is.)
  const mrl::Row* R = P.row + row;
  const int nrows = __ldg(P.nrows);
  const int feed = mk::ld_vol(&R->feed), pos = mk::ld_vol(&R->pos), prompt = mk::ld_vol(&R->prompt), plen = mk::ld_vol(&R->plen), gen = mk::ld_vol(&R->gen);
  const int ksQ = C.hp.ksQ;
  const size_t col = (size_t)row * mrl::QKVP + (warp < HPK ? (HPK * g + warp) * mrl::HD : mrl::D + g * mrl::HD) + lane;
  float x[4];                                                                   // x[i], x[i+32] of this warp's vector (warp w < 7: query head 7g + w; warp 7: the key); warp 7: also v[i], v[i+32]
#pragma unroll
  for (int i = 0; i < 4; i++) x[i] = 0.f;
  for (int cb = 0; cb < ksQ; cb += 4) {                                          // 4 chunks per round, added in chunk order
    float t[4][4]; const int nv = warp < HPK ? 2 : 4;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int i = 0; i < 4; i++) t[i][j] = 0.f;                                 // defined on every path (Registers, top of file)
    }
#pragma unroll
    for (int j = 0; j < 4; j++) if (cb + j < ksQ) {
      const float* src = (ksQ == 1 ? P.qkv : P.qkvp + (size_t)(cb + j) * mrl::MAXROWS * mrl::QKVP) + col;
      t[0][j] = ld1(src); t[1][j] = ld1(src + HH);
      if (nv == 4) { t[2][j] = ld1(src + mrl::NKV * mrl::HD); t[3][j] = ld1(src + mrl::NKV * mrl::HD + HH); }
    }
#pragma unroll
    for (int j = 0; j < 4; j++) if (cb + j < ksQ) {
#pragma unroll
      for (int i = 0; i < 4; i++) if (i < nv) x[i] = cb + j == 0 ? t[i][j] : __fadd_rn(x[i], t[i][j]);
    }
  }
  if (row >= nrows || feed == 0) return;
  // A fed row outside the contract (R_BEGIN, H_SAMPLE and the prompt-cache build never produce one). If its fields
  // would take this task outside the caches or the RoPE table, the task touches nothing; a position that is not
  // plen + gen - 1 is used as it is given. In both cases thread 0 counts the task in errors[ERR_ROW]: the row's other
  // phases treated it as fed, so its state is garbage and the run must fail its audit.
  const bool inside = (unsigned)prompt < (unsigned)C.prompts.n &&
                      (prefill ? pos >= 0 && pos < mrl::MAXPROMPT
                               : (layer >= mrl::LMID ? (plen >= 1 && plen <= mrl::MAXPROMPT && gen >= 0 && gen < mrl::MAXGEN && pos >= 0 && pos < mrl::MAXPOS)
                                                     : (plen >= 0 && plen <= mrl::MAXPROMPT && gen >= 1 && gen <= mrl::MAXGEN && pos >= 0 && pos < mrl::MAXPOS))
                      );
  if (tid == 0 && (!inside || (!prefill && pos != plen + gen - 1))) count_row_error(C);
  if (!inside) return;
  const float* segA = C.prompts.kv + ((size_t)prompt * mrl::NL + layer) * (mrl::MAXPROMPT * mrl::KVW);
  const float* segB = segA; float* wr; int lenA, n;
  if (prefill) {
    lenA = pos; n = pos + 1; wr = C.prompts.kv + ((size_t)prompt * mrl::NL + layer) * (mrl::MAXPROMPT * mrl::KVW) + (size_t)pos * mrl::KVW;
  } else {
    float* b = P.kv + ((size_t)row * mrl::NL + layer) * (mrl::MAXGEN * mrl::KVW);
    if (layer >= mrl::LMID) {
      segB = b; lenA = plen - 1; n = lenA + gen + 1; wr = b + (size_t)gen * mrl::KVW;
    } else
    {
      segB = b; lenA = plen; n = plen + gen; wr = b + (size_t)(gen - 1) * mrl::KVW;
    }
  }
  const int nold = n - 1;
  float* S = reinterpret_cast<float*>(smem);

  // 1. older keys -> shared memory (the copies are in flight while step 2 runs)
  for (int q = tid; q < nold * 16; q += 256) {
    const int j = q >> 4, c = q & 15;
    const float* src = (j < lenA ? segA + (size_t)j * mrl::KVW : segB + (size_t)(j - lenA) * mrl::KVW) + g * mrl::HD + 4 * c;
    lg::cp_async16(lg::smem_u32(S + AT_KV + j * AT_PITCH + 4 * c), src);
  }
  lg::cp_commit();

  // 2. RoPE of this position's q and k
  const float cs = __ldg(C.model.rope_cos + (size_t)pos * HH + lane), sn = __ldg(C.model.rope_sin + (size_t)pos * HH + lane);
  const float rl = __fadd_rn(__fmul_rn(x[0], cs), -__fmul_rn(x[1], sn));         // x'[i]      = x[i] c - x[i+32] s
  const float rh = __fadd_rn(__fmul_rn(x[1], cs), __fmul_rn(x[0], sn));          // x'[i + 32] = x[i+32] c + x[i] s
  if (warp < HPK) { S[AT_Q + warp * mrl::HD + lane] = rl; S[AT_Q + warp * mrl::HD + HH + lane] = rh; }
  else {
    float* kn = S + AT_KV + nold * AT_PITCH; kn[lane] = rl; kn[HH + lane] = rh;
    float* wk = wr + g * mrl::HD, *wv = wr + mrl::NKV * mrl::HD + g * mrl::HD;
    wk[lane] = rl; wk[HH + lane] = rh; wv[lane] = x[2]; wv[HH + lane] = x[3];
  }
  lg::cp_wait<0>();
  __syncthreads();

  // 3. scores of key `tid` for the 7 query heads
  float sc[HPK];
  if (tid < n) {
    const float* kr = S + AT_KV + tid * AT_PITCH; float acc[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) acc[a] = 0.f;
    for (int c = 0; c < mrl::HD / 4; c++) {
      const F4 kk = *reinterpret_cast<const F4*>(kr + 4 * c);
#pragma unroll
      for (int a = 0; a < HPK; a++) {
        const F4 qq = *reinterpret_cast<const F4*>(S + AT_Q + a * mrl::HD + 4 * c);
#pragma unroll
        for (int i = 0; i < 4; i++) acc[a] = fmaf(kk.v[i], qq.v[i], acc[a]);
      }
    }
#pragma unroll
    for (int a = 0; a < HPK; a++) sc[a] = __fmul_rn(0.125f, acc[a]);              // 1 / sqrt(head dimension), exact
  } else {
#pragma unroll
    for (int a = 0; a < HPK; a++) sc[a] = -INFINITY;
  }

  // 4. softmax, part 1: the maximum of each head
  {
    float m[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) m[a] = sc[a];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
      for (int a = 0; a < HPK; a++) m[a] = fmaxf(m[a], __shfl_xor_sync(0xffffffffu, m[a], o));
    }
    if (lane == 0) {
#pragma unroll
      for (int a = 0; a < HPK; a++) S[AT_R + warp * 8 + a] = m[a];
    }
  }
  __syncthreads();

  // 5. every thread is done with the keys: the region takes the values
  for (int q = tid; q < nold * 16; q += 256) {
    const int j = q >> 4, c = q & 15;
    const float* src = (j < lenA ? segA + (size_t)j * mrl::KVW : segB + (size_t)(j - lenA) * mrl::KVW) + mrl::NKV * mrl::HD + g * mrl::HD + 4 * c;
    lg::cp_async16(lg::smem_u32(S + AT_KV + j * AT_PITCH + 4 * c), src);
  }
  lg::cp_commit();
  if (warp == HPK) { float* vn = S + AT_KV + nold * AT_PITCH; vn[lane] = x[2]; vn[HH + lane] = x[3]; }

  // 4. softmax, part 2: exponentials, their sum, the probabilities
  float e[HPK];
  {
    float t[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) {
      float mm = S[AT_R + a];
#pragma unroll
      for (int w = 1; w < 8; w++) mm = fmaxf(mm, S[AT_R + w * 8 + a]);
      e[a] = tid < n ? expf(__fadd_rn(sc[a], -mm)) : 0.f;
      t[a] = e[a];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
      for (int a = 0; a < HPK; a++) t[a] = __fadd_rn(t[a], __shfl_xor_sync(0xffffffffu, t[a], o));
    }
    if (lane == 0) {
#pragma unroll
      for (int a = 0; a < HPK; a++) S[AT_R + 64 + warp * 8 + a] = t[a];
    }
  }
  __syncthreads();
#pragma unroll
  for (int a = 0; a < HPK; a++) {
    float tot = S[AT_R + 64 + a];
#pragma unroll
    for (int w = 1; w < 8; w++) tot = __fadd_rn(tot, S[AT_R + 64 + w * 8 + a]);
    if (tid < n) S[AT_P + a * 256 + tid] = __fdiv_rn(e[a], tot);
  }
  lg::cp_wait<0>();
  __syncthreads();

  // 6. weighted values: thread (jg, d) accumulates keys jg, jg + 4, ... for output column d of all 7 heads
  {
    const int jg = tid >> 6, d = tid & 63; float acc[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) acc[a] = 0.f;
    for (int j = jg; j < n; j += 4) {
      const float v = S[AT_KV + j * AT_PITCH + d];
#pragma unroll
      for (int a = 0; a < HPK; a++) acc[a] = fmaf(S[AT_P + a * 256 + j], v, acc[a]);
    }
#pragma unroll
    for (int a = 0; a < HPK; a++) S[AT_O + (jg * HPK + a) * mrl::HD + d] = acc[a];
  }
  __syncthreads();

  // 7. add the four chains in order, round to bf16, store 8 columns per thread
  if (tid < HPK * 8) {
    const int a = tid >> 3, d0 = (tid & 7) * 8; V8 out;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float* p = S + AT_O + a * mrl::HD + d0 + i;
      float s = p[0];
#pragma unroll
      for (int k = 1; k < 4; k++) s = __fadd_rn(s, p[k * HPK * mrl::HD]);
      out.v[i] = __float2bfloat16(s);
    }
    *reinterpret_cast<V8*>(P.ao + (size_t)row * mrl::D + (HPK * g + a) * mrl::HD + d0) = out;
  }
}

// ------------------------------------------------------------------------------------------------ dispatcher
// One task of a K_Q_ phase below K_Q_MID_COHORT. `local` is the task index inside one pipeline (the callers have
// divided out the lockstep multiplier), P the pipeline's buffers; pd.a = layer, pd.b = 1 in the prompt-cache build.
__device__ __forceinline__ void run(const mrl::Ctx& C, const mrl::Pipe& P, const mk::PhaseDesc& pd, int local, char* smem) {
  switch (pd.kind) {
    case mrl::K_Q_QKV: case mrl::K_Q_OPROJ: case mrl::K_Q_GU: case mrl::K_Q_DOWNP: gemm_task(C, P, pd.kind, pd.a, local, smem); break;
    case mrl::K_Q_ATTN: attn_task(C, P, pd.a, pd.b, local, smem); break;
    case mrl::K_Q_ACT: act_task(C, P, local, smem); break;
    case mrl::K_Q_NORM1: case mrl::K_Q_NORM2: case mrl::K_Q_DOWNR: case mrl::K_Q_NORM1F: case mrl::K_Q_EMBN:
      switch (rows_per_task(row_ks(pd.kind, C.hp))) {
        case 8: row_task<1>(C, P, pd.kind, pd.a, local, smem); break;
        case 4: row_task<2>(C, P, pd.kind, pd.a, local, smem); break;
        case 2: row_task<4>(C, P, pd.kind, pd.a, local, smem); break;
        default: row_task<8>(C, P, pd.kind, pd.a, local, smem); break;
      }
      break;
    default: break;
  }
}

// ------------------------------------------------------------------------------------------------ host: job graph
// The chunk counts the tasks support: every split-K partial must fit in P.qkvp, P.gu or P.part. Dies otherwise.
inline void check_trunk_hyper(const mrl::Hyper& hp) {
  auto rng = [&](const char* n, int v, int lo, int hi) { if (v < lo || v > hi) mrl::host::die(std::string("qw: ") + n + " = " + std::to_string(v) + " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + "]"); };
  rng("ksQ", hp.ksQ, 1, std::min(mrl::KS_MAX / 4, mrl::KQ / 32));               // partials in P.qkvp
  rng("ksO", hp.ksO, 1, std::min(mrl::KS_MAX, mrl::D / 32));                    // partials in P.part
  // chunk 0 in P.gu, the others in P.part, which holds KSG_MAX - 1 of them. mrl::host::check_hyper accepts up to
  // D / 32 = 28, so megarl stops here and not at its command line: say why.
  if (hp.ksG < 1 || hp.ksG > KSG_MAX) mrl::host::die("qw: ksG = " + std::to_string(hp.ksG) + " is outside [1, " + std::to_string(KSG_MAX) + "]: the gate/up chunks after the first are kept in P.part, which has room for " + std::to_string(KSG_MAX - 1));
  rng("ksD", hp.ksD, 1, std::min(mrl::KS_MAX, mrl::FF / 32));                   // partials in P.part
}
static_assert((KSG_MAX - 1) * (long long)mrl::GUP <= (long long)mrl::KS_MAX * mrl::DP && KSG_MAX >= 2, "gate/up partials of chunks 1 .. KSG_MAX - 1 must fit in P.part");
// The graph of the prompt-cache build: all NL layers as one chain (layer 0 starts with Q_EMBN, every later layer with
// Q_NORM1F[l]; there is no Q_DOWNP[LT]). The first phase depends on phase `dep` of the same step (dep < 0: on nothing), every other phase on
// its predecessor. Task counts are per pipeline times `mult`, which is stored in PhaseDesc.c; PhaseDesc.a = layer,
// PhaseDesc.b = prefill. Returns the index of the last phase (Q_ACT[LT]). The rollout graph is mt::add_roll_mid.
// The device code takes the chunk counts from Ctx.hp, so the graph and the Ctx must be built from the same Hyper.
// Likewise *P.nrows must be the `nrows` given here: the row and attention phases have just enough tasks for nrows
// rows (a task or row beyond *P.nrows does nothing; a row the graph does not reach is never processed).
inline int add_trunk(mk::Builder& b, int dep, int nrows, const mrl::Hyper& hp, int mult, int prefill = 0) {
  if (nrows < 1 || nrows > mrl::MAXROWS) mrl::host::die("qw::add_trunk: nrows must be between 1 and " + std::to_string(mrl::MAXROWS));
  if (mult < 1 || mult > mrl::MAXPIPE) mrl::host::die("qw::add_trunk: bad lockstep multiplier");
  check_trunk_hyper(hp);
  int prev = dep;
  auto add = [&](const std::string& name, int kind, int layer, int n) {
    std::vector<mk::Dep> d; if (prev >= 0) d.push_back(mk::Dep{prev, 0});
    prev = b.add(name, kind, n * mult, d, layer, prefill ? 1 : 0, mult);
  };
  auto rowt = [&](int kind) { const int r = rows_per_task(row_ks(kind, hp)); return (nrows + r - 1) / r; };
  for (int l = 0; l < mrl::NL; l++) {
    const std::string s = "[" + std::to_string(l) + "]";
    if (l == 0) add("Q_EMBN", mrl::K_Q_EMBN, 0, rowt(mrl::K_Q_EMBN));
    else add("Q_NORM1F" + s, mrl::K_Q_NORM1F, l, rowt(mrl::K_Q_NORM1F));
    add("Q_QKV" + s, mrl::K_Q_QKV, l, mrl::NT_QKV * hp.ksQ);
    add("Q_ATTN" + s, mrl::K_Q_ATTN, l, 2 * nrows);
    add("Q_OPROJ" + s, mrl::K_Q_OPROJ, l, mrl::NT_D * hp.ksO);
    add("Q_NORM2" + s, mrl::K_Q_NORM2, l, rowt(mrl::K_Q_NORM2));
    add("Q_GU" + s, mrl::K_Q_GU, l, mrl::NT_GU * hp.ksG);
    add("Q_ACT" + s, mrl::K_Q_ACT, l, mrl::NT_FF * act_split(hp.ksG));
    if (l < mrl::LT) {
      add("Q_DOWNP" + s, mrl::K_Q_DOWNP, l, mrl::NT_D * hp.ksD);
    }
  }
  return prev;
}

// ------------------------------------------------------------------------------------------------ ordinary kernels
// A K_Q_ phase as an ordinary kernel, one block per task: grid = the phase's task count (lockstep multiplier
// included; `pipes` then has pd.c entries). Used by the prompt-cache build. The Ctx is read through a pointer: a
// by-value Ctx would be copied to every thread's stack at every launch.
__global__ void __launch_bounds__(256, 2) k_phase(const mrl::Ctx* C, const mrl::Pipe* pipes, mk::PhaseDesc pd) {
  extern __shared__ __align__(1024) unsigned char dsm[];
  wg::check_smem_base(dsm);
  const int mult = pd.c > 1 ? pd.c : 1;
  run(*C, pipes[blockIdx.x % mult], pd, (int)(blockIdx.x / mult), reinterpret_cast<char*>(dsm));
}

// Prompt-cache build, per position: row r feeds token `pos` of prompt p0 + r while pos is inside the prompt.
// Rows r >= nb carry no prompt.
__global__ void k_prefill_rows(mrl::Row* row, const int* len, const int* tok, int p0, int nb, int pos) {
  const int r = threadIdx.x;
  if (r >= mrl::MAXROWS) return;
  mrl::Row w; w.active = 0; w.feed = 0; w.prompt = 0; w.plen = 0; w.gen = 0; w.tok = 0; w.pos = pos; w.buf = -1; w.seq = 0;
  if (r < nb) {
    const int p = p0 + r, plen = len[p];
    w.prompt = p; w.plen = plen;
    if (pos < plen) { w.feed = 1; w.tok = tok[(size_t)p * mrl::MAXPROMPT + pos]; }
  }
  row[r] = w;
}
// Stops feeding the rows whose prompt ends at this position, so that layers LMID.. leave its cache entries alone
// (the rollout computes them).
__global__ void k_prefill_trim_last(mrl::Row* row, int nb) {
  const int r = threadIdx.x;
  if (r < nb && row[r].feed && row[r].pos == row[r].plen - 1) row[r].feed = 0;
}
// Prompt-cache build, just before Q_QKV[LMID]: one block per row; a fed row whose prompt ends at this position copies
// its hres row (columns < D, the residual entering layer LMID) to hlast[prompt].
__global__ void k_prefill_save(const mrl::Row* row, const float* hres, float* hlast) {
  const mrl::Row* R = row + blockIdx.x;
  if (!R->feed || R->pos != R->plen - 1) return;
  const int p = R->prompt;
  for (int i = threadIdx.x; i < mrl::D / 4; i += blockDim.x)
    *reinterpret_cast<F4*>(hlast + (size_t)p * mrl::D + 4 * i) = *reinterpret_cast<const F4*>(hres + (size_t)blockIdx.x * mrl::DP + 4 * i);
}

inline void free_pipe(mrl::Pipe& p) {
  void* v[] = {p.hres, p.xn, p.qkv, p.qkvp, p.ao, p.gu, p.xm, p.part, p.xf, p.lmpart, p.kv, p.row, p.grp, p.nrows};
  for (void* q : v) if (q) MK_CK(cudaFree(q));
  memset(&p, 0, sizeof p);
}

// Host: fills P.kv and P.hlast for all P.n prompts with the base weights of M. The tasks run as ordinary kernels
// with rows = prompts, one position per pass; a row is fed while the position is inside its prompt. At the last
// position of a prompt, hres is saved to hlast just before Q_QKV[LMID] and the row stops being fed, so layers
// >= LMID cache positions 0 .. plen-2 only (see the header). Up to PREFILL_PIPES pipelines of 64 prompts each
// advance together in one launch per phase (the lockstep multiplier of add_trunk), so a pass costs one set of
// launches for up to 512 prompts; a row's arithmetic does not depend on which pipeline carries it or on the other
// rows. hp must be the Hyper of the rollout: its chunk counts fix the order of the partial sums. Returns after the
// device has finished.
constexpr int PREFILL_PIPES = mrl::MAXPIPE < 8 ? mrl::MAXPIPE : 8;
inline void build_prompt_cache(const mrl::Model& M, mrl::Prompts& P, const mrl::Hyper& hp) {
  check_trunk_hyper(hp);                                                        // before anything is allocated: megarl meets the trunk's limits here first
  if (P.n <= 0) return;
  if (!P.kv || !P.hlast) mrl::host::die("qw::build_prompt_cache: the prompt store has no cache buffers (upload_prompts with_cache = true)");
  const std::vector<int> len = mk::down(P.len, (size_t)P.n);
  for (int p = 0; p < P.n; p++) if (len[p] < 1 || len[p] > mrl::MAXPROMPT) mrl::host::die("qw::build_prompt_cache: prompt " + std::to_string(p) + " has length " + std::to_string(len[p]));
  const int np = std::min(PREFILL_PIPES, (P.n + mrl::MAXROWS - 1) / mrl::MAXROWS);   // pipelines in use
  std::vector<mrl::Pipe> pipe((size_t)np);
  for (int j = 0; j < np; j++) pipe[j] = mrl::host::alloc_pipe(mrl::MAXROWS);
  mrl::Ctx* hC = new mrl::Ctx; memset(hC, 0, sizeof(mrl::Ctx)); hC->model = M; hC->prompts = P; hC->hp = hp;
  mrl::Ctx* dC = mk::dalloc<mrl::Ctx>(1); MK_CK(cudaMemcpy(dC, hC, sizeof(mrl::Ctx), cudaMemcpyHostToDevice)); delete hC;
  mrl::Pipe* dP = mk::dalloc<mrl::Pipe>((size_t)np); MK_CK(cudaMemcpy(dP, pipe.data(), (size_t)np * sizeof(mrl::Pipe), cudaMemcpyHostToDevice));
  MK_CK(cudaFuncSetAttribute(k_phase, cudaFuncAttributeMaxDynamicSharedMemorySize, mrl::SMEM));
  mk::Builder b; add_trunk(b, -1, mrl::MAXROWS, hp, np, 1);                     // phase index order is dependency order; np pipelines in lockstep
  for (int p0 = 0; p0 < P.n; p0 += np * mrl::MAXROWS) {
    std::vector<int> nb((size_t)np); int maxlen = 0;
    for (int j = 0; j < np; j++) {                                              // pipeline j carries prompts p0 + 64 j ...; one beyond the store carries none
      nb[j] = std::max(0, std::min(mrl::MAXROWS, P.n - (p0 + j * mrl::MAXROWS)));
      for (int r = 0; r < nb[j]; r++) maxlen = std::max(maxlen, len[p0 + j * mrl::MAXROWS + r]);
      MK_CK(cudaMemcpy(pipe[j].nrows, &nb[j], sizeof(int), cudaMemcpyHostToDevice));
    }
    for (int pos = 0; pos < maxlen; pos++) {
      for (int j = 0; j < np; j++) k_prefill_rows<<<1, mrl::MAXROWS>>>(pipe[j].row, P.len, P.tok, p0 + j * mrl::MAXROWS, nb[j], pos);
      for (int i = 0; i < b.nph(); i++) {
        // Q_NORM1F[LMID] has just run: P.hres is the residual stream entering layer LMID.
        if (b.ph[i].a == mrl::LMID && (b.ph[i].kind == mrl::K_Q_QKV)) {
          for (int j = 0; j < np; j++) {
            bool ends = false; for (int r = 0; r < nb[j]; r++) ends = ends || len[p0 + j * mrl::MAXROWS + r] == pos + 1;
            if (ends) {
              k_prefill_save<<<nb[j], 256>>>(pipe[j].row, pipe[j].hres, P.hlast);
              k_prefill_trim_last<<<1, mrl::MAXROWS>>>(pipe[j].row, nb[j]);
            }
          }
        }
        k_phase<<<b.ntasks[i], 256, mrl::SMEM>>>(dC, dP, b.ph[i]);
      }
    }
    MK_CK(cudaGetLastError()); MK_CK(cudaDeviceSynchronize());
  }
  MK_CK(cudaFree(dC)); MK_CK(cudaFree(dP));
  for (int j = 0; j < np; j++) free_pipe(pipe[j]);
}

}  // namespace qw
