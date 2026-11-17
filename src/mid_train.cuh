// mid_train.cuh -- the trained layer LMID (namespace mt): its rollout GEMMs and the whole learner.
//
// Layer LMID (MRL_MID_LAYER, default 13) is trained in full: W_qkv [DQKV, KQ], W_o [D, D], W_gu [GUP, D] and
// W_down [D, FF], MID_NW = 14,946,304 bf16 values per snapshot. Column D of W_qkv (the q/k/v bias) and the padding
// columns after it keep their base values. The layer acts from the last prompt position on, so the prompt KV caches
// (positions 0..plen-2 of every layer) are valid for every policy version.
//
// Rollout: K_Q_MID_COHORT saves the residual stream entering layer LMID into C.store.hbase and groups the fed rows by
// the snapshot their group pinned; the K_Q_MID_* GEMMs then run layer LMID once per cohort with that snapshot.
// Learner: replays layers LMID..NL-1 from C.store.hbase with the rollout's tile geometry, split-K chunk counts,
// addition order, attention arithmetic and lm:: head functions, so at lag 0 (same weight bits) logp_new equals the
// generation-time logp bit for bit. It backpropagates through the LM head, the final norm and the frozen layers
// NL-1..LMID+1 into layer LMID, clips the global gradient norm and applies fp32-master AdamW to C.tr.W, which
// T_FINISH publishes into C.sh.ver.W. Every reduction runs in a fixed order, so an update is deterministic.
//
// Every task is run by all 256 threads of one block and uses only the dynamic shared memory `smem`; data passes from
// one phase to the next through the scheduler's dependency chain (mk_sched.cuh).
#pragma once
#include "mrl_common.cuh"
#include "mrl_lmhead.cuh"
#include "mrl_wgmma.cuh"
#include "qwen_fwd.cuh"
#include "head_sample.cuh"
#include "grpo_loss.cuh"


namespace mt {
using namespace mrl;

static_assert(LMID >= 1 && LMID < NL - 1, "MRL_MID_LAYER must be an interior layer 1 .. NL-2");
constexpr int NL_REM = NL - LMID;                 // layers the learner replays (11 for LMID = 13)
constexpr int HPK = NH / NKV;                     // 7 query heads per KV head
constexpr int AT_MAXN = MAXPROMPT + MAXGEN;       // 192, longest context
constexpr int AT_PITCH = HD + 4;                  // 68 floats: consecutive rows start 4 banks apart
constexpr int MID_MAX_COH = MAXGRP;               // a pipeline has at most one cohort per group slot
constexpr int NT_MID_ADAM = 128;                  // tasks of K_M_GNORM and K_M_ADAM, one slice of MID_NW each
static_assert(NT_MID_ADAM == 2 * NGNORM, "M_ADAM hands its partials to T_FINISH in pairs");

struct MidState {
  // rollout: the version cohorts of each pipeline's current step (written by K_Q_MID_COHORT)
  int* pipe_ncoh;                                 // [MAXPIPE] cohorts in use
  int* pipe_vslot;                                // [MAXPIPE * MID_MAX_COH] snapshot slot of cohort c (-1: unused)
  u64* pipe_rmask;                                // [MAXPIPE * MID_MAX_COH] fed rows of cohort c, bit r = row r

  // learner: optimizer state (C.tr.W is the bf16 working copy)
  float *W32, *mW, *vW;                           // [MID_NW] fp32 master weights, Adam first and second moments
  float* gsum;                                    // [MID_NW] weight gradient (sum of the split-K partials)
  float* gpart;                                   // [M_WG_KS][MID_NW] split-K partials of the weight gradient
  float* gnorm2_part;                             // [NT_MID_ADAM] squared gradient norm of each K_M_GNORM slice

  // learner: per-sequence metadata
  int* seq_prompt;                                // [TSEQ] prompt id of sequence s
  int* seq_plen;                                  // [TSEQ] prompt length plen (the prompt segment has lenA = plen - 1 positions)

  // learner forward, kept for the backward pass. Sequence rows (s * MAXGEN + t); li = l - LMID in 0..NL_REM-1
  float* h1;                                      // [NL_REM][TROWS, DP]   residual stream entering layer l
  float* rinv1;                                   // [NL_REM][TROWS]       1 / rms of h1
  bf16*  xn1_mid;                                 // [TROWS, KQ]           ln1 output of layer LMID (operand of dW_qkv)
  float* qkv;                                     // [NL_REM][TROWS, QKVP] q | k | v before RoPE, summed over the ksQ chunks
  bf16*  ao_mid;                                  // [TROWS, D]            attention output of layer LMID (operand of dW_o)
  float* h2;                                      // [NL_REM][TROWS, DP]   residual before ln2 (h1 + o_proj)
  float* rinv2;                                   // [NL_REM][TROWS]       1 / rms of h2
  bf16*  xn2_mid;                                 // [TROWS, KQ]           ln2 output of layer LMID (operand of dW_gu)
  float* gu;                                      // [NL_REM][TROWS, GUP]  gate | up before SwiGLU, summed over the ksG chunks
  bf16*  xm_mid;                                  // [TROWS, FF]           SwiGLU output of layer LMID (operand of dW_down)
  float* kv;                                      // [NL_REM][TROWS, KVW]  rotated keys and values of the sequence rows (positions plen-1 ..)

  // learner scratch, reused layer by layer in the forward and the backward pass
  bf16*  xn;                                      // [TROWS, KQ]           normalised GEMM input
  bf16*  ao;                                      // [TROWS, D]            attention output (layers > LMID)
  bf16*  xm;                                      // [TROWS, FF]           SwiGLU output (layers > LMID)
  float* part;                                    // [KS_MAX][TROWS, DP]   split-K partials: o_proj, down_proj, gate/up chunks 1.. (pitch
                                                  //   TROWS * GUP); backward dxm (fp32 [TROWS, FF]), dxn2, dxn1
  float* qkvp;                                    // [KS_MAX/4][TROWS, QKVP] split-K partials of q/k/v (ksQ > 1)
  float* dh;                                      // [TROWS, DP]           running dL/dh, from layer NL-1 down to LMID
  bf16*  dy_bf16;                                 // [TROWS, DP]           bf16(dh), operand of the backward GEMMs
  bf16*  dgu;                                     // [TROWS, GUP]          dL/d(gate | up)
  float* dao;                                     // [TROWS, D]            dL/d(ao)
  bf16*  dqkv;                                    // [TROWS, QKVP]         dL/d(q | k | v), before RoPE
  float* p_seq;                                   // [TSEQ][MAXGEN t][NKV][HPK][64 js] probabilities against the sequence keys
  float* ds_seq;                                  // [TSEQ][MAXGEN t][NKV][HPK][64 js] score gradients against the sequence keys
};

__device__ __forceinline__ const MidState* get(const Ctx& C) { return reinterpret_cast<const MidState*>(C.mid); }
__device__ __forceinline__ MidState* get_mut(const Ctx& C)   { return reinterpret_cast<MidState*>(C.mid); }

// ------------------------------------------------------------------------------------------------ GEMM wrappers
// One non-inlined function per tile instantiation, so each gets its own register allocation. The row-masked tile
// needs sm_90; any other compilation pass (the host pass included) gets an empty body.
__device__ __noinline__ void g_nt_f32(const lg::Args& a, int tm, int tn, char* smem)  { lg::gemm_tile<lg::NT, lg::EPI_F32>(a, tm, tn, smem); }
__device__ __noinline__ void g_nn_f32(const lg::Args& a, int tm, int tn, char* smem)  { lg::gemm_tile<lg::NN, lg::EPI_F32>(a, tm, tn, smem); }
__device__ __noinline__ void g_tn_f32(const lg::Args& a, int tm, int tn, char* smem)  { lg::gemm_tile<lg::TN, lg::EPI_F32>(a, tm, tn, smem); }
__device__ __noinline__ void g_nt_f32_masked(const lg::Args& a, int tn, u64 mask, char* smem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  wg::gemm_64x256_masked<lg::EPI_F32>(a, tn, mask, smem);
#else
  (void)a; (void)tn; (void)mask; (void)smem;
#endif
}

// ================================================================================================ ROLLOUT: LAYER LMID
// K_Q_MID_COHORT, one task per pipeline. All threads copy the residual row of every fed row into the store; thread 0
// then builds the pipeline's cohort table: the fed rows of GENERATING groups, grouped by the snapshot slot
// (GroupSlot.ver_slot) the group pinned at admission.
__device__ __noinline__ void roll_mid_cohort(const Ctx& C, const Pipe& P, int pipe_idx) {
  const MidState* M = get(C);
  const int tid = threadIdx.x, nrows = __ldg(P.nrows);
  // P.hres[r] is the residual stream entering layer LMID at the position fed now, which is sequence row gen of the
  // store (position plen - 1 + gen). On a group's first step (gen = 0) R_BEGIN copied it from Prompts.hlast; on later
  // steps the lower half of the previous step (layers 0..LMID-1 on the token sampled last, ending with
  // Q_DOWNR[LMID-1]) left it there.
  for (int r = 0; r < nrows; r++) {
    const Row* R = P.row + r;
    if (!mk::ld_vol(&R->feed)) continue;
    const int gen = mk::ld_vol(&R->gen), buf = mk::ld_vol(&R->buf), seq = mk::ld_vol(&R->seq);
    if ((unsigned)gen < (unsigned)MAXGEN && (unsigned)buf < (unsigned)C.store.nbuf) {
      float* dst = C.store.hbase + (((size_t)buf * GSZ + seq) * MAXGEN + gen) * D;
      const float* src = P.hres + (size_t)r * DP;
      for (int c = tid; c < D / 4; c += 256)
        *reinterpret_cast<F4*>(dst + 4 * c) = *reinterpret_cast<const F4*>(src + 4 * c);
    }
  }
  if (tid == 0) {
    int slots[MID_MAX_COH]; u64 masks[MID_MAX_COH]; int nc = 0;
    for (int g = 0; g < C.ngrp; g++) {
      const GroupSlot* G = P.grp + g;
      if (mk::ld_vol(&G->state) != 1) continue;
      u64 gm = 0ull;
      for (int j = 0; j < GSZ; j++) {
        const int r = GSZ * g + j;
        if (r < nrows && mk::ld_vol(&P.row[r].feed)) gm |= (1ull << r);
      }
      if (!gm) continue;
      const int vs = mk::ld_vol(&G->ver_slot);
      int found = -1;
      for (int c = 0; c < nc; c++) if (slots[c] == vs) { found = c; break; }
      if (found >= 0) masks[found] |= gm;
      else if (nc < MID_MAX_COH) { slots[nc] = vs; masks[nc] = gm; nc++; }
    }
    M->pipe_ncoh[pipe_idx] = nc;
    for (int c = 0; c < MID_MAX_COH; c++) {
      M->pipe_vslot[pipe_idx * MID_MAX_COH + c] = c < nc ? slots[c] : -1;
      M->pipe_rmask[pipe_idx * MID_MAX_COH + c] = c < nc ? masks[c] : 0ull;
    }
  }
}

// K_Q_MID_QKV, K_Q_MID_OPROJ, K_Q_MID_GU, K_Q_MID_DOWNP: the trunk's GEMM task (qw::gemm_task: same tiles, chunks and
// outputs) for layer LMID, run once per cohort with the cohort's row mask and the weights of its snapshot,
// C.sh.ver.W + vslot * MID_NW. Each row is thus multiplied by the version its group pinned. local = tn * ks + c.
__device__ __forceinline__ void roll_mid_gemm(const Ctx& C, const Pipe& P, int pipe_idx, int kind, int local, char* smem) {
  const MidState* M = get(C);
  const int ncoh = mk::ld_vol(M->pipe_ncoh + pipe_idx);
  if (ncoh <= 0) return;
  int ks, nk32; size_t woff;
  switch (kind) {
    case K_Q_MID_QKV:   ks = C.hp.ksQ; nk32 = KQ / 32; woff = MID_OFF_QKV; break;
    case K_Q_MID_OPROJ: ks = C.hp.ksO; nk32 = D / 32;  woff = MID_OFF_O;   break;
    case K_Q_MID_GU:    ks = C.hp.ksG; nk32 = D / 32;  woff = MID_OFF_GU;  break;
    default:            ks = C.hp.ksD; nk32 = FF / 32; woff = MID_OFF_DN;  break;
  }
  const int tn = local / ks, c = local % ks; int k0, kc; kchunk(nk32, ks, c, k0, kc);
  for (int coh = 0; coh < ncoh; coh++) {
    const int vs = M->pipe_vslot[pipe_idx * MID_MAX_COH + coh];
    const u64 rmask = M->pipe_rmask[pipe_idx * MID_MAX_COH + coh];
    if (!rmask || vs < 0) continue;
    const bf16* W = C.sh.ver.W + (size_t)vs * MID_NW + woff;
    lg::Args a = lg::args_default(); a.M = MAXROWS; a.K = kc;
    switch (kind) {
      case K_Q_MID_QKV:
        a.A = P.xn + k0; a.lda = KQ; a.B = W + k0; a.ldb = KQ; a.bvalid = DQKV;
        a.C = ks == 1 ? P.qkv : P.qkvp + (size_t)c * MAXROWS * QKVP; a.ldc = QKVP; a.cvalid = DQKV; a.N = QKVP; break;
      case K_Q_MID_OPROJ:
        a.A = P.ao + k0; a.lda = D; a.B = W + k0; a.ldb = D; a.bvalid = D;
        a.C = P.part + (size_t)c * MAXROWS * DP; a.ldc = DP; a.cvalid = D; a.N = DP; break;
      case K_Q_MID_GU:
        a.A = P.xn + k0; a.lda = KQ; a.B = W + k0; a.ldb = D;
        a.C = c == 0 ? P.gu : P.part + (size_t)(c - 1) * MAXROWS * GUP; a.ldc = GUP; a.N = GUP; break;
      default:
        a.A = P.xm + k0; a.lda = FF; a.B = W + k0; a.ldb = FF; a.bvalid = D;
        a.C = P.part + (size_t)c * MAXROWS * DP; a.ldc = DP; a.cvalid = D; a.N = DP; break;
    }
    g_nt_f32_masked(a, tn, rmask, smem);
  }
}

// ================================================================================================ LEARNER: FORWARD (LMID .. NL-1)
// sum_parts_rn: n split-K partials of 4 floats, `stride` apart, added in chunk order starting from chunk 0 with
// __fadd_rn (the trunk's order).
__device__ __forceinline__ F4 ldf4(const float* p) { return *reinterpret_cast<const F4*>(p); }
__device__ __forceinline__ void add4_ip(F4& a, const F4& b) {
#pragma unroll
  for (int i = 0; i < 4; i++) a.v[i] = __fadd_rn(a.v[i], b.v[i]);
}
__device__ __forceinline__ F4 sum_parts_rn(const float* p0, size_t stride, int n) {
  F4 s = ldf4(p0);
  for (int c = 1; c < n; c++) { F4 x = ldf4(p0 + (size_t)c * stride); add4_ip(s, x); }
  return s;
}

// K_M_GATHER, task s = 0..TSEQ-1. Copies sequence s from the store: hbase into h1[0], and the tokens, behaviour
// log-probabilities and sequence index into the packed rows off[s] .. off[s] + len - 1; records the prompt id and
// length. Also zeroes rows t >= len of xn1_mid, xn2_mid, ao_mid and xm_mid: the weight-gradient GEMMs reduce over all
// MAXGEN rows of every sequence, so those rows must be zero in both operands (dy_bf16, dgu and dqkv are zeroed by
// K_M_NORMB, K_M_BACT and K_M_BATTN_ROW).
__device__ __noinline__ void m_gather(const Ctx& C, int s) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int tid = threadIdx.x, g = s / GSZ, j = s % GSZ, buf = B->buf[g], len = B->len[s], off = B->off[s];
  const GroupRec* rec = C.store.rec + buf;
  if (tid == 0) {
    const int pr = rec->prompt;
    M->seq_prompt[s] = pr;
    M->seq_plen[s]   = C.prompts.len[pr];
  }
  const size_t sr = ((size_t)buf * GSZ + j) * MAXGEN;
  if (tid < len) {
    C.tr.y[off + tid]        = C.store.tokens[sr + tid];
    C.tr.logp_beh[off + tid] = C.store.logp[sr + tid];
    C.tr.rseq[off + tid]     = s;
  }
  // h1[0]: rows t >= len and columns D..DP-1 are written as zero
  float* h1_0 = M->h1 + (size_t)s * MAXGEN * DP;
  const float* sh = C.store.hbase + sr * D;
  for (int q = tid; q < MAXGEN * (DP / 4); q += 256) {
    const int t = q / (DP / 4), c4 = (q % (DP / 4)) * 4;
    F4 v; v.v[0] = v.v[1] = v.v[2] = v.v[3] = 0.f;
    if (t < len && c4 < D) v = ldf4(sh + (size_t)t * D + c4);
    *reinterpret_cast<F4*>(h1_0 + (size_t)t * DP + c4) = v;
  }
  V8 z;
#pragma unroll
  for (int i = 0; i < 8; i++) z.v[i] = __float2bfloat16(0.f);
  const int nz = MAXGEN - len;
  const size_t r0 = (size_t)s * MAXGEN + len;
  for (int q = tid; q < nz * (KQ / 8); q += 256) {
    size_t idx = (r0 + q / (KQ / 8)) * KQ + (q % (KQ / 8)) * 8;
    *reinterpret_cast<V8*>(M->xn1_mid + idx) = z;
    *reinterpret_cast<V8*>(M->xn2_mid + idx) = z;
  }
  for (int q = tid; q < nz * (D / 8); q += 256)
    *reinterpret_cast<V8*>(M->ao_mid + (r0 + q / (D / 8)) * D + (q % (D / 8)) * 8) = z;
  for (int q = tid; q < nz * (FF / 8); q += 256)
    *reinterpret_cast<V8*>(M->xm_mid + (r0 + q / (FF / 8)) * FF + (q % (FF / 8)) * 8) = z;
}

// K_M_FNORM1 / K_M_FNORM2 of layer l (PhaseDesc.a): one warp per sequence row, srow = 8 * local + warp, rows t < len.
// Forms the residual as qw::row_task does (FNORM1: h1 = h2 of layer l-1 plus its down_proj partials, or hbase from
// K_M_GATHER at l = LMID; FNORM2: h2 = h1 plus the o_proj partials) and normalises it into xn with
// lm::warp_rmsnorm_row (FNORM1 also sets the bias column xn[:, D] = 1). At l = LMID, xn is kept in xn1_mid / xn2_mid.
__device__ __noinline__ void m_fnorm(const Ctx& C, int kind, int l, int local) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int srow = 8 * local + warp;
  if (srow >= TROWS) return;
  const int s = srow / MAXGEN, t = srow % MAXGEN, len = B->len[s], li = l - LMID;
  if (t >= len) return;

  const bool first = (kind == K_M_FNORM1);
  float* h_dst = (first ? M->h1 : M->h2) + ((size_t)li * TROWS + srow) * DP;
  if (!first) {
    // h2[li] = h1[li] + sum_{c < ksO} part[c]
    const float* h_prev = M->h1 + ((size_t)li * TROWS + srow) * DP;
    for (int c4 = lane * 4; c4 < D; c4 += 128) {
      F4 x = ldf4(h_prev + c4);
      F4 p = sum_parts_rn(M->part + (size_t)srow * DP + c4, (size_t)TROWS * DP, C.hp.ksO);
      add4_ip(x, p);
      *reinterpret_cast<F4*>(h_dst + c4) = x;
    }
  } else if (li > 0) {
    // h1[li] = h2[li - 1] + sum_{c < ksD} part[c]   (layer l-1's down_proj)
    const float* h_prev = M->h2 + ((size_t)(li - 1) * TROWS + srow) * DP;
    for (int c4 = lane * 4; c4 < D; c4 += 128) {
      F4 x = ldf4(h_prev + c4);
      F4 p = sum_parts_rn(M->part + (size_t)srow * DP + c4, (size_t)TROWS * DP, C.hp.ksD);
      add4_ip(x, p);
      *reinterpret_cast<F4*>(h_dst + c4) = x;
    }
  }
  __syncwarp();

  const float* w = first ? C.model.ln1[l] : C.model.ln2[l];
  bf16* xn_out = M->xn + (size_t)srow * KQ;
  const float rinv = lm::warp_rmsnorm_row(h_dst, w, xn_out, first);
  if (lane == 0) {
    if (first) M->rinv1[(size_t)li * TROWS + srow] = rinv;
    else       M->rinv2[(size_t)li * TROWS + srow] = rinv;
  }
  if (li == 0) {
    bf16* save = (first ? M->xn1_mid : M->xn2_mid) + (size_t)srow * KQ;
    for (int c8 = lane * 8; c8 < KQ; c8 += 256)
      *reinterpret_cast<V8*>(save + c8) = *reinterpret_cast<const V8*>(xn_out + c8);
  }
}

// K_M_FQKV, K_M_FOPROJ, K_M_FGU, K_M_FDOWNP of layer l: the GEMMs of qw::gemm_task with the same tiles and chunk
// counts, one 64-row tile per sequence (layer LT's down_proj uses ksY, as H_Y0P does). A tile's result for a row does
// not depend on the other rows, so the rollout's pipeline tiles and these sequence tiles agree. Layer LMID reads
// C.tr.W, the other layers the frozen model. local = s * (ntiles * ks) + tn * ks + c.
__device__ __forceinline__ void m_fgemm(const Ctx& C, int kind, int l, int local, char* smem) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int li = l - LMID;
  int ntn, ks, nk32;
  switch (kind) {
    case K_M_FQKV:   ntn = NT_QKV; ks = C.hp.ksQ; nk32 = KQ / 32; break;
    case K_M_FOPROJ: ntn = NT_D;   ks = C.hp.ksO; nk32 = D / 32;  break;
    case K_M_FGU:    ntn = NT_GU;  ks = C.hp.ksG; nk32 = D / 32;  break;
    default:         ntn = NT_D;   ks = (l == LT ? C.hp.ksY : C.hp.ksD); nk32 = FF / 32; break;
  }
  const int per_s = ntn * ks, s = local / per_s, rem = local % per_s, tn = rem / ks, c = rem % ks;
  const int len = B->len[s];
  int k0, kc; kchunk(nk32, ks, c, k0, kc);
  const size_t r0 = (size_t)s * MAXGEN;
  const bool is_mid = (l == LMID);
  const bf16* W_mid = C.tr.W;

  lg::Args a = lg::args_default();
  a.M = TM; a.K = kc; a.avalid = len; a.rvalid = len;
  switch (kind) {
    case K_M_FQKV:
      a.A = M->xn + r0 * KQ + k0; a.lda = KQ;
      a.B = (is_mid ? W_mid + MID_OFF_QKV : C.model.qkv[l]) + k0; a.ldb = KQ; a.bvalid = DQKV;
      a.C = (ks == 1 ? M->qkv + ((size_t)li * TROWS + r0) * QKVP : M->qkvp + ((size_t)c * TROWS + r0) * QKVP);
      a.ldc = QKVP; a.cvalid = DQKV; a.N = QKVP; break;
    case K_M_FOPROJ:
      a.A = (is_mid ? M->ao_mid : M->ao) + r0 * D + k0; a.lda = D;
      a.B = (is_mid ? W_mid + MID_OFF_O : C.model.o[l]) + k0; a.ldb = D; a.bvalid = D;
      a.C = M->part + ((size_t)c * TROWS + r0) * DP; a.ldc = DP; a.cvalid = D; a.N = DP; break;
    case K_M_FGU:
      a.A = M->xn + r0 * KQ + k0; a.lda = KQ;
      a.B = (is_mid ? W_mid + MID_OFF_GU : C.model.gu[l]) + k0; a.ldb = D;
      a.C = (c == 0 ? M->gu + ((size_t)li * TROWS + r0) * GUP : reinterpret_cast<float*>(M->part) + ((size_t)(c - 1) * TROWS + r0) * GUP);
      a.ldc = GUP; a.N = GUP; break;
    default:
      a.A = (is_mid ? M->xm_mid : M->xm) + r0 * FF + k0; a.lda = FF;
      a.B = (is_mid ? W_mid + MID_OFF_DN : C.model.down[l]) + k0; a.ldb = FF; a.bvalid = D;
      a.C = M->part + ((size_t)c * TROWS + r0) * DP; a.ldc = DP; a.cvalid = D; a.N = DP; break;
  }
  g_nt_f32(a, 0, tn, smem);
}

// K_M_FATTN of layer l: task (srow, kv head g), local = 2 * srow + g, srow = s * MAXGEN + t. Causal GQA attention with
// the arithmetic of qw::attn_task; the steps are numbered as there. The context is the prompt segment
// prompts.kv[prompt][l][0, lenA), lenA = plen - 1, then sequence rows 0..t (positions plen-1 .. plen-1+t).
// The rollout produced the rows one step at a time, each reading the cache entries of earlier steps. Here all rows of
// a sequence run in the same phase, so a task must not read M->kv rows that another task is still writing. Instead it
// recomputes the rotated keys and the values of rows 0..t from q/k/v (same operations, same bits) and writes only row
// t's key and value into M->kv[li], for the backward pass.
__device__ __noinline__ void m_fattn(const Ctx& C, int l, int local, char* smem) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int srow = local >> 1, g = local & 1;
  const int s = srow / MAXGEN, t = srow % MAXGEN, len = B->len[s], li = l - LMID;
  if (t >= len) return;

  const int prompt = M->seq_prompt[s], plen = M->seq_plen[s], lenA = plen - 1;
  const int pos = lenA + t, n = pos + 1;
  const int ksQ = C.hp.ksQ;
  const float* qkv_seq = M->qkv + ((size_t)li * TROWS + (size_t)s * MAXGEN) * QKVP;

  // row t's q (warps 0..6: query head HPK * g + warp) or k and v (warp HPK), summed over the ksQ chunks in chunk order
  const size_t col_t = (size_t)t * QKVP + (warp < HPK ? (HPK * g + warp) * HD : D + g * HD) + lane;
  float x[4] = {0.f, 0.f, 0.f, 0.f};
  if (ksQ == 1) {
    const float* src = qkv_seq + col_t;
    x[0] = src[0]; x[1] = src[32];
    if (warp == HPK) { x[2] = src[NKV * HD]; x[3] = src[NKV * HD + 32]; }
  } else {
    for (int cb = 0; cb < ksQ; cb++) {
      const float* src = M->qkvp + ((size_t)cb * TROWS + (size_t)s * MAXGEN) * QKVP + col_t;
      x[0] = cb == 0 ? src[0] : __fadd_rn(x[0], src[0]);
      x[1] = cb == 0 ? src[32] : __fadd_rn(x[1], src[32]);
      if (warp == HPK) {
        x[2] = cb == 0 ? src[NKV * HD] : __fadd_rn(x[2], src[NKV * HD]);
        x[3] = cb == 0 ? src[NKV * HD + 32] : __fadd_rn(x[3], src[NKV * HD + 32]);
      }
    }
    // the backward pass reads the summed q/k/v from M->qkv[li]
    float* qdst = M->qkv + ((size_t)li * TROWS + (size_t)s * MAXGEN) * QKVP + col_t;
    qdst[0] = x[0]; qdst[32] = x[1];
    if (warp == HPK) { qdst[NKV * HD] = x[2]; qdst[NKV * HD + 32] = x[3]; }
  }

  const float* segA = C.prompts.kv + ((size_t)prompt * NL + l) * (MAXPROMPT * KVW);
  float* S = reinterpret_cast<float*>(smem);
  constexpr int O_KV = 0;                         // [AT_MAXN][AT_PITCH]
  constexpr int O_Q  = O_KV + AT_MAXN * AT_PITCH; // [HPK][HD]
  constexpr int O_P  = O_Q  + HPK * HD;           // [HPK][256]
  constexpr int O_O  = O_P  + HPK * 256;          // [4][HPK][HD]
  constexpr int O_R  = O_O  + 4 * HPK * HD;       // [2][8][8]

  // 1a. prompt keys j < lenA -> shared memory (cp.async)
  for (int q = tid; q < lenA * 16; q += 256) {
    const int j = q >> 4, c = q & 15;
    lg::cp_async16(lg::smem_u32(S + O_KV + j * AT_PITCH + 4 * c), segA + (size_t)j * KVW + g * HD + 4 * c);
  }
  lg::cp_commit();

  // 1b. rotated keys of sequence rows js = 0..t, recomputed from q/k/v
  for (int idx = tid; idx < (t + 1) * 32; idx += 256) {
    const int js = idx >> 5, pair = idx & 31, jpos = lenA + js;
    float k0 = 0.f, k1 = 0.f;
    if (ksQ == 1) {
      const float* ks_ptr = qkv_seq + (size_t)js * QKVP + D + g * HD + pair;
      k0 = ks_ptr[0]; k1 = ks_ptr[32];
    } else {
      for (int cb = 0; cb < ksQ; cb++) {
        const float* ks_ptr = M->qkvp + ((size_t)cb * TROWS + (size_t)s * MAXGEN + js) * QKVP + D + g * HD + pair;
        k0 = cb == 0 ? ks_ptr[0] : __fadd_rn(k0, ks_ptr[0]);
        k1 = cb == 0 ? ks_ptr[32] : __fadd_rn(k1, ks_ptr[32]);
      }
    }
    const float cs = __ldg(C.model.rope_cos + (size_t)jpos * 32 + pair);
    const float sn = __ldg(C.model.rope_sin + (size_t)jpos * 32 + pair);
    const float rl = __fadd_rn(__fmul_rn(k0, cs), -__fmul_rn(k1, sn));
    const float rh = __fadd_rn(__fmul_rn(k1, cs),  __fmul_rn(k0, sn));
    float* kj = S + O_KV + (lenA + js) * AT_PITCH;
    kj[pair] = rl; kj[pair + 32] = rh;
  }

  // 2. RoPE of row t's queries; warp HPK writes row t's rotated key and its value to M->kv[li]
  {
    const float cs = __ldg(C.model.rope_cos + (size_t)pos * 32 + lane);
    const float sn = __ldg(C.model.rope_sin + (size_t)pos * 32 + lane);
    const float rl = __fadd_rn(__fmul_rn(x[0], cs), -__fmul_rn(x[1], sn));
    const float rh = __fadd_rn(__fmul_rn(x[1], cs),  __fmul_rn(x[0], sn));
    if (warp < HPK) {
      S[O_Q + warp * HD + lane]      = rl;
      S[O_Q + warp * HD + 32 + lane] = rh;
    } else {
      float* kv_wr = M->kv + ((size_t)li * TROWS + srow) * KVW;
      kv_wr[g * HD + lane] = rl; kv_wr[g * HD + 32 + lane] = rh;
      kv_wr[NKV * HD + g * HD + lane] = x[2]; kv_wr[NKV * HD + g * HD + 32 + lane] = x[3];
    }
  }
  lg::cp_wait<0>();
  __syncthreads();

  // 3. scores of key tid for the 7 query heads
  float sc[HPK];
  if (tid < n) {
    const float* kr = S + O_KV + tid * AT_PITCH; float acc[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) acc[a] = 0.f;
    for (int c = 0; c < HD / 4; c++) {
      const F4 kk = *reinterpret_cast<const F4*>(kr + 4 * c);
#pragma unroll
      for (int a = 0; a < HPK; a++) {
        const F4 qq = *reinterpret_cast<const F4*>(S + O_Q + a * HD + 4 * c);
#pragma unroll
        for (int i = 0; i < 4; i++) acc[a] = fmaf(kk.v[i], qq.v[i], acc[a]);
      }
    }
#pragma unroll
    for (int a = 0; a < HPK; a++) sc[a] = __fmul_rn(0.125f, acc[a]);
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
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int a = 0; a < HPK; a++) m[a] = fmaxf(m[a], __shfl_xor_sync(0xffffffffu, m[a], o));
    if (lane == 0)
#pragma unroll
      for (int a = 0; a < HPK; a++) S[O_R + warp * 8 + a] = m[a];
  }
  __syncthreads();

  // 5. the values replace the keys: prompt values by cp.async, sequence values from q/k/v
  for (int q = tid; q < lenA * 16; q += 256) {
    const int j = q >> 4, c = q & 15;
    lg::cp_async16(lg::smem_u32(S + O_KV + j * AT_PITCH + 4 * c), segA + (size_t)j * KVW + NKV * HD + g * HD + 4 * c);
  }
  lg::cp_commit();
  for (int idx = tid; idx < (t + 1) * 64; idx += 256) {
    const int js = idx >> 6, d = idx & 63;
    float vv = 0.f;
    if (ksQ == 1) vv = qkv_seq[(size_t)js * QKVP + D + NKV * HD + g * HD + d];
    else {
      for (int cb = 0; cb < ksQ; cb++) {
        float u = M->qkvp[((size_t)cb * TROWS + (size_t)s * MAXGEN + js) * QKVP + D + NKV * HD + g * HD + d];
        vv = cb == 0 ? u : __fadd_rn(vv, u);
      }
    }
    S[O_KV + (lenA + js) * AT_PITCH + d] = vv;
  }

  // 4. softmax, part 2: exponentials, their sum, the probabilities
  float e[HPK];
  {
    float sm[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) {
      float mm = S[O_R + a];
#pragma unroll
      for (int w = 1; w < 8; w++) mm = fmaxf(mm, S[O_R + w * 8 + a]);
      e[a] = tid < n ? expf(__fadd_rn(sc[a], -mm)) : 0.f;
      sm[a] = e[a];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int a = 0; a < HPK; a++) sm[a] = __fadd_rn(sm[a], __shfl_xor_sync(0xffffffffu, sm[a], o));
    if (lane == 0)
#pragma unroll
      for (int a = 0; a < HPK; a++) S[O_R + 64 + warp * 8 + a] = sm[a];
  }
  __syncthreads();
#pragma unroll
  for (int a = 0; a < HPK; a++) {
    float tot = S[O_R + 64 + a];
#pragma unroll
    for (int w = 1; w < 8; w++) tot = __fadd_rn(tot, S[O_R + 64 + w * 8 + a]);
    if (tid < n) S[O_P + a * 256 + tid] = __fdiv_rn(e[a], tot);
  }
  lg::cp_wait<0>();
  __syncthreads();

  // 6. weighted values: thread (jg, d) folds keys jg, jg + 4, ... for column d of all 7 heads
  {
    const int jg = tid >> 6, d = tid & 63; float acc[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) acc[a] = 0.f;
    for (int j = jg; j < n; j += 4) {
      const float v = S[O_KV + j * AT_PITCH + d];
#pragma unroll
      for (int a = 0; a < HPK; a++) acc[a] = fmaf(S[O_P + a * 256 + j], v, acc[a]);
    }
#pragma unroll
    for (int a = 0; a < HPK; a++) S[O_O + (jg * HPK + a) * HD + d] = acc[a];
  }
  __syncthreads();

  // 7. add the four chains in order and round to bf16; layer LMID's output is kept in ao_mid
  if (tid < HPK * 8) {
    const int a = tid >> 3, d0 = (tid & 7) * 8; V8 out;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float* p = S + O_O + a * HD + d0 + i;
      float sv = p[0];
#pragma unroll
      for (int k = 1; k < 4; k++) sv = __fadd_rn(sv, p[k * HPK * HD]);
      out.v[i] = __float2bfloat16(sv);
    }
    bf16* dst = (l == LMID ? M->ao_mid : M->ao) + (size_t)srow * D + (HPK * g + a) * HD + d0;
    *reinterpret_cast<V8*>(dst) = out;
  }
}

// K_M_FACT of layer l: xm = bf16(silu(gate) * up) for one column slice of sequence s, rows t < len,
// local = s * (NT_FF * sp) + cl with sp = qw::act_split(ksG). The gate/up chunks are added in chunk order as in Q_ACT,
// and with ksG > 1 the sums are written back to M->gu[li] for the SwiGLU backward. Layer LMID's output goes to xm_mid.
__device__ __noinline__ void m_fact(const Ctx& C, int l, int local) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int ksG = C.hp.ksG, sp = (ksG <= 1 ? 2 : 4), per_s = NT_FF * sp;
  const int s = local / per_s, cl = local % per_s, W = TN / sp, c0 = cl * W;
  const int len = B->len[s], li = l - LMID;
  const size_t r0 = (size_t)s * MAXGEN;
  bf16* xm_dst = (l == LMID ? M->xm_mid : M->xm) + r0 * FF;
  const float* gu0 = M->gu + ((size_t)li * TROWS + r0) * GUP;

  for (int q = threadIdx.x; q < len * (W / 8); q += 256) {
    const int t = q / (W / 8), col = c0 + (q % (W / 8)) * 8;
    F4 g0 = ldf4(gu0 + (size_t)t * GUP + col), g1 = ldf4(gu0 + (size_t)t * GUP + col + 4);
    F4 u0 = ldf4(gu0 + (size_t)t * GUP + FF + col), u1 = ldf4(gu0 + (size_t)t * GUP + FF + col + 4);
    for (int cb = 1; cb < ksG; cb++) {
      const float* p = reinterpret_cast<const float*>(M->part) + ((size_t)(cb - 1) * TROWS + r0 + t) * GUP + col;
      add4_ip(g0, ldf4(p)); add4_ip(g1, ldf4(p + 4));
      add4_ip(u0, ldf4(p + FF)); add4_ip(u1, ldf4(p + FF + 4));
    }
    if (ksG > 1) {
      float* gd = M->gu + ((size_t)li * TROWS + r0 + t) * GUP + col;
      *reinterpret_cast<F4*>(gd) = g0; *reinterpret_cast<F4*>(gd + 4) = g1;
      *reinterpret_cast<F4*>(gd + FF) = u0; *reinterpret_cast<F4*>(gd + FF + 4) = u1;
    }
    V8 out;
#pragma unroll
    for (int i = 0; i < 4; i++) out.v[i]     = __float2bfloat16(__fmul_rn(silu(g0.v[i]), u0.v[i]));
#pragma unroll
    for (int i = 0; i < 4; i++) out.v[4 + i] = __float2bfloat16(__fmul_rn(silu(g1.v[i]), u1.v[i]));
    *reinterpret_cast<V8*>(xm_dst + (size_t)t * FF + col) = out;
  }
}

// K_M_NORMF: one warp per packed row r = 8 * local + warp, r < ntok. hfin = h2 of layer NL-1 + (part[0] + ... +
// part[ksY-1]), layer NL-1's down_proj partials summed first and added last, as H_Y0R does; then the final norm into
// C.tr.xf[r], as H_NORMF does. hfin and its 1 / rms are kept per sequence row for K_M_NORMB.
__device__ __noinline__ void m_normf(const Ctx& C, int local) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = 8 * local + warp;
  if (r >= B->ntok) return;
  const int s = C.tr.rseq[r]; const size_t srow = (size_t)s * MAXGEN + (r - B->off[s]);
  const float* h2_last = M->h2 + ((size_t)(NL_REM - 1) * TROWS + srow) * DP;
  float* hfin = C.tr.hfin + srow * DP;
  for (int c4 = lane * 4; c4 < D; c4 += 128) {
    F4 h = ldf4(h2_last + c4);
    F4 p = sum_parts_rn(M->part + srow * DP + c4, (size_t)TROWS * DP, C.hp.ksY);
    add4_ip(h, p);
    *reinterpret_cast<F4*>(hfin + c4) = h;
  }
  __syncwarp();
  const float rinv = lm::warp_rmsnorm_row(hfin, C.model.lnf, C.tr.xf + (size_t)r * D, false);
  if (lane == 0) C.tr.rinv[srow] = rinv;
}

// K_M_NORMB: final RMSNorm backward, one warp per sequence row srow = 8 * local + warp. Starts the running gradient:
// dh = dL/dhfin from C.tr.dxf of the row's packed index, and dy_bf16 = bf16(dh). Rows t >= len and columns D..DP-1
// are written as zero.
__device__ __noinline__ void m_normb_init(const Ctx& C, int local) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, srow = 8 * local + warp;
  if (srow >= TROWS) return;
  const int s = srow / MAXGEN, t = srow % MAXGEN, len = B->len[s];
  if (t >= len) {
    for (int c = lane; c < DP; c += 32) {
      M->dh[srow * DP + c] = 0.f;
      M->dy_bf16[srow * DP + c] = __float2bfloat16(0.f);
    }
    return;
  }
  const int r = B->off[s] + t;
  constexpr int PER = D / 32; // 28
  const float* x = C.tr.hfin + (size_t)srow * DP + lane * PER;
  const float* g = C.tr.dxf  + (size_t)r * DP + lane * PER;
  const float* w = C.model.lnf + lane * PER;
  const float rinv = C.tr.rinv[srow];
  F4 xx[PER / 4], uu[PER / 4]; float dot = 0.f;
#pragma unroll
  for (int i = 0; i < PER / 4; i++) {
    xx[i] = ldf4(x + 4 * i); const F4 gg = ldf4(g + 4 * i), ww = ldf4(w + 4 * i);
#pragma unroll
    for (int j = 0; j < 4; j++) { uu[i].v[j] = __fmul_rn(ww.v[j], gg.v[j]); dot = __fadd_rn(dot, __fmul_rn(uu[i].v[j], xx[i].v[j])); }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, o));
  const float proj = __fmul_rn(__fmul_rn(__fmul_rn(rinv, rinv), rinv), __fmul_rn(dot, 1.f / D));
  float* dh_dst = M->dh + (size_t)srow * DP + lane * PER;
  bf16*  dy_dst = M->dy_bf16 + (size_t)srow * DP + lane * PER;
#pragma unroll
  for (int i = 0; i < PER / 4; i++) {
    F4 f; V4 o;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      f.v[j] = __fadd_rn(__fmul_rn(rinv, uu[i].v[j]), -__fmul_rn(xx[i].v[j], proj));
      o.v[j] = __float2bfloat16(f.v[j]);
    }
    *reinterpret_cast<F4*>(dh_dst + 4 * i) = f;
    *reinterpret_cast<V4*>(dy_dst + 4 * i) = o;
  }
  if (lane == 0) {
    for (int c = D; c < DP; c++) { M->dh[(size_t)srow * DP + c] = 0.f; M->dy_bf16[(size_t)srow * DP + c] = __float2bfloat16(0.f); }
  }
}

// ================================================================================================ LEARNER: LAYER BACKWARD (NL-1 -> LMID)
// K_M_BDOWN of layer l: dxm = dy_bf16 * W_down (NN, [TM, D] x [D, FF]), one N tile of sequence s per task,
// local = s * NT_FF + tn. K = D is not split; the fp32 result goes to M->part, read as [TROWS, FF].
__device__ __forceinline__ void m_bdown(const Ctx& C, int l, int local, char* smem) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int s = local / NT_FF, tn = local % NT_FF, len = B->len[s];
  const size_t r0 = (size_t)s * MAXGEN;
  const bf16* W = (l == LMID) ? (C.tr.W + MID_OFF_DN) : C.model.down[l]; // [D, FF]
  lg::Args a = lg::args_default();
  a.A = M->dy_bf16 + r0 * DP; a.lda = DP; a.avalid = len;
  a.B = W; a.ldb = FF; a.bvalid = FF;
  a.C = reinterpret_cast<float*>(M->part) + r0 * FF; a.ldc = FF; a.rvalid = len; a.cvalid = FF;
  a.M = TM; a.N = FF; a.K = D;
  g_nn_f32(a, 0, tn, smem);
}

// K_M_BACT of layer l: SwiGLU backward for a 128-column slice of sequence s, local = s * (NT_FF * 2) + cl.
// Forward was xm = bf16(silu(g) * u), silu(g) = g * sig(g), d(silu)/dg = sig(g) * (1 + g * (1 - sig(g))).
// Reads dxm (fp32, M->part, from K_M_BDOWN) and the saved M->gu[li]; writes bf16 dgu, zero in rows t >= len.
__device__ __noinline__ void m_bact(const Ctx& C, int l, int local) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  constexpr int SP = 2, W = TN / SP, PER_S = NT_FF * SP;
  const int s = local / PER_S, cl = local % PER_S, c0 = cl * W, len = B->len[s], li = l - LMID;
  const size_t r0 = (size_t)s * MAXGEN;
  const float* dxm_src = reinterpret_cast<const float*>(M->part) + r0 * FF;
  const float* gu_src  = M->gu + ((size_t)li * TROWS + r0) * GUP;
  bf16* dgu_dst = M->dgu + r0 * GUP;

  for (int q = threadIdx.x; q < MAXGEN * (W / 8); q += 256) {
    const int t = q / (W / 8), col = c0 + (q % (W / 8)) * 8;
    V8 dg_out, du_out;
    if (t < len) {
      F4 dx0 = ldf4(dxm_src + (size_t)t * FF + col),     dx1 = ldf4(dxm_src + (size_t)t * FF + col + 4);
      F4 g0  = ldf4(gu_src  + (size_t)t * GUP + col),    g1  = ldf4(gu_src  + (size_t)t * GUP + col + 4);
      F4 u0  = ldf4(gu_src  + (size_t)t * GUP + FF + col), u1 = ldf4(gu_src + (size_t)t * GUP + FF + col + 4);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float gv = g0.v[i], uv = u0.v[i], dx = dx0.v[i];
        const float sig = 1.f / (1.f + __expf(-gv));
        const float sl  = gv * sig;
        const float dsl = sig * (1.f + gv * (1.f - sig));
        dg_out.v[i] = __float2bfloat16(dx * uv * dsl);
        du_out.v[i] = __float2bfloat16(dx * sl);
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float gv = g1.v[i], uv = u1.v[i], dx = dx1.v[i];
        const float sig = 1.f / (1.f + __expf(-gv));
        const float sl  = gv * sig;
        const float dsl = sig * (1.f + gv * (1.f - sig));
        dg_out.v[4 + i] = __float2bfloat16(dx * uv * dsl);
        du_out.v[4 + i] = __float2bfloat16(dx * sl);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) { dg_out.v[i] = __float2bfloat16(0.f); du_out.v[i] = __float2bfloat16(0.f); }
    }
    *reinterpret_cast<V8*>(dgu_dst + (size_t)t * GUP + col)      = dg_out;
    *reinterpret_cast<V8*>(dgu_dst + (size_t)t * GUP + FF + col) = du_out;
  }
}

// K_M_BGU of layer l: part[c] = dgu[:, chunk c] * W_gu[chunk c, :] (NN, [TM, GUP] x [GUP, D]), the BGU_KS = 8 split-K
// partials (K = 1216 each) of dxn2. local = (s * NT_D + tn) * BGU_KS + c.
constexpr int BGU_KS = 8;
__device__ __forceinline__ void m_bgu(const Ctx& C, int l, int local, char* smem) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int c = local % BGU_KS, stn = local / BGU_KS, tn = stn % NT_D, s = stn / NT_D;
  const int len = B->len[s];
  int k0, kc; kchunk(GUP / 32, BGU_KS, c, k0, kc);
  const size_t r0 = (size_t)s * MAXGEN;
  const bf16* W = (l == LMID) ? (C.tr.W + MID_OFF_GU) : C.model.gu[l]; // [GUP, D]
  lg::Args a = lg::args_default();
  a.A = M->dgu + r0 * GUP + k0; a.lda = GUP; a.avalid = len;
  a.B = W + (size_t)k0 * D; a.ldb = D; a.bvalid = D;
  a.C = M->part + ((size_t)c * TROWS + r0) * DP; a.ldc = DP; a.rvalid = len; a.cvalid = D;
  a.M = TM; a.N = DP; a.K = kc;
  g_nn_f32(a, 0, tn, smem);
}

// K_M_BNORM2 / K_M_BNORM1 of layer l: one warp per sequence row, rows t < len. Sums the partials of dxn (BNORM2: the
// BGU_KS chunks of K_M_BGU; BNORM1: chunk 0 from K_M_BQKV), applies the RMSNorm backward of ln2 / ln1, adds the result
// to dh and refreshes dy_bf16 = bf16(dh). Nothing below layer LMID is trained, so layer LMID has no BQKV / BNORM1.
__device__ __noinline__ void m_bnorm(const Ctx& C, int kind, int l, int local) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, srow = 8 * local + warp;
  if (srow >= TROWS) return;
  const int s = srow / MAXGEN, t = srow % MAXGEN, len = B->len[s], li = l - LMID;
  if (t >= len) return;

  const bool is_n2 = (kind == K_M_BNORM2);
  const int ks = is_n2 ? BGU_KS : 1;
  constexpr int PER = D / 32; // 28
  const float* x = (is_n2 ? M->h2 : M->h1) + ((size_t)li * TROWS + srow) * DP + lane * PER;
  const float* w = (is_n2 ? C.model.ln2[l] : C.model.ln1[l]) + lane * PER;
  const float rinv = (is_n2 ? M->rinv2 : M->rinv1)[(size_t)li * TROWS + srow];

  F4 xx[PER / 4], uu[PER / 4]; float dot = 0.f;
#pragma unroll
  for (int i = 0; i < PER / 4; i++) {
    xx[i] = ldf4(x + 4 * i);
    const F4 gg = sum_parts_rn(M->part + (size_t)srow * DP + lane * PER + 4 * i, (size_t)TROWS * DP, ks);
    const F4 ww = ldf4(w + 4 * i);
#pragma unroll
    for (int j = 0; j < 4; j++) { uu[i].v[j] = __fmul_rn(ww.v[j], gg.v[j]); dot = __fadd_rn(dot, __fmul_rn(uu[i].v[j], xx[i].v[j])); }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, o));
  const float proj = __fmul_rn(__fmul_rn(__fmul_rn(rinv, rinv), rinv), __fmul_rn(dot, 1.f / D));

  float* dh_dst = M->dh + (size_t)srow * DP + lane * PER;
  bf16*  dy_dst = M->dy_bf16 + (size_t)srow * DP + lane * PER;
#pragma unroll
  for (int i = 0; i < PER / 4; i++) {
    F4 cur = ldf4(dh_dst + 4 * i); V4 o;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float dx = __fadd_rn(__fmul_rn(rinv, uu[i].v[j]), -__fmul_rn(xx[i].v[j], proj));
      cur.v[j] = __fadd_rn(cur.v[j], dx);
      o.v[j] = __float2bfloat16(cur.v[j]);
    }
    *reinterpret_cast<F4*>(dh_dst + 4 * i) = cur;
    *reinterpret_cast<V4*>(dy_dst + 4 * i) = o;
  }
}

// K_M_BOPROJ of layer l: dao = dy_bf16 * W_o (NN, [TM, D] x [D, D]) in fp32, local = s * NT_D + tn.
__device__ __forceinline__ void m_boproj(const Ctx& C, int l, int local, char* smem) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int s = local / NT_D, tn = local % NT_D, len = B->len[s];
  const size_t r0 = (size_t)s * MAXGEN;
  const bf16* W = (l == LMID) ? (C.tr.W + MID_OFF_O) : C.model.o[l]; // [D, D]
  lg::Args a = lg::args_default();
  a.A = M->dy_bf16 + r0 * DP; a.lda = DP; a.avalid = len;
  a.B = W; a.ldb = D; a.bvalid = D;
  a.C = M->dao + r0 * D; a.ldc = D; a.rvalid = len; a.cvalid = D;
  a.M = TM; a.N = D; a.K = D;
  g_nn_f32(a, 0, tn, smem);
}

// K_M_BATTN_ROW of layer l: attention backward for one query row, task (srow, kv head g) as in K_M_FATTN.
// Recomputes the scores and probabilities P, forms dP[j] = dO . V[j] and dS = P (dP - sum_j P dP) / 8, writes the
// un-rotated dq of the 7 query heads into dqkv, and stores P and dS against the 64 sequence keys in p_seq / ds_seq for
// K_M_BATTN_KVR. The prompt keys are constant and get no gradient. A row t >= len only zeroes its dqkv entries.
__device__ __noinline__ void m_battn_row(const Ctx& C, int l, int local, char* smem) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int srow = local >> 1, g = local & 1;
  const int s = srow / MAXGEN, t = srow % MAXGEN, len = B->len[s], li = l - LMID;
  bf16* dqkv = M->dqkv + (size_t)s * MAXGEN * QKVP;
  if (t >= len) {
    for (int c = tid; c < HPK * HD; c += 256) dqkv[(size_t)t * QKVP + g * HPK * HD + c] = __float2bfloat16(0.f);
    if (tid < 2 * HD) { int w = tid >> 6, d = tid & 63; dqkv[(size_t)t * QKVP + D + (w * NKV + g) * HD + d] = __float2bfloat16(0.f); }
    return;
  }
  const int prompt = M->seq_prompt[s], plen = M->seq_plen[s], lenA = plen - 1, pos = lenA + t, n = pos + 1;
  const float* qkv = M->qkv + ((size_t)li * TROWS + (size_t)s * MAXGEN) * QKVP;
  const float* segA = C.prompts.kv + ((size_t)prompt * NL + l) * (MAXPROMPT * KVW);
  const float* segB = M->kv + ((size_t)li * TROWS + (size_t)s * MAXGEN) * KVW;
  const float* dao = M->dao + (size_t)s * MAXGEN * D;
  float* p_seq  = M->p_seq  + (size_t)s * (64 * NKV * HPK * 64);
  float* ds_seq = M->ds_seq + (size_t)s * (64 * NKV * HPK * 64);

  float* S = reinterpret_cast<float*>(smem);
  constexpr int O_KV = 0;                         // [192][68]
  constexpr int O_Q  = O_KV + AT_MAXN * AT_PITCH; // [7][64]
  constexpr int O_DO = O_Q  + HPK * HD;           // [7][64]
  constexpr int O_DS = O_DO + HPK * HD;           // [7][256]
  constexpr int O_DQ = O_DS + HPK * 256;          // [4][7][64]
  constexpr int O_R  = O_DQ + 4 * HPK * HD;       // [2][8][8]

  for (int q = tid; q < n * 16; q += 256) {
    const int j = q >> 4, c = q & 15;
    const float* src = (j < lenA ? segA + (size_t)j * KVW : segB + (size_t)(j - lenA) * KVW) + g * HD + 4 * c;
    lg::cp_async16(lg::smem_u32(S + O_KV + j * AT_PITCH + 4 * c), src);
  }
  lg::cp_commit();
  if (warp < HPK) {
    const int head = HPK * g + warp;
    const float cs = __ldg(C.model.rope_cos + (size_t)pos * 32 + lane), sn = __ldg(C.model.rope_sin + (size_t)pos * 32 + lane);
    const float x0 = qkv[(size_t)t * QKVP + head * HD + lane], x1 = qkv[(size_t)t * QKVP + head * HD + 32 + lane];
    S[O_Q + warp * HD + lane]      = x0 * cs - x1 * sn;
    S[O_Q + warp * HD + 32 + lane] = x1 * cs + x0 * sn;
    S[O_DO + warp * HD + lane]      = dao[(size_t)t * D + head * HD + lane];
    S[O_DO + warp * HD + 32 + lane] = dao[(size_t)t * D + head * HD + 32 + lane];
  }
  lg::cp_wait<0>();
  __syncthreads();

  float sc[HPK];
  if (tid < n) {
    const float* kr = S + O_KV + tid * AT_PITCH; float acc[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) acc[a] = 0.f;
    for (int c = 0; c < HD / 4; c++) {
      const F4 kk = *reinterpret_cast<const F4*>(kr + 4 * c);
#pragma unroll
      for (int a = 0; a < HPK; a++) {
        const F4 qq = *reinterpret_cast<const F4*>(S + O_Q + a * HD + 4 * c);
#pragma unroll
        for (int i = 0; i < 4; i++) acc[a] = fmaf(kk.v[i], qq.v[i], acc[a]);
      }
    }
#pragma unroll
    for (int a = 0; a < HPK; a++) sc[a] = 0.125f * acc[a];
  } else {
#pragma unroll
    for (int a = 0; a < HPK; a++) sc[a] = -INFINITY;
  }
  {
    float m[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) m[a] = sc[a];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int a = 0; a < HPK; a++) m[a] = fmaxf(m[a], __shfl_xor_sync(0xffffffffu, m[a], o));
    if (lane == 0)
#pragma unroll
      for (int a = 0; a < HPK; a++) S[O_R + warp * 8 + a] = m[a];
  }
  __syncthreads();

  for (int q = tid; q < n * 16; q += 256) {
    const int j = q >> 4, c = q & 15;
    const float* src = (j < lenA ? segA + (size_t)j * KVW : segB + (size_t)(j - lenA) * KVW) + NKV * HD + g * HD + 4 * c;
    lg::cp_async16(lg::smem_u32(S + O_KV + j * AT_PITCH + 4 * c), src);
  }
  lg::cp_commit();
  float pr[HPK];
  {
    float sm[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) {
      float mm = S[O_R + a];
#pragma unroll
      for (int w = 1; w < 8; w++) mm = fmaxf(mm, S[O_R + w * 8 + a]);
      pr[a] = tid < n ? expf(sc[a] - mm) : 0.f;
      sm[a] = pr[a];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int a = 0; a < HPK; a++) sm[a] += __shfl_xor_sync(0xffffffffu, sm[a], o);
    if (lane == 0)
#pragma unroll
      for (int a = 0; a < HPK; a++) S[O_R + 64 + warp * 8 + a] = sm[a];
  }
  __syncthreads();
#pragma unroll
  for (int a = 0; a < HPK; a++) {
    float tot = S[O_R + 64 + a];
#pragma unroll
    for (int w = 1; w < 8; w++) tot += S[O_R + 64 + w * 8 + a];
    pr[a] = tid < n ? (pr[a] / tot) : 0.f;
  }
  lg::cp_wait<0>();
  __syncthreads();

  float dp[HPK];
  if (tid < n) {
    const float* vr = S + O_KV + tid * AT_PITCH;
#pragma unroll
    for (int a = 0; a < HPK; a++) dp[a] = 0.f;
    for (int c = 0; c < HD / 4; c++) {
      const F4 vv = *reinterpret_cast<const F4*>(vr + 4 * c);
#pragma unroll
      for (int a = 0; a < HPK; a++) {
        const F4 dd = *reinterpret_cast<const F4*>(S + O_DO + a * HD + 4 * c);
#pragma unroll
        for (int i = 0; i < 4; i++) dp[a] = fmaf(vv.v[i], dd.v[i], dp[a]);
      }
    }
  } else {
#pragma unroll
    for (int a = 0; a < HPK; a++) dp[a] = 0.f;
  }
  {
    float pd[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) pd[a] = pr[a] * dp[a];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int a = 0; a < HPK; a++) pd[a] += __shfl_xor_sync(0xffffffffu, pd[a], o);
    if (lane == 0)
#pragma unroll
      for (int a = 0; a < HPK; a++) S[O_R + warp * 8 + a] = pd[a];
  }
  __syncthreads();

  for (int q = tid; q < n * 16; q += 256) {
    const int j = q >> 4, c = q & 15;
    const float* src = (j < lenA ? segA + (size_t)j * KVW : segB + (size_t)(j - lenA) * KVW) + g * HD + 4 * c;
    lg::cp_async16(lg::smem_u32(S + O_KV + j * AT_PITCH + 4 * c), src);
  }
  lg::cp_commit();
  if (tid >= lenA && tid < MAXPROMPT + 64) {
    const int js = tid - lenA;
    if (js < 64) {
#pragma unroll
      for (int a = 0; a < HPK; a++) {
        float Dt = S[O_R + a];
#pragma unroll
        for (int w = 1; w < 8; w++) Dt += S[O_R + w * 8 + a];
        const float ds = (tid < n) ? (0.125f * pr[a] * (dp[a] - Dt)) : 0.f;
        if (tid < n) S[O_DS + a * 256 + tid] = ds;
        const size_t off = (((size_t)t * NKV + g) * HPK + a) * 64 + js;
        p_seq[off]  = (tid < n) ? pr[a] : 0.f;
        ds_seq[off] = ds;
      }
    }
  } else if (tid < n) {
#pragma unroll
    for (int a = 0; a < HPK; a++) {
      float Dt = S[O_R + a];
#pragma unroll
      for (int w = 1; w < 8; w++) Dt += S[O_R + w * 8 + a];
      S[O_DS + a * 256 + tid] = 0.125f * pr[a] * (dp[a] - Dt);
    }
  }
  lg::cp_wait<0>();
  __syncthreads();

  {
    const int jg = tid >> 6, d = tid & 63; float acc[HPK];
#pragma unroll
    for (int a = 0; a < HPK; a++) acc[a] = 0.f;
    for (int j = jg; j < n; j += 4) {
      const float kv = S[O_KV + j * AT_PITCH + d];
#pragma unroll
      for (int a = 0; a < HPK; a++) acc[a] = fmaf(S[O_DS + a * 256 + j], kv, acc[a]);
    }
#pragma unroll
    for (int a = 0; a < HPK; a++) S[O_DQ + (jg * HPK + a) * HD + d] = acc[a];
  }
  __syncthreads();
  if (warp < HPK) {
    const int a = warp, head = HPK * g + a;
    float rl = S[O_DQ + a * HD + lane], rh = S[O_DQ + a * HD + 32 + lane];
#pragma unroll
    for (int k = 1; k < 4; k++) { rl += S[O_DQ + (k * HPK + a) * HD + lane]; rh += S[O_DQ + (k * HPK + a) * HD + 32 + lane]; }
    const float cs = __ldg(C.model.rope_cos + (size_t)pos * 32 + lane), sn = __ldg(C.model.rope_sin + (size_t)pos * 32 + lane);
    dqkv[(size_t)t * QKVP + head * HD + lane]      = __float2bfloat16(rl * cs + rh * sn);
    dqkv[(size_t)t * QKVP + head * HD + 32 + lane] = __float2bfloat16(-rl * sn + rh * cs);
  }
}

// K_M_BATTN_KVR of layer l: dK and dV of the sequence rows of sequence s for kv head g, local = 2 * s + g.
// Thread (js = tid / 4, d0 = 16 * (tid % 4)) owns 16 columns of key js and sums over the 7 heads and the query rows
// qt >= js: dK' += dS[qt, js] q'[qt] (rotated), dV += P[qt, js] dO[qt]. dK' is then un-rotated together with the
// thread that holds the other half of each RoPE pair (d0 +- 32).
__device__ __noinline__ void m_battn_kvr(const Ctx& C, int l, int local, char* smem) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int s = local >> 1, g = local & 1, li = l - LMID, len = B->len[s], plen = M->seq_plen[s], lenA = plen - 1;
  const int tid = threadIdx.x, js = tid >> 2, sub = tid & 3, d0 = sub * 16;
  const unsigned mask4 = 0xfu << (tid & 28);
  const float* qkv = M->qkv + ((size_t)li * TROWS + (size_t)s * MAXGEN) * QKVP;
  const float* dao = M->dao + (size_t)s * MAXGEN * D;
  const float* p_seq  = M->p_seq  + (size_t)s * (64 * NKV * HPK * 64);
  const float* ds_seq = M->ds_seq + (size_t)s * (64 * NKV * HPK * 64);
  bf16* dqkv = M->dqkv + (size_t)s * MAXGEN * QKVP;

  constexpr int PITCH = 68;
  float* S_Q  = reinterpret_cast<float*>(smem);
  float* S_DO = S_Q  + 64 * PITCH;
  float* S_P  = S_DO + 64 * PITCH;
  float* S_DS = S_P  + 64 * PITCH;

  float dk_rot[16], dv_reg[16];
#pragma unroll
  for (int i = 0; i < 16; i++) { dk_rot[i] = 0.f; dv_reg[i] = 0.f; }
  for (int a = 0; a < HPK; a++) {
    const int head = HPK * g + a;
    if (js < len) {
      const int pos = lenA + js;
      const F4* q4  = reinterpret_cast<const F4*>(qkv + (size_t)js * QKVP + head * HD);
      const F4* do4 = reinterpret_cast<const F4*>(dao + (size_t)js * D + head * HD + d0);
      const F4* p4  = reinterpret_cast<const F4*>(p_seq  + (((size_t)js * NKV + g) * HPK + a) * 64 + d0);
      const F4* ds4 = reinterpret_cast<const F4*>(ds_seq + (((size_t)js * NKV + g) * HPK + a) * 64 + d0);
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const int d = d0 + 4 * c, pair = d & 31;
        const F4 cs = *reinterpret_cast<const F4*>(C.model.rope_cos + pos * 32 + pair);
        const F4 sn = *reinterpret_cast<const F4*>(C.model.rope_sin + pos * 32 + pair);
        const F4 x0 = q4[pair >> 2], x1 = q4[(pair + 32) >> 2];
        F4 qr;
#pragma unroll
        for (int k = 0; k < 4; k++)
          qr.v[k] = (d < 32) ? (x0.v[k] * cs.v[k] - x1.v[k] * sn.v[k]) : (x1.v[k] * cs.v[k] + x0.v[k] * sn.v[k]);
        *reinterpret_cast<F4*>(S_Q  + js * PITCH + d) = qr;
        *reinterpret_cast<F4*>(S_DO + js * PITCH + d) = do4[c];
        *reinterpret_cast<F4*>(S_P  + js * PITCH + d) = p4[c];
        *reinterpret_cast<F4*>(S_DS + js * PITCH + d) = ds4[c];
      }
    }
    __syncthreads();
    if (js < len) {
      for (int qt = js; qt < len; qt++) {
        const float p  = S_P[qt * PITCH + js], ds = S_DS[qt * PITCH + js];
        const F4* qr4  = reinterpret_cast<const F4*>(S_Q  + qt * PITCH + d0);
        const F4* dor4 = reinterpret_cast<const F4*>(S_DO + qt * PITCH + d0);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          const F4 qv = qr4[c], dv = dor4[c];
#pragma unroll
          for (int k = 0; k < 4; k++) {
            dk_rot[4 * c + k] = fmaf(ds, qv.v[k], dk_rot[4 * c + k]);
            dv_reg[4 * c + k] = fmaf(p,  dv.v[k], dv_reg[4 * c + k]);
          }
        }
      }
    }
    __syncthreads();
  }
  if (js < len) {
    const int pos = lenA + js;
    bf16* dk_dst = dqkv + (size_t)js * QKVP + D + g * HD + d0;
    bf16* dv_dst = dqkv + (size_t)js * QKVP + D + NKV * HD + g * HD + d0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const float mate = __shfl_xor_sync(mask4, dk_rot[i], 2);
      const int d = d0 + i, pair = d & 31;
      const float cs = C.model.rope_cos[pos * 32 + pair], sn = C.model.rope_sin[pos * 32 + pair];
      const float rl = (d < 32) ? dk_rot[i] : mate, rh = (d < 32) ? mate : dk_rot[i];
      dk_dst[i] = __float2bfloat16((d < 32) ? (rl * cs + rh * sn) : (-rl * sn + rh * cs));
      dv_dst[i] = __float2bfloat16(dv_reg[i]);
    }
  }
  // Columns DQKV..QKVP-1 of dqkv are never written. They are zero from allocation, and neither K_M_BQKV (K = DQKV)
  // nor K_M_GQKV (avalid = DQKV) reads them.
}

// K_M_BQKV of layer l > LMID: dxn1 = dqkv * W_qkv (NN, [TM, DQKV] x [DQKV, KQ], output columns < D) into part[0];
// K = DQKV is not split. local = s * NT_D + tn.
__device__ __forceinline__ void m_bqkv(const Ctx& C, int l, int local, char* smem) {
  MidState* M = get_mut(C);
  const Batch* B = C.tr.batch;
  const int s = local / NT_D, tn = local % NT_D, len = B->len[s];
  const size_t r0 = (size_t)s * MAXGEN;
  lg::Args a = lg::args_default();
  a.A = M->dqkv + r0 * QKVP; a.lda = QKVP; a.avalid = len;
  a.B = C.model.qkv[l]; a.ldb = KQ; a.bvalid = D;
  a.C = M->part + r0 * DP; a.ldc = DP; a.rvalid = len; a.cvalid = D;
  a.M = TM; a.N = DP; a.K = DQKV;
  g_nn_f32(a, 0, tn, smem);
}

// ================================================================================================ LEARNER: LAYER LMID WEIGHT GRADIENTS & ADAMW
// TN GEMMs over all TROWS sequence rows of the batch (rows t >= len[s] are zero in both operands):
//   dW_down [D, FF]     = dy_bf16^T * xm_mid     (14 x 19 = 266 tiles)
//   dW_gu   [GUP, D]    = dgu^T     * xn2_mid    (152 x 4 = 608 tiles)
//   dW_o    [D, D]      = dy_bf16^T * ao_mid     (14 x 4  =  56 tiles)
//   dW_qkv  [DQKV, KQ]  = dqkv^T    * xn1_mid    (18 x 4  =  72 tiles; only columns < D are written)
// The q/k/v bias (column D of W_qkv) and the padding columns after it are not trained: K_M_GQKV does not write their
// gradient, K_M_GNORM sets it to zero and K_M_ADAM leaves those weights at their base values.
constexpr int M_TM_D    = D / TM;       // 14
constexpr int M_TM_GU   = GUP / TM;     // 152
constexpr int M_TM_QKV  = DQKV / TM;    // 18

// K_M_GDOWN, K_M_GGU, K_M_GO, K_M_GQKV: the rows are split into M_WG_KS chunks; chunk c of every output tile goes to
// gpart[c], and K_M_GNORM adds the chunks in order. local = (tm * ntn + tn) * M_WG_KS + c.
constexpr int M_WG_KS = 4;
static_assert(TROWS % (M_WG_KS * 32) == 0 || (TROWS / 32) >= M_WG_KS, "TROWS split-K steps");

__device__ __forceinline__ void m_wgrad(const Ctx& C, int kind, int local, char* smem) {
  MidState* M = get_mut(C);
  const int ntn = kind == K_M_GDOWN ? NT_FF : NT_D;
  const int c = local % M_WG_KS, tile = local / M_WG_KS, tm = tile / ntn, tn = tile % ntn;
  int k0, kc; kchunk(TROWS / 32, M_WG_KS, c, k0, kc);
  float* gp = M->gpart + (size_t)c * MID_NW;
  lg::Args a = lg::args_default();
  a.K = kc;
  switch (kind) {
    case K_M_GDOWN:
      a.A = M->dy_bf16 + (size_t)k0 * DP; a.lda = DP; a.avalid = D;
      a.B = M->xm_mid  + (size_t)k0 * FF; a.ldb = FF; a.bvalid = FF;
      a.C = gp + MID_OFF_DN; a.ldc = FF; a.rvalid = D; a.cvalid = FF; a.M = D; a.N = FF; break;
    case K_M_GGU:
      a.A = M->dgu     + (size_t)k0 * GUP; a.lda = GUP; a.avalid = GUP;
      a.B = M->xn2_mid + (size_t)k0 * KQ;  a.ldb = KQ;  a.bvalid = D;
      a.C = gp + MID_OFF_GU; a.ldc = D; a.rvalid = GUP; a.cvalid = D; a.M = GUP; a.N = D; break;
    case K_M_GO:
      a.A = M->dy_bf16 + (size_t)k0 * DP; a.lda = DP; a.avalid = D;
      a.B = M->ao_mid  + (size_t)k0 * D;  a.ldb = D;  a.bvalid = D;
      a.C = gp + MID_OFF_O; a.ldc = D; a.rvalid = D; a.cvalid = D; a.M = D; a.N = D; break;
    default:
      a.A = M->dqkv    + (size_t)k0 * QKVP; a.lda = QKVP; a.avalid = DQKV;
      a.B = M->xn1_mid + (size_t)k0 * KQ;   a.ldb = KQ;   a.bvalid = D;
      a.C = gp + MID_OFF_QKV; a.ldc = KQ; a.rvalid = DQKV; a.cvalid = D; a.M = DQKV; a.N = KQ; break;
  }
  g_tn_f32(a, tm, tn, smem);
}

// K_M_GNORM, task k of NT_MID_ADAM, each a contiguous slice of MID_NW: gsum = the M_WG_KS partials added in chunk
// order, with the bias and padding entries of W_qkv set to zero, and gnorm2_part[k] = the slice's squared norm (thread
// partials added in thread order). Uses 256 floats of smem.
__device__ __noinline__ void m_gnorm(const Ctx& C, int k, char* smem) {
  MidState* M = get_mut(C);
  const size_t chunk = ((MID_NW / 4 + NT_MID_ADAM - 1) / NT_MID_ADAM) * 4;
  const size_t i0 = (size_t)k * chunk, i1 = min(MID_NW, i0 + chunk);
  float g2 = 0.f;
  for (size_t i = i0 + (size_t)threadIdx.x * 4; i < i1; i += 256 * 4) {
    F4 g = sum_parts_rn(M->gpart + i, MID_NW, M_WG_KS);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const size_t idx = i + j;
      if (idx < MID_OFF_O && (idx % KQ) >= D) g.v[j] = 0.f;
      g2 = __fadd_rn(g2, __fmul_rn(g.v[j], g.v[j]));
    }
    *reinterpret_cast<F4*>(M->gsum + i) = g;
  }
  float* sf = reinterpret_cast<float*>(smem);
  sf[threadIdx.x] = g2;
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot = 0.f;
    for (int t = 0; t < 256; t++) tot = __fadd_rn(tot, sf[t]);
    M->gnorm2_part[k] = tot;
  }
}

// K_M_ADAM, task k: the slice of K_M_GNORM. Every task adds the NT_MID_ADAM squared norms in the same order, so all
// agree on the clip factor (used when hp.grad_clip > 0). AdamW with decoupled weight decay on the fp32 masters, bias
// corrections for update step + 1, then the bf16 copy into C.tr.W. Tasks k < NGNORM also hand the squared norms
// (before clipping) to T_FINISH in B->gnorm2, two partials each.
__device__ __forceinline__ double dmul_rn(double a, double b) { return __dmul_rn(a, b); }
// 1 - beta^t, with beta^t by repeated squaring in double
__device__ __forceinline__ float bias_corr_mid(float beta, u64 t) {
  double r = 1.0, b = (double)beta;
  for (u64 n = t; n != 0; n >>= 1) { if (n & 1ull) r = dmul_rn(r, b); b = dmul_rn(b, b); }
  return (float)__dsub_rn(1.0, r);
}

__device__ __noinline__ void m_adam(const Ctx& C, int k, u64 step) {
  MidState* M = get_mut(C);
  Batch* B = C.tr.batch;
  float tot_g2 = 0.f;
  for (int c = 0; c < NT_MID_ADAM; c++) tot_g2 = __fadd_rn(tot_g2, M->gnorm2_part[c]);
  const float gnorm = sqrtf(tot_g2);
  float clip_scale = 1.f;
  if (C.hp.grad_clip > 0.f && gnorm > C.hp.grad_clip) clip_scale = C.hp.grad_clip / (gnorm + 1e-12f);

  const float lrwd = __fmul_rn(C.hp.lr, C.hp.wd), b1 = C.hp.beta1, omb1 = __fadd_rn(1.f, -b1);
  const float b2 = C.hp.beta2, omb2 = __fadd_rn(1.f, -b2);
  const float step_sz = __fdiv_rn(C.hp.lr, bias_corr_mid(b1, step + 1ull));
  const float sq2 = sqrtf(bias_corr_mid(b2, step + 1ull)), eps = C.hp.adam_eps;

  const size_t chunk = ((MID_NW / 4 + NT_MID_ADAM - 1) / NT_MID_ADAM) * 4;
  const size_t i0 = (size_t)k * chunk, i1 = min(MID_NW, i0 + chunk);
  for (size_t i = i0 + (size_t)threadIdx.x * 4; i < i1; i += 256 * 4) {
    F4 g = ldf4(M->gsum + i), w = ldf4(M->W32 + i), m = ldf4(M->mW + i), v = ldf4(M->vW + i);
    V4 bo;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const size_t idx = i + j;
      if (idx < MID_OFF_O && (idx % KQ) >= D) {
        // q/k/v bias or padding column: unchanged (W32 holds the base bf16 value exactly)
        bo.v[j] = __float2bfloat16(w.v[j]);
      } else {
        const float gj = __fmul_rn(g.v[j], clip_scale);
        w.v[j] = __fadd_rn(w.v[j], -__fmul_rn(lrwd, w.v[j]));
        m.v[j] = __fadd_rn(__fmul_rn(b1, m.v[j]), __fmul_rn(omb1, gj));
        v.v[j] = __fadd_rn(__fmul_rn(b2, v.v[j]), __fmul_rn(__fmul_rn(omb2, gj), gj));
        w.v[j] = __fadd_rn(w.v[j], -__fmul_rn(step_sz, __fdiv_rn(m.v[j], __fadd_rn(__fdiv_rn(sqrtf(v.v[j]), sq2), eps))));
        bo.v[j] = __float2bfloat16(w.v[j]);
      }
    }
    *reinterpret_cast<F4*>(M->W32 + i) = w;
    *reinterpret_cast<F4*>(M->mW  + i) = m;
    *reinterpret_cast<F4*>(M->vW  + i) = v;
    *reinterpret_cast<V4*>(C.tr.W + i) = bo;
  }
  if (threadIdx.x == 0 && k < NGNORM) B->gnorm2[k] = M->gnorm2_part[2 * k] + M->gnorm2_part[2 * k + 1];
}

// ================================================================================================ DISPATCHERS & GRAPH BUILDERS
// One task of a K_Q_MID_ phase for pipeline pipe_idx (P); `local` without the lockstep multiplier.
__device__ __noinline__ void run_roll(const Ctx& C, const Pipe& P, int pipe_idx, const mk::PhaseDesc& pd, int local, char* smem) {
  if (pd.kind == K_Q_MID_COHORT) { roll_mid_cohort(C, P, pipe_idx); return; }
  roll_mid_gemm(C, P, pipe_idx, pd.kind, local, smem);
}

// One task of a K_M_ phase; `step` is the 0-based update index (K_M_ADAM bias-corrects for step + 1). Nothing runs for
// an invalid batch (shutdown).
__device__ __noinline__ void run_train(const Ctx& C, const mk::PhaseDesc& pd, int local, u64 step, char* smem) {
  Batch* B = C.tr.batch;
  if (B->valid == 0) return;
  switch (pd.kind) {
    case K_M_GATHER:    m_gather(C, local); break;
    case K_M_FNORM1:
    case K_M_FNORM2:    m_fnorm(C, pd.kind, pd.a, local); break;
    case K_M_FQKV:
    case K_M_FOPROJ:
    case K_M_FGU:
    case K_M_FDOWNP:    m_fgemm(C, pd.kind, pd.a, local, smem); break;
    case K_M_FATTN:     m_fattn(C, pd.a, local, smem); break;
    case K_M_FACT:      m_fact(C, pd.a, local); break;
    case K_M_NORMF:     m_normf(C, local); break;
    case K_M_LM1:       gt::t_lm1(C, B, local, smem); break;
    case K_M_LOSS:      gt::t_loss(C, B, local, smem); break;
    case K_M_LM2:       gt::t_lm2(C, B, local, smem); break;
    case K_M_LMB:       gt::t_lmb(C, B, local, smem); break;
    case K_M_LMBR:      gt::t_lmbr(C, B, local); break;
    case K_M_NORMB:     m_normb_init(C, local); break;
    case K_M_BDOWN:     m_bdown(C, pd.a, local, smem); break;
    case K_M_BACT:      m_bact(C, pd.a, local); break;
    case K_M_BGU:       m_bgu(C, pd.a, local, smem); break;
    case K_M_BNORM2:
    case K_M_BNORM1:    m_bnorm(C, pd.kind, pd.a, local); break;
    case K_M_BOPROJ:    m_boproj(C, pd.a, local, smem); break;
    case K_M_BATTN_ROW: m_battn_row(C, pd.a, local, smem); break;
    case K_M_BATTN_KVR: m_battn_kvr(C, pd.a, local, smem); break;
    case K_M_BQKV:      m_bqkv(C, pd.a, local, smem); break;
    case K_M_GDOWN:
    case K_M_GGU:
    case K_M_GO:
    case K_M_GQKV:      m_wgrad(C, pd.kind, local, smem); break;
    case K_M_GNORM:     m_gnorm(C, local, smem); break;
    case K_M_ADAM:      m_adam(C, local, step); break;
    default: break;
  }
}

// Host: the rollout job graph, one step per sampled token; task counts are per pipeline times the lockstep
// multiplier `mult`. In ticket order:
//   1. R_BEGIN (admission and cancellation)
//   2. Q_MID_COH, layer LMID (Q_NORM1 and the cohort GEMMs Q_MQKV, Q_MOPROJ, Q_MGU, Q_MDOWNP), then the frozen layers
//      LMID+1..NL-1 up to Q_ACT[NL-1]
//   3. the head: H_Y0P, H_Y0R, H_NORMF, H_LMHEAD, H_SAMPLE
//   4. after H_SAMPLE, side by side: R_ADVANCE, and the frozen layers 0..LMID-1 on the sampled tokens, ending with
//      Q_DOWNR[LMID-1], which leaves the residual entering layer LMID in P.hres for the next step
// R_BEGIN waits for both branches of step 4 of the previous step. Returns H_SAMPLE (also stored in ph_sample_out).
inline int add_roll_mid(mk::Builder& b, int ngrp, const Hyper& hp, int mult, int& ph_sample_out) {
  qw::check_trunk_hyper(hp);
  const int nrows = GSZ * ngrp;
  auto rowt = [&](int kind) { const int r = qw::rows_per_task(qw::row_ks(kind, hp)); return (nrows + r - 1) / r; };
  const int begin = b.add("R_BEGIN", K_R_BEGIN, 1, {});
  int prev = begin;
  auto add = [&](const std::string& name, int kind, int layer, int n) {
    prev = b.add(name, kind, n * mult, {{prev, 0}}, layer, 0, mult);
  };

  // upper half: layer LMID with the pinned snapshots, then the frozen layers up to Q_ACT[NL-1]
  add("Q_MID_COH", K_Q_MID_COHORT, LMID, 1);
  for (int l = LMID; l < NL; l++) {
    const std::string s = "[" + std::to_string(l) + "]";
    if (l == LMID) add("Q_NORM1" + s, K_Q_NORM1, l, rowt(K_Q_NORM1));
    else           add("Q_NORM1F" + s, K_Q_NORM1F, l, rowt(K_Q_NORM1F));
    add((l == LMID ? "Q_MQKV" : "Q_QKV") + s,   l == LMID ? K_Q_MID_QKV   : K_Q_QKV,   l, NT_QKV * hp.ksQ);
    add("Q_ATTN" + s,                           K_Q_ATTN,                              l, 2 * nrows);
    add((l == LMID ? "Q_MOPROJ" : "Q_OPROJ") + s, l == LMID ? K_Q_MID_OPROJ : K_Q_OPROJ, l, NT_D * hp.ksO);
    add("Q_NORM2" + s,                          K_Q_NORM2,                             l, rowt(K_Q_NORM2));
    add((l == LMID ? "Q_MGU" : "Q_GU") + s,     l == LMID ? K_Q_MID_GU    : K_Q_GU,    l, NT_GU * hp.ksG);
    add("Q_ACT" + s,                            K_Q_ACT,                               l, NT_FF * qw::act_split(hp.ksG));
    if (l < LT) {
      add((l == LMID ? "Q_MDOWNP" : "Q_DOWNP") + s, l == LMID ? K_Q_MID_DOWNP : K_Q_DOWNP, l, NT_D * hp.ksD);
    }
  }

  // sampling head: layer LT's down_proj, final norm, LM head, Gumbel-max sample
  const int ph_sample = hd::add_head(b, prev, ngrp, hp, mult);
  ph_sample_out = ph_sample;
  const int adv = b.add("R_ADVANCE", K_R_ADVANCE, 1, {{ph_sample, 0}});

  // lower half: layers 0..LMID-1 on the sampled tokens, beside R_ADVANCE
  prev = ph_sample;
  for (int l = 0; l < LMID; l++) {
    const std::string s = "[" + std::to_string(l) + "]";
    if (l == 0) add("Q_EMBN", K_Q_EMBN, 0, rowt(K_Q_EMBN));
    else        add("Q_NORM1F" + s, K_Q_NORM1F, l, rowt(K_Q_NORM1F));
    add("Q_QKV" + s,   K_Q_QKV,   l, NT_QKV * hp.ksQ);
    add("Q_ATTN" + s,  K_Q_ATTN,  l, 2 * nrows);
    add("Q_OPROJ" + s, K_Q_OPROJ, l, NT_D * hp.ksO);
    add("Q_NORM2" + s, K_Q_NORM2, l, rowt(K_Q_NORM2));
    add("Q_GU" + s,    K_Q_GU,    l, NT_GU * hp.ksG);
    add("Q_ACT" + s,   K_Q_ACT,   l, NT_FF * qw::act_split(hp.ksG));
    add("Q_DOWNP" + s, K_Q_DOWNP, l, NT_D * hp.ksD);
  }
  // A separate reduce of layer LMID-1's down_proj partials: the norm that would fold it in (Q_NORM1[LMID]) runs in the
  // next step, after Q_MID_COH has saved the residual.
  add("Q_DOWNR[" + std::to_string(LMID - 1) + "]", K_Q_DOWNR, LMID - 1, rowt(K_Q_DOWNR));

  b.dep(begin, mk::Dep{prev, -1});
  b.dep(begin, mk::Dep{adv, -1});
  b.finalize(1);
  return ph_sample;
}

// Host: the learner phases of one update, after `select_phase` (T_SELECT). last_a and last_b receive the phase that
// T_FINISH waits for (M_ADAM).
inline void add_train_mid(mk::Builder& b, int select_phase, const Hyper& hp, int& last_a, int& last_b) {
  typedef mk::Dep Dp;
  int prev = b.add("M_GATHER", K_M_GATHER, TSEQ, {Dp{select_phase, 0}}, 0, 0, 1);
  auto add = [&](const std::string& nm, int kind, int layer, int n) {
    prev = b.add(nm, kind, n, {Dp{prev, 0}}, layer, 0, 1);
    return prev;
  };

  // 1. forward replay through layers LMID..NL-1
  for (int l = LMID; l < NL; l++) {
    const std::string s = "[" + std::to_string(l) + "]";
    add("M_FNORM1" + s, K_M_FNORM1, l, TROWS / 8);
    add("M_FQKV" + s,   K_M_FQKV,   l, TSEQ * NT_QKV * hp.ksQ);
    add("M_FATTN" + s,  K_M_FATTN,  l, TROWS * 2);
    add("M_FOPROJ" + s, K_M_FOPROJ, l, TSEQ * NT_D * hp.ksO);
    add("M_FNORM2" + s, K_M_FNORM2, l, TROWS / 8);
    add("M_FGU" + s,    K_M_FGU,    l, TSEQ * NT_GU * hp.ksG);
    add("M_FACT" + s,   K_M_FACT,   l, TSEQ * NT_FF * qw::act_split(hp.ksG));
    const int ks_dn = (l == LT ? hp.ksY : hp.ksD);
    add("M_FDOWNP" + s, K_M_FDOWNP, l, TSEQ * NT_D * ks_dn);
  }

  // 2. final norm, LM head forward, GRPO clipped surrogate loss, LM head backward, final norm backward
  add("M_NORMF", K_M_NORMF, LT, TROWS / 8);
  add("M_LM1",   K_M_LM1,   LT, NT_V * TTILE);
  add("M_LOSS",  K_M_LOSS,  LT, TTILE);
  add("M_LM2",   K_M_LM2,   LT, NT_V * TTILE);
  add("M_LMB",   K_M_LMB,   LT, hp.tksV * TTILE * NT_D);
  add("M_LMBR",  K_M_LMBR,  LT, TTILE * NT_D);
  add("M_NORMB", K_M_NORMB, LT, TROWS / 8);

  // 3. backward through layers NL-1..LMID. At l = LMID each weight-gradient GEMM branches off the chain once its
  //    operands are ready; only M_GNORM waits for the four of them.
  int ph_gdown = -1, ph_ggu = -1, ph_go = -1, ph_gqkv = -1;
  for (int l = NL - 1; l >= LMID; l--) {
    const std::string s = "[" + std::to_string(l) + "]";
    const int dh_mlp_ready = prev;
    if (l == LMID) {
      ph_gdown = b.add("M_GDOWN", K_M_GDOWN, M_WG_KS * M_TM_D * NT_FF, {Dp{dh_mlp_ready, 0}}, LMID, 0, 1);
    }
    add("M_BDOWN" + s,  K_M_BDOWN,  l, TSEQ * NT_FF);
    const int bact = add("M_BACT" + s, K_M_BACT, l, TSEQ * NT_FF * 2);
    if (l == LMID) {
      ph_ggu = b.add("M_GGU", K_M_GGU, M_WG_KS * M_TM_GU * NT_D, {Dp{bact, 0}}, LMID, 0, 1);
    }
    add("M_BGU" + s,    K_M_BGU,    l, TSEQ * NT_D * BGU_KS);
    const int dh_attn_ready = add("M_BNORM2" + s, K_M_BNORM2, l, TROWS / 8);
    if (l == LMID) {
      b.dep(dh_attn_ready, Dp{ph_gdown, 0});     // M_BNORM2 overwrites the dy_bf16 that M_GDOWN reads
      ph_go = b.add("M_GO", K_M_GO, M_WG_KS * M_TM_D * NT_D, {Dp{dh_attn_ready, 0}}, LMID, 0, 1);
    }
    add("M_BOPROJ" + s,    K_M_BOPROJ,    l, TSEQ * NT_D);
    add("M_BATTN_ROW" + s, K_M_BATTN_ROW, l, TROWS * 2);
    const int kvr = add("M_BATTN_KVR" + s, K_M_BATTN_KVR, l, TSEQ * 2);
    if (l > LMID) {
      add("M_BQKV" + s,   K_M_BQKV,   l, TSEQ * NT_D);
      add("M_BNORM1" + s, K_M_BNORM1, l, TROWS / 8);
    } else {
      ph_gqkv = b.add("M_GQKV", K_M_GQKV, M_WG_KS * M_TM_QKV * NT_D, {Dp{kvr, 0}}, LMID, 0, 1);
    }
  }

  // 4. gradient sum and norm, then clipped fp32-master AdamW
  const int gn = b.add("M_GNORM", K_M_GNORM, NT_MID_ADAM,
                       {Dp{ph_gdown, 0}, Dp{ph_ggu, 0}, Dp{ph_go, 0}, Dp{ph_gqkv, 0}}, LMID, 0, 1);
  const int ad = b.add("M_ADAM",  K_M_ADAM,  NT_MID_ADAM, {Dp{gn, 0}}, LMID, 0, 1);
  last_a = ad;
  last_b = ad;
}

// Host: allocates the learner state (mk::dalloc: zero-filled, so the Adam moments start at 0, and owned by the
// caller's open mk::dscope) and initialises C.tr.W and the fp32 masters from the base weights of layer LMID.
// Returns the device copy of the MidState.
inline MidState* alloc_mid_state(const Model& model, const Train& tr) {
  MidState h; memset(&h, 0, sizeof h);
  h.pipe_ncoh  = mk::dalloc<int>(MAXPIPE);
  h.pipe_vslot = mk::dalloc<int>(MAXPIPE * MID_MAX_COH);
  h.pipe_rmask = mk::dalloc<u64>(MAXPIPE * MID_MAX_COH);

  h.W32  = mk::dalloc<float>(MID_NW);
  h.mW   = mk::dalloc<float>(MID_NW);
  h.vW   = mk::dalloc<float>(MID_NW);
  h.gsum = mk::dalloc<float>(MID_NW);
  h.gpart = mk::dalloc<float>((size_t)M_WG_KS * MID_NW);
  h.gnorm2_part = mk::dalloc<float>(NT_MID_ADAM);

  h.seq_prompt = mk::dalloc<int>(TSEQ);
  h.seq_plen   = mk::dalloc<int>(TSEQ);

  h.h1      = mk::dalloc<float>((size_t)NL_REM * TROWS * DP);
  h.rinv1   = mk::dalloc<float>((size_t)NL_REM * TROWS);
  h.xn1_mid = mk::dalloc<bf16>((size_t)TROWS * KQ);
  h.qkv     = mk::dalloc<float>((size_t)NL_REM * TROWS * QKVP);
  h.ao_mid  = mk::dalloc<bf16>((size_t)TROWS * D);
  h.h2      = mk::dalloc<float>((size_t)NL_REM * TROWS * DP);
  h.rinv2   = mk::dalloc<float>((size_t)NL_REM * TROWS);
  h.xn2_mid = mk::dalloc<bf16>((size_t)TROWS * KQ);
  h.gu      = mk::dalloc<float>((size_t)NL_REM * TROWS * GUP);
  h.xm_mid  = mk::dalloc<bf16>((size_t)TROWS * FF);
  h.kv      = mk::dalloc<float>((size_t)NL_REM * TROWS * KVW);

  h.xn      = mk::dalloc<bf16>((size_t)TROWS * KQ);
  h.ao      = mk::dalloc<bf16>((size_t)TROWS * D);
  h.xm      = mk::dalloc<bf16>((size_t)TROWS * FF);
  h.part    = mk::dalloc<float>((size_t)KS_MAX * TROWS * DP);
  h.qkvp    = mk::dalloc<float>((size_t)(KS_MAX / 4) * TROWS * QKVP);
  h.dh      = mk::dalloc<float>((size_t)TROWS * DP);
  h.dy_bf16 = mk::dalloc<bf16>((size_t)TROWS * DP);
  h.dgu     = mk::dalloc<bf16>((size_t)TROWS * GUP);
  h.dao     = mk::dalloc<float>((size_t)TROWS * D);
  h.dqkv    = mk::dalloc<bf16>((size_t)TROWS * QKVP);
  h.p_seq   = mk::dalloc<float>((size_t)TSEQ * 64 * NKV * HPK * 64);
  h.ds_seq  = mk::dalloc<float>((size_t)TSEQ * 64 * NKV * HPK * 64);

  MK_CK(cudaMemcpy(tr.W + MID_OFF_QKV, model.qkv[LMID],  (size_t)DQKV * KQ * sizeof(bf16), cudaMemcpyDeviceToDevice));
  MK_CK(cudaMemcpy(tr.W + MID_OFF_O,   model.o[LMID],    (size_t)D * D     * sizeof(bf16), cudaMemcpyDeviceToDevice));
  MK_CK(cudaMemcpy(tr.W + MID_OFF_GU,  model.gu[LMID],   (size_t)GUP * D   * sizeof(bf16), cudaMemcpyDeviceToDevice));
  MK_CK(cudaMemcpy(tr.W + MID_OFF_DN,  model.down[LMID], (size_t)D * FF    * sizeof(bf16), cudaMemcpyDeviceToDevice));
  std::vector<bf16> hw = mk::down(tr.W, MID_NW);
  std::vector<float> fw = host::bf16_to_float(hw);
  mk::up(h.W32, fw);

  MidState* dM = mk::dalloc<MidState>(1);
  MK_CK(cudaMemcpy(dM, &h, sizeof(MidState), cudaMemcpyHostToDevice));
  return dM;
}

} // namespace mt

