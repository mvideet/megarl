// grpo_loss.cuh -- the learner's GRPO loss through the LM head (tasks dispatched by mt::run_train, mid_train.cuh).
//
// Packed rows: one row per trained token, sequences back to back (Batch.off). Per update:
//   M_LM1   log-sum-exp partials of logits = xf E^T / T, one (vocabulary tile, packed tile) per task
//   M_LOSS  merge the partials, logp_new, clipped-ratio surrogate and dLoss/dlogp per token
//   M_LM2   dlogits = coef / T (onehot - softmax), bf16, stored per K chunk of the next GEMM
//   M_LMB   dxf partials = dlogits E, split over tksV chunks of the vocabulary
//   M_LMBR  dxf = sum of the partials in chunk order
// The forward path uses the same lm:: functions as the sampling head, so a group trained with the weights that
// sampled it reproduces its behaviour log-probabilities bit for bit (rho == 1 exactly at lag 0).
#pragma once
#include "mrl_common.cuh"
#include "mrl_lmhead.cuh"

namespace gt {
using mrl::bf16; using mrl::u64;
using mrl::D; using mrl::DP; using mrl::V; using mrl::VP;
using mrl::MAXGEN; using mrl::TSEQ; using mrl::TROWS; using mrl::TTILE; using mrl::NT_D; using mrl::NT_V;

constexpr float DLOGP_CLAMP = 20.f;                         // |logp_new - logp_beh| is clamped to this before the exponential
static_assert(MAXGEN == mrl::TM && TTILE == TSEQ && TROWS % 8 == 0, "one sequence is one M tile; packed tiles are M tiles");
static_assert(D % 32 == 0 && V % 32 == 0, "whole K steps");
static_assert(sizeof(mrl::LsePart) == 8, "stride of the log-sum-exp partials");

__device__ __noinline__ void g_nn_f32(const lg::Args& a, int tm, int tn, char* smem) { lg::gemm_tile<lg::NN, lg::EPI_F32>(a, tm, tn, smem); }

__device__ __forceinline__ mrl::F4 ldf4(const float* p) { return *reinterpret_cast<const mrl::F4*>(p); }

// Sum of n fp32 partials (4 floats each, `stride` floats apart) in index order starting from partial 0, with __fadd_rn:
// ((p0 + p1) + p2) + ... The loads are issued four at a time (independent of each other), the additions stay ordered.
__device__ __forceinline__ mrl::F4 sum_parts(const float* p, size_t stride, int n) {
  mrl::F4 a = ldf4(p); int c = 1;
  for (; c + 4 <= n; c += 4) {
    const mrl::F4 x0 = ldf4(p + (size_t)c * stride), x1 = ldf4(p + (size_t)(c + 1) * stride), x2 = ldf4(p + (size_t)(c + 2) * stride), x3 = ldf4(p + (size_t)(c + 3) * stride);
#pragma unroll
    for (int j = 0; j < 4; j++) a.v[j] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(a.v[j], x0.v[j]), x1.v[j]), x2.v[j]), x3.v[j]);
  }
  for (; c < n; c++) {
    const mrl::F4 x = ldf4(p + (size_t)c * stride);
#pragma unroll
    for (int j = 0; j < 4; j++) a.v[j] = __fadd_rn(a.v[j], x.v[j]);
  }
  return a;
}

// ------------------------------------------------------------------------------------------------ LM head, pass 1: log-sum-exp
// Vocabulary tile n of packed tile m. Warp w takes the rows w, 8 + w, 16 + w, ... of the tile.
__device__ __forceinline__ void t_lm1(const mrl::Ctx& C, const mrl::Batch* B, int local, char* smem) {
  const int n = local / TTILE, m = local % TTILE, r0 = m * mrl::TM, ntok = B->ntok;
  if (r0 >= ntok) return;
  const int nrows = min(mrl::TM, ntok - r0), n0 = n * mrl::TN, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  lm::lmhead_tile(C.tr.xf + (size_t)r0 * D, D, nrows, C.model.emb, n, smem);
  const float* T = reinterpret_cast<const float*>(smem); const float inv_t = C.hp.inv_temperature;
  // z of the taken token: lane i < 8 looks after row 8 i + warp; exactly one vocabulary tile holds y[r]
  if (lane < 8) {
    const int row = lane * 8 + warp;
    if (row < nrows) { float z; if (lm::row_logit_of(T + row * mrl::TN, n0, inv_t, C.tr.y[r0 + row], z)) C.tr.logit_y[r0 + row] = z; }
  }
  mrl::LsePart* out = C.tr.lsepart + ((size_t)m * NT_V + n) * mrl::TM;
  for (int i = 0; i < 8; i++) {
    const int row = i * 8 + warp;
    if (row >= nrows) break;
    const lm::RowLse l = lm::warp_row_lse(T + row * mrl::TN, n0, inv_t);
    if (lane == 0) { out[row].lmax = l.lmax; out[row].lsum = l.lsum; }
  }
}

// ------------------------------------------------------------------------------------------------ loss and dLoss/dlogp
// Packed tile m. For each row: merge the 594 partials (lm::block_merge_lse, the head's merge), logp_new, then
//   d = logp_new - logp_beh;  rho = exp(clamp(d, -20, 20));  l = -min(rho A, clip(rho, 1 - eps, 1 + eps) A)
//   coef = dLoss/dlogp = -w A rho if the unclipped term is the minimum (ties included) and d was not clamped, else 0
// with w = 1 / (TSEQ MAXGEN) (norm_mode 0) or 1 / (TSEQ len[s]) (norm_mode 1). The clamp is part of the function, so
// its derivative is zero where it acts: a token whose log-ratio left [-20, 20] cannot produce an infinite ratio nor a
// gradient. loss_part = sum of w l in row order; nclip_part = tokens with zero gradient (clip, clamp or NaN);
// maxdlogp_part = largest |d| before the clamp (+inf if a d is NaN). Tiles without rows store zeros.
// A NaN d fails the `inside` test (no gradient) and is dropped by fmaxf, so dc = -20: its loss term is finite.
__device__ __forceinline__ void t_loss(const mrl::Ctx& C, mrl::Batch* B, int m, char* smem) {
  const int tid = threadIdx.x, r0 = m * mrl::TM, ntok = B->ntok;
  if (r0 >= ntok) {
    if (tid == 0) { B->loss_part[m] = 0.f; B->nclip_part[m] = 0; B->maxdlogp_part[m] = 0.f; }
    return;
  }
  const int nrows = min(mrl::TM, ntok - r0);
  float* scr = reinterpret_cast<float*>(smem);              // [0, 16): scratch of the merge; [64, 256): three slots per row
  float* s_wl = scr + 64; float* s_ad = scr + 128; int* s_cl = reinterpret_cast<int*>(scr + 192);
  const char* part0 = reinterpret_cast<const char*>(C.tr.lsepart + (size_t)m * NT_V * mrl::TM);
  float my_max = 0.f, my_sum = 1.f;
  for (int i = 0; i < nrows; i++) {
    float lmax, lsum;
    lm::block_merge_lse(part0 + (size_t)i * sizeof(mrl::LsePart), (size_t)mrl::TM * sizeof(mrl::LsePart), scr, lmax, lsum);
    if (tid == i) { my_max = lmax; my_sum = lsum; }
  }
  if (tid < nrows) {                                        // thread i finishes row i
    const int r = r0 + tid, s = C.tr.rseq[r];
    const float lp = lm::logp_from(C.tr.logit_y[r], my_max, my_sum), lb = C.tr.logp_beh[r], adv = B->adv[s];
    const float w = C.hp.norm_mode == 1 ? __fdiv_rn(1.f, (float)(TSEQ * B->len[s])) : 1.f / (float)(TSEQ * MAXGEN);
    const float eps = C.hp.clip_eps, lo = __fadd_rn(1.f, -eps), hi = __fadd_rn(1.f, eps);
    const float d = __fadd_rn(lp, -lb);
    const bool inside = d >= -DLOGP_CLAMP && d <= DLOGP_CLAMP;            // false for NaN too
    const float dc = fminf(fmaxf(d, -DLOGP_CLAMP), DLOGP_CLAMP);
    const float rho = dc == 0.f ? 1.f : expf(dc);                         // lag 0: exactly 1
    const float ua = __fmul_rn(rho, adv), ca = __fmul_rn(fminf(fmaxf(rho, lo), hi), adv);
    const bool grad = ua <= ca && inside;
    C.tr.lse[r] = lm::lse_from(my_max, my_sum); C.tr.logp_new[r] = lp;
    C.tr.coef[r] = grad ? -__fmul_rn(__fmul_rn(w, adv), rho) : 0.f;
    s_wl[tid] = __fmul_rn(w, -fminf(ua, ca)); s_ad[tid] = d == d ? fabsf(d) : INFINITY; s_cl[tid] = grad ? 0 : 1;
  }
  __syncthreads();
  if (tid == 0) {
    float loss = 0.f, md = 0.f; int nc = 0;
    for (int i = 0; i < nrows; i++) { loss = __fadd_rn(loss, s_wl[i]); md = fmaxf(md, s_ad[i]); nc += s_cl[i]; }
    B->loss_part[m] = loss; B->nclip_part[m] = nc; B->maxdlogp_part[m] = md;
  }
}

// ------------------------------------------------------------------------------------------------ LM head, pass 2: dlogits
// dlogits[r, tile n] = bf16(coef / T (onehot_y - softmax)); ids >= V are written as zero (lm::warp_row_dlogits).
__device__ __forceinline__ void t_lm2(const mrl::Ctx& C, const mrl::Batch* B, int local, char* smem) {
  const int n = local / TTILE, m = local % TTILE, r0 = m * mrl::TM, ntok = B->ntok;
  if (r0 >= ntok) return;
  const int nrows = min(mrl::TM, ntok - r0), n0 = n * mrl::TN, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  lm::lmhead_tile(C.tr.xf + (size_t)r0 * D, D, nrows, C.model.emb, n, smem);
  const float* T = reinterpret_cast<const float*>(smem); const float inv_t = C.hp.inv_temperature;
  // lane i < 8 fetches the three scalars of row 8 i + warp once; they are handed to the warp with a shuffle
  float my_lse = 0.f, my_coef = 0.f; int my_y = 0;
  if (lane < 8 && lane * 8 + warp < nrows) { const int r = r0 + lane * 8 + warp; my_lse = C.tr.lse[r]; my_coef = C.tr.coef[r]; my_y = C.tr.y[r]; }
  for (int i = 0; i < 8; i++) {
    const int row = i * 8 + warp;
    if (row >= nrows) break;
    const float lse = __shfl_sync(0xffffffffu, my_lse, i), coef = __shfl_sync(0xffffffffu, my_coef, i); const int y = __shfl_sync(0xffffffffu, my_y, i);
    if (C.hp.tksV <= mrl::KS_MAX)
      lm::warp_row_dlogits_chunked(T + row * mrl::TN, n0, inv_t, lse, coef, y, C.tr.dlogits, r0 + row, C.hp.tksV);
    else
      lm::warp_row_dlogits(T + row * mrl::TN, n0, inv_t, lse, coef, y, C.tr.dlogits + (size_t)(r0 + row) * VP + n0);
  }
}

// ------------------------------------------------------------------------------------------------ LM-head backward
// dxpart[c][rows of m] = dlogits[rows of m, chunk c of V] emb[chunk c, :]   (NN: reduce over exactly V = 4748 x 32 ids)
__device__ __forceinline__ void t_lmb(const mrl::Ctx& C, const mrl::Batch* B, int local, char* smem) {
  const int tn = local % NT_D, cm = local / NT_D, m = cm % TTILE, c = cm / TTILE, r0 = m * mrl::TM, ntok = B->ntok;
  if (r0 >= ntok) return;
  const int nrows = min(mrl::TM, ntok - r0); int k0, kc; mrl::kchunk(V / 32, C.hp.tksV, c, k0, kc);
  lg::Args a = lg::args_default();
  if (C.hp.tksV <= mrl::KS_MAX) {
    const int kc_max = ((V / 32 + C.hp.tksV - 1) / C.hp.tksV) * 32;
    a.A = C.tr.dlogits + ((size_t)c * TROWS + r0) * kc_max; a.lda = kc_max; a.avalid = nrows;
  } else {
    a.A = C.tr.dlogits + (size_t)r0 * VP + k0; a.lda = VP; a.avalid = nrows;
  }
  a.B = C.model.emb + (size_t)k0 * D; a.ldb = D; a.bvalid = D;
  a.C = C.tr.dxpart + ((size_t)c * TROWS + r0) * DP; a.ldc = DP; a.rvalid = nrows; a.cvalid = D;
  a.M = mrl::TM; a.N = DP; a.K = kc;
  g_nn_f32(a, 0, tn, smem);
}
// dxf[rows of m, columns of tile tn below D] = sum over the tksV chunks, in chunk order
__device__ __forceinline__ void t_lmbr(const mrl::Ctx& C, const mrl::Batch* B, int local) {
  const int m = local / NT_D, tn = local % NT_D, r0 = m * mrl::TM, ntok = B->ntok;
  if (r0 >= ntok) return;
  const int nrows = min(mrl::TM, ntok - r0), c0 = tn * mrl::TN, nq = (min(D, c0 + mrl::TN) - c0) / 4, nch = C.hp.tksV;
  for (int q = threadIdx.x; q < nrows * nq; q += 256) {
    const size_t e = (size_t)(r0 + q / nq) * DP + c0 + (size_t)(q % nq) * 4;
    *reinterpret_cast<mrl::F4*>(C.tr.dxf + e) = sum_parts(C.tr.dxpart + e, (size_t)TROWS * DP, nch);
  }
}

}  // namespace gt
