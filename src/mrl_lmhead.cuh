// mrl_lmhead.cuh -- the arithmetic that the rollout head and the learner must share exactly:
// RMSNorm of a row, the LM-head tile, the reductions over a tile's logits (log-sum-exp, Gumbel-max), the merge of the
// per-tile partials of a row, and the gradient of the log-probability with respect to the logits.
// Both sides call these functions (head_sample.cuh; mid_train.cuh and grpo_loss.cuh), so for a group trained at lag 0
// the learner's log-probabilities equal the generation-time ones bit for bit (same operands, same order of operations).
//
// Conventions: z = logit * inv_temperature; vocabulary ids >= V do not exist and are skipped everywhere;
// the tile of an LM-head task is float T[TM][TN] in shared memory (lg::EPI_SMEM), tile n covers ids [n*TN, n*TN + TN).
#pragma once
#include "mrl_common.cuh"

namespace lm {
using mrl::bf16; using mrl::u64;

// ------------------------------------------------------------------------------------------------ RMSNorm of one row
// Called by all 32 lanes of one warp for one row: out = bf16((h * (1 / sqrt(mean(h^2) + eps))) * w), D elements.
// Lane l owns elements [28 l, 28 l + 28): it sums their squares in order, then the lanes are combined with an
// xor-shuffle tree. h and w must be 16-byte aligned, out 8-byte aligned.
// If bias_one, out[D] = 1.0 is written too (the q/k/v bias column of mrl::Pipe::xn). Returns 1 / rms (same in all lanes).
__device__ __forceinline__ float warp_rmsnorm_row(const float* h, const float* w, bf16* out, bool bias_one) {
  const int lane = threadIdx.x & 31;
  constexpr int PER = mrl::D / 32;                           // 28 consecutive elements per lane
  static_assert(mrl::D % 32 == 0 && PER % 4 == 0, "row split");
  mrl::F4 x[PER / 4];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < PER / 4; i++) {
    x[i] = *reinterpret_cast<const mrl::F4*>(h + lane * PER + 4 * i);
#pragma unroll
    for (int j = 0; j < 4; j++) ss = __fadd_rn(ss, __fmul_rn(x[i].v[j], x[i].v[j]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
  const float rinv = 1.f / sqrtf(__fadd_rn(ss * (1.f / mrl::D), mrl::RMS_EPS));
#pragma unroll
  for (int i = 0; i < PER / 4; i++) {
    const mrl::F4 ww = *reinterpret_cast<const mrl::F4*>(w + lane * PER + 4 * i); mrl::V4 o;
#pragma unroll
    for (int j = 0; j < 4; j++) o.v[j] = __float2bfloat16(__fmul_rn(__fmul_rn(x[i].v[j], rinv), ww.v[j]));
    *reinterpret_cast<mrl::V4*>(out + lane * PER + 4 * i) = o;
  }
  if (bias_one && lane == 0) out[mrl::D] = __float2bfloat16(1.f);
  return rinv;
}

// ------------------------------------------------------------------------------------------------ the LM-head tile
// One block: logits of rows [0, nrows) of xf against vocabulary tile `tile_n`, left in shared memory. Rows >= nrows
// and ids >= V read as zero.
__device__ __noinline__ void lmhead_tile(const bf16* xf, int ldx, int nrows, const bf16* emb, int tile_n, char* smem) {
  lg::Args a = lg::args_default();
  a.A = xf; a.lda = ldx; a.avalid = nrows; a.B = emb; a.ldb = mrl::D; a.bvalid = mrl::V; a.M = mrl::TM; a.N = mrl::VP; a.K = mrl::D;
  lg::gemm_tile<lg::NT, lg::EPI_SMEM>(a, 0, tile_n, smem);
}

// ------------------------------------------------------------------------------------------------ reductions over one tile row
// Each is called by all 32 lanes of one warp with the same arguments. Lane l reads columns l, l+32, ..., l+224 (no bank
// conflicts) in that order, then the lanes are combined with a fixed xor-shuffle tree.
struct RowLse { float lmax, lsum; };                         // max of z over the tile's valid ids; sum of exp(z - lmax)
__device__ __forceinline__ RowLse warp_row_lse(const float* Trow, int n0, float inv_t) {
  const int lane = threadIdx.x & 31;
  float z[8]; float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < 8; k++) { const int c = lane + 32 * k; z[k] = (n0 + c < mrl::V) ? __fmul_rn(Trow[c], inv_t) : -INFINITY; m = fmaxf(m, z[k]); }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) s = __fadd_rn(s, expf(__fadd_rn(z[k], -m)));     // exp(-inf) = 0 for the ids that do not exist
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
  RowLse r; r.lmax = m; r.lsum = s; return r;
}

// Order of sampling candidates: the higher score wins, equal scores go to the lower id, NaN never wins.
// If every score of a row is NaN, the Gumbel reductions below return their start value idx = NO_WINNER (0x7fffffff),
// which is not a vocabulary id. A caller that stores or indexes with the id must test idx >= V first
// (head_sample.cuh ends the row with <|endoftext|> and stores the NaN log-probability).
constexpr int NO_WINNER = 0x7fffffff;
__device__ __forceinline__ bool better(float sa, int ia, float sb, int ib) { return sa > sb || (sa == sb && ia < ib); }

struct RowGumbel { float score; int idx; float z; };         // best z + noise, its vocabulary id, its z
// warp_row_lse plus the Gumbel-max candidate of the tile row (best z + noise, its id, its z) in one pass over Trow.
// The (lmax, lsum) it returns are bit for bit those of warp_row_lse, which the learner calls: the learner needs no
// sample, but must reproduce the head's log-probabilities exactly.
__device__ __forceinline__ void warp_row_lse_gumbel(const float* Trow, int n0, float inv_t, u64 key, RowLse& rl, RowGumbel& rg) {
  const int lane = threadIdx.x & 31;
  float z[8]; float m = -INFINITY;
  RowGumbel b; b.score = -INFINITY; b.idx = NO_WINNER; b.z = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int c = lane + 32 * k, v = n0 + c;
    if (v < mrl::V) {
      const float zv = __fmul_rn(Trow[c], inv_t);
      z[k] = zv; m = fmaxf(m, zv);
      const float sc = __fadd_rn(zv, mrl::gumbel(key, v));
      if (better(sc, v, b.score, b.idx)) { b.score = sc; b.idx = v; b.z = zv; }
    } else z[k] = -INFINITY;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) s = __fadd_rn(s, expf(__fadd_rn(z[k], -m)));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float os = __shfl_xor_sync(0xffffffffu, b.score, o), oz = __shfl_xor_sync(0xffffffffu, b.z, o);
    const int oi = __shfl_xor_sync(0xffffffffu, b.idx, o);
    if (better(os, oi, b.score, b.idx)) { b.score = os; b.idx = oi; b.z = oz; }
  }
  rl.lmax = m; rl.lsum = s; rg = b;
}

__device__ __forceinline__ void lse_join(float& m, float& s, float m2, float s2) {       // (m, s) <- (m, s) joined with (m2, s2)
  const float mm = fmaxf(m, m2);
  s = __fadd_rn(__fmul_rn(s, expf(__fadd_rn(m, -mm))), __fmul_rn(s2, expf(__fadd_rn(m2, -mm)))); m = mm;
}

// z of vocabulary id y if it lies in this tile (valid only when the function returns true; any lane may call it)
__device__ __forceinline__ bool row_logit_of(const float* Trow, int n0, float inv_t, int y, float& z) {
  if (y < n0 || y >= n0 + mrl::TN) return false;
  z = __fmul_rn(Trow[y - n0], inv_t); return true;
}

// dlogits of one tile row: out[c] = bf16(coef * inv_t * ([id == y] - exp(z - lse))), zero for ids that do not exist.
// Called by all 32 lanes of one warp; lane l writes columns 8l .. 8l+7 with one 16-byte store.
__device__ __forceinline__ void warp_row_dlogits(const float* Trow, int n0, float inv_t, float lse, float coef, int y, bf16* out) {
  const int lane = threadIdx.x & 31; mrl::V8 o; const float g = __fmul_rn(coef, inv_t);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int c = 8 * lane + j, v = n0 + c;
    float d = 0.f;
    if (v < mrl::V) { const float p = expf(__fadd_rn(__fmul_rn(Trow[c], inv_t), -lse)); d = __fmul_rn(g, (v == y ? 1.f : 0.f) - p); }
    o.v[j] = __float2bfloat16(d);
  }
  *reinterpret_cast<mrl::V8*>(out + 8 * lane) = o;
}

// The same elements as warp_row_dlogits, stored chunk-major for M_LMB: K chunk c of the vocabulary (mrl::kchunk over
// V / 32 steps, tksV chunks) is a [TROWS, kc_max] matrix with kc_max = 32 * ceil((V / 32) / tksV), at
// dlogits_base + c * TROWS * kc_max. The GEMM then reads rows kc_max apart instead of VP (304 KB in bf16).
// row_abs is the packed row; ids >= V are not stored. Lane l still writes columns 8l .. 8l+7 of the tile row.
__device__ __forceinline__ void warp_row_dlogits_chunked(const float* Trow, int n0, float inv_t, float lse, float coef, int y,
                                                         bf16* dlogits_base, int row_abs, int tksV) {
  const int lane = threadIdx.x & 31; mrl::V8 o; const float g = __fmul_rn(coef, inv_t);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int c = 8 * lane + j, v = n0 + c;
    float d = 0.f;
    if (v < mrl::V) { const float p = expf(__fadd_rn(__fmul_rn(Trow[c], inv_t), -lse)); d = __fmul_rn(g, (v == y ? 1.f : 0.f) - p); }
    o.v[j] = __float2bfloat16(d);
  }
  const int kt = (n0 >> 5) + (lane >> 2), nk32 = mrl::V / 32;
  if (kt < nk32) {
    const int c = (int)(((long long)(kt + 1) * tksV - 1) / nk32);
    const int a_c = (int)((long long)c * nk32 / tksV);
    const int kc_max = ((nk32 + tksV - 1) / tksV) * 32;
    const size_t off = ((size_t)c * mrl::TROWS + row_abs) * kc_max + (kt - a_c) * 32 + (lane & 3) * 8;
    *reinterpret_cast<mrl::V8*>(dlogits_base + off) = o;
  }
}

// ------------------------------------------------------------------------------------------------ merging the tiles of a row
// Called by all 256 threads of a block for one row. Thread t merges tiles t, t+256, t+512 in that order, the lanes of a
// warp are combined with a fixed xor-shuffle tree, and the 8 warps are merged in order through `scratch`, shared
// memory that nothing else uses meanwhile (16 floats for block_merge_lse, 24 for block_merge_gumbel). The result is
// returned to every thread.
// first: pointer to {lmax, lsum} (two consecutive floats, 4-byte aligned) of tile 0 of this row; stride: bytes between
// consecutive tiles. The pair is read as two scalar loads: inside mrl::SampPart it sits at offset 12, so an 8-byte
// load (float2) would fault.
__device__ __forceinline__ void block_merge_lse(const char* first, size_t stride, float* scratch, float& lmax, float& lsum) {
  const int tid = threadIdx.x;
  const float* p0 = reinterpret_cast<const float*>(first + (size_t)tid * stride);
  float m = p0[0], s = p0[1];
#pragma unroll
  for (int k = 1; k < 3; k++) {
    const int n = tid + 256 * k;
    if (n < mrl::NT_V) { const float* p = reinterpret_cast<const float*>(first + (size_t)n * stride); lse_join(m, s, p[0], p[1]); }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) { const float om = __shfl_xor_sync(0xffffffffu, m, o), os = __shfl_xor_sync(0xffffffffu, s, o); lse_join(m, s, om, os); }
  if ((tid & 31) == 0) { scratch[2 * (tid >> 5)] = m; scratch[2 * (tid >> 5) + 1] = s; }
  __syncthreads();
  m = scratch[0]; s = scratch[1];
#pragma unroll
  for (int w = 1; w < 8; w++) lse_join(m, s, scratch[2 * w], scratch[2 * w + 1]);
  __syncthreads();
  lmax = m; lsum = s;
}
// first: the mrl::SampPart of tile 0 of this row; stride: bytes between consecutive tiles.
// idx is NO_WINNER (not a vocabulary id) if no candidate has a comparable score; see the note at `better`.
__device__ __forceinline__ void block_merge_gumbel(const char* first, size_t stride, float* scratch, float& score, int& idx, float& z) {
  const int tid = threadIdx.x;
  float bs = -INFINITY, bz = 0.f; int bi = NO_WINNER;
#pragma unroll
  for (int k = 0; k < 3; k++) {
    const int n = tid + 256 * k;
    if (n < mrl::NT_V) {
      const mrl::SampPart* p = reinterpret_cast<const mrl::SampPart*>(first + (size_t)n * stride);
      const float ps = p->gmax, pz = p->glogit; const int pi = p->gidx;
      if (better(ps, pi, bs, bi)) { bs = ps; bi = pi; bz = pz; }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float os = __shfl_xor_sync(0xffffffffu, bs, o), oz = __shfl_xor_sync(0xffffffffu, bz, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (better(os, oi, bs, bi)) { bs = os; bi = oi; bz = oz; }
  }
  int* si = reinterpret_cast<int*>(scratch);
  if ((tid & 31) == 0) { scratch[3 * (tid >> 5)] = bs; scratch[3 * (tid >> 5) + 1] = bz; si[3 * (tid >> 5) + 2] = bi; }
  __syncthreads();
  bs = scratch[0]; bz = scratch[1]; bi = si[2];
#pragma unroll
  for (int w = 1; w < 8; w++) if (better(scratch[3 * w], si[3 * w + 2], bs, bi)) { bs = scratch[3 * w]; bz = scratch[3 * w + 1]; bi = si[3 * w + 2]; }
  __syncthreads();
  score = bs; idx = bi; z = bz;
}

// log-probability of the token whose z is z_tok, given the merged (lmax, lsum) of its row
__device__ __forceinline__ float logp_from(float z_tok, float lmax, float lsum) { return __fadd_rn(z_tok, -__fadd_rn(lmax, logf(lsum))); }
__device__ __forceinline__ float lse_from(float lmax, float lsum) { return __fadd_rn(lmax, logf(lsum)); }

}  // namespace lm
