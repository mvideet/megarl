// mrl_wgmma.cuh -- Hopper (sm_90a) warpgroup tensor-core GEMM tiles (wgmma.mma_async) behind lg::gemm_tile.
//
// Included by lgemm.cuh after lg::Args / lg::epi_store are defined; include lgemm.cuh (or mrl_common.cuh) first, not
// this file. Deterministic bf16 -> fp32 tiles for 256-thread blocks (2 warpgroups of 128 threads,
// __launch_bounds__(256, 2)), each using 81,920 B of dynamic shared memory:
//   wg::gemm_64x256<LAYOUT, EPI>: 64 x 256 output tile; warpgroup 0 computes columns 0..127, warpgroup 1 columns
//     128..255. A fixed K order per output element, so the bits do not depend on scheduling.
//   wg::gemm_64x256_masked<EPI>: row-masked NT tile for the per-version cohorts of the trained layer: only rows
//     with bit r set in row_mask are loaded and stored, and a row's result does not depend on which other rows are
//     in the mask.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include "lgemm.cuh"

namespace wg {
using bf16 = __nv_bfloat16;

// Hopper WGMMA shared-memory matrix descriptor (PTX ISA 8.x):
//   bits  0..13: start_address >> 4
//   bits 16..29: leading_dimension_byte_offset >> 4
//   bits 32..45: stride_dimension_byte_offset >> 4
//   bits 62..63: swizzle mode (0 = none, 1 = 128B, 2 = 64B, 3 = 32B)
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr, uint32_t lbo, uint32_t sbo, uint64_t swz) {
  return (uint64_t)((smem_addr >> 4) & 0x3fff)
       | ((uint64_t)((lbo >> 4) & 0x3fff) << 16)
       | ((uint64_t)((sbo >> 4) & 0x3fff) << 32)
       | (swz << 62);
}

// Swizzle modes 1 and 2 are defined on absolute shared-memory addresses, so every operand tile must start on a
// 1024-byte boundary. Kernels declare their dynamic shared memory __align__(1024) and check it once at entry.
__device__ __forceinline__ void check_smem_base(const void* p) {
  if (threadIdx.x == 0 && (__cvta_generic_to_shared(p) & 1023u)) { printf("wg: dynamic shared memory is not 1024-byte aligned\n"); __trap(); }
}

__device__ __forceinline__ void fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
#endif
}

template <int TRANS_A, int TRANS_B>
__device__ __forceinline__ void mma_m64n128k16(float (&d)[16][4], uint64_t desc_a, uint64_t desc_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
    "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
    "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
    "%64, %65, p, 1, 1, %66, %67;\n"
    "}\n"
    : "+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),
      "+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),
      "+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),
      "+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),
      "+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),
      "+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),
      "+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),
      "+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]),
      "+f"(d[8][0]),"+f"(d[8][1]),"+f"(d[8][2]),"+f"(d[8][3]),
      "+f"(d[9][0]),"+f"(d[9][1]),"+f"(d[9][2]),"+f"(d[9][3]),
      "+f"(d[10][0]),"+f"(d[10][1]),"+f"(d[10][2]),"+f"(d[10][3]),
      "+f"(d[11][0]),"+f"(d[11][1]),"+f"(d[11][2]),"+f"(d[11][3]),
      "+f"(d[12][0]),"+f"(d[12][1]),"+f"(d[12][2]),"+f"(d[12][3]),
      "+f"(d[13][0]),"+f"(d[13][1]),"+f"(d[13][2]),"+f"(d[13][3]),
      "+f"(d[14][0]),"+f"(d[14][1]),"+f"(d[14][2]),"+f"(d[14][3]),
      "+f"(d[15][0]),"+f"(d[15][1]),"+f"(d[15][2]),"+f"(d[15][3])
    : "l"(desc_a), "l"(desc_b), "n"(TRANS_A), "n"(TRANS_B)
  );
#endif
}

// K-contiguous operand, [R][32] with 64-byte rows in the 64B swizzle.
// Rows >= valid are zero-filled without a global load.
template <int R>
__device__ __forceinline__ void load_kc_64b(uint32_t sbase, const bf16* g, int ld, int mn0, int k0, int tid, int valid) {
#pragma unroll
  for (int i = 0; i < R * 4 / 256; i++) {
    const int c = tid + i * 256, row = c >> 2, kc = c & 3;
    const uint32_t dst = sbase + row * 64 + ((kc ^ ((row >> 1) & 3)) << 4);
    if (mn0 + row < valid) lg::cp_async16(dst, g + (size_t)(mn0 + row) * ld + k0 + kc * 8);
    else lg::st_zero16(dst);
  }
}

__device__ __forceinline__ void load_kc_64b_masked(uint32_t sbase, const bf16* g, int ld, int k0, int tid, uint64_t row_mask) {
  const int row = tid >> 2, kc = tid & 3;
  const uint32_t dst = sbase + row * 64 + ((kc ^ ((row >> 1) & 3)) << 4);
  if ((row_mask >> row) & 1ull) lg::cp_async16(dst, g + (size_t)row * ld + k0 + kc * 8);
  else lg::st_zero16(dst);
}

// MN-contiguous operand, [32][R] in the 128B swizzle: per 64 MN indices one block of 32 k-rows x 128 B (4,096 B).
// An 8-index chunk is loaded whole if it starts below valid and zero-filled otherwise: valid must be a multiple of 8.
template <int R>
__device__ __forceinline__ void load_mnc_128b(uint32_t sbase, const bf16* g, int ld, int mn0, int k0, int tid, int valid) {
  constexpr int CH = R / 8;
#pragma unroll
  for (int i = 0; i < 32 * CH / 256; i++) {
    const int c = tid + i * 256, k = c / CH, mc = c % CH;
    const int blk64 = mc >> 3, mc_in = mc & 7;
    const uint32_t dst = sbase + blk64 * 4096 + k * 128 + ((mc_in ^ (k & 7)) << 4);
    if (mn0 + mc * 8 < valid) lg::cp_async16(dst, g + (size_t)(k0 + k) * ld + mn0 + mc * 8);
    else lg::st_zero16(dst);
  }
}

// ------------------------------------------------------------------------------------------------ 64 x 256 WGMMA tile
template <int LAYOUT, int EPI, int STAGES = 4>
__device__ __forceinline__ void gemm_64x256(const lg::Args& p_in, int tile_m, int tile_n, char* smem) {
  const lg::Args p = p_in;
  constexpr bool A_KC = (LAYOUT != lg::TN);
  constexpr bool B_KC = (LAYOUT == lg::NT);
  constexpr int A_BYTES = 64 * 32 * 2;          // 4,096 B
  constexpr int B_BYTES = 256 * 32 * 2;         // 16,384 B
  constexpr int STAGE_BYTES = A_BYTES + B_BYTES; // 20,480 B * 4 = 81,920 B
  const int tid = threadIdx.x, wg = tid >> 7, tid_wg = tid & 127;
  const int warp_wg = tid_wg >> 5, lane = tid_wg & 31;
  const int m0 = tile_m * 64, n0 = tile_n * 256;
  const uint32_t sbase = lg::smem_u32(smem);
  const int KT = p.K / 32;

  auto load_stage = [&](int kt) {
    const uint32_t s = sbase + (kt % STAGES) * STAGE_BYTES;
    const int k0 = kt * 32;
    if (A_KC) load_kc_64b<64>(s, p.A, p.lda, m0, k0, tid, p.avalid);
    else      load_mnc_128b<64>(s, p.A, p.lda, m0, k0, tid, p.avalid);
    if (B_KC) load_kc_64b<256>(s + A_BYTES, p.B, p.ldb, n0, k0, tid, p.bvalid);
    else      load_mnc_128b<256>(s + A_BYTES, p.B, p.ldb, n0, k0, tid, p.bvalid);
  };

  float acc[16][4];
#pragma unroll
  for (int i = 0; i < 16; i++)
#pragma unroll
    for (int e = 0; e < 4; e++) acc[i][e] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < KT) load_stage(s); lg::cp_commit(); }

  for (int kt = 0; kt < KT; kt++) {
    lg::cp_wait<STAGES - 2>();
    __syncthreads();
    const uint32_t sA = sbase + (kt % STAGES) * STAGE_BYTES;
    const uint32_t sB = sA + A_BYTES + wg * (B_BYTES / 2);
    fence();
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      const uint64_t da = A_KC ? make_desc(sA + ks * 32, 16, 512, 2) : make_desc(sA + ks * 2048, 4096, 1024, 1);
      const uint64_t db = B_KC ? make_desc(sB + ks * 32, 16, 512, 2) : make_desc(sB + ks * 2048, 4096, 1024, 1);
      mma_m64n128k16<A_KC ? 0 : 1, B_KC ? 0 : 1>(acc, da, db);
    }
    commit();
    const int nk = kt + STAGES - 1;
    if (nk < KT) load_stage(nk);
    lg::cp_commit();
    wait<1>();
  }
  wait<0>();
  lg::cp_wait<0>();
  __syncthreads();

  const int g = lane >> 2, t = lane & 3;
  if constexpr (EPI == lg::EPI_SMEM) {
    float* T = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int ns = 0; ns < 16; ns++) {
      const int rl = warp_wg * 16 + g, cl = wg * 128 + ns * 8 + 2 * t;
      *reinterpret_cast<float2*>(T + rl * 256 + cl) = make_float2(acc[ns][0], acc[ns][1]);
      *reinterpret_cast<float2*>(T + (rl + 8) * 256 + cl) = make_float2(acc[ns][2], acc[ns][3]);
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int ns = 0; ns < 16; ns++) {
      const int r = m0 + warp_wg * 16 + g, c = n0 + wg * 128 + ns * 8 + 2 * t;
      lg::epi_store<EPI>(p, r, c, acc[ns][0], acc[ns][1]);
      lg::epi_store<EPI>(p, r + 8, c, acc[ns][2], acc[ns][3]);
    }
  }
}

// ------------------------------------------------------------------------------------------------ 64 x 256 row-masked NT tile
// A is rows 0..63 of p.A (there is no tile_m) and p.avalid is ignored: row r takes part iff bit r of row_mask is set.
// B, bvalid, rvalid and cvalid as for gemm_64x256<lg::NT, EPI>.
template <int EPI, int STAGES = 4>
__device__ __forceinline__ void gemm_64x256_masked(const lg::Args& p_in, int tile_n, uint64_t row_mask, char* smem) {
  const lg::Args p = p_in;
  constexpr int A_BYTES = 64 * 32 * 2, B_BYTES = 256 * 32 * 2, STAGE_BYTES = A_BYTES + B_BYTES;
  const int tid = threadIdx.x, wg = tid >> 7, tid_wg = tid & 127;
  const int warp_wg = tid_wg >> 5, lane = tid_wg & 31;
  const int n0 = tile_n * 256;
  const uint32_t sbase = lg::smem_u32(smem);
  const int KT = p.K / 32;

  auto load_stage = [&](int kt) {
    const uint32_t s = sbase + (kt % STAGES) * STAGE_BYTES;
    const int k0 = kt * 32;
    load_kc_64b_masked(s, p.A, p.lda, k0, tid, row_mask);
    load_kc_64b<256>(s + A_BYTES, p.B, p.ldb, n0, k0, tid, p.bvalid);
  };

  float acc[16][4];
#pragma unroll
  for (int i = 0; i < 16; i++)
#pragma unroll
    for (int e = 0; e < 4; e++) acc[i][e] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < KT) load_stage(s); lg::cp_commit(); }

  for (int kt = 0; kt < KT; kt++) {
    lg::cp_wait<STAGES - 2>();
    __syncthreads();
    const uint32_t sA = sbase + (kt % STAGES) * STAGE_BYTES;
    const uint32_t sB = sA + A_BYTES + wg * (B_BYTES / 2);
    fence();
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      mma_m64n128k16<0, 0>(acc, make_desc(sA + ks * 32, 16, 512, 2), make_desc(sB + ks * 32, 16, 512, 2));
    }
    commit();
    const int nk = kt + STAGES - 1;
    if (nk < KT) load_stage(nk);
    lg::cp_commit();
    wait<1>();
  }
  wait<0>();
  lg::cp_wait<0>();
  __syncthreads();

  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int ns = 0; ns < 16; ns++) {
    const int r = warp_wg * 16 + g, c = n0 + wg * 128 + ns * 8 + 2 * t;
    if ((row_mask >> r) & 1ull) lg::epi_store<EPI>(p, r, c, acc[ns][0], acc[ns][1]);
    if ((row_mask >> (r + 8)) & 1ull) lg::epi_store<EPI>(p, r + 8, c, acc[ns][2], acc[ns][3]);
  }
}

}  // namespace wg
