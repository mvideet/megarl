// lgemm.cuh -- the bf16 tensor-core GEMM tile of MegaRL (lg::gemm_tile) and its argument contract (lg::Args).
// Derived from open-flapping-airplanes llm/gemm/gemm.cuh (MIT, (c) 2026 Jerry Zhu).
//
// One task = one 64 x 256 output tile computed by all 256 threads of a block; mrl_common.cuh derives TM and TN from
// it. No blockIdx/gridDim use, no atomics, and a fixed reduction order over K, so results are deterministic and a
// row's result does not depend on the other rows of the tile. The tile itself is wg::gemm_64x256 (mrl_wgmma.cuh,
// Hopper wgmma). Additions to the original contract:
//   - operands can declare how many of their M (for A) / N (for B) indices exist: the rest is zero-filled in shared
//     memory with no global load (avalid, bvalid), for both storage orders;
//   - outputs can be limited to the first rvalid rows / cvalid columns (padding is then never stored);
//   - EPI_SMEM leaves the fp32 tile in shared memory for the caller to post-process (the LM head's row reductions).
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace lg {

typedef __nv_bfloat16 bf16;

// Layouts. Output C[M,N] (row-major) = sum_k Aop(m,k) * Bop(k,n); all matrices row-major with explicit ld.
// MegaRL's weights are stored [out, in]:
//   NT: A stored [M,K] (lda>=K),  B stored [N,K] (ldb>=K)       forward linear Y = X.W^T, logits = x.emb^T
//   NN: A stored [M,K],           B stored [K,N] (ldb>=N)       dX = dY.W
//   TN: A stored [K,M] (lda>=M),  B stored [K,N]                dW = dY^T.X  (reduce over the token dim K)
enum Layout { NN = 0, NT = 1, TN = 2 };

enum Epi {
  EPI_F32  = 0,   // C(fp32): accumulate ? C += acc : C = acc
  EPI_SMEM = 1    // nothing is stored to global memory: the tile is left in shared memory as float T[BM][BN]
                  // (T[r * BN + c], tile-local r, c) at the base of `smem`; rvalid / cvalid are ignored
};

constexpr int BM = 64, BN = 256, BK = 32, STAGES = 4;
constexpr int SMEM_BYTES = STAGES * (BM * BK + BN * BK) * 2;   // 81,920 B of dynamic shared memory per block
constexpr int SMEM_TILE_BYTES = BM * BN * 4;                    // what EPI_SMEM leaves at the base of smem (65,536 B)
static_assert(SMEM_TILE_BYTES <= SMEM_BYTES, "EPI_SMEM tile must fit in the pipeline's shared memory");

struct Args {
  const bf16* A;  // see Layout
  const bf16* B;
  float* C;       // EPI_F32 output; unused for EPI_SMEM
  int lda, ldb, ldc;  // leading dims, in elements, of the respective row-major arrays
  int M, N, K;    // GEMM dims (informational except K): the tile reduces over K, a multiple of 32, K >= 32
  int accumulate; // EPI_F32 only: 1 -> C += acc, 0 -> C = acc
  int avalid;     // A has M-indices [0, avalid): rows of A[M,K] (NN, NT) or columns of A[K,M] (TN); the rest reads as zero.
  int bvalid;     // B has N-indices [0, bvalid): rows of B[N,K] (NT) or columns of B[K,N] (NN, TN); the rest reads as zero.
                  // In the column forms (A in TN, B in NN and TN) the count must be a multiple of 8.
  int rvalid, cvalid;   // output rows / columns >= these are not stored, so the caller decides what
                        // C holds there. cvalid must be even: columns are stored in pairs.
};
// All-valid defaults; set A, B, C, the leading dims and K afterwards.
__host__ __device__ inline Args args_default() {
  Args a; a.A = nullptr; a.B = nullptr; a.C = nullptr; a.lda = a.ldb = a.ldc = 8; a.M = a.N = a.K = 0; a.accumulate = 0;
  a.avalid = a.bvalid = a.rvalid = a.cvalid = 0x7fffffff; return a;
}

// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void st_zero16(uint32_t dst) {
  asm volatile("st.shared.v2.u64 [%0], {%1, %1};\n" ::"l"((unsigned long long)dst), "l"(0ull) : "memory");
}
// Thread stores of two adjacent output columns (c, c + 1) of row r.
template <int EPI>
__device__ __forceinline__ void epi_store(const Args& p, int r, int c, float v0, float v1) {
  static_assert(EPI == EPI_F32, "EPI_SMEM tiles are not stored through epi_store");
  if (r >= p.rvalid || c >= p.cvalid) return;
  float2* dst = reinterpret_cast<float2*>(p.C + (size_t)r * p.ldc + c);
  if (p.accumulate) { float2 o = *dst; v0 += o.x; v1 += o.y; }
  *dst = make_float2(v0, v1);
}

}  // namespace lg

#include "mrl_wgmma.cuh"

namespace lg {

// gemm_tile: compute output tile (tile_m, tile_n) of C[M,N]. Called by all 256 threads of a block.
// smem: dynamic shared memory base, SMEM_BYTES bytes, 1024-byte aligned (the swizzled operand layouts need it; see
// wg::check_smem_base). No thread touches smem after its last barrier, so the caller may reuse it as soon as the call
// returns; with EPI_SMEM the tile is complete and visible to all threads by then.
template <int LAYOUT, int EPI>
__device__ __forceinline__ void gemm_tile(const Args& p, int tile_m, int tile_n, char* smem) {
  wg::gemm_64x256<LAYOUT, EPI, STAGES>(p, tile_m, tile_n, smem);
}

}  // namespace lg
