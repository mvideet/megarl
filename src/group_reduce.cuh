// group_reduce.cuh -- group-relative advantages of the GSZ rewards of one prompt group (computed in R_ADVANCE):
//
//   mean  = (r_0 + ... + r_{G-1}) / G                        G = mrl::GSZ
//   std   = sqrt( sum_i (r_i - mean)^2 / (G - 1) )           unbiased (divisor G - 1), as in TRL's GRPOTrainer
//   adv_i = (r_i - mean) / (std + eps)                       eps = hp.adv_eps (default 1e-4), eps >= 0
//
// A group whose rewards are all equal carries no learning signal: adv = 0 and the function returns false, and the caller
// drops the group (state DROPPED_ZEROVAR) instead of training on it. "Equal" is tested on the rewards themselves
// (r_i == r_0 for all i), not on std == 0.
//
// Arithmetic. The rewards are float32; everything in between is float64, every operation one correctly rounded IEEE-754
// operation in a fixed order (sums in index order starting from 0, the squares rounded on their own so that no compiler
// fuses them into the following addition), and the quotient is rounded once to float32. Therefore
//   * host and device return the same bits for the same input;
//   * for 0/1 rewards (and G = 4) the mean, the deviations and the sum of squares are exact, so the result is bit for
//     bit the float32 rounding of the formula evaluated in float64 (as numpy does it): one 1 among four gives
//     +1.4997001 / -0.4999000, two give +-0.8658754, three +0.4999000 / -1.4997001 (bits 3fbff62c, 3efff2e5,
//     3f5daa03 and their negatives). With eps = 1e-4 the same bits come out whether eps is taken as the float32 the
//     kernel holds or as the float64 literal.
//     Plain float32 arithmetic happens to give the same bits for 0/1 rewards at eps = 1e-4, but not for every eps
//     (at 1e-3 or 1e-8 some values differ by one ulp);
//   * for arbitrary finite rewards the result is within half a float32 ulp, plus 2^-47 max|r| / (std + eps), of any
//     other float64 evaluation of the formula. float32 arithmetic would lose (r - mean) to cancellation as soon as the
//     rewards are large compared with their spread.
//
// Outside the contract (rw::score_group produces only 0 and 1, and eps is a positive constant): the function never
// returns a NaN or an infinity. Rewards that are not finite, and an advantage that would not be finite (possible only
// with a negative or NaN eps), are treated like a zero-variance group: adv = 0, return false.
//
// A few dozen scalar operations by one thread (thread 0 of the control task); no shared memory, no barrier. reward and
// adv need 4-byte alignment only: they are accessed one float at a time through volatile lvalues, which the compiler
// cannot fuse into 8- or 16-byte vector accesses (see reward.cuh). All rewards are read before the first advantage is
// written, so adv may be the same array as reward.
#pragma once
#include <cfloat>
#include <cmath>
#include "mrl_common.cuh"

namespace gr {

static_assert(mrl::GSZ >= 2, "the unbiased standard deviation needs at least two rewards");

// a * b rounded to float64 on its own, never contracted with a following addition into a fused multiply-add
// (nvcc contracts by default on the device; a host compiler may, depending on its flags and target).
__host__ __device__ inline double mul_rn(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dmul_rn(a, b);
#else
  volatile double p = a * b; return p;
#endif
}

// reward[0..GSZ) -> adv[0..GSZ). Returns true if the group is usable: its rewards are finite and not all equal
// (then every advantage is finite); otherwise adv = 0 and the result is false.
__host__ __device__ inline bool advantages(const float* reward, float* adv, float eps) {
  const volatile float* vr = reward; volatile float* va = adv;
  float r[mrl::GSZ], a[mrl::GSZ]; bool equal = true, finite = true;
  for (int i = 0; i < mrl::GSZ; i++) {
    r[i] = vr[i]; a[i] = 0.f;
    equal = equal && (r[i] == r[0]);
    finite = finite && (fabsf(r[i]) <= FLT_MAX);               // false for NaN and for +-inf
  }
  if (!equal && finite) {
    double sum = 0.0;
    for (int i = 0; i < mrl::GSZ; i++) sum += (double)r[i];
    const double mean = sum / (double)mrl::GSZ;
    double d[mrl::GSZ], ss = 0.0;
    for (int i = 0; i < mrl::GSZ; i++) { d[i] = (double)r[i] - mean; ss += mul_rn(d[i], d[i]); }
    const double den = sqrt(ss / (double)(mrl::GSZ - 1)) + (double)eps;
    for (int i = 0; i < mrl::GSZ; i++) { a[i] = (float)(d[i] / den); finite = finite && (fabsf(a[i]) <= FLT_MAX); }
  }
  const bool usable = !equal && finite;
  for (int i = 0; i < mrl::GSZ; i++) va[i] = usable ? a[i] : 0.f;
  return usable;
}

}  // namespace gr
