// reward.cuh -- the reward of the GSZ answers of one finished group, computed in the rollout control task R_ADVANCE.
//
// Rule. y[0..n-1] are the generated tokens of one answer, the end token included; key (0..3) is the correct option.
//   verifier_mode 0 ("Answer: X"):   the largest i with 2 <= i < n, tok_letter[y[i]] >= 0, tok_colon[y[i-1]] == 1 and
//                                    tok_answer[y[i-2]] == 1 decides: the reward is 1 if such an i exists and
//                                    tok_letter[y[i]] == key, else 0. End tokens have no special role.
//   verifier_mode 1 ("first letter"): the reward is 1 if n >= 1 and tok_letter[y[0]] == key, else 0.
//
// Inputs the rule does not cover (none can occur in a run):
//   * a token id outside [0, V) inside an answer has no class: it is never a letter, a colon or "Answer", and it never
//     indexes a table;
//   * a key outside 0..3 equals no letter of a valid table (reward 0);
//   * len < 0 counts as 0 and len > MAXGEN as MAXGEN (a sequence has MAXGEN token slots);
//   * a verifier_mode other than 0 or 1 gives reward 0.
// Table bytes are taken literally: a letter is any value >= 0, a colon / "Answer" token is the value 1 exactly.
//
// The Scorer in megarl.cu applies hp.require_eos on top of score_group.
#pragma once
#include "mrl_common.cuh"

namespace rw {

// score_group uses bytes [0, SMEM_BYTES) of the `smem` it is given: four 32-bit masks per warp.
constexpr int SMEM_BYTES = 128;
static_assert(mrl::GSZ * mrl::MAXGEN == 256 && mrl::MAXGEN == 64,
              "score_group: one thread per token slot, one sequence = two warps = one 64-bit mask");

// Rewards (0.f or 1.f) of the GSZ sequences of one group into out[0..GSZ).
//
// Calling convention: called by all 256 threads of a block, under block-uniform control flow (the function contains two
// block barriers and warp ballots). It declares no shared memory of its own, never uses blockIdx and never waits.
//   P       prompt store: the three token tables (constant during the launch) and verifier_mode
//   tokens  [GSZ][MAXGEN] token ids of the group (mrl::Store::tokens of its buffer). The slots at index >= len of a
//           sequence are stale and are never read.
//   len     [GSZ] generated tokens per sequence, end token included (mrl::GroupRec::len); read with scalar loads
//   key     correct option of the group's prompt (mrl::GroupRec::key)
//   out     [GSZ] floats, global or shared memory, 4-byte aligned. Written by thread 0 only, after its last read of
//           the scratch (so out may even lie inside the scratch), then fenced; the function returns after a block
//           barrier, so every thread of the block may read out[] right after the call. out may be the reward[] of the
//           group's record itself: then the caller needs neither a copy nor a barrier of its own for the rewards.
//   smem    4-byte aligned shared memory; bytes [0, SMEM_BYTES) are overwritten, nothing else is touched. What they
//           hold at entry does not matter.
// Precondition on the scratch (the caller's side of the contract; the function cannot check it). The first thread of
// each warp stores its warp's four masks as soon as the 32 threads of that warp have voted, i.e. before the
// function's first block barrier: warp 0 writes bytes [0, 16) while threads of other warps may not have entered the
// function yet. So when the first thread of the block calls, [smem, smem + SMEM_BYTES) must hold nothing that any
// thread of the block has still to read. A caller that hands words to its threads through these bytes
// (ctl::roll_advance keeps the command of its round in bytes 0..11 of the block's scratch, and megarl.cu's Scorer
// passes that same pointer) needs a block barrier between its threads' last read of them and the call
// (rl_control.cuh: "Second barrier"), or must pass a scratch that holds no live data (any other 128 bytes of the block's
// dynamic shared memory). "Every thread copies the words into registers before it calls" is not enough without that
// barrier: the threads of a block do not leave the caller's previous barrier at the same instant, so a late thread
// would load a mask where it expects its command.
// The scratch words and out[] are accessed through volatile lvalues, one 4-byte access each. Without that nvcc can fuse
// the neighbouring loads into ld.shared.v2 / v4 and, when the pointer is a select between an aligned and an unaligned
// place, get the alignment wrong (a misaligned-address fault).
//
// How: thread 64*j + t owns token slot t of sequence j. If t < len[j] it loads its token and that token's three table
// bytes. Warp ballots turn the classes into bit masks with bit t = token t: L (a letter), E (the letter that equals
// key), C (a colon), A (the word "Answer"). Slots at or beyond len contribute zero bits. Then
//   mode 0:  match = L & (C << 1) & (A << 2)      bit i set <=> letter at i, colon at i-1, "Answer" at i-2 (so i >= 2)
//   mode 1:  match = L & 1                         (only the threads of slot 0 look at anything)
// and the reward is E at the highest set bit of match: the largest i of the rule.
// Cost: per thread at most three dependent global loads (len, token, table bytes).
__device__ __forceinline__ void score_group(const mrl::Prompts& P, const int* tokens /*[GSZ][MAXGEN]*/, const int* len, int key, float* out, char* smem) {
  const int tid = threadIdx.x, seq = tid >> 6, t = tid & 63, warp = tid >> 5;
  const int mode = P.verifier_mode;
  bool is_l = false, is_e = false, is_c = false, is_a = false;
  if (mode == 0 || (mode == 1 && t == 0)) {                  // mode 1 needs slot 0 only; an unknown mode needs nothing
    const int n = len[seq];
    if (t < n) {                                             // t <= 63, so a len beyond MAXGEN acts as MAXGEN
      const int y = tokens[seq * mrl::MAXGEN + t];
      if ((unsigned)y < (unsigned)mrl::V) {
        const int l = __ldg(P.tok_letter + y);
        is_l = l >= 0; is_e = is_l && l == key;
        if (mode == 0) { is_c = __ldg(P.tok_colon + y) == 1; is_a = __ldg(P.tok_answer + y) == 1; }
      }
    }
  }
  const unsigned ml = __ballot_sync(0xffffffffu, is_l), me = __ballot_sync(0xffffffffu, is_e);
  const unsigned mc = __ballot_sync(0xffffffffu, is_c), ma = __ballot_sync(0xffffffffu, is_a);
  volatile unsigned* M = reinterpret_cast<volatile unsigned*>(smem);   // M[4 * warp + {0, 1, 2, 3}] = L, E, C, A of slots [32 * (warp & 1), +32) of sequence warp / 2
  if ((tid & 31) == 0) { M[4 * warp] = ml; M[4 * warp + 1] = me; M[4 * warp + 2] = mc; M[4 * warp + 3] = ma; }
  __syncthreads();
  if (tid == 0) {
    float r[mrl::GSZ];
#pragma unroll
    for (int j = 0; j < mrl::GSZ; j++) {
      const volatile unsigned* m = M + 8 * j;                // low warp of the sequence, then high warp
      const mrl::u64 L = (mrl::u64)m[0] | ((mrl::u64)m[4] << 32), E = (mrl::u64)m[1] | ((mrl::u64)m[5] << 32);
      const mrl::u64 C = (mrl::u64)m[2] | ((mrl::u64)m[6] << 32), A = (mrl::u64)m[3] | ((mrl::u64)m[7] << 32);
      const mrl::u64 match = mode == 0 ? (L & (C << 1) & (A << 2)) : (L & 1ull);
      r[j] = 0.f;
      if (match != 0ull) { const int i = 63 - __clzll((long long)match); r[j] = ((E >> i) & 1ull) != 0ull ? 1.f : 0.f; }
    }
    volatile float* vo = out;
#pragma unroll
    for (int j = 0; j < mrl::GSZ; j++) vo[j] = r[j];         // every scratch word was read above
    __threadfence_block();
  }
  __syncthreads();
}

}  // namespace rw
