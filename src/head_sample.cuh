// head_sample.cuh -- the sampling head of the rollout (namespace hd, phase kinds K_H_*).
//
// The upper half of the trunk leaves, for every fed row, P.xm = the bf16 input of layer LT's down_proj and P.hres = the
// residual stream before that down_proj is added. The head finishes layer LT, applies the final norm and the LM head,
// and samples one token with its log-probability per active row. All its weights are frozen.
//
// Phases in chain order (hd::add_head). Task counts are per pipeline; add_head multiplies them by the lockstep multiplier.
//   kind      tasks        what                                                                       depends on
//   H_Y0P     NT_D * ksY   part[c] = xm[:, K chunk c] * down[LT]^T, fp32, rows < nrows                dep (Q_ACT[LT])
//   H_Y0R     NT_D         active rows: hres += sum_c part[c], columns < D                            H_Y0P
//   H_NORMF   MAXROWS / 8  xf = bf16(rmsnorm(hres) * lnf), rows < nrows (8 rows per task)             H_Y0R
//   H_LMHEAD  NT_V         LM-head tile; per active row lmpart[tile][row] (Gumbel-max and lse partial) H_NORMF
//   H_SAMPLE  ngrp         per active row: token, log-probability, store entry, Row update            H_LMHEAD
// H_Y0P's task index is local = tile * ksY + chunk.
//
// nrows is GSZ * C.ngrp (by contract equal to *P.nrows; taking it from the kernel parameter saves a global load at the
// start of every task).
//
// What the head touches (nothing else of Ctx or Pipe):
//   reads   C.hp.{ksY, inv_temperature, seed}, C.ngrp, C.model.{down[LT], lnf, emb}, P.xm, P.hres, P.row, P.grp[].uid;
//   writes  P.part (chunks < ksY), P.hres (active rows, columns < D), P.xf, P.lmpart;
//           C.store.{tokens, logp} at [buf][seq][gen] of the active rows, C.store.rec[buf].len[seq] of the rows that
//           end; P.row[r].{gen, tok, pos, active, feed} of the active rows; C.sh.cnt.tokens_generated (atomicAdd);
//           C.sh.cnt.errors[ERR_NO_WINNER] (atomicAdd) if a row has no token to sample, which a correct run never has
//           (the note above t_sample).
// All of these buffers must be 16-byte aligned with the row lengths of mrl_common.cuh (the allocations of mrl_host.cuh
// are): rows are copied and reduced with 16-byte loads.
//
// What the caller must guarantee (mt::add_roll_mid and the control tasks do):
//   * The graph: `dep` of add_head is Q_ACT[LT], the last phase of the upper half; R_ADVANCE and the lower half's first
//     phase (Q_EMBN) depend on the returned H_SAMPLE. The job's phases form one chain per step, so nothing else of the
//     job runs between Q_ACT[LT] and H_SAMPLE, and Row, xm, hres and part belong to the head.
//   * Row.active == 0 in every row of a slot that is not GENERATING. An active row has 0 <= gen < MAXGEN, buf and seq
//     name its store rows, plen is its prompt length, and its xm and hres rows come from the upper half of this step.
//   * GSZ * C.ngrp == *P.nrows; hp.ksY is the value add_head was called with.
//   * C.sh.cnt.tokens_generated and the 16 counters C.sh.cnt.errors exist (rlc::host_alloc). The second is only written
//     for a row without a token to sample.
// What the head leaves: P.hres of a row that was active is the residual stream after layer LT (Q_EMBN overwrites it
// when the row is fed again; a row that ended keeps it); P.part chunks < ksY are overwritten.
//
// Exactness. The learner (mid_train.cuh, grpo_loss.cuh) recomputes the log-probability from store.hbase through layers
// LMID..NL-1 and this head. For the same weight bits it gets the same bits as the rollout because
//   * Y0 is reduced over the hp.ksY chunks of mrl::kchunk, one EPI_F32 tile per chunk, and the partials are summed in
//     chunk order starting from chunk 0 with __fadd_rn before they are added to the residual (sum_partials below);
//   * norm, LM-head tile, row reductions, tile merges and logp are the lm:: functions of mrl_lmhead.cuh, nothing else;
//   * a GEMM tile's result for a row depends neither on the row's position in the tile nor on the other rows, so the
//     pipeline's tiles here and the learner's sequence tiles agree.
// Do not change one of these on this side only.
//
// Rules followed by every task function: called by all 256 threads of one block, only the dynamic shared memory `smem`
// (no __shared__ variables), no blockIdx, no waiting. Rows of free slots and ended rows are carried through the GEMM
// tiles but nothing of theirs is written to hres, the store, lmpart or Row.
// Shared memory: H_Y0P uses all of it inside the tile; H_LMHEAD keeps the tile (float T[64][256], 65,536 B) at its
// base; H_SAMPLE uses 24 floats at its base (scratch of the two merges); the other tasks use none.
//
// Memory visibility. Everything the head reads that a block wrote during the launch is read with ordinary loads (or
// cp.async.cg inside a tile) behind the scheduler's release / acquire chain, and is correct because of the chain:
//   Row and GroupSlot           <- R_BEGIN (admission, cancel) and H_SAMPLE of the previous step;
//   P.xm, P.hres                <- the upper half of this step;
//   part, xf, lmpart            <- the head's own phases, by add_head's edges.
// H_SAMPLE writes the Row fields that H_Y0R and H_LMHEAD read, and depends on both. The two counters the head writes
// are the only words without a chain: atomicAdd only. The head polls nothing, and nothing here is read with
// __ldg / .nc. Weights (down[LT], lnf, emb) are constant during the launch.
//
// Registers: every local of a task function is defined on every path before it is read, for the reason given at the
// top of qwen_fwd.cuh. The zeros below that are never used are for that, not for the arithmetic.
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "mrl_common.cuh"
#include "mrl_lmhead.cuh"

namespace hd {
using mrl::bf16; using mrl::u64;

// Slot of C.sh.cnt.errors (16 slots that must all stay 0) that H_SAMPLE increments for a row without a token to
// sample (the note above t_sample). rl_state.cuh names the same slot rls::ERR_HEAD_NO_WINNER for the host audit, which
// fails a run on any non-zero slot.
constexpr int ERR_NO_WINNER = 15;

// one non-inlined function per gemm_tile instantiation: each gets its own register allocation
__device__ __noinline__ void g_nt_f32(const lg::Args& a, int tm, int tn, char* smem) { lg::gemm_tile<lg::NT, lg::EPI_F32>(a, tm, tn, smem); }

// Sum of the n fp32 split-K partials of 4 consecutive elements: partial c is at p0 + c * stride, and the partials are
// added in chunk order starting from chunk 0 (((p0 + p1) + p2) + ...), the order the learner's M_NORMF uses too.
// The loads of up to 8 chunks are issued before the first addition needs one, so they are in flight together instead of
// each waiting for the previous one. n >= 1.
// A slot of t[] without a chunk is set to zero. That zero is never added (the second loop has the same guard): it only
// makes every register written on every path before it is read (Registers, top of this file).
__device__ __forceinline__ mrl::F4 sum_partials(const float* p0, size_t stride, int n) {
  mrl::F4 s; s.v[0] = s.v[1] = s.v[2] = s.v[3] = 0.f;
  for (int c = 0; c < n; c += 8) {
    mrl::F4 t[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      if (c + j < n) t[j] = *reinterpret_cast<const mrl::F4*>(p0 + (size_t)(c + j) * stride);
      else { t[j].v[0] = 0.f; t[j].v[1] = 0.f; t[j].v[2] = 0.f; t[j].v[3] = 0.f; }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) if (c + j < n) {
      if (c + j == 0) s = t[j];                                   // start from chunk 0, not from +0: a -0 partial stays -0
      else {
#pragma unroll
        for (int e = 0; e < 4; e++) s.v[e] = __fadd_rn(s.v[e], t[j].v[e]);
      }
    }
  }
  return s;
}

// ------------------------------------------------------------------------------------------------ H_Y0P
// One (N tile, K chunk) of layer LT's down_proj: part[c][rows, tile] = xm[:, chunk c] * down[LT]^T.
// Rows >= nrows are zero-filled in shared memory and never stored. local = tile * ksY + chunk.
__device__ __forceinline__ void t_y0p(const mrl::Ctx& C, const mrl::Pipe& P, int local, char* smem) {
  const int ks = C.hp.ksY, tn = local / ks, c = local % ks, nrows = mrl::GSZ * C.ngrp;
  int k0, kc; mrl::kchunk(mrl::FF / 32, ks, c, k0, kc);
  lg::Args a = lg::args_default();
  a.A = P.xm + k0; a.lda = mrl::FF; a.avalid = nrows;
  a.B = C.model.down[mrl::LT] + k0; a.ldb = mrl::FF; a.bvalid = mrl::D;          // [D, FF]: one row per output
  a.C = P.part + (size_t)c * mrl::MAXROWS * mrl::DP; a.ldc = mrl::DP; a.rvalid = nrows; a.cvalid = mrl::D;
  a.M = mrl::MAXROWS; a.N = mrl::DP; a.K = kc; a.accumulate = 0;
  g_nt_f32(a, 0, tn, smem);
}

// ------------------------------------------------------------------------------------------------ H_Y0R
// Task tn: columns [tn * TN, tn * TN + TN) below D of the active rows: hres = hres + (part[0] + part[1] + ...).
// The partials are summed first and added to the residual stream last, as in the learner's M_NORMF.
// Thread t owns 4 columns (t & 63) in the 16 rows (t >> 6) + 4i, so the two warps that share t >> 6 read the same 16
// activity flags (one broadcast load each), and every 16-byte load is issued by exactly one thread.
__device__ __noinline__ void t_y0r(const mrl::Ctx& C, const mrl::Pipe& P, int tn) {
  const int tid = threadIdx.x, ks = C.hp.ksY, nrows = mrl::GSZ * C.ngrp;
  const int col = tn * mrl::TN + 4 * (tid & 63), r0 = tid >> 6;
  static_assert(mrl::D % 4 == 0 && mrl::TN == 256 && mrl::MAXROWS == 64, "reduce layout: 64 groups of 4 columns per row and tile");
  if (col >= mrl::D) return;                                      // padding columns of the last tile: never written
  const mrl::Row* rows = P.row; float* hres = P.hres; const float* part = P.part;
  unsigned act = 0;
#pragma unroll
  for (int i = 0; i < mrl::MAXROWS / 4; i++) { const int row = r0 + 4 * i; if (row < nrows && rows[row].active) act |= 1u << i; }
  const size_t stride = (size_t)mrl::MAXROWS * mrl::DP;
  for (int i = 0; i < mrl::MAXROWS / 4; i++) {
    if (!((act >> i) & 1u)) continue;
    const size_t e = (size_t)(r0 + 4 * i) * mrl::DP + col;
    mrl::F4 h = *reinterpret_cast<const mrl::F4*>(hres + e);
    const mrl::F4 s = sum_partials(part + e, stride, ks);
#pragma unroll
    for (int j = 0; j < 4; j++) h.v[j] = __fadd_rn(h.v[j], s.v[j]);
    *reinterpret_cast<mrl::F4*>(hres + e) = h;
  }
}

// ------------------------------------------------------------------------------------------------ H_NORMF
// Task t: rows 8t .. 8t+7, one warp per row.
__device__ __forceinline__ void t_normf(const mrl::Ctx& C, const mrl::Pipe& P, int local) {
  const int row = 8 * local + (threadIdx.x >> 5);
  if (row >= mrl::GSZ * C.ngrp) return;                           // uniform within the warp
  lm::warp_rmsnorm_row(P.hres + (size_t)row * mrl::DP, C.model.lnf, P.xf + (size_t)row * mrl::D, false);
}

// ------------------------------------------------------------------------------------------------ H_LMHEAD
// Vocabulary tile n: logits of all rows < nrows (lm::lmhead_tile), then for each active row one warp computes the
// log-sum-exp partial and the Gumbel-max partial of the tile and lane 0 writes lmpart[n][row].
// The sampling key of a row is sample_key(seed, uid of its group, seq, gen): gen is the index of the token being sampled.
// The reductions (per id: one expf, the noise hash and two logf) are not cheap next to the tile, and a warp does its rows
// one after the other, so the rows are dealt to the warps as a Latin square: row 8a + b belongs to warp (a + b) mod 8.
// Every warp has one row of each block of 8 rows, so a pipeline of 4 slots gives every warp 2 rows (not 8 rows to two
// warps), and prefixes, alternating slots and every fourth slot all spread evenly. Which warp reduces a row does not
// change the result: the lm:: row functions do not depend on the warp.
// What a warp needs to know about its 8 rows (active, gen, seq, uid) is one load instruction before the tile, one word
// per lane (lane 8f + a: field f of the warp's row in block a), in flight during the GEMM; afterwards the words are
// broadcast inside the warp with shuffles.
__device__ __noinline__ void t_lmhead(const mrl::Ctx& C, const mrl::Pipe& P, int n, char* smem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nrows = mrl::GSZ * C.ngrp;
  static_assert(mrl::MAXROWS == 64, "8 warps x 8 rows; 4 metadata words per row in 32 lanes");
  int meta = 0;
  {
    const int a = lane & 7, f = lane >> 3, mrow = 8 * a + ((warp - a) & 7);
    if (mrow < nrows) {
      const mrl::Row* R = P.row + mrow;
      const int* src = f == 0 ? &R->active : f == 1 ? &R->gen : f == 2 ? &R->seq : &P.grp[mrow / mrl::GSZ].uid;
      meta = *src;
    }
  }
  lm::lmhead_tile(P.xf, mrl::D, nrows, C.model.emb, n, smem);
  const float* T = reinterpret_cast<const float*>(smem);
  const float inv_t = C.hp.inv_temperature; const u64 seed = C.hp.seed; const int n0 = n * mrl::TN;
  mrl::SampPart* out = P.lmpart + (size_t)n * mrl::MAXROWS;
  for (int a = 0; a < 8; a++) {
    if (!__shfl_sync(0xffffffffu, meta, a)) continue;            // inactive row or row >= nrows; uniform within the warp
    const int gen = __shfl_sync(0xffffffffu, meta, 8 + a), seq = __shfl_sync(0xffffffffu, meta, 16 + a), uid = __shfl_sync(0xffffffffu, meta, 24 + a);
    const int row = 8 * a + ((warp - a) & 7); const float* Trow = T + row * mrl::TN;
    lm::RowLse l; lm::RowGumbel b;
    lm::warp_row_lse_gumbel(Trow, n0, inv_t, mrl::sample_key(seed, uid, seq, gen), l, b);
    if (lane == 0) {
      mrl::SampPart p; p.gmax = b.score; p.gidx = b.idx; p.glogit = b.z; p.lmax = l.lmax; p.lsum = l.lsum; p.pad[0] = p.pad[1] = p.pad[2] = 0;
      out[row] = p;
    }
  }
}

// ------------------------------------------------------------------------------------------------ H_SAMPLE
// Group slot g: for each active row merge the per-tile partials (whole block per row), then thread 0 stores the token
// and its log-probability at index gen and advances the Row:
//   gen++; tok = token; pos = plen + gen - 1;
//   end token or gen == MAXGEN: active = 0, feed = 0, rec[buf].len[seq] = gen;   otherwise feed = 1.
// The activity flags are read by all threads before the first barrier and written by thread 0 only after the last
// barrier of the same row, so the block's control flow is uniform.
// A row without a token to sample. lm::better never prefers a NaN score, so if every score of the row is NaN (one NaN
// or infinity in the row's hres is enough: the final norm spreads it over the row, and then every logit is NaN) the
// merge returns lm::NO_WINNER, which is not a vocabulary id. Nothing >= V may leave the head: the token indexes tables
// in the trunk, the reward and the learner. Such a row
//   * is given <|endoftext|>, which ends its sequence here instead of carrying it to MAXGEN;
//   * keeps the log-probability the merges give it: NaN when the row's logits are NaN;
//   * is counted in cnt.errors[ERR_NO_WINNER], which fails the host audit. With finite inputs the count stays 0.
// Nothing downstream filters such a token: its group is scored like any other and, if its rewards differ, trained on.
// The learner gives the token coef = 0 (a NaN log-ratio fails the clamp test of t_loss in grpo_loss.cuh), but
// lm::warp_row_dlogits forms coef * (onehot - softmax) with a NaN softmax, and 0 * NaN is NaN. The NaN reaches the
// weight gradients, so T_FINISH finds a gradient norm that is not finite and stops the run without publishing the
// update (ERR_NONFINITE).
__device__ __noinline__ void t_sample(const mrl::Ctx& C, const mrl::Pipe& P, int g, char* smem) {
  const int tid = threadIdx.x;
  float* scratch = reinterpret_cast<float*>(smem);                // 24 floats, used by the two merges one after the other
  mrl::Row* R0 = P.row + mrl::GSZ * g; const mrl::SampPart* lmpart = P.lmpart;
  int act[mrl::GSZ];
#pragma unroll
  for (int j = 0; j < mrl::GSZ; j++) act[j] = R0[j].active;
  const size_t stride = (size_t)mrl::MAXROWS * sizeof(mrl::SampPart);              // lmpart is [NT_V][MAXROWS]
  int nsamp = 0;
  for (int j = 0; j < mrl::GSZ; j++) {
    if (!act[j]) continue;
    const char* first = reinterpret_cast<const char*>(lmpart + mrl::GSZ * g + j);  // tile 0 of this row
    float lmax, lsum, score, z; int tok;
    lm::block_merge_lse(first + offsetof(mrl::SampPart, lmax), stride, scratch, lmax, lsum);
    lm::block_merge_gumbel(first, stride, scratch, score, tok, z);
    if (tid == 0) {
      mrl::Row* R = R0 + j;
      if ((unsigned)tok >= (unsigned)mrl::V) {                                     // no candidate won (all scores NaN): end the sequence, count it
        tok = mrl::EOS_ENDOFTEXT; atomicAdd(C.sh.cnt.errors + ERR_NO_WINNER, 1);
      }
      const int gen0 = R->gen, buf = R->buf, seq = R->seq, plen = R->plen;
      const size_t idx = ((size_t)buf * mrl::GSZ + seq) * mrl::MAXGEN + gen0;
      C.store.tokens[idx] = tok; C.store.logp[idx] = lm::logp_from(z, lmax, lsum);
      const int gen = gen0 + 1;
      R->gen = gen; R->tok = tok; R->pos = plen + gen - 1;
      if (tok == mrl::EOS_IM_END || tok == mrl::EOS_ENDOFTEXT || gen == mrl::MAXGEN) { R->active = 0; R->feed = 0; C.store.rec[buf].len[seq] = gen; }
      else R->feed = 1;
      nsamp++;
    }
  }
  if (tid == 0 && nsamp) atomicAdd(C.sh.cnt.tokens_generated, (u64)nsamp);         // other pipelines' sample tasks add concurrently
}

// ------------------------------------------------------------------------------------------------ dispatcher
// pd.kind in K_H_Y0P .. K_H_SAMPLE; `local` is the task index inside the phase for this pipeline (megarl.cu has already
// divided out the lockstep multiplier); P is the pipeline. `step` is not used: nothing in the head depends on the step
// number (the sampling key uses the row's own token index).
__device__ void run(const mrl::Ctx& C, const mrl::Pipe& P, const mk::PhaseDesc& pd, int local, mk::u64 step, char* smem) {
  (void)step;
  switch (pd.kind) {
    case mrl::K_H_Y0P:    t_y0p(C, P, local, smem); break;
    case mrl::K_H_Y0R:    t_y0r(C, P, local); break;
    case mrl::K_H_NORMF:  t_normf(C, P, local); break;
    case mrl::K_H_LMHEAD: t_lmhead(C, P, local, smem); break;
    case mrl::K_H_SAMPLE: t_sample(C, P, local, smem); break;
    default: break;
  }
}

// ------------------------------------------------------------------------------------------------ host: job graph
// Adds the five head phases to a rollout job graph as a chain. H_Y0P depends on `dep` (Q_ACT[LT] of the same step);
// every task count is multiplied by the lockstep multiplier `mult`, which is also stored in PhaseDesc.c (PhaseDesc.a =
// LT, the layer whose down_proj the head completes). Returns the index of H_SAMPLE.
inline int add_head(mk::Builder& b, int dep, int ngrp, const mrl::Hyper& hp, int mult) {
  if (ngrp < 1 || ngrp > mrl::MAXGRP || mult < 1 || hp.ksY < 1 || hp.ksY > mrl::KS_MAX || dep < 0 || dep >= b.nph()) {
    fprintf(stderr, "hd::add_head: bad arguments (dep %d of %d phases, ngrp %d, mult %d, ksY %d; it must be in [1, %d])\n", dep, b.nph(), ngrp, mult, hp.ksY, mrl::KS_MAX);
    exit(1);
  }
  auto add = [&](const char* name, int kind, int n, std::vector<mk::Dep> d) { return b.add(name, kind, n * mult, d, mrl::LT, 0, mult); };
  const int y0p = add("H_Y0P", mrl::K_H_Y0P, mrl::NT_D * hp.ksY, {mk::Dep{dep, 0}});
  const int y0r = add("H_Y0R", mrl::K_H_Y0R, mrl::NT_D, {mk::Dep{y0p, 0}});
  const int nrm = add("H_NORMF", mrl::K_H_NORMF, mrl::MAXROWS / 8, {mk::Dep{y0r, 0}});
  const int lmh = add("H_LMHEAD", mrl::K_H_LMHEAD, mrl::NT_V, {mk::Dep{nrm, 0}});
  return add("H_SAMPLE", mrl::K_H_SAMPLE, ngrp, {mk::Dep{lmh, 0}});
}

}  // namespace hd
