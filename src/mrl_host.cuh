// mrl_host.cuh -- host-side helpers: the error exit, raw file I/O, the readers of the weight directory and of the
// prompt store, device allocation of the structs of mrl_common.cuh, and the default and the check of Hyper.
// Everything allocated here comes from mk::dalloc and is zero-initialised.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "mrl_common.cuh"

namespace mrl { namespace host {

using mk::dalloc; using mk::up; using mk::down;

// die() ends the program with a message. die_context() is put in front of it: megarl names the run it is working on
// there, so that a bad option of the seventh run of a process says which run it belongs to.
inline std::string& die_context() { static std::string c; return c; }
[[noreturn]] inline void die(const std::string& m) { fprintf(stderr, "mrl::host: %s%s\n", die_context().c_str(), m.c_str()); exit(1); }

// ---------------------------------------------------------------- raw files
template <class X> inline std::vector<X> read_file(const std::string& path, size_t expect = 0) {
  FILE* f = fopen(path.c_str(), "rb"); if (!f) die("cannot open " + path);
  fseek(f, 0, SEEK_END); const size_t bytes = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
  if (bytes % sizeof(X)) die("size of " + path + " is not a multiple of the element size");
  std::vector<X> v(bytes / sizeof(X));
  if (!v.empty() && fread(v.data(), sizeof(X), v.size(), f) != v.size()) die("short read " + path);
  fclose(f);
  if (expect && v.size() != expect) die(path + ": expected " + std::to_string(expect) + " elements, found " + std::to_string(v.size()));
  return v;
}
template <class X> inline void write_file(const std::string& path, const std::vector<X>& v) {
  FILE* f = fopen(path.c_str(), "wb"); if (!f) die("cannot write " + path);
  if (!v.empty()) fwrite(v.data(), sizeof(X), v.size(), f); fclose(f);
}
inline std::vector<float> bf16_to_float(const std::vector<bf16>& b) { std::vector<float> f(b.size()); for (size_t i = 0; i < b.size(); i++) f[i] = __bfloat162float(b[i]); return f; }

// ---------------------------------------------------------------- weights
// A weight directory (tools/export_qwen.py) holds weights.bin, the tensors back to back, and weights.idx, one line per
// tensor: name dtype rows cols offset nbytes (dtype bf16 or f32, offset and nbytes in bytes).
struct TensorInfo { std::string dtype; long long rows, cols, offset, nbytes; };
struct WeightFile {
  std::map<std::string, TensorInfo> idx; std::string bin_path;
  explicit WeightFile(const std::string& dir) : bin_path(dir + "/weights.bin") {
    FILE* f = fopen((dir + "/weights.idx").c_str(), "r"); if (!f) die("cannot open " + dir + "/weights.idx");
    char name[128], dt[16]; long long r, c, off, nb;
    while (fscanf(f, "%127s %15s %lld %lld %lld %lld", name, dt, &r, &c, &off, &nb) == 6) idx[name] = TensorInfo{dt, r, c, off, nb};
    fclose(f);
    if (idx.empty()) die("empty weights.idx in " + dir);
  }
  const TensorInfo& info(const std::string& name, const char* dtype, long long rows, long long cols) const {
    auto it = idx.find(name); if (it == idx.end()) die("tensor " + name + " missing in weights.idx");
    const TensorInfo& t = it->second;
    if (t.dtype != dtype || t.rows != rows || t.cols != cols) die("tensor " + name + ": expected " + dtype + " [" + std::to_string(rows) + "," + std::to_string(cols) + "], found " + t.dtype + " [" + std::to_string(t.rows) + "," + std::to_string(t.cols) + "]");
    return t;
  }
  // read one tensor into host memory
  template <class X> std::vector<X> read(const std::string& name, const char* dtype, long long rows, long long cols) const {
    const TensorInfo& t = info(name, dtype, rows, cols);
    if ((size_t)t.nbytes != (size_t)rows * cols * sizeof(X)) die("tensor " + name + ": byte count does not match its shape");
    FILE* f = fopen(bin_path.c_str(), "rb"); if (!f) die("cannot open " + bin_path);
    std::vector<X> v((size_t)rows * cols); fseek(f, (long)t.offset, SEEK_SET);
    if (fread(v.data(), sizeof(X), v.size(), f) != v.size()) die("short read of " + name);
    fclose(f); return v;
  }
  template <class X> X* upload(const std::string& name, const char* dtype, long long rows, long long cols) const {
    std::vector<X> v = read<X>(name, dtype, rows, cols); X* d = dalloc<X>(v.size()); up(d, v); return d;
  }
};

// Upload the whole model. With `layers` < NL only the first `layers` layers are loaded (the others stay null).
inline Model load_model(const std::string& dir, int layers = NL) {
  WeightFile w(dir); Model m; memset(&m, 0, sizeof m);
  { // the embedding gets VP rows (rows >= V zero): a reduction that forgets to stop at V reads zeros, not foreign memory
    const std::vector<bf16> e = w.read<bf16>("emb", "bf16", V, D); bf16* d = dalloc<bf16>((size_t)VP * D); up(d, e); m.emb = d;
  }
  for (int l = 0; l < layers; l++) {
    const std::string p = "L" + std::to_string(l) + ".";
    m.qkv[l] = w.upload<bf16>(p + "qkv", "bf16", DQKV, KQ);
    m.o[l] = w.upload<bf16>(p + "o", "bf16", D, D);
    m.gu[l] = w.upload<bf16>(p + "gu", "bf16", 2 * FF, D);
    m.down[l] = w.upload<bf16>(p + "down", "bf16", D, FF);
    m.ln1[l] = w.upload<float>(p + "ln1", "f32", 1, D);
    m.ln2[l] = w.upload<float>(p + "ln2", "f32", 1, D);
  }
  m.lnf = w.upload<float>("lnf", "f32", 1, D);
  m.rope_cos = w.upload<float>("rope_cos", "f32", MAXPOS, HD / 2);
  m.rope_sin = w.upload<float>("rope_sin", "f32", MAXPOS, HD / 2);
  return m;
}

// ---------------------------------------------------------------- prompt store
// A prompt store (tools/stores.py) holds prompts.idx (a header line "n max_len verifier_mode", then one line
// "id task key len offset" per prompt), prompts.i32 (the prompts' tokens back to back; offset counts tokens) and the
// reward's per-token tables tok_letter.i8, tok_colon.u8 and tok_answer.u8 (V entries each).
struct PromptsHost {                             // host copy; megarl keeps it to set each run's hand-out order
  int n = 0, max_len = 0, verifier_mode = 0;
  std::vector<int> id, task, key, len, offset, tokens;      // tokens: concatenated
  std::vector<signed char> tok_letter; std::vector<unsigned char> tok_colon, tok_answer;
  const int* prompt_tokens(int p) const { return tokens.data() + offset[p]; }
};
inline PromptsHost read_prompts(const std::string& dir) {
  PromptsHost h; FILE* f = fopen((dir + "/prompts.idx").c_str(), "r"); if (!f) die("cannot open " + dir + "/prompts.idx");
  if (fscanf(f, "%d %d %d", &h.n, &h.max_len, &h.verifier_mode) != 3) die("bad header in prompts.idx");
  h.id.resize(h.n); h.task.resize(h.n); h.key.resize(h.n); h.len.resize(h.n); h.offset.resize(h.n);
  for (int i = 0; i < h.n; i++) if (fscanf(f, "%d %d %d %d %d", &h.id[i], &h.task[i], &h.key[i], &h.len[i], &h.offset[i]) != 5) die("bad line in prompts.idx");
  fclose(f);
  h.tokens = read_file<int>(dir + "/prompts.i32");
  h.tok_letter = read_file<signed char>(dir + "/tok_letter.i8", V); h.tok_colon = read_file<unsigned char>(dir + "/tok_colon.u8", V); h.tok_answer = read_file<unsigned char>(dir + "/tok_answer.u8", V);
  // The device cannot raise an error: with an unknown mode rw::score_group scores every answer 0, every group is
  // dropped for equal rewards and the run ends "learner starved" with no hint of the cause. So the host rejects it.
  if (h.verifier_mode != 0 && h.verifier_mode != 1) die("prompts.idx: unknown verifier_mode " + std::to_string(h.verifier_mode));
  for (int v = 0; v < V; v++)
    if (h.tok_letter[v] < -1 || h.tok_letter[v] > 3 || h.tok_colon[v] > 1 || h.tok_answer[v] > 1) die("token table entry " + std::to_string(v) + " is outside its range");
  for (size_t k = 0; k < h.tokens.size(); k++) if (h.tokens[k] < 0 || h.tokens[k] >= V) die("prompts.i32: token id outside the vocabulary at index " + std::to_string(k));
  for (int i = 0; i < h.n; i++) {
    if (h.len[i] < 1 || h.len[i] > MAXPROMPT) die("prompt " + std::to_string(i) + " has length " + std::to_string(h.len[i]) + " (limit " + std::to_string(MAXPROMPT) + ")");
    if (h.task[i] < 0 || h.task[i] >= NTASKS || h.key[i] < 0 || h.key[i] > 3) die("prompt " + std::to_string(i) + ": bad task or key");
    if ((size_t)h.offset[i] + h.len[i] > h.tokens.size()) die("prompt " + std::to_string(i) + " runs past prompts.i32");
  }
  return h;
}
// Optional task names from <prompts_dir>/envs.json ({"names": ["math", "science", ...]}), falling back to standard
// defaults. A minimal reader: the names must not contain quotes.
inline std::vector<std::string> read_env_names(const std::string& dir) {
  std::vector<std::string> out(NTASKS);
  const char* def4[4] = {"math", "science", "sciq", "csqa"};
  for (int t = 0; t < NTASKS; t++) out[t] = t < 4 ? def4[t] : ("task" + std::to_string(t));
  FILE* f = fopen((dir + "/envs.json").c_str(), "r");
  if (!f) return out;
  char buf[4096]; const size_t n = fread(buf, 1, sizeof(buf) - 1, f); fclose(f); buf[n] = 0;
  const char* p = strstr(buf, "\"names\"");
  if (!p) return out;
  p = strchr(p, '['); if (!p) return out;
  int t = 0;
  while (*p && *p != ']' && t < NTASKS) {
    const char* q1 = strchr(p, '"'); if (!q1) break;
    const char* q2 = strchr(q1 + 1, '"'); if (!q2) break;
    out[t++] = std::string(q1 + 1, (size_t)(q2 - q1 - 1));
    p = q2 + 1;
  }
  return out;
}

// The prompts a run uses, as indices into the store: the first `use_per_task` of each task (0 = all), in store order.
inline std::vector<int> kept_prompts(const PromptsHost& h, int use_per_task) {
  std::vector<int> keep; int cnt[NTASKS] = {0};
  for (int i = 0; i < h.n; i++) { if (use_per_task && cnt[h.task[i]] >= use_per_task) continue; cnt[h.task[i]]++; keep.push_back(i); }
  return keep;
}
// The order in which each task's prompts are handed out: by[t] lists the task's prompts as indices into the kept
// prompts (= device prompt indices). order_seed 0 is store order; any other seed is a Fisher-Yates permutation with
// the kernel's hash, so that the order depends on nothing but the seed.
inline void prompt_order(const PromptsHost& h, int use_per_task, u64 order_seed, std::vector<int>* by) {
  const std::vector<int> keep = kept_prompts(h, use_per_task);
  for (int t = 0; t < NTASKS; t++) by[t].clear();
  for (int k = 0; k < (int)keep.size(); k++) by[h.task[keep[k]]].push_back(k);
  if (order_seed)
    for (int t = 0; t < NTASKS; t++)
      for (size_t i = by[t].size(); i > 1; i--) std::swap(by[t][i - 1], by[t][(size_t)(mix64(mix64(order_seed + ((u64)t << 48)) + (u64)i) % (u64)i)]);
}
// Device copy. The prompt cache (kv, hlast) is allocated here if with_cache and filled later by the cache build.
// `use_per_task` limits the store to the first that many prompts of each task (0 = all). `order_seed` != 0 permutes the
// order in which each task's prompts are handed out (the by_task lists); prompt indices, and with them the prompt cache
// and the logs, are unchanged. `order_out`, if given, receives the NTASKS lists as uploaded.
inline Prompts upload_prompts(const PromptsHost& h, int use_per_task = 0, bool with_cache = true, u64 order_seed = 0, std::vector<int>* order_out = nullptr) {
  Prompts P; memset(&P, 0, sizeof P);
  const std::vector<int> keep = kept_prompts(h, use_per_task);
  const int n = (int)keep.size(); P.n = n;
  std::vector<int> len(n), task(n), key(n), tok((size_t)n * MAXPROMPT, 0), by[NTASKS];
  for (int k = 0; k < n; k++) {
    const int i = keep[k]; len[k] = h.len[i]; task[k] = h.task[i]; key[k] = h.key[i];
    memcpy(tok.data() + (size_t)k * MAXPROMPT, h.prompt_tokens(i), (size_t)h.len[i] * sizeof(int));
  }
  int* dlen = dalloc<int>(n); up(dlen, len); P.len = dlen; int* dtask = dalloc<int>(n); up(dtask, task); P.task = dtask;
  int* dkey = dalloc<int>(n); up(dkey, key); P.key = dkey; int* dtok = dalloc<int>(tok.size()); up(dtok, tok); P.tok = dtok;
  signed char* dl = dalloc<signed char>(V); up(dl, h.tok_letter); P.tok_letter = dl;
  unsigned char* dc = dalloc<unsigned char>(V); up(dc, h.tok_colon); P.tok_colon = dc;
  unsigned char* da = dalloc<unsigned char>(V); up(da, h.tok_answer); P.tok_answer = da;
  P.verifier_mode = h.verifier_mode;
  prompt_order(h, use_per_task, order_seed, by);
  for (int t = 0; t < NTASKS; t++) { P.ntask[t] = (int)by[t].size(); int* d = dalloc<int>(by[t].size()); up(d, by[t]); P.by_task[t] = d; if (order_out) order_out[t] = by[t]; }
  if (with_cache) {
    P.kv = dalloc<float>((size_t)n * NL * MAXPROMPT * KVW);
    P.hlast = dalloc<float>((size_t)n * D);
  }
  return P;
}
// Give an uploaded store another hand-out order (megarl: one per run of a process). `h` and `use_per_task` must be
// those of the upload; the lists keep their lengths, so nothing else of the device copy changes.
inline void set_prompt_order(const Prompts& P, const PromptsHost& h, int use_per_task, u64 order_seed, std::vector<int>* order_out = nullptr) {
  std::vector<int> by[NTASKS]; prompt_order(h, use_per_task, order_seed, by);
  for (int t = 0; t < NTASKS; t++) {
    if ((int)by[t].size() != P.ntask[t]) die("set_prompt_order: the store was uploaded with another selection of prompts");
    up(const_cast<int*>(P.by_task[t]), by[t]); if (order_out) order_out[t] = by[t];
  }
}

// ---------------------------------------------------------------- device structs
// Buffers for MAXROWS rows. nrows: rows the pipe will use (rollout: GSZ * ngrp; the prompt-cache build rewrites it per
// batch of prompts). Constant during a launch.
inline Pipe alloc_pipe(int nrows) {
  if (nrows < 1 || nrows > MAXROWS) die("alloc_pipe: nrows out of range");
  Pipe p; memset(&p, 0, sizeof p);
  p.hres = dalloc<float>((size_t)MAXROWS * DP);
  p.xn = dalloc<bf16>((size_t)MAXROWS * KQ); p.qkv = dalloc<float>((size_t)MAXROWS * QKVP); p.qkvp = dalloc<float>((size_t)(KS_MAX / 4) * MAXROWS * QKVP);
  p.ao = dalloc<bf16>((size_t)MAXROWS * D); p.gu = dalloc<float>((size_t)MAXROWS * GUP); p.xm = dalloc<bf16>((size_t)MAXROWS * FF);
  p.part = dalloc<float>((size_t)KS_MAX * MAXROWS * DP); p.xf = dalloc<bf16>((size_t)MAXROWS * D);
  p.lmpart = dalloc<SampPart>((size_t)NT_V * MAXROWS);
  p.kv = dalloc<float>((size_t)MAXROWS * NL * MAXGEN * KVW);
  p.row = dalloc<Row>(MAXROWS); p.grp = dalloc<GroupSlot>(MAXGRP); p.nrows = dalloc<int>(1);
  std::vector<int> n(1, nrows); up(p.nrows, n);
  return p;
}
inline Store alloc_store(int nbuf) {
  Store s; memset(&s, 0, sizeof s); s.nbuf = nbuf;
  s.rec = dalloc<GroupRec>(nbuf); s.tokens = dalloc<int>((size_t)nbuf * GSZ * MAXGEN); s.logp = dalloc<float>((size_t)nbuf * GSZ * MAXGEN);
  s.hbase = dalloc<float>((size_t)nbuf * GSZ * MAXGEN * D);
  return s;
}
inline Train alloc_train(int steps, int tksV) {
  Train t; memset(&t, 0, sizeof t); const size_t st = (size_t)std::max(1, steps);
  t.W = dalloc<bf16>(MID_NW);
  t.batch = dalloc<Batch>(1);
  t.hfin = dalloc<float>((size_t)TROWS * DP); t.rinv = dalloc<float>(TROWS);
  t.y = dalloc<int>(TROWS); t.logp_beh = dalloc<float>(TROWS); t.rseq = dalloc<int>(TROWS);
  t.xf = dalloc<bf16>((size_t)TROWS * D); t.lsepart = dalloc<LsePart>((size_t)TTILE * NT_V * TM);
  t.logit_y = dalloc<float>(TROWS); t.lse = dalloc<float>(TROWS); t.logp_new = dalloc<float>(TROWS); t.coef = dalloc<float>(TROWS);
  t.dlogits = dalloc<bf16>((size_t)TROWS * (VP + KS_MAX * 32));
  t.dxpart = dalloc<float>((size_t)std::max(1, tksV) * TROWS * DP); t.dxf = dalloc<float>((size_t)TROWS * DP);
  t.loss_hist = dalloc<float>(st); t.step_done = dalloc<u64>(st); t.step_info = dalloc<int>(st * 16); t.step_stat = dalloc<float>(st * 4);
  return t;
}
inline Hyper default_hyper() {
  Hyper h; memset(&h, 0, sizeof h);
  h.temperature = 1.f; h.inv_temperature = 1.f; h.clip_eps = 0.2f; h.adv_eps = 1e-4f;
  h.lr = 4e-5f; h.beta1 = 0.9f; h.beta2 = 0.999f; h.adam_eps = 1e-8f; h.wd = 0.f; h.grad_clip = 1.f; h.require_eos = 1;
  h.lag = 8; h.norm_mode = 0; h.pair_mode = 2; h.select_wait_ns = 2000000ull; h.starve_ns = 5000000000ull;
  h.ksY = 4; h.ksD = 4; h.ksQ = 2; h.ksO = 2; h.ksG = 1; h.tksV = 16; h.target_per_task = 8;
  h.seed = 1; h.deadline_ns = 600ull * 1000000000ull;
  return h;
}
// Exits with a message if a chunk count or policy parameter cannot work; warns if the target admits more groups than
// the lag rule lets the learner use.
inline void check_hyper(const Hyper& h) {
  auto rng = [&](const char* n, int v, int lo, int hi) { if (v < lo || v > hi) die(std::string(n) + " = " + std::to_string(v) + " is outside [" + std::to_string(lo) + ", " + std::to_string(hi) + "]"); };
  rng("ksY", h.ksY, 1, KS_MAX); rng("ksD", h.ksD, 1, KS_MAX);
  rng("ksQ", h.ksQ, 1, KS_MAX / 4); rng("ksO", h.ksO, 1, D / 32); rng("ksG", h.ksG, 1, D / 32);
  rng("tksV", h.tksV, 1, V / 32); rng("lag", h.lag, 0, 1 << 20); rng("target_per_task", h.target_per_task, 1, 1 << 20);
  rng("norm_mode", h.norm_mode, 0, 1); rng("pair_mode", h.pair_mode, 0, 2);
  rng("count_held", h.count_held, 0, 1); rng("alternate", h.alternate, 0, 1); rng("no_train", h.no_train, 0, 1);
  rng("require_eos", h.require_eos, 0, 1);
  if (h.grad_clip < 0.f) die("grad_clip must be >= 0");
  if (h.pair_mode == 0 && TGRP % NTASKS != 0) die("pair_mode 0 needs TGRP to be a multiple of the number of tasks");
  if (h.lag < (1 << 20) && (long long)h.target_per_task > (long long)std::max(1, h.lag) * std::max(1, TGRP / NTASKS))
    fprintf(stderr, "mrl::host: warning: target_per_task %d exceeds lag * groups per task per update = %d: groups will be generated that can no longer be trained on\n", h.target_per_task, std::max(1, h.lag) * std::max(1, TGRP / NTASKS));
  if (!(h.temperature > 0.f) || fabsf(h.inv_temperature * h.temperature - 1.f) > 1e-5f) die("temperature / inv_temperature inconsistent");
}

}}  // namespace mrl::host
