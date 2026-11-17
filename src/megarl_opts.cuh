// megarl_opts.cuh -- the host-only half of megarl: its options and the list of runs of one process, the record of a
// run (run.json), the digest of a prompt store, process timing and the per-run console file. Nothing in this file
// calls CUDA.
#pragma once
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include "mrl_host.cuh"
#include "rl_state.cuh"

namespace mgo {
using namespace mrl;

// ------------------------------------------------------------------------------------------------ small helpers
inline std::string jlist(const std::vector<long long>& v) { std::string s = "["; for (size_t i = 0; i < v.size(); i++) s += (i ? ", " : "") + std::to_string(v[i]); return s + "]"; }
template <class X> inline std::vector<long long> ll(const std::vector<X>& v, size_t n = ~(size_t)0) { std::vector<long long> o; for (size_t i = 0; i < v.size() && i < n; i++) o.push_back((long long)v[i]); return o; }
// Length of the well-formed UTF-8 sequence that starts at x[i] (1 to 4), or 0 if there is none.
inline int utf8_len(const std::string& x, size_t i) {
  const unsigned char c = (unsigned char)x[i]; const size_t n = x.size();
  auto cont = [&](size_t k) { return k < n && ((unsigned char)x[k] & 0xC0) == 0x80; };
  if (c < 0x80) return 1;
  if (c >= 0xC2 && c <= 0xDF) return cont(i + 1) ? 2 : 0;
  if (c >= 0xE0 && c <= 0xEF) { if (!cont(i + 1) || !cont(i + 2)) return 0; const unsigned char d = (unsigned char)x[i + 1]; return (c == 0xE0 && d < 0xA0) || (c == 0xED && d > 0x9F) ? 0 : 3; }
  if (c >= 0xF0 && c <= 0xF4) { if (!cont(i + 1) || !cont(i + 2) || !cont(i + 3)) return 0; const unsigned char d = (unsigned char)x[i + 1]; return (c == 0xF0 && d < 0x90) || (c == 0xF4 && d > 0x8F) ? 0 : 4; }
  return 0;
}
// A JSON string. Control characters are escaped; a byte that is not part of well-formed UTF-8 (a file name in another
// encoding) is written as \udcXX, the form Python's "surrogateescape" reads back to the same byte, so that run.json
// stays a file every reader can open whatever the command line held.
inline std::string jstr(const std::string& x) {
  std::string s = "\"";
  for (size_t i = 0; i < x.size();) {
    const unsigned char c = (unsigned char)x[i]; char b[8];
    if (c == '"' || c == '\\') { s += '\\'; s += (char)c; i++; }
    else if (c < 0x20 || c == 0x7f) { snprintf(b, sizeof b, "\\u%04x", c); s += b; i++; }
    else if (const int k = utf8_len(x, i)) { s.append(x, i, (size_t)k); i += (size_t)k; }
    else { snprintf(b, sizeof b, "\\udc%02x", c); s += b; i++; }
  }
  return s + "\"";
}
// A list of JSON strings without its closing bracket: the format string of the caller ends it (as for "argv").
inline std::string jstrs_open(const std::vector<std::string>& v) { std::string a = "["; for (size_t i = 0; i < v.size(); i++) a += (i ? ", " : "") + jstr(v[i]); return a; }
inline std::vector<std::string> split_words(const std::string& s) {
  std::vector<std::string> w; std::string cur;
  for (char c : s) { if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { if (!cur.empty()) { w.push_back(cur); cur.clear(); } } else cur += c; }
  if (!cur.empty()) w.push_back(cur);
  return w;
}
// A path as written, without repeated or trailing slashes and without "./" steps: enough to notice that two runs of
// one process name the same directory (links are not followed).
inline std::string plain_path(const std::string& p) {
  std::vector<std::string> parts; std::string cur;
  for (size_t i = 0; i <= p.size(); i++) { if (i == p.size() || p[i] == '/') { if (!cur.empty() && cur != ".") parts.push_back(cur); cur.clear(); } else cur += p[i]; }
  std::string o = !p.empty() && p[0] == '/' ? "/" : "";
  for (size_t i = 0; i < parts.size(); i++) o += (i ? "/" : "") + parts[i];
  return o.empty() ? "." : o;
}
// The directory a path names, for the question whether two runs of one process would write to the same place: the
// real path if it exists, else the real path of its parent with the last name appended, else the plain spelling.
// Links and '..' steps are followed as far as the file system has them.
inline std::string canon_dir(const std::string& p) {
  char buf[PATH_MAX];
  if (realpath(p.c_str(), buf)) return buf;
  const std::string q = plain_path(p); const size_t cut = q.rfind('/');
  const std::string parent = cut == std::string::npos ? "." : (cut == 0 ? "/" : q.substr(0, cut)), last = cut == std::string::npos ? q : q.substr(cut + 1);
  if (last != ".." && realpath(parent.c_str(), buf)) return std::string(buf) + (std::string(buf) == "/" ? "" : "/") + last;
  return q;
}
inline bool file_exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode); }
inline void mkdir_p(const std::string& d) {
  if (d.empty()) host::die("an empty directory name");
  for (size_t i = 1; i <= d.size(); i++)
    if (i == d.size() || d[i] == '/') {
      const std::string p = d.substr(0, i);
      if (mkdir(p.c_str(), 0777) != 0 && errno != EEXIST) host::die("cannot create " + p + ": " + strerror(errno));
    }
  struct stat st; if (stat(d.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) host::die(d + " is not a directory");
}

// ------------------------------------------------------------------------------------------------ digest of a prompt store
// SHA-256 (FIPS 180-4), host only.
struct Sha256 {
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  unsigned char buf[64]; size_t nbuf = 0; unsigned long long total = 0;
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const unsigned char* p) {
    static const uint32_t K[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) | ((uint32_t)p[4 * i + 2] << 8) | (uint32_t)p[4 * i + 3];
    for (int i = 16; i < 64; i++) { const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3), s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10); w[i] = w[i - 16] + s0 + w[i - 7] + s1; }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const void* data, size_t n) {
    const unsigned char* p = (const unsigned char*)data; total += n;
    while (n) { const size_t k = std::min(n, (size_t)64 - nbuf); memcpy(buf + nbuf, p, k); nbuf += k; p += k; n -= k; if (nbuf == 64) { block(buf); nbuf = 0; } }
  }
  std::string hex() {
    const unsigned long long bits = total * 8ull; const unsigned char one = 0x80, zero = 0;
    update(&one, 1); while (nbuf != 56) update(&zero, 1);
    unsigned char len[8]; for (int i = 0; i < 8; i++) len[i] = (unsigned char)(bits >> (56 - 8 * i));
    update(len, 8);
    char out[65]; for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
  }
};
// The digest of a prompt store as tools/stores.py Store.sha256() computes it (DIGEST_FILES there): for each of the
// files below, in this order, the file name and then the file's bytes (a missing file contributes its name only).
// prompts.jsonl is not read by the kernel; it is hashed because the tools read the questions from it, and the digest
// in run.json is what pins the store a run trained on.
static const char* const STORE_FILES[] = {"prompts.idx", "prompts.i32", "tok_letter.i8", "tok_colon.u8", "tok_answer.u8", "prompts.jsonl"};
inline std::string store_sha256(const std::string& dir) {
  Sha256 H; std::vector<unsigned char> chunk(1 << 20);
  for (const char* name : STORE_FILES) {
    H.update(name, strlen(name));
    FILE* f = fopen((dir + "/" + name).c_str(), "rb"); if (!f) continue;
    for (size_t k; (k = fread(chunk.data(), 1, chunk.size(), f)) > 0;) H.update(chunk.data(), k);
    fclose(f);
  }
  return H.hex();
}

// ------------------------------------------------------------------------------------------------ process timing
// Clock ticks since boot at which the operating system created this process (field 22 of /proc/self/stat); 0 if the
// file cannot be read.
inline unsigned long long process_start_ticks() {
  FILE* f = fopen("/proc/self/stat", "r"); if (!f) return 0;
  char buf[4096]; const size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
  const char* p = strrchr(buf, ')'); if (!p) return 0;        // field 2, the command name, may itself hold spaces and brackets
  int field = 2;
  for (p++; *p;) {
    while (*p == ' ') p++;
    if (!*p || *p == '\n') break;
    if (++field == 22) return strtoull(p, nullptr, 10);
    while (*p && *p != ' ') p++;
  }
  return 0;
}
// Seconds since the operating system created this process (the pid), to one clock tick (usually 10 ms); -1 if it
// cannot be known. Counted from there, set-up includes loading the program and its libraries. A shell that ends with
// megarl (bash -c '...; megarl ...') replaces itself with it, and megarl then inherits the shell's age: megarl must be
// started as a process of its own (main records the age and warns).
inline double process_age_s() {
  const unsigned long long st = process_start_ticks(); const long hz = sysconf(_SC_CLK_TCK); struct timespec ts;
  if (!st || hz <= 0 || clock_gettime(CLOCK_BOOTTIME, &ts) != 0) return -1;
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9 - (double)st / (double)hz;
}
// A name for this process that two processes of an experiment will not share: its pid and the second it was created.
inline std::string default_process_id() {
  const double age = process_age_s(); struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  const long long born = llround((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9 - (age > 0 ? age : 0.0));
  return "p" + std::to_string((long long)getpid()) + "-" + std::to_string(born);
}

// ------------------------------------------------------------------------------------------------ console file of a run
// While open, everything the program prints through stdout and stderr also goes to a file, so that a run directory
// keeps the text of its own run (several runs of one process share the terminal). The two streams are replaced by
// streams that hand every piece to the terminal and to the file in the order in which it is written, in the printing
// thread itself: what the terminal shows and what the file holds are the same text in the same order, and a program
// that ends through exit() inside a run (host::die, MK_CK) loses nothing, because exit() flushes these streams like
// any other. Only C stdio output is seen (megarl has no other); assigning stdout and stderr is a property of glibc.
struct Console {
  struct Sink { int term, file; int* warned; };
  int file = -1, warned = 0; Sink sink[2]; FILE* own[2] = {nullptr, nullptr}; FILE* saved[2] = {nullptr, nullptr};
  static bool write_all(int fd, const char* p, size_t n) { while (n > 0) { const ssize_t k = ::write(fd, p, n); if (k < 0) { if (errno == EINTR) continue; return false; } p += k; n -= (size_t)k; } return true; }
  // The file first: if the terminal side is gone (a closed pipe ends the program with SIGPIPE), the run directory still
  // has the text. A file that cannot be written (a full disk) is said once on the terminal, not passed over.
  static ssize_t put(void* c, const char* buf, size_t n) {
    Sink* s = (Sink*)c;
    if (!write_all(s->file, buf, n) && !*s->warned) { *s->warned = 1; const char m[] = "megarl: warning: console.log cannot be written; the text of this run goes to the terminal only from here on\n"; write_all(2, m, sizeof m - 1); }
    write_all(s->term, buf, n);
    return (ssize_t)n;
  }
  ~Console() { close(); }                       // a return out of a run must not leave stdout and stderr pointing into this object
  bool open(const std::string& path) {
    if (file >= 0) return false;
    warned = 0;
    file = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0666); if (file < 0) return false;
    fflush(stdout); fflush(stderr);
    cookie_io_functions_t io; memset(&io, 0, sizeof io); io.write = put;
    for (int s = 0; s < 2; s++) { sink[s].term = s + 1; sink[s].file = file; sink[s].warned = &warned; own[s] = fopencookie(&sink[s], "w", io); }
    if (!own[0] || !own[1]) { for (int s = 0; s < 2; s++) if (own[s]) { fclose(own[s]); own[s] = nullptr; } ::close(file); file = -1; return false; }
    setvbuf(own[0], nullptr, _IOLBF, 0); setvbuf(own[1], nullptr, _IONBF, 0);             // as megarl sets the real ones: whole lines, errors at once
    saved[0] = stdout; saved[1] = stderr; stdout = own[0]; stderr = own[1];
    return true;
  }
  void close() {
    if (file < 0) return;
    fflush(stdout); fflush(stderr);
    stdout = saved[0]; stderr = saved[1];
    for (int s = 0; s < 2; s++) { fclose(own[s]); own[s] = nullptr; }
    ::close(file); file = -1;
  }
};

// ------------------------------------------------------------------------------------------------ options
struct Opt {
  std::string weights = "data/qwen05b", prompts = "data/prompts/multitask4_train", out = "results/run";
  int updates = 250, nroll = 4, ppj = 2, ngrp = 16, bind = 0, use_per_task = 0;
  std::string layout = "share"; int train_sms = -1;
  long long trace_cap = 1 << 22, log_cap = 1 << 18, tok_cap = -1;
  double deadline_s = 600; int dry = 0, audit_verbose = 1, overwrite = 0;
  unsigned long long prompt_seed = 1; double budget_s = 0;
  Hyper hp = host::default_hyper(); bool target_set = false, out_set = false;
  std::vector<std::string> run_argv;            // the words of this run's --run option; empty if the run is the command line itself
};
struct Proc {
  std::vector<std::string> argv;                // the command line of the process
  std::string id;                               // --process-id, or default_process_id()
  Opt base;                                     // the options outside --run, with nothing derived from them yet
  std::vector<Opt> runs;                        // the runs, each complete and checked (at least one)
  bool listed = false;                          // true if the runs came from --run options
};

inline void usage() {
  printf(
    "megarl: persistent-kernel RL (rollout engines + GRPO learner in one launch)\n"
    "The defaults are the configuration of the shipped headline run.\n"
    "  --weights DIR        exported model (default data/qwen05b)\n"
    "  --prompts DIR        prompt store to train on (default data/prompts/multitask4_train)\n"
    "  --out DIR            run directory to write (default results/run)\n"
    "  --updates N          optimiser updates to run (default 250)\n"
    "  --nroll N            rollout engines, each its own job with its own blocks (default 4)\n"
    "  --ppj N              pipelines advanced in lockstep inside one engine (default 2)\n"
    "  --ngrp N             prompt groups per pipeline, 1..16 (default 16); a group is 4 answers\n"
    "  --bind 0|1           1: pipeline p only admits environment p mod (number of environments); 0: any environment (default)\n"
    "  --lag N              largest policy lag a trained group may have (default 8; 0 = strictly on-policy)\n"
    "  --target N           groups (generating + queued) allowed per environment (default: lag * groups per environment per update, at least 4)\n"
    "  --pair-mode 0|1|2    0: balanced batches only; 1: fall back to any eligible groups after --select-wait-us;\n"
    "                       2: admit and select the environment that has been trained on least (default)\n"
    "  --select-wait-us N --starve-ms N --deadline-s S\n"
    "  --layout share|split share: first block of every SM generates, second block of --train-sms SMs trains (the rest generate)\n"
    "                       split: the last --train-sms SMs train with both blocks, the others generate\n"
    "  --train-sms N        default: half of the SMs (share), a quarter (split)\n"
    "  --temperature T --lr X --clip X --wd X --grad-clip X --norm-mode 0|1 --seed N\n"
    "                       defaults 1, 4e-5, 0.2, 0, 1.0, 0, 1\n"
    "  --require-eos 0|1    1: a sequence that does not end with an end-of-sequence token gets reward 0 (default)\n"
    "  --ksY N --ksD N --ksQ N --ksO N --ksG N --tksV N    split-K chunk counts (defaults 4, 4, 2, 2, 1, 16)\n"
    "  --use-per-task N     use only the first N prompts of each environment (0 = all)\n"
    "  --prompt-seed N      permute the order in which each environment's prompts are used (0 = store order; default 1)\n"
    "  --budget-s S         stop after S seconds of kernel time and count that as a complete run; --updates is then an upper bound\n"
    "  --count-held 0|1     1: groups the learner holds count against --target (with --lag 0 and --target = groups per environment\n"
    "                       per update this is the synchronous loop: generate a batch, train on it, publish, repeat)\n"
    "  --alternate 0|1      1: the engines wait while the learner trains on a batch (generation and training take turns)\n"
    "  --no-train 0|1       1: the learner's own phases do nothing; batches are consumed and the unchanged weights are published\n"
    "  --trace-cap N --log-cap N --tok-cap N\n"
    "  --dry 1              build everything, print the job graphs of the first run, do not launch, write nothing\n"
    "  --audit-verbose 0|1  1: print the end-of-run audit summary and its first violations of each kind (default)\n"
    "  --overwrite 1        megarl refuses an --out directory that already holds a run (run.json); with this it replaces it\n"
    "  --run \"OPTIONS\"      one run of several in this process (repeat it). The model, the prompt store and the prompt caches are\n"
    "                       loaded once; every run gets freshly allocated device state. OPTIONS are the options above, split at\n"
    "                       blanks (there is no quoting: a value cannot hold a blank), on top of those given outside --run,\n"
    "                       which hold for every run wherever they stand on the line; each run needs its own --out. Not inside --run:\n"
    "                       --weights --prompts --use-per-task --dry --overwrite --process-id\n"
    "  --process-id NAME    name of this process in run.json (default: pid and start time)\n");
}

// Numbers are read strictly: the whole word must be the number, so that "--lr 1e-4," or "--lag 8x" stop the program
// instead of becoming some other value.
inline bool digit(char c) { return c >= '0' && c <= '9'; }
inline long long to_ll(const std::string& k, const char* v) {
  errno = 0; char* e = nullptr; const long long x = strtoll(v, &e, 10);
  if (!(digit(v[0]) || (v[0] == '-' && digit(v[1]))) || *e || errno) host::die(k + " wants an integer, not '" + v + "'");
  return x;
}
inline int to_i(const std::string& k, const char* v) {
  const long long x = to_ll(k, v);
  if (x < -2147483647ll - 1 || x > 2147483647ll) host::die(k + ": " + v + " does not fit an int");
  return (int)x;
}
inline unsigned long long to_ull(const std::string& k, const char* v) {
  errno = 0; char* e = nullptr; const unsigned long long x = strtoull(v, &e, 10);
  if (!digit(v[0]) || *e || errno) host::die(k + " wants a non-negative integer, not '" + v + "'");
  return x;
}
inline double to_d(const std::string& k, const char* v) {
  errno = 0; char* e = nullptr; const double x = strtod(v, &e);
  const bool starts = digit(v[0]) || ((v[0] == '-' || v[0] == '.') && digit(v[1])) || (v[0] == '-' && v[1] == '.' && digit(v[2]));
  const bool hex = strchr(v, 'x') || strchr(v, 'X');             // strtod reads "0x10" as 16
  if (!starts || hex || *e || errno || !std::isfinite(x)) host::die(k + " wants a number, not '" + v + "'");
  return x;
}
// A number that is stored as a float must be one: 1e39 is a double and an infinite float (run.json would then hold
// 'inf', which no reader loads), and 1e-50 would become 0 without a word.
inline float to_f(const std::string& k, const char* v) {
  const double x = to_d(k, v); const float f = (float)x;
  if (!std::isfinite(f) || (x != 0 && f == 0)) host::die(k + ": " + v + " is not a single-precision number");
  return f;
}

// Apply the options a[first..] to o. Outside --run (in_run false) the process-level options are taken too: --run
// strings are appended to run_specs and --process-id is stored in process_id.
inline void apply_args(Opt& o, const std::vector<std::string>& a, size_t first, bool in_run, std::string* process_id, std::vector<std::string>* run_specs) {
  for (size_t i = first; i < a.size(); i++) {
    const std::string& k = a[i];
    if (k == "--help" || k == "-h") { if (in_run) host::die(k + " cannot be given inside --run"); usage(); exit(0); }
    if (i + 1 >= a.size()) host::die("missing value after " + k);
    const char* v = a[++i].c_str();
    if (v[0] == '-' && v[1] == '-' && k != "--run") host::die(k + " is followed by the option " + v + ", not by its value");
    if (in_run && (k == "--weights" || k == "--prompts" || k == "--use-per-task" || k == "--dry" || k == "--overwrite" || k == "--run" || k == "--process-id"))
      host::die(k + " belongs to the process and cannot be given inside --run");
    if (k == "--weights") o.weights = v; else if (k == "--prompts") o.prompts = v; else if (k == "--out") { o.out = v; o.out_set = true; }
    else if (k == "--updates") o.updates = to_i(k, v); else if (k == "--nroll") o.nroll = to_i(k, v); else if (k == "--ppj") o.ppj = to_i(k, v);
    else if (k == "--ngrp") o.ngrp = to_i(k, v); else if (k == "--bind") o.bind = to_i(k, v); else if (k == "--use-per-task") o.use_per_task = to_i(k, v);
    else if (k == "--lag") o.hp.lag = to_i(k, v); else if (k == "--target") { o.hp.target_per_task = to_i(k, v); o.target_set = true; }
    else if (k == "--pair-mode") o.hp.pair_mode = to_i(k, v); else if (k == "--select-wait-us") o.hp.select_wait_ns = 1000ull * to_ull(k, v);
    else if (k == "--starve-ms") o.hp.starve_ns = 1000000ull * to_ull(k, v); else if (k == "--deadline-s") o.deadline_s = to_d(k, v);
    else if (k == "--layout") o.layout = v; else if (k == "--train-sms") o.train_sms = to_i(k, v);
    else if (k == "--temperature") { o.hp.temperature = to_f(k, v); o.hp.inv_temperature = 1.f / o.hp.temperature; }
    else if (k == "--lr") o.hp.lr = to_f(k, v); else if (k == "--clip") o.hp.clip_eps = to_f(k, v);
    else if (k == "--wd") o.hp.wd = to_f(k, v); else if (k == "--norm-mode") o.hp.norm_mode = to_i(k, v); else if (k == "--seed") o.hp.seed = to_ull(k, v);
    else if (k == "--ksY") o.hp.ksY = to_i(k, v); else if (k == "--ksD") o.hp.ksD = to_i(k, v);
    else if (k == "--ksQ") o.hp.ksQ = to_i(k, v); else if (k == "--ksO") o.hp.ksO = to_i(k, v); else if (k == "--ksG") o.hp.ksG = to_i(k, v);
    else if (k == "--tksV") o.hp.tksV = to_i(k, v);
    else if (k == "--trace-cap") o.trace_cap = to_ll(k, v); else if (k == "--log-cap") o.log_cap = to_ll(k, v); else if (k == "--tok-cap") o.tok_cap = to_ll(k, v);
    else if (k == "--dry") o.dry = to_i(k, v); else if (k == "--audit-verbose") o.audit_verbose = to_i(k, v); else if (k == "--overwrite") o.overwrite = to_i(k, v);
    else if (k == "--prompt-seed") o.prompt_seed = to_ull(k, v);
    else if (k == "--budget-s") o.budget_s = to_d(k, v);
    else if (k == "--count-held") o.hp.count_held = to_i(k, v); else if (k == "--alternate") o.hp.alternate = to_i(k, v); else if (k == "--no-train") o.hp.no_train = to_i(k, v);
    else if (k == "--require-eos") o.hp.require_eos = to_i(k, v); else if (k == "--grad-clip") o.hp.grad_clip = to_f(k, v);
    else if (k == "--run" && run_specs) run_specs->push_back(v); else if (k == "--process-id" && process_id) *process_id = v;
    else host::die("unknown option " + k + " (try --help)");
  }
}
// Check a run's options and derive what follows from them (the admission limit when --target was not given, the
// deadline, the token-log capacity). The number of training SMs waits for the device (megarl.cu).
inline void finish(Opt& o) {
  if (o.nroll < 1 || o.ppj < 1 || o.nroll * o.ppj > MAXPIPE) host::die("nroll * ppj must be between 1 and " + std::to_string(MAXPIPE));
  if (o.ngrp < 1 || o.ngrp > MAXGRP) host::die("ngrp must be between 1 and " + std::to_string(MAXGRP));
  if (o.updates < 1) host::die("updates must be at least 1");
  if (o.layout != "share" && o.layout != "split") host::die("layout must be share or split");
  if (o.out.empty()) host::die("--out names no directory");
  if (o.use_per_task < 0) host::die("use-per-task must not be negative");
  if (o.trace_cap < 0 || o.log_cap < 0) host::die("trace-cap and log-cap must not be negative");
  if (!(o.hp.temperature > 0.f)) host::die("temperature must be positive");
  const int per_update = std::max(1, TGRP / NTASKS);
  if (!o.target_set) o.hp.target_per_task = std::max(per_update, std::min(1 << 20, std::max(1, o.hp.lag) * per_update));
  // Only with --count-held 1 does the limit bound what the learner can collect: without it a group the learner has
  // taken no longer counts, the engines admit again, and any limit of at least 1 fills a batch.
  if (o.hp.count_held && (long long)o.hp.target_per_task * NTASKS < TGRP)
    host::die("target " + std::to_string(o.hp.target_per_task) + " per environment with --count-held 1: fewer groups can be in the system than the " + std::to_string(TGRP) + " of one update, so the learner could never fill a batch");
  if (o.budget_s < 0) host::die("budget-s must not be negative");
  if (!(o.deadline_s > 0)) host::die("deadline-s must be positive");
  // the budget, when there is one, is the deadline: megarl counts that stop as a complete run
  const double limit_s = o.budget_s > 0 ? o.budget_s : o.deadline_s;
  if (limit_s > 1.8e10) host::die("the deadline does not fit 64-bit nanoseconds");
  o.hp.deadline_ns = (u64)(limit_s * 1e9);
  if (o.tok_cap < 0) o.tok_cap = std::min(o.log_cap, 1ll << 22);
  if (o.tok_cap > o.log_cap || o.tok_cap > (1ll << 22)) host::die("tok-cap must be at most log-cap and at most 2^22 (rlc::host_alloc)");
  host::check_hyper(o.hp);
}
inline Proc parse_process(int argc, char** argv) {
  Proc pr; for (int i = 0; i < argc; i++) pr.argv.push_back(argv[i]);
  std::vector<std::string> specs;
  host::die_context() = "";
  apply_args(pr.base, pr.argv, 1, false, &pr.id, &specs);
  if (pr.id.empty()) pr.id = default_process_id();
  if (specs.empty()) { Opt o = pr.base; finish(o); pr.runs.push_back(o); return pr; }
  pr.listed = true;
  if (pr.base.out_set) host::die("--out outside --run has no meaning once --run is used: every --run names its own --out");
  for (size_t r = 0; r < specs.size(); r++) {
    host::die_context() = "--run number " + std::to_string(r + 1) + " (\"" + specs[r] + "\"): ";
    Opt o = pr.base; o.run_argv = split_words(specs[r]);
    apply_args(o, o.run_argv, 0, true, nullptr, nullptr);
    if (!o.out_set) host::die("it needs its own --out");
    finish(o); pr.runs.push_back(o);
  }
  host::die_context() = "";
  for (size_t i = 0; i < pr.runs.size(); i++) for (size_t j = i + 1; j < pr.runs.size(); j++)
    if (canon_dir(pr.runs[i].out) == canon_dir(pr.runs[j].out))
      host::die("--run number " + std::to_string(i + 1) + " and number " + std::to_string(j + 1) + " write to the same directory " + pr.runs[i].out);
  return pr;
}

// ------------------------------------------------------------------------------------------------ the record of a run
struct JobRec { std::string name; int blocks = 0; double steps = 0; u64 busy_ns = 0; std::vector<std::string> ph_name; std::vector<int> ph_kind, ph_tasks; std::vector<u64> ph_busy; };
struct RunRecord {
  Opt o;                                        // as finished, with train_sms resolved
  int npipe = 0, nbuf = 0, nslots = 0; std::vector<long long> pipe_task, prompt_ids, prompt_order[NTASKS];
  std::string prompt_sha, process_id; std::vector<std::string> argv; int run_index = 0, run_count = 1;
  const rlc::Snapshot* S = nullptr; int violations = 0, steps_done = 0;
  u64 t_end = 0, t_exit = 0; double wall = 0;
  // Set-up, in seconds. pre_load_s: creation of the process (or entry of main, see from_creation) to the start of
  // loading; load_s: model and prompt store; cache_s: building the prompt caches this run used; prep_s: this run's own
  // device state and job graphs, up to the launch; setup_s: their sum = creation of the process to the launch as if
  // this run were the only one of its process.
  double pre_load_s = 0, load_s = 0, cache_s = 0, prep_s = 0, setup_s = 0; int from_creation = 0, cache_shared = 0;
  double pre_main_s = 0;                        // how old the process was when main was entered (part of pre_load_s); main treats
                                                // more than 1 s as a shell's inherited age
  int train_blocks = 0, bad_sm = 0; std::vector<long long> roll_blocks; std::vector<JobRec> jobs;     // jobs: the engines, then the learner
};

inline void write_run_json(const std::string& dir, const RunRecord& R) {
  const Opt& o = R.o; const Hyper& hp = o.hp; const rlc::Snapshot& S = *R.S;
  { FILE* f = fopen((dir + "/run.json").c_str(), "w"); if (!f) host::die("cannot write run.json");
    fprintf(f, "{\n \"config\": {\"temperature\": %.9g, \"clip_eps\": %.9g, \"adv_eps\": %.9g, \"lr\": %.9g, \"beta1\": %.9g, \"beta2\": %.9g, \"adam_eps\": %.9g, \"wd\": %.9g, \"grad_clip\": %.9g, \"require_eos\": %d,\n",
            hp.temperature, hp.clip_eps, hp.adv_eps, hp.lr, hp.beta1, hp.beta2, hp.adam_eps, hp.wd, hp.grad_clip, hp.require_eos);
    fprintf(f, "  \"lag\": %d, \"norm_mode\": %d, \"pair_mode\": %d, \"select_wait_ns\": %llu, \"starve_ns\": %llu, \"target_per_task\": %d, \"seed\": %llu, \"deadline_ns\": %llu,\n",
            hp.lag, hp.norm_mode, hp.pair_mode, (unsigned long long)hp.select_wait_ns, (unsigned long long)hp.starve_ns, hp.target_per_task, (unsigned long long)hp.seed, (unsigned long long)hp.deadline_ns);
    fprintf(f, "  \"ksY\": %d, \"ksD\": %d, \"ksQ\": %d, \"ksO\": %d, \"ksG\": %d, \"tksV\": %d,\n", hp.ksY, hp.ksD, hp.ksQ, hp.ksO, hp.ksG, hp.tksV);
    fprintf(f, "  \"nroll\": %d, \"ppj\": %d, \"npipe\": %d, \"ngrp\": %d, \"pipe_task\": %s, \"layout\": \"%s\", \"train_sms\": %d, \"updates\": %d, \"tgrp\": %d, \"gsz\": %d, \"maxgen\": %d, \"nbuf\": %d, \"nslots\": %d,\n",
            o.nroll, o.ppj, R.npipe, o.ngrp, jlist(R.pipe_task).c_str(), o.layout.c_str(), o.train_sms, o.updates, TGRP, GSZ, MAXGEN, R.nbuf, R.nslots);
    fprintf(f, "  \"bind\": %d, \"use_per_task\": %d, \"prompt_seed\": %llu, \"budget_s\": %.17g,\n", o.bind, o.use_per_task, o.prompt_seed, o.budget_s);
    fprintf(f, "  \"count_held\": %d, \"alternate\": %d, \"no_train\": %d,\n", hp.count_held, hp.alternate, hp.no_train);
    fprintf(f, "  \"weights\": %s, \"prompt_set\": %s, \"prompt_set_sha256\": \"%s\"},\n", jstr(o.weights).c_str(), jstr(o.prompts).c_str(), R.prompt_sha.c_str());
    fprintf(f, " \"argv\": %s],\n", jstrs_open(R.argv).c_str());
    fprintf(f, " \"run_argv\": %s],\n", jstrs_open(o.run_argv).c_str());
    fprintf(f, " \"process\": {\"id\": %s, \"runs\": %d, \"index\": %d},\n", jstr(R.process_id).c_str(), R.run_count, R.run_index);
    fprintf(f, " \"build\": {\"tgrp\": %d, \"ntasks\": %d, \"mid_layer\": %d, \"compiled\": \"%s %s\"},\n", TGRP, NTASKS, LMID, __DATE__, __TIME__);
    fprintf(f, " \"prompt_ids\": %s,\n", jlist(R.prompt_ids).c_str());
    fprintf(f, " \"prompt_order\": [");
    for (int t = 0; t < NTASKS; t++) fprintf(f, "%s%s", t ? ", " : "", jlist(R.prompt_order[t]).c_str());
    fprintf(f, "],\n");
    fprintf(f, " \"counters\": {\"admitted\": %s, \"finished\": %s, \"drop_zerovar\": %s, \"drop_stale_inflight\": %s, \"drop_stale_queued\": %s, \"consumed\": %s, \"inflight\": %s, \"queued\": %s,\n",
            jlist(ll(S.admitted)).c_str(), jlist(ll(S.finished)).c_str(), jlist(ll(S.drop_zerovar)).c_str(), jlist(ll(S.drop_stale_inflight)).c_str(), jlist(ll(S.drop_stale_queued)).c_str(),
            jlist(ll(S.consumed)).c_str(), jlist(ll(S.inflight)).c_str(), jlist(ll(S.queued)).c_str());
    fprintf(f, "  \"held\": %d, \"errors\": %s, \"stats\": %s, \"uid_next\": %d, \"tokens_generated\": %llu, \"stop_reason\": %d, \"stop_reason_dev\": %d, \"train_ver\": %d, \"audit_violations\": %d},\n",
            S.held, jlist(ll(S.errors)).c_str(), jlist(ll(S.stats)).c_str(), S.uid_next, (unsigned long long)S.tokens_generated, S.stop_reason, S.stop_reason_dev, S.train_ver, R.violations);
    fprintf(f, " \"sizes\": {\"groups\": %zu, \"groups_total\": %llu, \"groups_with_tokens\": %zu, \"trace\": %zu, \"trace_total\": %llu, \"steps\": %d, \"versions\": %zu},\n",
            S.log.size(), (unsigned long long)S.log_total, (size_t)S.groups_with_tokens, S.trace.size(), (unsigned long long)S.trace_total, R.steps_done, (size_t)S.versions);
    fprintf(f, " \"timing\": {\"kernel_ns\": %llu, \"exit_ns\": %llu, \"wall_s\": %.6f, \"load_s\": %.3f, \"prompt_cache_s\": %.3f,\n", (unsigned long long)R.t_end, (unsigned long long)R.t_exit, R.wall, R.load_s, R.cache_s);
    fprintf(f, "  \"pre_main_s\": %.3f, \"pre_load_s\": %.3f, \"prep_s\": %.3f, \"setup_s\": %.3f, \"setup_from_process_creation\": %d, \"prompt_cache_shared\": %d,\n", R.pre_main_s, R.pre_load_s, R.prep_s, R.setup_s, R.from_creation, R.cache_shared);
    fprintf(f, "  \"train_blocks\": %d, \"roll_blocks\": %s, \"sms_without_two_blocks\": %d,\n  \"jobs\": [\n", R.train_blocks, jlist(R.roll_blocks).c_str(), R.bad_sm);
    // per job: blocks, steps completed, total busy time of its blocks, and per phase the task count and the time spent inside tasks
    for (size_t j = 0; j < R.jobs.size(); j++) {
      const JobRec& J = R.jobs[j];
      fprintf(f, "   {\"name\": \"%s\", \"blocks\": %d, \"steps\": %.0f, \"busy_ns\": %llu, \"phases\": [", J.name.c_str(), J.blocks, J.steps, (unsigned long long)J.busy_ns);
      for (size_t i = 0; i < J.ph_name.size(); i++) fprintf(f, "%s[%s, %d, %d, %llu]", i ? ", " : "", jstr(J.ph_name[i]).c_str(), J.ph_kind[i], J.ph_tasks[i], (unsigned long long)J.ph_busy[i]);
      fprintf(f, "]}%s\n", j + 1 < R.jobs.size() ? "," : "");
    }
    fprintf(f, "  ]}\n}\n");
    const bool bad = ferror(f) != 0;                             // a full disk must not leave a cut run.json behind a run that says it is complete
    if (fclose(f) != 0 || bad) host::die("run.json could not be written in full (" + std::string(strerror(errno)) + ")"); }
}

}  // namespace mgo
