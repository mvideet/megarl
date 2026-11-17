#!/usr/bin/env python3
"""Reader of a MegaRL prompt store (docs/FORMATS.md section 2) and the token-hash test that keeps stores apart.
numpy only.

    from stores import Store, overlap
    s = Store("data/prompts/multitask4_train")
    s.n, s.ids, s.task, s.key, s.len            # per prompt
    s.names                                     # environment names in task order (envs.json)
    s.prompt(i)                                 # token ids of prompt i (int32 view)
    s.hashes()                                  # one 64-bit hash per prompt, of its token ids
    s.reward(tokens, key)                       # the reward rule of docs/FORMATS.md section 2 (both verifier modes)
    overlap(a, b)                               # prompts of store a whose token hash also occurs in store b
    run_overlap(run_json, b)                    # prompts a run trained on that also occur in store b

As a program:
    python tools/stores.py info <store>...
    python tools/stores.py overlap <store_a> <store_b>     exit status 1 if any prompt is shared
    python tools/stores.py selftest
"""
import hashlib
import json
import os
import sys

import numpy as np

VOCAB = 151936
MAXGEN = 64                                   # generated tokens of one response; a longer token list is cut here
TASK_NAMES = ("math", "science", "sciq", "csqa")   # the kernel's names for a store without envs.json

# The files of the store digest, in the order they are hashed (store_sha256 in src/megarl_opts.cuh hashes the same).
# The kernel does not read prompts.jsonl; it is covered so that a digest also pins the questions.
DIGEST_FILES = ("prompts.idx", "prompts.i32", "tok_letter.i8", "tok_colon.u8", "tok_answer.u8", "prompts.jsonl")


class StoreError(Exception):
    pass


class Store:
    def __init__(self, path):
        self.path = path
        idx = os.path.join(path, "prompts.idx")
        if not os.path.isfile(idx):
            raise StoreError(f"{path}: no prompts.idx")
        with open(idx) as f:
            head = f.readline().split()
            if len(head) != 3:
                raise StoreError(f"{idx}: first line must be 'n_prompts max_len verifier_mode'")
            self.n, self.max_len, self.verifier_mode = (int(x) for x in head)
            rows = [ln.split() for ln in f if ln.strip()]
        if len(rows) != self.n or any(len(r) != 5 for r in rows):
            raise StoreError(f"{idx}: expected {self.n} lines of 'id task key len offset'")
        a = np.asarray(rows, dtype=np.int64).reshape(self.n, 5)
        self.ids, self.task, self.key, self.len, self.offset = (a[:, k].copy() for k in range(5))
        p = os.path.join(path, "prompts.i32")
        self.tok = np.memmap(p, dtype="<i4", mode="r") if os.path.getsize(p) else np.zeros(0, "<i4")
        envs = os.path.join(path, "envs.json")
        if os.path.isfile(envs):
            with open(envs) as f:
                self.names = list(json.load(f)["names"])
        else:
            self.names = list(TASK_NAMES)
        self._tables = None
        self._hashes = None

    def prompt(self, i):
        o, n = int(self.offset[i]), int(self.len[i])
        return self.tok[o:o + n]

    def hashes(self):
        """uint64 per prompt: the first 8 bytes of sha256 over the prompt's little-endian int32 token ids."""
        if self._hashes is None:
            h = np.empty(self.n, dtype=np.uint64)
            for i in range(self.n):
                d = hashlib.sha256(np.ascontiguousarray(self.prompt(i), dtype="<i4").tobytes()).digest()
                h[i] = int.from_bytes(d[:8], "little")
            self._hashes = h
        return self._hashes

    def sha256(self):
        """Digest of the whole store (docs/FORMATS.md section 2): for each file of DIGEST_FILES, in that order, its
        name and then its bytes; a missing file contributes its name only. run.json records the digest megarl
        computes as config.prompt_set_sha256."""
        h = hashlib.sha256()
        for name in DIGEST_FILES:
            p = os.path.join(self.path, name)
            h.update(name.encode())
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
        return h.hexdigest()

    def tables(self):
        """(tok_letter int8 [V], tok_colon uint8 [V], tok_answer uint8 [V])"""
        if self._tables is None:
            t = (np.fromfile(os.path.join(self.path, "tok_letter.i8"), dtype=np.int8),
                 np.fromfile(os.path.join(self.path, "tok_colon.u8"), dtype=np.uint8),
                 np.fromfile(os.path.join(self.path, "tok_answer.u8"), dtype=np.uint8))
            if any(len(x) != VOCAB for x in t):
                raise StoreError(f"{self.path}: token tables must have {VOCAB} entries")
            self._tables = t
        return self._tables

    def _classes(self, y):
        """(letter, colon, answer) class of each of the first MAXGEN tokens of y. A token id outside the vocabulary
        has no class, as in the kernel."""
        letter, colon, answer = self.tables()
        y = np.asarray(y, dtype=np.int64).reshape(-1)[:MAXGEN]
        ok = (y >= 0) & (y < VOCAB)
        yy = np.where(ok, y, 0)
        return np.where(ok, letter[yy], -1), np.where(ok, colon[yy], 0), np.where(ok, answer[yy], 0)

    def find_answer(self, y):
        """Position and letter (0..3) of the scored answer in the generated tokens y under verifier mode 0, or
        (-1, -1): the largest i >= 2 with letter[y[i]] >= 0, colon[y[i-1]] and answer[y[i-2]]. Only the first
        MAXGEN tokens count."""
        letter, colon, answer = self._classes(y)
        for i in range(len(letter) - 1, 1, -1):
            if letter[i] >= 0 and colon[i - 1] == 1 and answer[i - 2] == 1:
                return i, int(letter[i])
        return -1, -1

    def reward(self, y, key, mode=None):
        """Reward of one response (generated tokens, including the end token if produced), as the kernel gives it
        (docs/FORMATS.md section 2): 0 for a key outside 0..3 and for a verifier mode other than 0 or 1."""
        mode = self.verifier_mode if mode is None else mode
        if int(key) not in (0, 1, 2, 3) or mode not in (0, 1):
            return 0.0
        if mode == 1:
            letter = self._classes(y)[0]
            return 1.0 if len(letter) >= 1 and int(letter[0]) == int(key) else 0.0
        _, got = self.find_answer(y)
        return 1.0 if got >= 0 and got == int(key) else 0.0

    def check(self):
        """Structural problems of the store (empty list if none)."""
        bad = []
        if len(set(self.ids.tolist())) != self.n:
            bad.append("duplicate prompt id")
        if self.n and (self.len.max() > self.max_len or self.len.min() < 1):
            bad.append(f"prompt length outside 1..{self.max_len}")
        if self.n and int((self.offset + self.len).max()) > len(self.tok):
            bad.append("prompts.i32 is shorter than the index says")
        if not np.isin(self.task, np.arange(len(self.names))).all():
            bad.append(f"task outside 0..{len(self.names) - 1}")
        if not np.isin(self.key, (0, 1, 2, 3)).all():
            bad.append("key outside 0..3")
        if self.n and int((self.offset + self.len).max()) <= len(self.tok):
            used = np.asarray(self.tok[: int((self.offset + self.len).max())])
            if (used >= VOCAB).any() or (used < 0).any():
                bad.append("token id outside the vocabulary")
        if self.n and (self.offset < 0).any():
            bad.append("negative offset")
        h = self.hashes()
        if len(np.unique(h)) != self.n:
            bad.append(f"{self.n - len(np.unique(h))} prompt(s) repeat another prompt of the same store token for token")
        return bad


def overlap(a, b):
    """Indices of the prompts of store a whose token hash also occurs in store b."""
    return np.nonzero(np.isin(a.hashes(), b.hashes()))[0]


def run_overlap(run_json_path, score_store, train_store=None):
    """Prompts a run trained on that also occur in score_store. run.json names the training store in
    config.prompt_set and lists the ids it used in prompt_ids. train_store overrides the stored path (use it when
    the run was made on another machine). Returns the array of shared training prompt ids."""
    with open(run_json_path) as f:
        meta = json.load(f)
    ts = train_store if train_store is not None else Store(meta["config"]["prompt_set"])
    used = np.asarray(meta.get("prompt_ids", []), dtype=np.int64)
    pos = {int(i): k for k, i in enumerate(ts.ids.tolist())}
    miss = [int(i) for i in used.tolist() if int(i) not in pos]
    if miss:
        raise StoreError(f"run used prompt ids missing from {ts.path}: {miss[:5]}")
    rows = np.asarray([pos[int(i)] for i in used.tolist()], dtype=np.int64)
    shared = np.isin(ts.hashes()[rows], score_store.hashes())
    return used[shared]


def _selftest():
    import tempfile
    rng = np.random.default_rng(0)

    def make(path, prompts, mode=0):
        os.makedirs(path)
        off, lines, toks = 0, [], []
        for i, (task, key, p) in enumerate(prompts):
            lines.append(f"{100 + i} {task} {key} {len(p)} {off}")
            toks.extend(p)
            off += len(p)
        with open(os.path.join(path, "prompts.idx"), "w") as f:
            f.write(f"{len(prompts)} {max(len(p) for _, _, p in prompts)} {mode}\n" + "\n".join(lines) + "\n")
        np.asarray(toks, dtype="<i4").tofile(os.path.join(path, "prompts.i32"))
        letter = np.full(VOCAB, -1, np.int8)
        letter[[32, 33, 34, 35]] = [0, 1, 2, 3]
        colon = np.zeros(VOCAB, np.uint8)
        colon[25] = 1
        answer = np.zeros(VOCAB, np.uint8)
        answer[[16141, 21806]] = 1
        letter.tofile(os.path.join(path, "tok_letter.i8"))
        colon.tofile(os.path.join(path, "tok_colon.u8"))
        answer.tofile(os.path.join(path, "tok_answer.u8"))

    ok = True

    def expect(name, cond):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + name)
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as d:
        pa = [(i % 2, i % 4, rng.integers(0, 1000, size=5 + i).tolist()) for i in range(12)]
        pb = [(i % 2, i % 4, rng.integers(0, 1000, size=7 + i).tolist()) for i in range(8)]
        pb[3] = (1, 2, list(pa[5][2]))                        # one shared prompt
        make(os.path.join(d, "a"), pa)
        make(os.path.join(d, "b"), pb)
        a, b = Store(os.path.join(d, "a")), Store(os.path.join(d, "b"))
        expect("stores load and pass the structural check", not a.check() and not b.check())
        expect("prompt tokens round-trip", a.prompt(5).tolist() == pa[5][2] and b.prompt(7).tolist() == pb[7][2])
        expect("the one shared prompt is found, in both directions",
               overlap(a, b).tolist() == [5] and overlap(b, a).tolist() == [3])
        expect("hashes differ for prompts that differ in one token",
               len(np.unique(a.hashes())) == a.n)
        # reward rule, mode 0: the last 'Answer' ':' letter triple decides
        y = [7, 16141, 25, 32, 9, 21806, 25, 34, 151645]
        expect("mode 0 takes the last pattern", a.find_answer(y) == (7, 2) and a.reward(y, 2) == 1.0 and a.reward(y, 0) == 0.0)
        expect("mode 0 needs all three tokens in a row", a.find_answer([16141, 9, 25, 32]) == (-1, -1) and a.reward([16141, 9, 25, 32], 0) == 0.0)
        expect("mode 0 with the pattern at the very start", a.find_answer([16141, 25, 33]) == (2, 1))
        expect("mode 1 reads the first token", a.reward([33, 1, 2], 1, mode=1) == 1.0 and a.reward([1, 33], 1, mode=1) == 0.0 and a.reward([], 1, mode=1) == 0.0)
        # inputs outside the format get what the kernel gives them
        expect("a token id outside the vocabulary has no class: negative ids do not wrap around, ids >= V do not raise",
               a.find_answer([16141, 25, 32 - VOCAB]) == (-1, -1) and a.find_answer([16141 - VOCAB, 25, 32]) == (-1, -1)
               and a.find_answer([16141, 25, VOCAB + 32]) == (-1, -1) and a.reward([32 - VOCAB], 0, mode=1) == 0.0
               and a.find_answer([-1, 16141, 25, 33, 2 ** 40]) == (3, 1))
        expect("a key outside 0..3 gives reward 0 in both modes", a.reward([16141, 25, 32], -1) == 0.0 and a.reward([16141, 25, 32], 4) == 0.0
               and a.reward([7], -1, mode=1) == 0.0 and a.reward([32], 0, mode=1) == 1.0)
        expect("an unknown verifier mode gives reward 0", a.reward([16141, 25, 32], 0, mode=2) == 0.0 and a.reward([16141, 25, 32], 0, mode=0) == 1.0)
        long_y = [7] * 62 + [16141, 25, 32]
        expect("only the first 64 tokens count", a.find_answer(long_y) == (-1, -1) and a.find_answer(long_y[1:]) == (63, 0) and a.reward(long_y, 0) == 0.0)
        # run_overlap
        rj = os.path.join(d, "run.json")
        with open(rj, "w") as f:
            json.dump({"config": {"prompt_set": os.path.join(d, "a")}, "prompt_ids": [100, 105, 111]}, f)
        expect("a run that trained on the shared prompt is caught", run_overlap(rj, b).tolist() == [105])
        with open(rj, "w") as f:
            json.dump({"config": {"prompt_set": os.path.join(d, "a")}, "prompt_ids": [100, 101, 111]}, f)
        expect("a run that did not is clean", run_overlap(rj, b).tolist() == [])
        pc = list(pa)
        pc[2] = (0, 1, list(pa[9][2]))
        make(os.path.join(d, "c"), pc)
        expect("a repeated prompt inside one store fails the check", len(Store(os.path.join(d, "c")).check()) == 1)
    print("stores selftest:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[1] == "selftest":
        return _selftest()
    if argv[1] == "info":
        rc = 0
        for p in argv[2:]:
            if not os.path.isfile(os.path.join(p, "prompts.idx")):
                print(f"{p}: not a prompt store (no prompts.idx), skipped")
                continue
            s = Store(p)
            bad = s.check()
            per = ", ".join(f"{name} {int((s.task == t).sum())}" for t, name in enumerate(s.names))
            keys = "/".join(str(int((s.key == k).sum())) for k in range(4))
            print(f"{p}: {s.n} prompts ({per}), keys A/B/C/D {keys}, length {int(s.len.min()) if s.n else 0}..{int(s.len.max()) if s.n else 0} tokens "
                  f"(header says longest {s.max_len}), verifier mode {s.verifier_mode}, sha256 {s.sha256()[:16]}")
            for b in bad:
                print("  PROBLEM:", b)
            rc |= 1 if bad else 0
        return rc
    if argv[1] == "overlap" and len(argv) == 4:
        a, b = Store(argv[2]), Store(argv[3])
        sh = overlap(a, b)
        print(f"{len(sh)} of {a.n} prompts of {argv[2]} also occur in {argv[3]}" + (f": ids {a.ids[sh][:10].tolist()}" if len(sh) else ""))
        return 1 if len(sh) else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
