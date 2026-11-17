#!/usr/bin/env python3
"""Reader of a MegaRL run directory (docs/FORMATS.md section 3). numpy only.

src/megarl.cu writes the binary files and mgo::write_run_json in src/megarl_opts.cuh writes run.json; this reader
follows both.

    from rundir import Run
    r = Run("results/headline")
    r.cfg["lr"], r.counters["consumed"], r.timing["kernel_ns"]
    r.groups["terminal"], r.steps["loss"], r.trace["type"]
    r.tokens[i, j, :r.groups["len"][i, j]]      # needs group_tokens.i32
    r.batches()                                 # per update: the uids of its groups in batch order

As a program it prints a short summary and, with --check, runs the consistency checks:
    python tools/rundir.py results/headline [--check]
"""
import json
import os
import sys

import numpy as np

GSZ, MAXGEN = 4, 64

GROUP_DT = np.dtype([("uid", "<i4"), ("task", "<i4"), ("prompt", "<i4"), ("key", "<i4"), ("ver_id", "<i4"),
                     ("terminal", "<i4"), ("train_step", "<i4"), ("lag", "<i4"), ("len", "<i4", (4,)),
                     ("reward", "<f4", (4,)), ("adv", "<f4", (4,)), ("t_admit", "<u8"), ("t_done", "<u8"),
                     ("t_end", "<u8")])
TRACE_DT = np.dtype([("t", "<u8"), ("type", "<i4"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4"), ("d", "<i4"),
                     ("e", "<i4")])
STEP_DT = np.dtype([("t_done", "<u8"), ("loss", "<f4"), ("gnorm", "<f4"), ("reserved", "<f4"),
                    ("max_dlogp", "<f4"), ("info", "<i4", (16,))])
assert GROUP_DT.itemsize == 104 and TRACE_DT.itemsize == 32 and STEP_DT.itemsize == 88

# terminal states of a group (groups.bin 'terminal')
T_CONSUMED, T_ZEROVAR, T_STALE_QUEUED, T_CANCELLED = 5, 6, 7, 8
TERMINAL = {5: "trained", 6: "equal rewards", 7: "stale in queue", 8: "cancelled"}
# trace event types (trace.bin 'type')
EV_ADMIT, EV_GROUP_DONE, EV_ENQUEUE, EV_DROP_ZEROVAR, EV_CANCEL, EV_DROP_STALE = 1, 2, 3, 4, 5, 6
EV_SELECT, EV_PUBLISH, EV_STEP, EV_IDLE, EV_SELECT_WAIT, EV_STOP = 7, 8, 9, 10, 11, 12
STOP = {1: "all updates done", 2: "deadline", 3: "learner starved", 4: "internal error"}
TASKS = ("math", "science", "sciq", "csqa")   # environment names in task order; run.json does not record them


class RunError(Exception):
    pass


class Run:
    """A run directory, loaded lazily. Arrays are read-only memory maps or small in-memory arrays."""

    def __init__(self, path):
        self.path = path
        p = os.path.join(path, "run.json")
        if not os.path.isfile(p):
            raise RunError(f"{path}: no run.json")
        with open(p) as f:
            self.meta = json.load(f)
        self.cfg = self.meta["config"]
        self.counters = self.meta["counters"]
        self.sizes = self.meta["sizes"]
        self.timing = self.meta.get("timing", {})
        self.tgrp = int(self.cfg["tgrp"])
        self.gsz = int(self.cfg.get("gsz", GSZ))
        self.maxgen = int(self.cfg.get("maxgen", MAXGEN))
        self._cache = {}

    # ---- raw files
    def _file(self, name):
        return os.path.join(self.path, name)

    def _map(self, name, dtype, shape_tail=()):
        if name in self._cache:
            return self._cache[name]
        p = self._file(name)
        if not os.path.isfile(p):
            raise RunError(f"{self.path}: missing {name}")
        per = int(np.prod(shape_tail)) * np.dtype(dtype).itemsize if shape_tail else np.dtype(dtype).itemsize
        size = os.path.getsize(p)
        if size % per:
            raise RunError(f"{name}: size {size} is not a multiple of the record size {per}")
        n = size // per
        a = np.memmap(p, dtype=dtype, mode="r", shape=(n,) + tuple(shape_tail)) if n else \
            np.zeros((0,) + tuple(shape_tail), dtype=dtype)
        self._cache[name] = a
        return a

    @property
    def groups(self):
        """groups.bin: one record per group that reached a terminal state, in append order."""
        return self._map("groups.bin", GROUP_DT)

    @property
    def tokens(self):
        """[groups_with_tokens, 4, 64] generated tokens of the first records of groups.bin."""
        return self._map("group_tokens.i32", "<i4", (self.gsz, self.maxgen))

    @property
    def logp(self):
        """[groups_with_tokens, 4, 64] generation-time log-probabilities (same rows as tokens)."""
        return self._map("group_logp.f32", "<f4", (self.gsz, self.maxgen))

    @property
    def trace(self):
        """trace.bin sorted by time (stable), as an in-memory structured array."""
        if "trace_sorted" not in self._cache:
            t = np.array(self._map("trace.bin", TRACE_DT))
            self._cache["trace_sorted"] = t[np.argsort(t["t"], kind="stable")]
        return self._cache["trace_sorted"]

    @property
    def steps(self):
        """steps.bin: one record per completed update."""
        return self._map("steps.bin", STEP_DT)

    # ---- derived views
    def events(self, typ):
        t = self.trace
        return t[t["type"] == typ]

    def batches(self):
        """List over updates of the uids of the update's groups in batch order, from the SELECT trace events
        (a = train step, b = uid, e = index in the batch). Works for any number of groups per update, unlike the
        12 uid fields of steps.bin. An update whose SELECT events are incomplete (trace overflow) gets None."""
        sel = self.events(EV_SELECT)
        n = len(self.steps)
        out = [dict() for _ in range(n)]
        for s, uid, idx in zip(sel["a"].tolist(), sel["b"].tolist(), sel["e"].tolist()):
            if 0 <= s < n:
                out[s][idx] = uid
        res = []
        for d in out:
            res.append([d[i] for i in range(self.tgrp)] if len(d) == self.tgrp and all(i in d for i in range(self.tgrp))
                       else None)
        return res

    def kernel_seconds(self):
        """Device time of the last completed update, in seconds (megarl's 'kernel time')."""
        return float(self.timing.get("kernel_ns", 0)) * 1e-9

    def setup_seconds(self):
        """Seconds from the creation of the megarl process to the launch of the kernel, counted as for the only run
        of a process (timing.setup_s = pre_load_s + load_s + prompt_cache_s + prep_s): a run that shared the model
        and the prompt caches with earlier runs of its process is charged for them as if it had built them itself.
        None if run.json has no setup_s."""
        t = self.timing.get("setup_s")
        return None if t is None else float(t)

    def setup_from_creation(self):
        """True if setup_seconds() starts at the creation of the process, False if megarl could only count from the
        entry of its main function (/proc was not readable), None if run.json does not say."""
        v = self.timing.get("setup_from_process_creation")
        return None if v is None else bool(v)

    def process(self):
        """The megarl process the run was made in: {"id": name, "runs": how many runs the process made, "index":
        this run's place among them, from 0}, or None if run.json has no such object."""
        p = self.meta.get("process")
        return dict(p) if isinstance(p, dict) else None

    def control(self):
        """The control options, 0 where run.json does not have them: count_held (groups the learner holds count
        against the admission limit), alternate (the engines wait while a batch is trained), no_train (the
        learner's own phases are skipped). megarl writes 0 or 1; check() reports anything else."""
        return {k: self.cfg.get(k, 0) for k in ("count_held", "alternate", "no_train")}

    def exit_seconds(self):
        """Device time at which the kernel stopped, in seconds: timing.exit_ns, else the time of the last trace
        event, and never less than kernel_seconds(). The counters of run.json are those at exit, so rates of
        tokens and groups are taken over this time. It differs from kernel_seconds() by more than microseconds
        only for a run that did not end with its last update (a time budget, the deadline, a starved learner)."""
        t = self.timing.get("exit_ns")
        if not t:
            tr = self._map("trace.bin", TRACE_DT)
            t = int(np.asarray(tr["t"]).max()) if len(tr) else 0
        return max(float(t), float(self.timing.get("kernel_ns", 0))) * 1e-9

    # ---- consistency of the files with each other and with run.json (cheap; no model)
    def check(self):
        """Returns a list of problems (empty if none). Structural checks only. group_tokens.i32 and group_logp.f32
        are checked where present; the runs under results/ leave them out."""
        bad = []
        g, st, sz = self.groups, self.steps, self.sizes

        def want(name, got, exp):
            if got != exp:
                bad.append(f"{name}: {got}, expected {exp}")

        want("groups.bin records", len(g), sz["groups"])
        want("steps.bin records", len(st), sz["steps"])
        want("trace.bin records", len(self._map("trace.bin", TRACE_DT)), sz["trace"])
        if os.path.isfile(self._file("group_tokens.i32")):
            want("group_tokens.i32 groups", len(self.tokens), sz["groups_with_tokens"])
        if os.path.isfile(self._file("group_logp.f32")):
            want("group_logp.f32 groups", len(self.logp), sz["groups_with_tokens"])
        want("counters.train_ver", int(self.counters["train_ver"]), len(st))
        if len(g):
            term = np.asarray(g["terminal"])
            if not np.isin(term, tuple(TERMINAL)).all():
                bad.append("groups.bin: terminal outside 5..8")
            if len(set(g["uid"].tolist())) != len(g):
                bad.append("groups.bin: duplicate uid")
            ln = np.asarray(g["len"])
            if (ln < 0).any() or (ln > self.maxgen).any():
                bad.append("groups.bin: len outside 0..maxgen")
            tr = term == T_CONSUMED
            if sz.get("groups_total", len(g)) == len(g):
                want("groups trained on (terminal 5)", int(tr.sum()), int(sum(self.counters["consumed"])))
            if tr.any() and ((g["train_step"][tr] < 0).any() or (g["train_step"][tr] >= max(1, len(st))).any()):
                bad.append("groups.bin: train_step of a trained group outside the completed updates")
        full_trace = sz.get("trace_total", sz["trace"]) == sz["trace"]
        if len(st) and full_trace:
            nb = sum(1 for b in self.batches() if b is not None)
            want("updates with a complete SELECT record", nb, len(st))
        # prompt order: the n-th group admitted for an environment gets entry n mod length of its list, and every
        # counter value is used by exactly one admission, so the prompts of the first N admissions are a known multiset
        order = self.meta.get("prompt_order")
        if order is not None and full_trace:
            adm = self.events(EV_ADMIT)
            for t, name in enumerate(TASKS[:len(order)]):
                got = np.asarray(adm["c"])[np.asarray(adm["b"]) == t]
                want(f"ADMIT events of {name}", len(got), int(self.counters["admitted"][t]))
                lst = np.asarray(order[t], dtype=np.int64)
                if len(lst) == 0:
                    if len(got):
                        bad.append(f"prompt_order of {name} is empty but groups were admitted")
                    continue
                exp = lst[np.arange(len(got)) % len(lst)]
                if not np.array_equal(np.sort(got), np.sort(exp)):
                    bad.append(f"{name}: the prompts of the admitted groups are not the first {len(got)} entries of prompt_order (taken cyclically)")
        pr = self.process()
        whole = lambda x: type(x) is int                    # not a bool, not a fraction
        if "process" in self.meta and pr is None:
            bad.append(f"process is not an object with id, runs and index: {self.meta['process']!r}")
        elif pr is not None and not isinstance(pr.get("id"), str):
            bad.append(f"process: no id that is a string ({pr.get('id')!r})")
        elif pr is not None and not (whole(pr.get("runs")) and whole(pr.get("index")) and 0 <= pr["index"] < pr["runs"]):
            bad.append(f"process: index {pr.get('index')!r} is not a place among {pr.get('runs')!r} runs")
        if self.setup_seconds() is not None:
            parts = [self.timing.get(k) for k in ("pre_load_s", "load_s", "prompt_cache_s", "prep_s")]
            if any(x is None for x in parts) or any(float(x) < 0 for x in parts) or abs(sum(float(x) for x in parts) - self.setup_seconds()) > 0.0025:
                bad.append(f"timing.setup_s {self.setup_seconds()} is not the sum of pre_load_s, load_s, prompt_cache_s and prep_s {parts}")
        # megarl counts the set-up from the entry of main when its process is older than a second there
        # (setup_from_process_creation 0), so 'from creation' with such an age contradicts itself
        pm = self.timing.get("pre_main_s")
        if pm is not None and float(pm) > 1.0 and self.setup_from_creation():
            bad.append(f"timing.pre_main_s {pm}: the process was that old when main was entered (it inherited the age of a shell), "
                       f"so pre_load_s and setup_s contain time that is not megarl's set-up")
        ctl = self.control()
        if any(not (whole(v) and v in (0, 1)) for v in ctl.values()):
            bad.append(f"control options are not 0 or 1: {ctl}")
        return bad


def summary(r):
    g, st = r.groups, r.steps
    ks, es = r.kernel_seconds(), r.exit_seconds()
    lines = [f"run {r.path}",
             f"  {r.cfg.get('nroll')} engine(s) x {r.cfg.get('ppj')} pipeline(s) x {r.cfg.get('ngrp')} groups x {r.gsz} answers; "
             f"{r.tgrp} groups per update; lag bound {r.cfg.get('lag')}; admission limit {r.cfg.get('target_per_task')} per environment",
             f"  stop: {STOP.get(r.counters.get('stop_reason'), r.counters.get('stop_reason'))}; "
             f"in-binary violations {r.counters.get('audit_violations', 'n/a')}; updates {len(st)}; "
             f"last update at {ks:.3f} s, kernel stopped at {es:.3f} s"]
    pr, su, ctl = r.process(), r.setup_seconds(), r.control()
    if pr is not None or su is not None:
        ok = pr is not None and type(pr.get("index")) is int and type(pr.get("runs")) is int
        lines.append("  " + (f"run {pr['index'] + 1} of {pr['runs']} of process {pr.get('id')}" if ok else "process not on record" if pr is None else f"process record not as expected: {pr}")
                     + ("; set-up not on record as one figure" if su is None else
                        f"; set-up {su:.2f} s from {'the creation of the process' if r.setup_from_creation() else 'the entry of main'} to the launch, "
                        f"counted as for the only run of a process"))
    if any(ctl.values()):
        lines.append("  control options: " + ", ".join(k if v == 1 else f"{k}={v!r}" for k, v in ctl.items() if v))
    if len(g):
        term = np.asarray(g["terminal"])
        for t, name in enumerate(TASKS[:len(r.counters["admitted"])]):
            m = np.asarray(g["task"]) == t
            parts = ", ".join(f"{TERMINAL[k]} {int((term[m] == k).sum())}" for k in TERMINAL)
            lines.append(f"  {name:8s} groups logged {int(m.sum())}: {parts}")
        tr = term == T_CONSUMED
        if tr.any():
            lag = np.asarray(g["lag"])[tr]
            lines.append(f"  lag of trained groups: mean {lag.mean():.2f}, median {np.median(lag):.0f}, "
                         f"95th percentile {np.percentile(lag, 95):.0f}, max {lag.max()}")
        if ks > 0 and es > 0:
            lines.append(f"  per second until the kernel stopped: tokens {r.counters['tokens_generated'] / es:.0f}, "
                         f"groups trained {tr.sum() / es:.1f}; updates per second until the last update {len(st) / ks:.2f}")
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    r = Run(argv[1])
    print(summary(r))
    if "--check" in argv[2:]:
        bad = r.check()
        for b in bad:
            print("  PROBLEM:", b)
        print("rundir check:", "PASSED" if not bad else f"FAILED ({len(bad)} problem(s))")
        return 0 if not bad else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
