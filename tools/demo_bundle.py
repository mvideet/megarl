#!/usr/bin/env python3
"""Pack the data of the race page (demo/race.src.html) into one JSON bundle.

    python tools/demo_bundle.py [--out demo/bundle.json]

MegaRL side: the headline run and its held-out accuracy curve (tools/eval_multitask.py). PyTorch side: the HF
transformers trace of tools/record_pytorch_race.py --engine hf; the CUDA-graph trace (--engine rolling) only
contributes its summary numbers. MegaRL's generated tokens are put on a 10 ms grid. Exits with status 1 if a
consistency check fails.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rundir

TASK_KEYS = ("math", "science", "sciq", "csqa")
GRID_US = 10000


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def us(t_ns):
    return (np.asarray(t_ns, dtype=np.uint64) // np.uint64(1000)).astype(np.int64)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/headline")
    ap.add_argument("--eval", default="results/headline_eval.json")
    ap.add_argument("--pytorch-hf", default="results/pytorch_hf_trace.json")
    ap.add_argument("--pytorch-tuned", default="results/pytorch_rolling_trace.json")
    ap.add_argument("--out", default="demo/bundle.json")
    a = ap.parse_args(argv)
    fails = []

    def check(ok, what):
        print(("  ok    " if ok else "  FAIL  ") + what)
        if not ok:
            fails.append(what)

    r = rundir.Run(a.run)
    tr, cn = r.trace, r.counters
    t_end = int(r.timing["exit_ns"]) // 1000
    kernel_s = t_end / 1e6

    # tokens of each finished group, spread evenly between its admission and its completion, on a 10 ms grid
    adm = tr[tr["type"] == rundir.EV_ADMIT]
    dn = tr[tr["type"] == rundir.EV_GROUP_DONE]
    n = int(adm["a"].max()) + 1
    admit = np.zeros(n, np.int64)
    done = np.zeros(n, np.int64)
    admit[adm["a"]] = us(adm["t"])
    done[dn["a"]] = us(dn["t"])
    g = r.groups
    gid = np.asarray(g["uid"])
    glen = np.clip(np.asarray(g["len"]), 0, r.maxgen).sum(1).astype(np.float64)
    a0, a1 = admit[gid].astype(np.float64), np.maximum(done[gid], admit[gid] + 1).astype(np.float64)
    grid = np.arange(0, t_end + GRID_US, GRID_US, dtype=np.float64)
    tok = np.zeros(len(grid))
    order = np.argsort(a0)
    for i0 in range(0, len(order), 2048):
        ix = order[i0:i0 + 2048]
        fr = np.clip((grid[None, :] - a0[ix, None]) / (a1[ix, None] - a0[ix, None]), 0.0, 1.0)
        tok += (fr * glen[ix, None]).sum(0)
    tok_total_counter = int(cn["tokens_generated"])
    check(abs(tok[-1] - glen.sum()) < 1, f"token series ends at the finished groups' {int(glen.sum()):,} tokens")
    check(0.97 < glen.sum() / tok_total_counter <= 1.0,
          f"finished groups hold {glen.sum() / tok_total_counter:.1%} of the kernel's {tok_total_counter:,} generated tokens")
    # groups still generating or queued for the learner at exit: their lengths are not recorded, so share the
    # remaining tokens equally and spread each share from the group's admission to its completion (or the exit)
    open_uid = np.setdiff1d(np.asarray(adm["a"]), gid)
    rest = tok_total_counter - glen.sum()
    n_open = int(np.sum(cn["inflight"])) + int(np.sum(cn["queued"]))
    check(len(open_uid) == n_open and rest >= 0,
          f"{len(open_uid)} groups in flight or queued at exit hold the other {int(rest):,} tokens")
    if len(open_uid):
        b0 = admit[open_uid].astype(np.float64)
        b1 = np.where(done[open_uid] > 0, done[open_uid], t_end).astype(np.float64)
        fr = np.clip((grid[None, :] - b0[:, None]) / np.maximum(b1[:, None] - b0[:, None], 1.0), 0.0, 1.0)
        tok += (fr * (rest / len(open_uid))).sum(0)
    check(abs(tok[-1] - tok_total_counter) < 1, f"token series ends at the kernel's {tok_total_counter:,} tokens")

    tk = tr[tr["type"] == rundir.EV_SELECT]
    trn_t = np.sort(us(tk["t"]))
    pb = tr[tr["type"] == rundir.EV_PUBLISH]
    pub = np.zeros(int(r.cfg["updates"]), np.int64)
    pub[pb["c"]] = us(pb["t"])
    check(len(trn_t) == int(r.cfg["updates"]) * int(r.cfg["tgrp"]), f"{len(trn_t)} trained groups = {r.cfg['updates']} updates x {r.cfg['tgrp']}")
    check(bool(np.all(np.diff(pub) > 0)), "publications in order")

    eo = json.load(open(a.eval))
    mr_eval = []
    for p in eo["points"]:
        v = int(p["version"])
        ts = p["tasks"]
        mr_eval.append({"ver": v, "t": 0 if v == 0 else int(pub[v - 1]), "macro": round(float(ts["macro_mean"]["greedy_acc"]), 4),
                        **{k: round(float(ts[k]["greedy_acc"]), 4) for k in TASK_KEYS}})
    arch_ids = np.fromfile(os.path.join(a.run, "mid_arch_ids.i32"), dtype="<i4")
    check({e["ver"] for e in mr_eval} <= set(arch_ids.tolist()), f"held-out curve scores versions archived by {a.run}")
    check(len(mr_eval) >= 10 and mr_eval[0]["ver"] == 0 and mr_eval[-1]["ver"] == int(r.cfg["updates"]), f"{len(mr_eval)} held-out points, v0..v{mr_eval[-1]['ver']}")

    pt = json.load(open(a.pytorch_hf))
    check(pt.get("engine") == "hf", "PyTorch trace is the HF transformers engine")
    check(pt["eval_store"] == eo["store"], f"both scored on {eo['store']}")
    ev = np.asarray(pt["events"], dtype=np.float64)
    s0, s1 = [], []
    for i in range(len(ev) - 1):
        if int(ev[i, 1]) == 1 and int(ev[i + 1, 1]) == 1 and ev[i + 1, 0] > ev[i, 0]:
            s0.append(int(round(ev[i, 0] * 1e6)))
            s1.append(int(round(ev[i + 1, 0] * 1e6)))
    check(len(s0) == pt["updates"], f"PyTorch: {len(s0)} backprop pauses = {pt['updates']} updates")
    check(mr_eval[0]["macro"] == pt["eval"][0]["macro"], f"same starting accuracy {mr_eval[0]['macro']:.4f}")
    tuned = json.load(open(a.pytorch_tuned))

    bundle = {
        "format": "mrl-brrr-1",
        "t_end": max(t_end, int(round(pt["wall_s"] * 1e6))),
        "grid_us": GRID_US,
        "meta": {"model": "Qwen2.5-0.5B-Instruct", "device": "NVIDIA H100 80GB HBM3", "layer": 13, "params_m": 14.95,
                 "store": eo["store"], "n_eval": 1600},
        "mr": {
            "source": a.run,
            "kernel_s": kernel_s,
            "setup_s": float(r.timing.get("setup_s", 0.0)),
            "tokens": tok_total_counter,
            "tokens_per_s": tok_total_counter / kernel_s,
            "trained_per_s": len(trn_t) / kernel_s,
            "updates": int(r.cfg["updates"]),
            "violations": int(cn["audit_violations"]),
            "tok": b64(np.rint(tok), "<u4"),
            "trn_t": b64(trn_t, "<u4"),
            "pub": b64(pub, "<u4"),
            "eval": mr_eval,
        },
        "pt": {
            "name": pt["name"],
            "wall_s": pt["wall_s"],
            "setup_s": pt["setup_s"],
            "tokens_per_s": pt["tokens_per_s"],
            "trained_per_s": pt["groups_trained_per_s"],
            "updates": pt["updates"],
            "bwd_frac": pt["bwd_frac"],
            "kernel_launches": pt["kernel_launches"],
            "t": b64(np.clip(np.rint(ev[:, 0] * 1e6), 0, 60_000_000), "<u4"),
            "tok": b64(ev[:, 2], "<u4"),
            "gtrn": b64(ev[:, 4], "<u4"),
            "upd": b64(ev[:, 5], "<u2"),
            "lnch": b64(ev[:, 6], "<u4"),
            "s0": b64(np.asarray(s0), "<u4"),
            "s1": b64(np.asarray(s1), "<u4"),
            "eval": [{"ver": int(e["ver"]), "t": int(round(e["t_s"] * 1e6)), "macro": e["macro"], **{k: e[k] for k in TASK_KEYS}} for e in pt["eval"]],
        },
        "tuned": {"name": tuned["name"], "tokens_per_s": tuned["tokens_per_s"], "trained_per_s": tuned["groups_trained_per_s"],
                  "macro": tuned["eval"][-1]["macro"], "updates": tuned["updates"]},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e6:.2f} MB | MegaRL {bundle['mr']['tokens_per_s']:,.0f} tok/s, "
          f"PyTorch {pt['tokens_per_s']:,.0f} tok/s ({bundle['mr']['tokens_per_s'] / pt['tokens_per_s']:.1f}x)")
    if fails:
        print("demo_bundle: FAILED:", "; ".join(fails))
        return 1
    print("demo_bundle: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
