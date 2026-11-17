#!/usr/bin/env python3
"""Build the blog post: figures from the recorded runs, then one self-contained HTML page.

    python tools/blog_build.py

Writes blog/fig/*.svg and blog/index.html from blog/index.src.html. The source names a figure with <!--FIG name-->
and a computed number with {{key}}; [[SHOT]], [[OFA]] and [[DEMO]] become the embedded race screenshot, dashboard
image and race page (demo/race.html), [[READ_MIN]] the reading time. Every number drawn in a data figure is read
from a result file. Exits with status 1 if a check fails.
"""
import base64
import html
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rundir

RUN = "results/headline"
EVAL = "results/headline_eval.json"
PT_HF = "results/pytorch_hf_trace.json"
PT_TUNED = "results/pytorch_rolling_trace.json"
LAG0, LAG8 = "results/lag0", "results/lag8"
SHOT = "blog/assets/race.png"
DEMO = "demo/race.html"
OFA_IMG = "blog/assets/ofa_dashboard.png"
EMBED_CSS = ("<style>#wrap{padding:0 10px}#controls{flex-wrap:nowrap;gap:6px;padding:6px 0}"
             "#controls button,#controls select{font-size:12px;padding:4px 9px}#bhide{display:none}"
             "#tlabel{min-width:0;font-size:11px;white-space:nowrap}</style>")
SRC, OUT, FIGDIR = "blog/index.src.html", "blog/index.html", "blog/fig"

INK, MUTE, DIM, GRID = "#1d232f", "#5b6475", "#8a93a3", "#e4e8ef"
HF, TUNED, STOP, STOPPED = "#f08c00", "#c2410c", "#e03131", "#262a31"
MRL, LRN, LRN_WAIT = "#1971c2", "#7048e8", "#d9d0fb"
ENG = ["#1864ab", "#1c7ed6", "#4dabf7", "#a5d8ff"]
DROP, OK = "#adb5bd", "#2f9e44"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

fails = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        fails.append(what)


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.p = w, h, []

    def rect(self, x, y, w, h, fill, rx=0, stroke=None, sw=1):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{s}/>')

    def line(self, x1, y1, x2, y2, stroke=GRID, sw=1, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        a = ' marker-end="url(#arr)"' if arrow else ""
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{a}/>')

    def path(self, d, stroke, sw=1.5, dash=None, arrow=False):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        a = ' marker-end="url(#arr)"' if arrow else ""
        self.p.append(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="none" stroke-linejoin="round" stroke-linecap="round"{da}{a}/>')

    def circle(self, cx, cy, r, fill, stroke=None):
        s = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
        self.p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"{s}/>')

    def text(self, x, y, s, size=13, fill=INK, anchor="start", weight=400, mono=False):
        f = f' font-family="{MONO}"' if mono else ""
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{f}>{html.escape(s)}</text>')

    def render(self, label):
        defs = (f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTE}"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{html.escape(label)}" '
                f'font-family="{FONT}">{defs}{"".join(self.p)}</svg>')


def fmt_k(x):
    return f"{x / 1000:.1f}k"


def legend(s, x, y, items):
    for col, name in items:
        s.rect(x, y - 10, 14, 10, col, rx=2)
        s.text(x + 20, y, name, 12, MUTE)
        x += 26 + len(name) * 6.6


# ------------------------------------------------------------------------------------------------------ data
def load():
    r = rundir.Run(RUN)
    cn, tm, cfg = r.counters, r.timing, r.cfg
    kernel_s = tm["kernel_ns"] / 1e9
    tr = r.trace
    D = {"run": r, "kernel_s": kernel_s, "setup_s": tm["setup_s"]}
    D["tok"] = int(cn["tokens_generated"])
    D["tok_s"] = D["tok"] / kernel_s
    D["updates"] = int(cfg["updates"])
    D["consumed"] = int(sum(cn["consumed"]))
    D["trained_s"] = D["consumed"] / kernel_s
    D["admitted"] = int(sum(cn["admitted"]))
    D["finished"] = int(sum(cn["finished"]))
    D["zerovar"] = int(sum(cn["drop_zerovar"]))
    D["stale"] = int(sum(cn["drop_stale_inflight"]) + sum(cn["drop_stale_queued"]))
    D["inflight"] = int(sum(cn["inflight"]))
    D["queued"] = int(sum(cn["queued"]))
    D["violations"] = int(cn["audit_violations"])
    D["cfg"] = cfg
    check(D["admitted"] == D["finished"] + D["inflight"] + int(sum(cn["drop_stale_inflight"])), "admitted = finished + in flight + cancelled")
    check(D["finished"] == D["zerovar"] + D["consumed"] + int(sum(cn["drop_stale_queued"])) + D["queued"], "finished = equal-reward + trained + stale + queued")
    g = r.groups
    tg = g[g["terminal"] == rundir.T_CONSUMED]
    D["lag_hist"] = np.bincount(tg["lag"], minlength=cfg["lag"] + 1).tolist()
    D["lag_mean"] = float(tg["lag"].mean())
    check(len(tg) == D["consumed"] and max(tg["lag"]) <= cfg["lag"], f"{len(tg)} trained groups, all within lag {cfg['lag']}")
    sw = tr[tr["type"] == rundir.EV_SELECT_WAIT]
    pb = tr[tr["type"] == rundir.EV_PUBLISH]
    swt = np.zeros(D["updates"])
    swt[sw["a"]] = sw["t"] / 1e9
    pbt = np.zeros(D["updates"])
    pbt[pb["c"]] = pb["t"] / 1e9
    D["learn"] = list(zip(swt.tolist(), pbt.tolist()))
    D["learn_ms"] = float(np.mean(pbt - swt) * 1e3)
    D["wait_ms"] = float(np.mean(swt[1:] - pbt[:-1]) * 1e3)
    traced_wait = float(np.mean(sw["b"][1:]) / 1e3)
    check(abs(traced_wait - D["wait_ms"]) < 5, f"learner wait {D['wait_ms']:.1f} ms matches the traced wait {traced_wait:.1f} ms")
    D["mid_nw"] = int(json.load(open(os.path.join(RUN, "mid_layer.json")))["mid_nw"])
    D["idle_events"] = int((tr["type"] == rundir.EV_IDLE).sum())
    check(np.all(pbt > swt), "every update publishes after its batch was complete")
    D["mr_upd_3s"] = int((pbt < 3.0).sum())

    eo = json.load(open(EVAL))
    arch_ids = np.fromfile(os.path.join(RUN, "mid_arch_ids.i32"), dtype="<i4")
    check({p["version"] for p in eo["points"]} <= set(arch_ids.tolist()), "held-out curve scores versions the headline run archived")
    D["mr_eval"] = [(0.0 if p["version"] == 0 else float(pbt[p["version"] - 1]), float(p["tasks"]["macro_mean"]["greedy_acc"])) for p in eo["points"]]

    for key, path in (("hf", PT_HF), ("tuned", PT_TUNED)):
        t = json.load(open(path))
        check(t["eval_store"] == eo["store"], f"{key}: same held-out store")
        ev = np.asarray(t["events"], float)
        pauses = [(ev[i, 0], ev[i + 1, 0]) for i in range(len(ev) - 1) if ev[i, 1] == 1 and ev[i + 1, 1] == 1 and ev[i + 1, 0] > ev[i, 0]]
        check(len(pauses) == t["updates"], f"{key}: {len(pauses)} generation stops = {t['updates']} updates")
        i3 = np.searchsorted(ev[:, 0], 3.0) - 1
        D[key] = {"t": t, "pauses": pauses, "stop_s": float(sum(b - a for a, b in pauses)), "upd_3s": int(ev[i3, 5]),
                  "eval": [(float(e["t_s"]), float(e["macro"])) for e in t["eval"]]}
        check(abs(D[key]["eval"][0][1] - D["mr_eval"][0][1]) < 1e-4, f"{key}: same starting accuracy")

    D["race_s"] = float(np.mean([D["kernel_s"], D["hf"]["t"]["wall_s"], D["tuned"]["t"]["wall_s"]]))
    l0, l8 = rundir.Run(LAG0), rundir.Run(LAG8)
    c0, c8 = dict(l0.cfg), dict(l8.cfg)
    diff = sorted(k for k in c0 if c0[k] != c8.get(k))
    check(set(diff) <= {"lag", "target_per_task", "updates", "nbuf", "nslots", "count_held"}, f"lag-0 and lag-8 runs differ only in {diff}")
    D["lag0"] = {"s": l0.timing["kernel_ns"] / 1e9, "upd": int(c0["updates"]), "tok": int(l0.counters["tokens_generated"])}
    D["lag8"] = {"s": l8.timing["kernel_ns"] / 1e9, "upd": int(c8["updates"]), "tok": int(l8.counters["tokens_generated"])}
    for k in ("lag0", "lag8"):
        D[k]["ms_upd"] = 1e3 * D[k]["s"] / D[k]["upd"]
        D[k]["tok_s"] = D[k]["tok"] / D[k]["s"]
    return D


def first_at(pts, thr):
    for t, m in pts:
        if m >= thr:
            return t
    return None


# ------------------------------------------------------------------------------------------------------ figures
def fig_group():
    s = Svg(720, 236)
    s.rect(0.5, 18.5, 236, 200, "#f6f8fb", rx=10, stroke=GRID)
    s.text(16, 44, "One question", 12, MUTE, weight=600)
    s.text(16, 72, "What is 251 + 345?", 15, INK, weight=600)
    for i, o in enumerate(["A. 638", "B. 941", "C. 596", "D. 586"]):
        s.text(24, 100 + 22 * i, o, 14, OK if i == 2 else INK, weight=600 if i == 2 else 400)
    s.text(16, 204, "asked 4 times", 12, MUTE)
    ans = [("...so the sum is 596. Answer: C", 1, "+0.87"), ("...about 586. Answer: D", 0, "-0.87"),
           ("251 + 345 = 596. Answer: C", 1, "+0.87"), ("...probably 638. Answer: A", 0, "-0.87")]
    s.text(292, 14, "4 sampled answers", 12, MUTE, weight=600)
    s.text(588, 14, "reward", 12, MUTE, weight=600, anchor="middle")
    s.text(672, 14, "advantage", 12, MUTE, weight=600, anchor="middle")
    for i, (a, r, adv) in enumerate(ans):
        y = 30 + 48 * i
        s.path(f"M238,118 C262,118 262,{y + 18} 282,{y + 18}", DIM, 1.2, arrow=True)
        s.rect(286, y, 270, 36, "#ffffff", rx=7, stroke=GRID)
        s.text(298, y + 23, a, 12.5, INK, mono=True)
        s.text(588, y + 24, str(r), 16, OK if r else STOP, anchor="middle", weight=700)
        s.text(672, y + 24, adv, 16, OK if r else STOP, anchor="middle", weight=700, mono=True)
    return s.render("A question is answered four times; correct answers get reward 1 and a positive advantage, wrong ones a negative advantage.")


def fig_timeline(D):
    T = 3.0
    s = Svg(720, 262)
    x0, x1 = 150, 640
    px = lambda t: x0 + (x1 - x0) * t / T
    lx = x0
    for col in (HF, TUNED, MRL):
        s.rect(lx, 6, 9, 10, col, rx=1.5)
        lx += 11
    s.text(lx + 5, 16, "generating", 12, MUTE)
    legend(s, lx + 84, 16, [(STOPPED, "generation stopped to train"), (LRN, "learner computing"), (LRN_WAIT, "learner waiting")])
    rows = [("HF PyTorch", "hf", 48), ("Tuned PyTorch", "tuned", 98)]
    for name, key, y in rows:
        col = HF if key == "hf" else TUNED
        s.text(x0 - 12, y + 15, name, 13, INK, anchor="end", weight=600)
        s.rect(x0, y, x1 - x0, 22, col, rx=3)
        for a, b in D[key]["pauses"]:
            if a < T:
                s.rect(px(a), y, max(px(min(b, T)) - px(a), 1.2), 22, STOPPED)
        s.text(x1 + 10, y + 15, f"{D[key]['upd_3s']} updates", 12.5, MUTE)
    y = 148
    s.text(x0 - 12, y + 15, "MegaRL engines", 13, INK, anchor="end", weight=600)
    s.rect(x0, y, x1 - x0, 22, MRL, rx=3)
    y2 = y + 30
    s.text(x0 - 12, y2 + 15, "MegaRL learner", 13, INK, anchor="end", weight=600)
    s.rect(x0, y2, x1 - x0, 22, LRN_WAIT, rx=3)
    for a, b in D["learn"]:
        if a < T:
            s.rect(px(a), y2, px(min(b, T)) - px(a), 22, LRN)
    s.text(x1 + 10, y + 34, f"{D['mr_upd_3s']} updates", 12.5, MUTE)
    for t in np.arange(0, T + 1e-9, 0.5):
        s.line(px(t), 230, px(t), 236, DIM)
        s.text(px(t), 252, f"{t:.1f} s", 11.5, MUTE, anchor="middle")
    s.line(x0, 230, x1, 230, DIM)
    return s.render("First three seconds of each run: both PyTorch loops stop generating for every update; MegaRL's engines never stop while the learner computes.")


def fig_launches(D):
    s = Svg(720, 196)
    s.text(0, 18, "PyTorch: the CPU launches every operation as its own kernel", 13, INK, weight=600)
    x = 0
    rng = np.random.default_rng(3)
    while x < 712:
        w = float(rng.uniform(5, 16))
        gap = float(rng.uniform(2, 6))
        s.rect(x, 52, w, 26, HF, rx=1.5)
        s.line(x + 0.5, 34, x + 0.5, 48, DIM, 1)
        x += w + gap
    s.text(0, 98, f"HF PyTorch: {D['hf']['t']['kernel_launches']:,} launches in {D['hf']['t']['wall_s']:.1f} s; each gap is time the GPU waits for the next one", 12, MUTE)
    s.text(0, 132, "MegaRL: one launch; its blocks run every task of every job until the run ends", 13, INK, weight=600)
    s.rect(0, 144, 712, 34, "#ffffff", rx=4, stroke=MRL, sw=2)
    x = 3
    while x < 708:
        w = float(rng.uniform(4, 12))
        col = LRN if rng.uniform() < 0.28 else MRL
        s.rect(x, 147, min(w, 709 - x), 28, col, rx=1)
        x += w + 0.6
    return s.render("PyTorch issues one small kernel after another from the CPU; MegaRL is a single kernel that packs generation and learner tasks back to back.")


def fig_tickets():
    s = Svg(720, 250)
    phases = [("QKV", 4, "done"), ("attention", 6, "run"), ("MLP", 4, "wait"), ("LM head", 3, "wait"), ("sample", 1, "wait")]
    x, tk = 20, 0
    s.text(20, 18, "one step of a generation engine = the next token for each of its 128 answers", 12.5, MUTE, weight=600)
    centers = []
    for name, n, st in phases:
        w = 22 * n + 6 * (n - 1) if n > 1 else 34
        s.text(x + w / 2, 46, name, 12.5, INK, anchor="middle", weight=600)
        for i in range(n):
            bx = x + i * 28
            running = st == "run" and i >= 3
            fill = MRL if st == "done" or (st == "run" and i < 3) else ("#ffffff")
            stroke = MRL if st != "wait" else DIM
            s.rect(bx, 58, 22, 30, fill, rx=3, stroke=stroke, sw=1.5)
            s.text(bx + 11, 78, str(tk), 11, "#ffffff" if fill == MRL else (MRL if running else DIM), anchor="middle", mono=True, weight=600)
            tk += 1
        done = n if st == "done" else 3 if st == "run" else 0
        s.text(x, 110, f"{done}/{n} done", 11.5, OK if done == n else MUTE, mono=True)
        centers.append((x, w))
        x += w + 34
    for (xa, wa), (xb, wb) in zip(centers, centers[1:]):
        s.line(xa + wa + 4, 73, xb - 6, 73, DIM, 1.2, arrow=True)
    bx0 = centers[1][0] + 3 * 28
    for i, smn in enumerate([7, 41, 88]):
        cx = 70 + i * 150
        s.rect(cx, 160, 116, 40, "#f6f8fb", rx=8, stroke=GRID)
        s.text(cx + 58, 177, f"block on SM {smn}", 12, INK, anchor="middle", weight=600)
        s.text(cx + 58, 193, f"took ticket {7 + i}", 11.5, MRL, anchor="middle", mono=True)
        s.path(f"M{cx + 58},{158} C{cx + 58},{130} {bx0 + 11 + 28 * i},{120} {bx0 + 11 + 28 * i},{92}", MRL, 1.3, arrow=True)
    s.rect(530, 146, 186, 70, "#ffffff", rx=8, stroke=GRID)
    s.text(542, 166, "ticket counter: 10", 12.5, INK, mono=True, weight=600)
    s.text(542, 184, "ticket 10 (MLP) is handed", 11.5, MUTE)
    s.text(542, 200, "out once attention is 6/6", 11.5, MUTE)
    s.text(20, 238, "A phase opens once every task it depends on has bumped its completion counter.", 12, MUTE)
    return s.render("Tasks are numbered by tickets; blocks take the next ticket, run that task, and bump a completion counter that unlocks the next phase.")


def fig_smmap(D):
    nsm, nroll, train_sms = 132, int(D["cfg"]["nroll"]), int(D["cfg"]["train_sms"])
    roles = []
    for sm in range(nsm):
        job = sm * nroll // nsm
        trains = ((sm + 1) * train_sms // nsm) != (sm * train_sms // nsm)
        roles.append((job, nroll if trains else job))
    tb = sum(1 for a, b in roles for x in (a, b) if x == nroll)
    rb = [sum(1 for a, b in roles for x in (a, b) if x == e) for e in range(nroll)]
    tm = D["run"].timing
    check(tb == tm["train_blocks"] and rb == list(tm["roll_blocks"]), f"SM map reproduces the run's blocks: learner {tb}, engines {rb}")
    s = Svg(720, 290)
    cols, tw, th = 22, 30, 15
    for sm, (a, b) in enumerate(roles):
        x = 6 + (sm % cols) * (tw + 2.6)
        y = 34 + (sm // cols) * (2 * th + 8)
        s.rect(x, y, tw, th, ENG[a], rx=2)
        s.rect(x, y + th + 1, tw, th, LRN if b == nroll else ENG[b], rx=2)
    legend(s, 6, 16, [(ENG[0], "engine 1"), (ENG[1], "engine 2"), (ENG[2], "engine 3"), (ENG[3], "engine 4"), (LRN, "learner")])
    s.text(6, 282, f"132 SMs, 2 blocks each (top and bottom half). {sum(rb)} blocks generate, {tb} learn.", 12, MUTE)
    return s.render("Map of the H100's 132 SMs: the first block of every SM belongs to one of four generation engines; the second block of every other SM belongs to the learner.")


def fig_versions():
    s = Svg(720, 250)
    x0, x1 = 60, 700
    px = lambda v: x0 + (x1 - x0) * v / 12
    s.text(0, 24, "learner", 12, MUTE, weight=600)
    s.line(x0, 20, x1, 20, DIM)
    for v in range(13):
        s.line(px(v), 14, px(v), 26, LRN, 2)
        s.text(px(v), 42, f"v{v}", 11, LRN, anchor="middle", mono=True)
    groups = [(3.2, 4.4, 5.0, "trained", "pinned v3, trained at v5: 2 behind"),
              (4.1, 5.3, 9.0, "trained", "pinned v4, trained at v9: 5 behind"),
              (2.1, 3.6, 11.0, "dropped", "pinned v2, reached at v11: 9 behind, dropped")]
    for i, (a, b, c, fate, lab) in enumerate(groups):
        y = 70 + 52 * i
        s.rect(px(a), y, px(b) - px(a), 18, MRL, rx=3)
        s.rect(px(b), y, px(c) - px(b), 18, "#e7f0fb", rx=3, stroke="#a5c8ee")
        s.circle(px(a), y + 9, 5, "#ffffff", MRL)
        if fate == "trained":
            s.circle(px(c), y + 9, 6, OK)
        else:
            s.line(px(c) - 5, y + 4, px(c) + 5, y + 14, STOP, 2.5)
            s.line(px(c) - 5, y + 14, px(c) + 5, y + 4, STOP, 2.5)
        s.text(px(a), y + 34, lab, 12, INK if fate == "trained" else STOP)
    legend(s, x0, 236, [(MRL, "generating with its pinned weights"), ("#e7f0fb", "waiting in the queue")])
    return s.render("Each group pins the newest weights when it starts; the learner trains on it only if it is at most 8 versions behind.")


def fig_lag(D):
    h = D["lag_hist"]
    n = sum(h)
    s = Svg(720, 230)
    x0, x1, y0, y1 = 50, 700, 30, 180
    m = max(h)
    bw = (x1 - x0) / len(h)
    for i, c in enumerate(h):
        bh = (y1 - y0) * c / m
        s.rect(x0 + i * bw + 8, y1 - bh, bw - 16, bh, LRN, rx=3)
        s.text(x0 + (i + 0.5) * bw, y1 - bh - 7, f"{c:,}", 12, INK, anchor="middle", mono=True)
        s.text(x0 + (i + 0.5) * bw, y1 + 18, str(i), 12.5, MUTE, anchor="middle", mono=True)
    s.line(x0, y1, x1, y1, DIM)
    s.text((x0 + x1) / 2, 222, f"updates behind when trained ({n:,} trained groups, mean {D['lag_mean']:.2f}, limit {D['cfg']['lag']})", 12.5, MUTE, anchor="middle")
    return s.render("Histogram of how many updates behind each trained group was.")


def fig_lag08(D):
    a, b = D["lag0"], D["lag8"]
    s = Svg(720, 178)
    x0, x1 = 210, 640
    def bars(y, title, v0, v8, f, unit):
        s.text(0, y, title, 13, INK, weight=600)
        m = max(v0, v8)
        for j, (lab, v, col) in enumerate((("lag 0: wait for fresh answers", v0, DROP), ("lag 8: MegaRL default", v8, MRL))):
            yy = y + 12 + j * 28
            s.text(x0 - 10, yy + 15, lab, 12, MUTE, anchor="end")
            s.rect(x0, yy, (x1 - x0) * v / m, 20, col, rx=3)
            s.text(x0 + (x1 - x0) * v / m + 8, yy + 15, f(v) + unit, 12.5, INK, mono=True, weight=600)
    bars(18, "Time per update", a["ms_upd"], b["ms_upd"], lambda v: f"{v:.0f}", " ms")
    bars(104, "Tokens generated per second", a["tok_s"], b["tok_s"], fmt_k, "")
    return s.render("The same kernel with lag 0 and lag 8: lag 8 is about eleven times faster per update.")


def fig_flow(D):
    s = Svg(720, 250)
    usable = D["consumed"] + D["stale"] + D["queued"]
    def box(x, y, w, title, num, col, sub=None):
        s.rect(x, y, w, 64 if sub else 50, "#ffffff", rx=8, stroke=col, sw=2)
        s.text(x + 12, y + 20, title, 12, MUTE, weight=600)
        s.text(x + 12, y + 41, num, 17, INK, weight=700, mono=True)
        if sub:
            s.text(x + 12, y + 56, sub, 10.5, MUTE)
    box(0, 90, 140, "groups started", f"{D['admitted']:,}", MRL)
    box(176, 90, 160, "answered and scored", f"{D['finished']:,}", MRL)
    box(380, 18, 340, "all 4 rewards equal: no signal, dropped", f"{D['zerovar']:,}  ({D['zerovar'] / D['finished']:.0%})", DROP)
    box(380, 104, 140, "usable, queued", f"{usable:,}", LRN)
    box(556, 104, 164, "trained on", f"{D['consumed']:,}", OK)
    box(556, 176, 164, "too stale", f"{D['stale']:,}", STOP, f"+ {D['queued']} still queued at the end")
    s.line(142, 115, 172, 115, DIM, 1.3, arrow=True)
    s.path("M338,110 C360,110 356,44 376,44", DIM, 1.3, arrow=True)
    s.path("M338,120 C360,120 356,130 376,130", DIM, 1.3, arrow=True)
    s.line(522, 130, 552, 130, DIM, 1.3, arrow=True)
    s.path("M522,138 C540,138 536,200 552,200", DIM, 1.3, arrow=True)
    s.text(176, 168, f"{D['inflight']} still generating at the end", 11.5, MUTE)
    return s.render("What happened to every group in the run: most were dropped because all four answers scored the same.")


def fig_ladder(D):
    rows = [("PyTorch", "HF transformers", D["hf"]["t"]["tokens_per_s"], HF, D["hf"]["t"]["kernel_launches"], D["hf"]["t"]["updates"]),
            ("PyTorch, hand-tuned", "CUDA graphs, static KV cache", D["tuned"]["t"]["tokens_per_s"], TUNED, D["tuned"]["t"]["kernel_launches"], D["tuned"]["t"]["updates"]),
            ("MegaRL", "one persistent kernel", D["tok_s"], MRL, 1, D["updates"])]
    s = Svg(720, 214)
    x0, x1 = 190, 520
    s.text(x0, 16, "tokens generated per second", 12, MUTE, weight=600)
    s.text(600, 16, "kernel launches", 12, MUTE, weight=600, anchor="end")
    s.text(712, 16, "updates", 12, MUTE, weight=600, anchor="end")
    m = D["tok_s"]
    for i, (a, b, v, col, l, u) in enumerate(rows):
        y = 34 + 52 * i
        s.text(0, y + 16, a, 14, INK, weight=600)
        s.text(0, y + 33, b, 11.5, MUTE)
        s.rect(x0, y + 4, (x1 - x0 - 70) * v / m, 28, col, rx=4)
        s.text(x0 + (x1 - x0 - 70) * v / m + 8, y + 24, fmt_k(v), 15, INK, mono=True, weight=700)
        s.text(600, y + 24, f"{l:,}", 13, INK, mono=True, anchor="end")
        s.text(712, y + 24, str(u), 13, INK, mono=True, anchor="end")
    s.text(0, 206, f"One H100, same model, prompts and training step; about {D['race_s']:.0f} s of training each, set-up excluded.", 11.5, MUTE)
    return s.render("Tokens per second, kernel launches and updates for the three systems.")


def fig_accuracy(D):
    s = Svg(720, 300)
    x0, x1, y0, y1 = 48, 562, 20, 250
    T = max(D["hf"]["t"]["wall_s"], D["kernel_s"])
    lo, hi = 0.55, 0.80
    px = lambda t: x0 + (x1 - x0) * t / T
    py = lambda v: y1 - (y1 - y0) * (v - lo) / (hi - lo)
    for v in np.arange(lo, hi + 1e-9, 0.05):
        s.line(x0, py(v), x1, py(v), GRID)
        s.text(x0 - 8, py(v) + 4, f"{v * 100:.0f}%", 11.5, MUTE, anchor="end", mono=True)
    for t in range(0, int(T) + 1, 5):
        s.text(px(t), y1 + 20, f"{t} s", 11.5, MUTE, anchor="middle", mono=True)
    s.line(x0, py(0.70), x1, py(0.70), DIM, 1, dash="4 4")
    series = [("HF PyTorch", D["hf"]["eval"], HF, "6 4"), ("Tuned PyTorch", D["tuned"]["eval"], TUNED, "6 4"), ("MegaRL", D["mr_eval"], MRL, None)]
    ends = []
    for name, pts, col, dash in series:
        d = " ".join(("M" if i == 0 else "L") + f"{px(t):.1f},{py(v):.1f}" for i, (t, v) in enumerate(pts))
        s.path(d, col, 2.5, dash=dash)
        for t, v in pts:
            s.circle(px(t), py(v), 3.2, col)
        ends.append([py(pts[-1][1]), name, pts[-1][1], col])
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 17:
            ends[i][0] = ends[i - 1][0] + 17
    for y, name, v, col in ends:
        s.text(x1 + 10, y + 4, f"{name} {v * 100:.1f}%", 12, col, weight=700)
    s.text(x1 - 4, py(0.70) - 6, "70%", 11, DIM, anchor="end", mono=True)
    s.text((x0 + x1) / 2, 296, "seconds of training", 11.5, MUTE, anchor="middle")
    return s.render("Held-out accuracy over training time for the three systems.")


# ------------------------------------------------------------------------------------------------------ page
def main():
    D = load()
    figs = {"group": fig_group(), "timeline": fig_timeline(D), "launches": fig_launches(D), "tickets": fig_tickets(),
            "smmap": fig_smmap(D), "versions": fig_versions(), "lag": fig_lag(D), "lag08": fig_lag08(D),
            "flow": fig_flow(D), "ladder": fig_ladder(D), "accuracy": fig_accuracy(D)}
    os.makedirs(FIGDIR, exist_ok=True)
    for k, v in figs.items():
        with open(os.path.join(FIGDIR, k + ".svg"), "w", encoding="utf-8") as f:
            f.write(v)
    hf, tu = D["hf"]["t"], D["tuned"]["t"]
    mr70, hf70 = first_at(D["mr_eval"], 0.70), first_at(D["hf"]["eval"], 0.70)
    nums = {
        "mr_tok_s": fmt_k(D["tok_s"]), "tuned_tok_s": fmt_k(tu["tokens_per_s"]),
        "x_hf": f"{D['tok_s'] / hf['tokens_per_s']:.0f}", "x_tuned": f"{D['tok_s'] / tu['tokens_per_s']:.1f}",
        "mr_trained_s": f"{D['trained_s']:.0f}", "hf_trained_s": f"{hf['groups_trained_per_s']:.0f}", "tuned_trained_s": f"{tu['groups_trained_per_s']:.0f}",
        "mr_updates": str(D["updates"]), "hf_updates": str(hf["updates"]), "tuned_updates": str(tu["updates"]),
        "mr_s": f"{D['kernel_s']:.1f}", "race_s": f"{D['race_s']:.0f}", "hf_s": f"{hf['wall_s']:.1f}",
        "hf_launches": f"{hf['kernel_launches']:,}", "tuned_launches": f"{tu['kernel_launches']:,}",
        "hf_launch_rate": f"{hf['kernel_launches'] / hf['wall_s'] / 1000:.0f},000",
        "roll_blocks": str(sum(D["run"].timing["roll_blocks"])), "train_blocks": str(D["run"].timing["train_blocks"]),
        "params": f"{D['mid_nw']:,}", "params_m": f"{D['mid_nw'] / 1e6:.1f}M", "snap_mb": f"{2 * D['mid_nw'] / 1e6:.1f}",
        "tuned_stop_pct": f"{100 * D['tuned']['stop_s'] / tu['wall_s']:.0f}%", "tuned_stops": str(len(D["tuned"]["pauses"])),
        "tuned_stop_ms": f"{1e3 * np.median([b - a for a, b in D['tuned']['pauses']]):.0f}",
        "base_acc": f"{100 * D['mr_eval'][0][1]:.1f}%",
        "mr_70": f"{mr70:.1f}", "hf_70": f"{hf70:.1f}",
        "mr_setup": f"{D['setup_s']:.1f}", "hf_setup": f"{hf['setup_s']:.1f}",
        "learn_ms": f"{D['learn_ms']:.0f}", "wait_ms": f"{D['wait_ms']:.0f}",
        "lag_mean": f"{D['lag_mean']:.2f}", "stale": str(D["stale"]), "zerovar_pct": f"{D['zerovar'] / D['finished']:.0%}",
        "x_lag": f"{D['lag0']['ms_upd'] / D['lag8']['ms_upd']:.0f}",
        "lag0_upd": str(D["lag0"]["upd"]), "lag8_upd": str(D["lag8"]["upd"]),
        "violations": str(D["violations"]),
    }
    check(D["idle_events"] == 0, "the engines never went idle")
    src = open(SRC, encoding="utf-8").read()
    used = set(re.findall(r"\{\{(\w+)\}\}", src))
    check(used <= set(nums), f"every number the page asks for is computed (missing: {sorted(used - set(nums))})")
    page = re.sub(r"\{\{(\w+)\}\}", lambda m: nums.get(m.group(1), "??"), src)
    wanted = set(re.findall(r"<!--FIG (\w+)-->", page))
    check(wanted <= set(figs), f"every figure the page asks for exists (missing: {sorted(wanted - set(figs))})")
    page = re.sub(r"<!--FIG (\w+)-->", lambda m: figs.get(m.group(1), ""), page)
    b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
    check(os.path.getsize(DEMO) < 200_000 and "/*BUNDLES*/" not in open(DEMO, encoding="utf-8").read(), f"{DEMO} is the built standalone page")
    page = page.replace("[[SHOT]]", "data:image/png;base64," + b64(SHOT)).replace("[[OFA]]", "data:image/png;base64," + b64(OFA_IMG))
    demo = open(DEMO, encoding="utf-8").read()
    check(demo.count("</head>") == 1, "the replay page has one </head> for the embed style")
    demo = demo.replace("</head>", EMBED_CSS + "</head>")
    page = page.replace("[[DEMO]]", base64.b64encode(demo.encode("utf-8")).decode())
    words = len(re.findall(r"\w+", re.sub(r"(?s)<svg.*?</svg>|<style.*?</style>|<script.*?</script>|<[^>]+>", " ", page)))
    page = page.replace("[[READ_MIN]]", str(max(1, round(words / 230))))
    check("[[" not in page and "{{" not in page, "no placeholder left in the page")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {OUT}: {len(page) / 1e3:.0f} kB, {words:,} words, {len(figs)} figures")
    if fails:
        print("blog_build: FAILED:", "; ".join(fails))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
