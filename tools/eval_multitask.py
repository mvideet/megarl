#!/usr/bin/env python3
"""Score the base model, or archived versions of a run's trained layer, on a prompt store.

    python tools/eval_multitask.py --run base --store STORE --out EVAL.json
    python tools/eval_multitask.py --run RUN --store STORE [--versions 0,10,250] --out EVAL.json

RUN is a run directory (docs/FORMATS.md section 3) with mid_layer.json, mid_arch_ids.i32 and mid_arch.bf16; the
base weights are the export named by config.weights in its run.json. Version v is scored as the float32
transformers model plus (W_v - W_base) @ x on the outputs of the trained layer's seven linear maps. As in the kernel
(from_last=1), the prompt up to its second-to-last token goes through the base layer and the last prompt token and
the answer through the trained one. The tool refuses a run whose training store (config.prompt_set) shares a prompt
token hash with STORE.

Every prompt is answered twice, greedily and sampled at temperature 1, up to 64 tokens. The sampling uniforms come
from a fixed seed, drawn batch by batch, so sampled scores depend on --batch-size. Per task the output holds
greedy_acc (mean greedy reward), found (an answer was found), acc_when_found, mean_len, no_end (no end token
within 64 tokens), sampled_reward and sampled_found; macro_mean averages greedy_acc, acc_when_found and
sampled_reward over the tasks.

results/headline_eval.json was written by (with the headline run's mid_arch.bf16, which the repository omits):

    python tools/eval_multitask.py --run results/headline --store data/prompts/multitask4_dev_twin \\
        --versions 0,10,20,30,40,60,80,100,120,150,180,210,250 --batch-size 256 --out results/headline_eval.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stores

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
END_TOKENS = (151645, 151643)                   # <|im_end|>, <|endoftext|>
MAX_NEW = 64
SAMPLE_SEED = 20261116
LINEARS = ("q", "k", "v", "o", "gate", "up", "down")


def task_names(store_dir, ntasks):
    p = os.path.join(store_dir, "envs.json")
    if os.path.isfile(p):
        names = json.load(open(p)).get("names", [])
        if len(names) >= ntasks:
            return list(names[:ntasks])
    defaults = ["math", "science", "sciq", "csqa"]
    return [defaults[t] if t < len(defaults) else f"task{t}" for t in range(ntasks)]


def linears(layer):
    """The seven linear maps of a decoder layer, by the names in LINEARS."""
    a, m = layer.self_attn, layer.mlp
    return dict(zip(LINEARS, (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj)))


def install_mid_hooks(model, lmid):
    """Forward hooks that add x @ delta.T to the outputs of layer `lmid`'s linear maps.

    Returns (mid, handles). mid["delta"][name] is a float32 [out, in] tensor or None (no change); while mid["on"] is
    False the layer is the base layer. Pass `mid` to eval_model_on_store.
    """
    mid = {"on": True, "delta": dict.fromkeys(LINEARS)}

    def hook(name):
        def add(module, args, output):
            d = mid["delta"][name]
            return output if d is None or not mid["on"] else output + F.linear(args[0], d)
        return add
    handles = [lin.register_forward_hook(hook(name)) for name, lin in linears(model.model.layers[lmid]).items()]
    return mid, handles


@torch.no_grad()
def eval_model_on_store(model, store, batch_size=128, mid=None):
    """Per-task scores of `model` on every prompt of `store` (see the module docstring).

    With `mid` (install_mid_hooks) the prompt up to its second-to-last token is run with the hooks off.
    """
    device = next(model.parameters()).device
    n = store.n
    greedy_rew = np.zeros(n, dtype=np.float32)
    greedy_fnd = np.zeros(n, dtype=np.float32)
    greedy_len = np.zeros(n, dtype=np.int32)
    greedy_noe = np.zeros(n, dtype=np.float32)
    samp_rew = np.zeros(n, dtype=np.float32)
    samp_fnd = np.zeros(n, dtype=np.float32)
    eos = torch.tensor(END_TOKENS, dtype=torch.long, device=device)

    for greedy in (True, False):
        rng = np.random.default_rng(SAMPLE_SEED)
        for i0 in range(0, n, batch_size):
            prompts = [store.prompt(i) for i in range(i0, min(n, i0 + batch_size))]
            B, P = len(prompts), max(len(p) for p in prompts)
            ids = torch.zeros((B, P), dtype=torch.long, device=device)
            mask = torch.zeros((B, P + MAX_NEW), dtype=torch.long, device=device)
            for b, p in enumerate(prompts):
                ids[b, P - len(p):] = torch.as_tensor(np.array(p, copy=True), dtype=torch.long, device=device)
                mask[b, P - len(p):P] = 1

            if mid is not None and P > 1:
                mid["on"] = False
                past = model(input_ids=ids[:, :-1], attention_mask=mask[:, :P - 1], use_cache=True).past_key_values
                mid["on"] = True
                out = model(input_ids=ids[:, -1:], attention_mask=mask[:, :P], past_key_values=past, use_cache=True)
            else:
                out = model(input_ids=ids, attention_mask=mask[:, :P], use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()

            u = None if greedy else rng.random((MAX_NEW, B))
            toks = np.full((B, MAX_NEW), -1, dtype=np.int32)
            lens = np.zeros(B, dtype=np.int32)
            done = torch.zeros(B, dtype=torch.bool, device=device)
            for t in range(MAX_NEW):
                if greedy:
                    nxt = logits.argmax(dim=-1)
                else:
                    cdf = torch.softmax(logits.double(), dim=-1).cumsum(dim=-1)
                    ut = torch.as_tensor(u[t], dtype=torch.float64, device=device).unsqueeze(-1) * cdf[:, -1:]
                    nxt = torch.searchsorted(cdf, ut, right=True).squeeze(-1).clamp_(max=cdf.shape[-1] - 1)
                live = ~done.cpu().numpy()
                toks[live, t] = nxt.cpu().numpy()[live]
                lens[live] = t + 1
                done |= torch.isin(nxt, eos)
                if t == MAX_NEW - 1 or bool(done.all()):
                    break
                mask[:, P + t] = 1
                out = model(input_ids=nxt.unsqueeze(-1), attention_mask=mask[:, :P + t + 1],
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()

            for b in range(B):
                i, seq = i0 + b, toks[b, :lens[b]]
                rew = float(store.reward(seq, int(store.key[i])))
                fnd = 1.0 if store.find_answer(seq)[0] >= 0 else 0.0
                if greedy:
                    greedy_rew[i], greedy_fnd[i], greedy_len[i] = rew, fnd, lens[b]
                    greedy_noe[i] = 0.0 if lens[b] > 0 and int(seq[-1]) in END_TOKENS else 1.0
                else:
                    samp_rew[i], samp_fnd[i] = rew, fnd

    names = task_names(store.path, int(store.task.max()) + 1)
    summary = {}
    for t, name in enumerate(names):
        m = store.task == t
        summary[name] = {
            "n": int(m.sum()),
            "greedy_acc": float(greedy_rew[m].mean()),
            "found": float(greedy_fnd[m].mean()),
            "acc_when_found": float(greedy_rew[m].sum() / max(1e-9, float(greedy_fnd[m].sum()))),
            "mean_len": float(greedy_len[m].mean()),
            "no_end": float(greedy_noe[m].mean()),
            "sampled_reward": float(samp_rew[m].mean()),
            "sampled_found": float(samp_fnd[m].mean()),
        }
    summary["macro_mean"] = {k: float(np.mean([summary[nm][k] for nm in names]))
                             for k in ("greedy_acc", "acc_when_found", "sampled_reward")}
    return summary


def read_base_layer(weights_dir, lmid):
    """qkv, o, gu, down of layer `lmid` from an export (docs/FORMATS.md section 1), float32 on the GPU."""
    idx = {}
    for line in open(os.path.join(weights_dir, "weights.idx")):
        name, _, rows, cols, off, _ = line.split()
        idx[name] = (int(rows), int(cols), int(off))
    wbin = np.memmap(os.path.join(weights_dir, "weights.bin"), dtype="<u2", mode="r")
    out = []
    for part in ("qkv", "o", "gu", "down"):
        rows, cols, off = idx[f"L{lmid}.{part}"]
        out.append(bf16_to_cuda(wbin[off // 2:off // 2 + rows * cols]).reshape(rows, cols))
    return out


def bf16_to_cuda(bits):
    return torch.from_numpy(np.array(bits, copy=True)).view(torch.bfloat16).float().cuda()


def summary_line(v, s, seconds):
    tasks = " | ".join(f"{k}: {s[k]['greedy_acc']:.3f} (samp {s[k]['sampled_reward']:.3f}, no_end {s[k]['no_end']:.2f})"
                       for k in s if k != "macro_mean")
    return f"  version {v:4d} [{seconds:.1f}s] | macro greedy {s['macro_mean']['greedy_acc']:.3f} | {tasks}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory, or 'base' for the untrained model")
    ap.add_argument("--store", required=True, help="prompt store to score on")
    ap.add_argument("--versions", default="", help="comma-separated versions (default: 0 and the last archived one)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", required=True, help="output JSON")
    args = ap.parse_args()

    store = stores.Store(args.store)
    if args.run != "base":
        cfg = json.load(open(os.path.join(args.run, "run.json")))["config"]
        shared = len(stores.overlap(stores.Store(cfg["prompt_set"]), store))
        if shared:
            sys.exit(f"refused: {shared} prompts of the training store {cfg['prompt_set']} share a token hash "
                     f"with {args.store}")
        print(f"leak check passed: no prompt of {cfg['prompt_set']} occurs in {args.store}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).cuda().eval()
    results = {"run": args.run, "store": args.store, "points": []}

    if args.run == "base":
        t0 = time.perf_counter()
        s = eval_model_on_store(model, store, args.batch_size)
        results["points"].append({"version": 0, "tasks": s})
        print(summary_line(0, s, time.perf_counter() - t0), flush=True)
    else:
        mj = json.load(open(os.path.join(args.run, "mid_layer.json")))
        lmid, nw = mj["mid_layer"], mj["mid_nw"]
        ids = np.fromfile(os.path.join(args.run, "mid_arch_ids.i32"), dtype="<i4")
        arch = np.memmap(os.path.join(args.run, "mid_arch.bf16"), dtype="<u2", mode="r").reshape(-1, nw)
        slot = {int(v): k for k, v in enumerate(ids) if v >= 0}
        versions = [int(x) for x in args.versions.split(",") if x.strip()] if args.versions else sorted({0, max(slot)})
        missing = [v for v in versions if v not in slot]
        if missing:
            sys.exit(f"versions {missing} are not in mid_arch_ids.i32 {sorted(slot)}")

        base = read_base_layer(cfg["weights"], lmid)
        sizes = [b.numel() for b in base]
        assert sum(sizes) == nw, (sizes, nw)
        lin = linears(model.model.layers[lmid])
        H, nq, nk, ff = lin["q"].in_features, lin["q"].out_features, lin["k"].out_features, lin["gate"].out_features
        mid, handles = install_mid_hooks(model, lmid)
        for v in versions:
            dqkv, do, dgu, ddown = (w.reshape(b.shape) - b for w, b in
                                    zip(bf16_to_cuda(arch[slot[v]]).split(sizes), base))
            parts = (dqkv[:nq, :H], dqkv[nq:nq + nk, :H], dqkv[nq + nk:, :H], do, dgu[:ff], dgu[ff:], ddown)
            mid["delta"] = {name: d.contiguous() for name, d in zip(LINEARS, parts)}
            t0 = time.perf_counter()
            s = eval_model_on_store(model, store, args.batch_size, mid)
            results["points"].append({"version": v, "tasks": s})
            print(summary_line(v, s, time.perf_counter() - t0), flush=True)
        for h in handles:
            h.remove()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
