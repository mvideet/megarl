#!/usr/bin/env python3
"""Record a PyTorch baseline trace for the race: layer-13 GRPO on Qwen2.5-0.5B-Instruct with the headline settings.

    python tools/record_pytorch_race.py --engine hf --duration-s 26.53 --save-every 4 \\
        --out results/pytorch_hf_trace.json
    python tools/record_pytorch_race.py --engine rolling --duration-s 26.53 --save-every 25 \\
        --out results/pytorch_rolling_trace.json

The settings are those of results/headline: bf16 model, the seven weight matrices of layer 13 trained, groups of 4
answers sampled at temperature 1 with at most 64 tokens, reward 0 for an answer without an end token, groups with
equal rewards dropped, groups admitted to the task with the fewest admissions, 16 groups per update taken from the
tasks with the fewest trained groups, groups more than 8 versions old dropped, PPO-clipped GRPO (clip 0.2), AdamW
lr 4e-5, gradient clip 1.0. As in the kernel (from_last=1), the prompt up to its second-to-last token goes through
the base layer 13, and the backward pass runs through layers 23..13 only, from the saved layer-13 inputs.

Engines:
  hf       transformers forward with DynamicCache and SDPA. 128 groups (512 answers) are generated as one batch,
           then the queued groups are trained.
  rolling  512 answer slots decoded by one captured CUDA graph with a static KV cache and precomputed prompt caches.
           The graph is replayed 4 steps at a time; finished groups are scored and replaced between bursts, and
           decoding stops for training whenever 16 groups are queued.

Times count from the start of the race; setup (model load, prompt caches, graph capture) and copying snapshots are
excluded. Layer 13 is copied to host memory every --save-every updates and at the end; after the race the copies
are scored on --eval-store with tools/eval_multitask.py (batch size 256). Launch counts are the profiler's kernel
counts of one decode step, prefill and update times how often each ran, plus the few small ops between them; a
graph replay counts as one host launch.

Output JSON: totals and rates, launch counts, `eval` (per snapshot: ver, t_s, greedy accuracy per task, macro,
sampled_macro) and `events`, one row per host check with the columns named in `events_schema`.
"""
import argparse
import collections
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_multitask
import stores

MODEL_ID, END_TOKENS, MAX_NEW = eval_multitask.MODEL_ID, eval_multitask.END_TOKENS, eval_multitask.MAX_NEW
LMID = 13
NL, D, NH, NKV, HD = 24, 896, 14, 2, 64
NGROUPS = 128
BS = NGROUPS * 4          # answer rows in flight
TGRP = 16                 # groups per update
LAG = 8
LR = 4e-5
MAX_PREFIX = 120          # prompt tokens before the last one; multitask4 prompts have at most 120 tokens
MAX_SEQ = 192             # >= MAX_PREFIX + 1 + MAX_NEW
BURST = 4
ENGINE_NAMES = {"rolling": "PyTorch Continuous-Rolling CUDA-Graph + Prompt Cache + SDPA L13",
                "hf": "PyTorch HF transformers (DynamicCache + SDPA), 512-row generate-then-train"}


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RollingQwenEngine(nn.Module):
    """Graph-capturable decode of BS slots over all 24 layers, and the layers 13..23 forward used for training."""

    def __init__(self, hf_model):
        super().__init__()
        self.embed = hf_model.model.embed_tokens
        self.layers = hf_model.model.layers
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head

        for p in self.parameters():
            p.requires_grad_(False)
        self.train_params = [p for mod in (self.layers[LMID].self_attn, self.layers[LMID].mlp)
                             for p in mod.parameters() if p.ndim == 2]
        for p in self.train_params:
            p.requires_grad_(True)

        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32, device="cuda") / HD))
        t = torch.arange(MAX_SEQ, dtype=torch.float32, device="cuda")
        emb = torch.cat((torch.outer(t, inv_freq), torch.outer(t, inv_freq)), dim=-1)
        buffers = {
            "cos_tab": emb.cos().to(torch.bfloat16),
            "sin_tab": emb.sin().to(torch.bfloat16),
            "k_cache": torch.zeros((NL, BS, NKV, MAX_SEQ, HD), dtype=torch.bfloat16, device="cuda"),
            "v_cache": torch.zeros((NL, BS, NKV, MAX_SEQ, HD), dtype=torch.bfloat16, device="cuda"),
            "h13_hist": torch.zeros((BS, MAX_NEW, D), dtype=torch.bfloat16, device="cuda"),   # layer-13 inputs
            # per-slot state, advanced inside the graph
            "s_tok_in": torch.zeros(BS, dtype=torch.long, device="cuda"),
            "s_h13_ov": torch.zeros((BS, D), dtype=torch.bfloat16, device="cuda"),
            "s_use_ov": torch.zeros(BS, dtype=torch.bool, device="cuda"),
            "s_pos": torch.zeros(BS, dtype=torch.long, device="cuda"),
            "s_step": torch.zeros(BS, dtype=torch.long, device="cuda"),
            "s_done": torch.zeros(BS, dtype=torch.bool, device="cuda"),
            "s_lens": torch.zeros(BS, dtype=torch.long, device="cuda"),
            "s_toks": torch.full((BS, MAX_NEW), -1, dtype=torch.long, device="cuda"),
            "s_logp": torch.zeros((BS, MAX_NEW), dtype=torch.float32, device="cuda"),
            "s_new_toks": torch.zeros(1, dtype=torch.long, device="cuda"),
            "ar_seq": torch.arange(MAX_SEQ, device="cuda").view(1, 1, 1, MAX_SEQ),
        }
        for name, buf in buffers.items():
            self.register_buffer(name, buf, persistent=False)

    def decode_step_inplace(self):
        """One token for every active slot, then the slot bookkeeping; reads and writes only the buffers."""
        pos = self.s_pos.clamp(0, MAX_SEQ - 1)
        step = self.s_step.clamp(0, MAX_NEW - 1)
        active = ~self.s_done

        cos = self.cos_tab.index_select(0, pos).view(BS, 1, 1, HD)
        sin = self.sin_tab.index_select(0, pos).view(BS, 1, 1, HD)
        neg_mask = ~(self.ar_seq <= pos.view(BS, 1, 1, 1))
        scatter_idx = pos.view(BS, 1, 1, 1).expand(BS, NKV, 1, HD)

        h = self.embed(self.s_tok_in).unsqueeze(1)
        for li, layer in enumerate(self.layers):
            if li == LMID:
                # a new slot's first token is the prompt's last; its layer-13 input comes from the prompt cache
                h = torch.where(self.s_use_ov.view(BS, 1, 1), self.s_h13_ov.unsqueeze(1), h)
                step_idx = step.view(BS, 1, 1).expand(BS, 1, D)
                self.h13_hist.scatter_(1, step_idx, torch.where(active.view(BS, 1, 1), h,
                                                                self.h13_hist.gather(1, step_idx)))
            res = h
            x = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(x).view(BS, 1, NH, HD).transpose(1, 2)
            k = layer.self_attn.k_proj(x).view(BS, 1, NKV, HD).transpose(1, 2)
            v = layer.self_attn.v_proj(x).view(BS, 1, NKV, HD).transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)
            self.k_cache[li].scatter_(2, scatter_idx, k)
            self.v_cache[li].scatter_(2, scatter_idx, v)
            scores = torch.matmul(q.view(BS, NKV, NH // NKV, HD), self.k_cache[li].transpose(-1, -2)) * HD ** -0.5
            scores.masked_fill_(neg_mask, -65504.0)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            ao = torch.matmul(probs, self.v_cache[li]).reshape(BS, D).unsqueeze(1)
            h = res + layer.self_attn.o_proj(ao)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        logp_all = F.log_softmax(self.lm_head(self.norm(h[:, 0, :])).float(), dim=-1)
        nxt = torch.multinomial(torch.exp(logp_all), num_samples=1).squeeze(1)
        chosen_lp = logp_all.gather(1, nxt.unsqueeze(1)).squeeze(1)

        step_col = step.unsqueeze(1)
        prev_tok = self.s_toks.gather(1, step_col).squeeze(1)
        prev_lp = self.s_logp.gather(1, step_col).squeeze(1)
        self.s_toks.scatter_(1, step_col, torch.where(active, nxt, prev_tok).unsqueeze(1))
        self.s_logp.scatter_(1, step_col, torch.where(active, chosen_lp, prev_lp).unsqueeze(1))
        self.s_new_toks.add_(active.long().sum())

        next_step = self.s_step + active.long()
        self.s_lens.copy_(torch.where(active, next_step, self.s_lens))
        hit_eos = (nxt == END_TOKENS[0]) | (nxt == END_TOKENS[1]) | (next_step >= MAX_NEW)
        now_done = self.s_done | hit_eos
        still_active = ~now_done

        self.s_tok_in.copy_(torch.where(still_active, nxt, self.s_tok_in))
        self.s_use_ov.masked_fill_(active, False)
        self.s_pos.add_(still_active.long())
        self.s_step.copy_(next_step)
        self.s_done.copy_(now_done)

    def forward_train_l13_23(self, h13_seq, pos_ids, pref_k13, pref_v13, pref_mask):
        """Logits of layers 13..23 + norm + LM head for [B, S] answer tokens from their layer-13 inputs.

        pref_k13 / pref_v13 [B, 11, NKV, MAX_PREFIX, HD] are the prompt-prefix keys and values of layers 13..23,
        pref_mask [B, MAX_PREFIX] marks the valid prefix positions.
        """
        B, S, _ = h13_seq.shape
        P = pref_k13.shape[3]
        cos = self.cos_tab[pos_ids].unsqueeze(1)
        sin = self.sin_tab[pos_ids].unsqueeze(1)
        p_ok = pref_mask.view(B, 1, 1, P).expand(B, 1, S, P)
        s_ok = torch.tril(torch.ones((S, S), dtype=torch.bool, device=h13_seq.device)).expand(B, 1, S, S)
        attn_mask = torch.cat([p_ok, s_ok], dim=-1)

        h = h13_seq
        for idx, layer in enumerate(self.layers[LMID:]):
            res = h
            x = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(x).view(B, S, NH, HD).transpose(1, 2)
            k = layer.self_attn.k_proj(x).view(B, S, NKV, HD).transpose(1, 2)
            v = layer.self_attn.v_proj(x).view(B, S, NKV, HD).transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)
            kc = torch.cat([pref_k13[:, idx], k], dim=2).repeat_interleave(NH // NKV, dim=1)
            vc = torch.cat([pref_v13[:, idx], v], dim=2).repeat_interleave(NH // NKV, dim=1)
            ao = F.scaled_dot_product_attention(q, kc, vc, attn_mask=attn_mask)
            h = res + layer.self_attn.o_proj(ao.transpose(1, 2).contiguous().view(B, S, D))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.lm_head(self.norm(h))


@torch.no_grad()
def build_all_prompt_caches(hf_model, store):
    """Keys/values of every layer for positions 0..len-2 of every prompt, the layer-13 input at the last prompt
    token, the last token and the prompt lengths."""
    n = store.n
    pk_all = torch.zeros((n, NL, NKV, MAX_PREFIX, HD), dtype=torch.bfloat16, device="cuda")
    pv_all = torch.zeros((n, NL, NKV, MAX_PREFIX, HD), dtype=torch.bfloat16, device="cuda")
    h13_last = torch.zeros((n, D), dtype=torch.bfloat16, device="cuda")
    last_tok = torch.zeros(n, dtype=torch.long, device="cuda")
    plens = np.zeros(n, dtype=np.int32)

    cap = {}

    def grab(module, args):
        cap["h"] = args[0][:, -1, :].detach().clone()
    hook = hf_model.model.layers[LMID].register_forward_pre_hook(grab)
    bs = 128
    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        ps = [store.prompt(i) for i in range(i0, i1)]
        Ls = [len(p) for p in ps]
        PMAX = max(Ls)
        ids = torch.zeros((i1 - i0, PMAX), dtype=torch.long, device="cuda")
        am = torch.zeros((i1 - i0, PMAX), dtype=torch.long, device="cuda")
        for b, p in enumerate(ps):
            L = len(p)
            plens[i0 + b] = L
            t_p = torch.as_tensor(np.array(p, copy=True), dtype=torch.long, device="cuda")
            ids[b, PMAX - L:] = t_p
            am[b, PMAX - L:] = 1
            last_tok[i0 + b] = t_p[-1]
        pos_ids = (am.long().cumsum(-1) - 1).clamp(min=0)
        out = hf_model(input_ids=ids, attention_mask=am, position_ids=pos_ids, use_cache=True)
        h13_last[i0:i1] = cap["h"]
        for b, L in enumerate(Ls):
            if L > 1:
                for li, kv in enumerate(out.past_key_values.layers):
                    pk_all[i0 + b, li, :, :L - 1, :] = kv.keys[b, :, PMAX - L:PMAX - 1, :]
                    pv_all[i0 + b, li, :, :L - 1, :] = kv.values[b, :, PMAX - L:PMAX - 1, :]
    hook.remove()
    return pk_all, pv_all, h13_last, last_tok, plens


def count_cuda_ops(fn):
    """CUDA kernels launched by fn(), from the profiler."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return max(1, sum(int(e.count) for e in prof.key_averages() if str(e.device_type).endswith("CUDA") and e.count > 0))


@torch.no_grad()
def hf_prefill(hf_model, base_l13_mod, prompts):
    """HF forward over the left-padded prompts without their last token, with the base layer 13 swapped in.

    Returns the DynamicCache, a [rows, prefix + MAX_NEW + 1] attention mask with the prefix marked, and the prefix
    width.
    """
    n = len(prompts)
    pfx = max(len(p) for p in prompts) - 1
    ids_np = np.zeros((n, pfx), dtype=np.int64)
    am_np = np.zeros((n, pfx + MAX_NEW + 1), dtype=np.int64)
    for r, p in enumerate(prompts):
        L = len(p) - 1
        if L > 0:
            ids_np[r, pfx - L:] = p[:-1]
            am_np[r, pfx - L:pfx] = 1
    ids = torch.from_numpy(ids_np).cuda()
    am = torch.from_numpy(am_np).cuda()
    pos = (am[:, :pfx].cumsum(-1) - 1).clamp(min=0)
    layers = hf_model.model.layers
    train_l13 = layers[LMID]
    layers[LMID] = base_l13_mod
    try:
        out = hf_model.model(input_ids=ids, attention_mask=am[:, :pfx], position_ids=pos, use_cache=True)
    finally:
        layers[LMID] = train_l13
    return out.past_key_values, am, pfx


@torch.no_grad()
def hf_new_state(hf_model, base_l13_mod, prompts):
    past, am, pfx = hf_prefill(hf_model, base_l13_mod, prompts)
    n = len(prompts)
    return {"past": past, "am": am, "pfx": pfx, "step": 0,
            "cur": torch.tensor([p[-1] for p in prompts], dtype=torch.long, device="cuda"),
            "pos": torch.tensor([len(p) - 1 for p in prompts], dtype=torch.long, device="cuda"),
            "toks": torch.full((n, MAX_NEW), -1, dtype=torch.long, device="cuda"),
            "logp": torch.zeros((n, MAX_NEW), dtype=torch.float32, device="cuda"),
            "lens": torch.zeros(n, dtype=torch.long, device="cuda"),
            "done": torch.zeros(n, dtype=torch.bool, device="cuda"),
            "h13": torch.zeros((n, MAX_NEW, D), dtype=torch.bfloat16, device="cuda")}


@torch.no_grad()
def hf_decode_step(hf_model, st, cap):
    """One HF forward of every row with the DynamicCache, then sampling; rows that have ended keep their tokens."""
    s, pfx = st["step"], st["pfx"]
    st["am"][:, pfx + s] = 1
    out = hf_model(input_ids=st["cur"].unsqueeze(1), attention_mask=st["am"][:, :pfx + s + 1],
                   position_ids=st["pos"].unsqueeze(1), past_key_values=st["past"], use_cache=True)
    active = ~st["done"]
    st["h13"][:, s] = torch.where(active.unsqueeze(1), cap["h"], st["h13"][:, s])
    lp_all = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    nxt = torch.multinomial(lp_all.exp(), num_samples=1).squeeze(1)
    st["toks"][:, s] = torch.where(active, nxt, st["toks"][:, s])
    st["logp"][:, s] = torch.where(active, lp_all.gather(1, nxt.unsqueeze(1)).squeeze(1), st["logp"][:, s])
    st["lens"] += active.long()
    ended = (nxt == END_TOKENS[0]) | (nxt == END_TOKENS[1])
    if s == MAX_NEW - 1:
        ended = torch.ones_like(ended)
    st["done"] |= active & ended
    st["cur"] = nxt
    st["pos"] = st["pos"] + 1
    st["step"] = s + 1


def snapshot_l13_cpu(layer13):
    return {name: lin.weight.detach().cpu().clone() for name, lin in eval_multitask.linears(layer13).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["rolling", "hf"], default="rolling")
    ap.add_argument("--store", default="data/prompts/multitask4_train", help="training prompts")
    ap.add_argument("--eval-store", default="data/prompts/multitask4_dev_twin", help="held-out prompts")
    ap.add_argument("--duration-s", type=float, default=26.6, help="race length, excluding setup")
    ap.add_argument("--save-every", type=int, default=30, help="snapshot layer 13 every this many updates")
    ap.add_argument("--out", help="output JSON (default: results/pytorch_<engine>_trace.json)")
    args = ap.parse_args()
    out_path = args.out or f"results/pytorch_{args.engine}_trace.json"

    train_store = stores.Store(args.store)
    dev_store = stores.Store(args.eval_store)
    if len(stores.overlap(train_store, dev_store)):
        sys.exit(f"refused: {args.store} and {args.eval_store} share prompts")
    ntasks = int(train_store.task.max()) + 1
    task_prompts = [np.flatnonzero(train_store.task == t) for t in range(ntasks)]
    rng = np.random.default_rng(1)
    for t in range(ntasks):
        rng.shuffle(task_prompts[t])
    task_cursor = [0] * ntasks

    def next_prompt(t):
        idx = int(task_prompts[t][task_cursor[t] % len(task_prompts[t])])
        task_cursor[t] += 1
        return idx

    t_setup0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa")
    hf_model = hf_model.cuda().eval()
    pk_all, pv_all, h13_last, last_tok, plens = build_all_prompt_caches(hf_model, train_store)
    plens_gpu = torch.from_numpy(plens).long().cuda()
    eng = RollingQwenEngine(hf_model).cuda()
    opt = torch.optim.AdamW(eng.train_params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    hf_prefill_kernels = 0
    if args.engine == "rolling":
        cap_stream = torch.cuda.Stream()
        cap_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap_stream), torch.no_grad():
            for _ in range(3):
                eng.decode_step_inplace()
        torch.cuda.current_stream().wait_stream(cap_stream)
        torch.cuda.synchronize()
        g_decode = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g_decode, stream=cap_stream):
            eng.decode_step_inplace()
        torch.cuda.synchronize()
        with torch.no_grad():
            eager_decode_kernels = count_cuda_ops(eng.decode_step_inplace)
    else:
        base_l13_mod = copy.deepcopy(hf_model.model.layers[LMID]).requires_grad_(False)
        hf_cap = {}

        def grab(module, args):
            hf_cap["h"] = args[0][:, -1, :]
        hf_model.model.layers[LMID].register_forward_pre_hook(grab)
        hf_rows = [train_store.prompt(i) for i in range(BS)]
        with torch.no_grad():
            for _ in range(2):
                st_c = hf_new_state(hf_model, base_l13_mod, hf_rows)
                for _ in range(3):
                    hf_decode_step(hf_model, st_c, hf_cap)
            hf_prefill_kernels = count_cuda_ops(lambda: hf_new_state(hf_model, base_l13_mod, hf_rows))
            st_c = hf_new_state(hf_model, base_l13_mod, hf_rows)
            hf_decode_step(hf_model, st_c, hf_cap)
            # + 2: the count of unfinished rows read after each step
            eager_decode_kernels = count_cuda_ops(lambda: hf_decode_step(hf_model, st_c, hf_cap)) + 2
            del st_c

    def dummy_bwd():
        opt.zero_grad(set_to_none=True)
        dh = torch.randn((64, 64, D), dtype=torch.bfloat16, device="cuda")
        dp = torch.arange(64, device="cuda").unsqueeze(0).expand(64, 64) + 80
        dpk = torch.zeros((64, NL - LMID, NKV, MAX_PREFIX, HD), dtype=torch.bfloat16, device="cuda")
        dpv = torch.zeros((64, NL - LMID, NKV, MAX_PREFIX, HD), dtype=torch.bfloat16, device="cuda")
        dpm = torch.ones((64, MAX_PREFIX), dtype=torch.bool, device="cuda")
        lg = eng.forward_train_l13_23(dh, dp, dpk, dpv, dpm)
        tok = torch.zeros((64, 64, 1), dtype=torch.long, device="cuda")
        (-F.log_softmax(lg.float(), dim=-1).gather(-1, tok).squeeze(-1).mean()).backward()
        torch.nn.utils.clip_grad_norm_(eng.train_params, 1.0)
        opt.step()

    base_snap = snapshot_l13_cpu(eng.layers[LMID])
    dummy_bwd()
    bwd_kernels = count_cuda_ops(dummy_bwd)
    with torch.no_grad():
        for name, lin in eval_multitask.linears(eng.layers[LMID]).items():
            lin.weight.copy_(base_snap[name])
    opt = torch.optim.AdamW(eng.train_params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    setup_s = time.perf_counter() - t_setup0
    print(f"setup {setup_s:.2f}s | kernels per decode step {eager_decode_kernels}, per update {bwd_kernels}",
          flush=True)

    admitted_per_task = [0] * ntasks
    trained_per_task = [0] * ntasks
    queues = [collections.deque() for _ in range(ntasks)]
    grp_pid = np.zeros(NGROUPS, dtype=np.int32)
    grp_task = np.zeros(NGROUPS, dtype=np.int32)
    grp_ver = np.zeros(NGROUPS, dtype=np.int32)

    launches_host = launches_cuda_ops = 0
    tokens_cum = groups_gen = groups_trained = updates = 0
    gen_time_s = bwd_time_s = max_lag0_dlogp = snap_overhead_s = 0.0
    events = []
    saved_snaps = [(0, 0.0, base_snap)]

    def race_time(t):
        return t - t_race0 - snap_overhead_s

    def log_event(t, mode):
        events.append([round(t, 5), mode, tokens_cum, groups_gen, groups_trained, updates, launches_host,
                       launches_cuda_ops])

    def new_group(g, ver):
        t = int(np.argmin(admitted_per_task))
        admitted_per_task[t] += 1
        grp_pid[g], grp_task[g], grp_ver[g] = next_prompt(t), t, ver
        return int(grp_pid[g])

    @torch.no_grad()
    def admit_groups(g_indices, ver):
        """Start new groups in the rolling engine's slots of groups g_indices."""
        if not g_indices:
            return
        pids = [new_group(g, ver) for g in g_indices]
        rows_t = torch.tensor([g * 4 + j for g in g_indices for j in range(4)], dtype=torch.long, device="cuda")
        pids_t = torch.tensor([p for p in pids for _ in range(4)], dtype=torch.long, device="cuda")
        eng.k_cache[:, rows_t, :, :MAX_PREFIX, :] = pk_all.index_select(0, pids_t).transpose(0, 1)
        eng.v_cache[:, rows_t, :, :MAX_PREFIX, :] = pv_all.index_select(0, pids_t).transpose(0, 1)
        eng.s_tok_in.index_copy_(0, rows_t, last_tok.index_select(0, pids_t))
        eng.s_h13_ov.index_copy_(0, rows_t, h13_last.index_select(0, pids_t))
        eng.s_use_ov.index_fill_(0, rows_t, True)
        eng.s_pos.index_copy_(0, rows_t, plens_gpu.index_select(0, pids_t) - 1)
        eng.s_step.index_fill_(0, rows_t, 0)
        eng.s_done.index_fill_(0, rows_t, False)
        eng.s_lens.index_fill_(0, rows_t, 0)

    def score_groups(fin_groups, toks_fin, logp_fin, lens_fin, h13_fin):
        """Rewards of finished groups; groups whose four rewards differ go to their task's queue."""
        for gi, g in enumerate(fin_groups):
            p = int(grp_pid[g])
            key = int(train_store.key[p])
            rews = np.zeros(4, dtype=np.float32)
            for j in range(4):
                seq = toks_fin[gi, j, :int(lens_fin[gi, j])]
                ended = len(seq) > 0 and int(seq[-1]) in END_TOKENS
                rews[j] = float(train_store.reward(seq, key)) if ended else 0.0
            if float(rews.max() - rews.min()) > 1e-6:
                adv = (rews - float(rews.mean())) / (float(rews.std(ddof=1)) + 1e-4)
                queues[int(grp_task[g])].append({"pid": p, "ver": int(grp_ver[g]), "lens": lens_fin[gi],
                                                 "toks": toks_fin[gi], "logp_beh": logp_fin[gi], "adv": adv,
                                                 "h13": h13_fin[gi]})

    def train_ready():
        """GRPO updates, with generation stopped, while TGRP groups are queued."""
        nonlocal updates, groups_trained, trained_per_task, launches_host, launches_cuda_ops, bwd_time_s
        nonlocal max_lag0_dlogp, snap_overhead_s
        while True:
            for q in queues:
                while q and updates - q[0]["ver"] > LAG:
                    q.popleft()
            if sum(len(q) for q in queues) < TGRP:
                return

            tb0 = time.perf_counter()
            log_event(race_time(tb0), 1)
            batch = []
            sim_tr = list(trained_per_task)
            for _ in range(TGRP):
                pick = min((k for k in range(ntasks) if queues[k]), key=lambda k: sim_tr[k])
                sim_tr[pick] += 1
                batch.append(queues[pick].popleft())

            rows = TGRP * 4
            S_max = int(max(int(item["lens"].max()) for item in batch))
            h13_b = torch.stack([item["h13"][:, :S_max, :] for item in batch], dim=0).view(rows, S_max, D)
            pids_bt = torch.tensor([item["pid"] for item in batch for _ in range(4)], dtype=torch.long, device="cuda")
            pref_k13 = pk_all.index_select(0, pids_bt)[:, LMID:, :, :, :]
            pref_v13 = pv_all.index_select(0, pids_bt)[:, LMID:, :, :, :]
            pl_b = plens_gpu.index_select(0, pids_bt)
            pref_mask = torch.arange(MAX_PREFIX, device="cuda").unsqueeze(0) < (pl_b - 1).unsqueeze(1)
            pos_ids = (pl_b - 1).unsqueeze(1) + torch.arange(S_max, device="cuda").unsqueeze(0)

            toks_np = np.stack([item["toks"][:, :S_max] for item in batch], axis=0).reshape(rows, S_max)
            toks_b = torch.from_numpy(np.maximum(toks_np, 0)).long().cuda()
            beh_b = torch.from_numpy(np.stack([item["logp_beh"][:, :S_max] for item in batch], axis=0)
                                     .reshape(rows, S_max)).cuda()
            lens_b = torch.from_numpy(np.concatenate([item["lens"] for item in batch], axis=0)).long().cuda()
            adv_b = torch.from_numpy(np.concatenate([item["adv"] for item in batch], axis=0)).float().cuda()
            valid_m = torch.arange(S_max, device="cuda").unsqueeze(0) < lens_b.unsqueeze(1)

            opt.zero_grad(set_to_none=True)
            logits = eng.forward_train_l13_23(h13_b, pos_ids, pref_k13, pref_v13, pref_mask)
            logp_new = F.log_softmax(logits.float(), dim=-1).gather(-1, toks_b.unsqueeze(-1)).squeeze(-1)
            if updates == 0:
                max_lag0_dlogp = float(((logp_new - beh_b).abs() * valid_m.float()).max().item())
            rho = torch.exp((logp_new - beh_b).clamp(-20.0, 20.0))
            adv_e = adv_b.unsqueeze(1)
            tok_loss = -torch.minimum(rho * adv_e, torch.clamp(rho, 1.0 - 0.2, 1.0 + 0.2) * adv_e) * valid_m.float()
            (tok_loss.sum() / (rows * MAX_NEW)).backward()
            torch.nn.utils.clip_grad_norm_(eng.train_params, 1.0)
            opt.step()
            torch.cuda.synchronize()

            updates += 1
            groups_trained += TGRP
            trained_per_task = sim_tr
            launches_host += bwd_kernels
            launches_cuda_ops += bwd_kernels
            tb1 = time.perf_counter()
            bwd_time_s += tb1 - tb0
            t_cur = race_time(tb1)
            log_event(t_cur, 1)
            log_event(t_cur, 0)

            if updates % args.save_every == 0:
                ts0 = time.perf_counter()
                saved_snaps.append((updates, round(t_cur, 4), snapshot_l13_cpu(eng.layers[LMID])))
                torch.cuda.synchronize()
                snap_overhead_s += time.perf_counter() - ts0
            if t_cur >= args.duration_s:
                return

    def run_hf():
        """Generate 128 groups as one HF batch, score them, train on what is queued, repeat."""
        nonlocal tokens_cum, groups_gen, launches_host, launches_cuda_ops, gen_time_s
        while race_time(time.perf_counter()) < args.duration_s:
            tg0 = time.perf_counter()
            prompts = [train_store.prompt(new_group(g, updates)) for g in range(NGROUPS)]
            st = hf_new_state(hf_model, base_l13_mod, [p for p in prompts for _ in range(4)])
            launches_host += hf_prefill_kernels
            launches_cuda_ops += hf_prefill_kernels
            remaining = BS
            while remaining > 0:
                hf_decode_step(hf_model, st, hf_cap)
                tokens_cum += remaining
                remaining = int((~st["done"]).sum().item())
                launches_host += eager_decode_kernels
                launches_cuda_ops += eager_decode_kernels
                t_cur = race_time(time.perf_counter())
                log_event(t_cur, 0)
                if t_cur >= args.duration_s:
                    gen_time_s += time.perf_counter() - tg0
                    return
            gen_time_s += time.perf_counter() - tg0
            groups_gen += NGROUPS
            with torch.no_grad():
                toks_fin = st["toks"].cpu().numpy().reshape(NGROUPS, 4, MAX_NEW)
                logp_fin = st["logp"].cpu().numpy().reshape(NGROUPS, 4, MAX_NEW)
                lens_fin = st["lens"].cpu().numpy().reshape(NGROUPS, 4)
                h13_fin = st["h13"].view(NGROUPS, 4, MAX_NEW, D)
            del st
            score_groups(list(range(NGROUPS)), toks_fin, logp_fin, lens_fin, h13_fin)
            log_event(race_time(time.perf_counter()), 0)
            train_ready()

    def run_rolling():
        """Replay BURST decode steps, retire and replace finished groups, train when TGRP groups are queued."""
        nonlocal tokens_cum, groups_gen, launches_host, launches_cuda_ops, gen_time_s
        while race_time(time.perf_counter()) < args.duration_s:
            tg0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(BURST):
                    g_decode.replay()
                fin_gpu = torch.nonzero(eng.s_done.view(NGROUPS, 4).all(dim=1), as_tuple=False).squeeze(1)
                tokens_cum = int(eng.s_new_toks.item())
                fin_groups = fin_gpu.cpu().tolist()
            # the replays, and the three ops that find the finished groups
            launches_host += BURST + 3
            launches_cuda_ops += BURST * eager_decode_kernels + 3

            if fin_groups:
                groups_gen += len(fin_groups)
                nfin = len(fin_groups)
                rf_t = torch.tensor([g * 4 + j for g in fin_groups for j in range(4)], dtype=torch.long, device="cuda")
                with torch.no_grad():
                    toks_fin = eng.s_toks.index_select(0, rf_t).cpu().numpy().reshape(nfin, 4, MAX_NEW)
                    logp_fin = eng.s_logp.index_select(0, rf_t).cpu().numpy().reshape(nfin, 4, MAX_NEW)
                    lens_fin = eng.s_lens.index_select(0, rf_t).cpu().numpy().reshape(nfin, 4)
                    h13_fin = eng.h13_hist.index_select(0, rf_t).view(nfin, 4, MAX_NEW, D).clone()
                score_groups(fin_groups, toks_fin, logp_fin, lens_fin, h13_fin)
                admit_groups(fin_groups, updates)
                # the five copies out of the slots, the nine writes of admit_groups
                launches_host += 5 + 9
                launches_cuda_ops += 5 + 9

            torch.cuda.synchronize()
            tg1 = time.perf_counter()
            gen_time_s += tg1 - tg0
            log_event(race_time(tg1), 0)
            train_ready()

    eng.s_new_toks.zero_()
    if args.engine == "rolling":
        admit_groups(list(range(NGROUPS)), 0)
    torch.cuda.synchronize()
    t_race0 = time.perf_counter()
    log_event(0.0, 0)
    if args.engine == "hf":
        run_hf()
    else:
        run_rolling()

    wall_s = race_time(time.perf_counter())
    if saved_snaps[-1][0] != updates:
        saved_snaps.append((updates, round(wall_s, 4), snapshot_l13_cpu(eng.layers[LMID])))
    print(f"{ENGINE_NAMES[args.engine]}: {wall_s:.3f}s (update-0 max |dlogp| {max_lag0_dlogp:.4f})\n"
          f"  tokens {tokens_cum} ({tokens_cum / wall_s:.1f}/s) | groups generated {groups_gen} "
          f"({groups_gen / wall_s:.1f}/s) | trained {groups_trained} ({groups_trained / wall_s:.1f}/s) | "
          f"updates {updates} ({updates / wall_s:.2f}/s)\n"
          f"  generating {gen_time_s:.3f}s ({100 * gen_time_s / wall_s:.1f}%) | training {bwd_time_s:.3f}s "
          f"({100 * bwd_time_s / wall_s:.1f}%) | trained per task {trained_per_task}\n"
          f"  kernel launches {launches_host:,} (CUDA kernels including those inside graphs {launches_cuda_ops:,})",
          flush=True)

    del eng, pk_all, pv_all, h13_last, hf_model
    torch.cuda.empty_cache()
    print(f"scoring {len(saved_snaps)} snapshots on {args.eval_store}", flush=True)
    eval_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).cuda().eval()
    base_l13 = {k: v.float().cuda() for k, v in base_snap.items()}
    mid, handles = eval_multitask.install_mid_hooks(eval_model, LMID)
    eval_points = []
    for ver, t_s, snap in saved_snaps:
        mid["delta"] = {k: (snap[k].float().cuda() - base_l13[k]).contiguous() if ver > 0 else None for k in snap}
        te0 = time.perf_counter()
        s = eval_multitask.eval_model_on_store(eval_model, dev_store, batch_size=256, mid=mid)
        tasks = [k for k in s if k != "macro_mean"]
        eval_points.append({"ver": int(ver), "t_s": float(t_s),
                            **{k: round(float(s[k]["greedy_acc"]), 4) for k in tasks},
                            "macro": round(float(s["macro_mean"]["greedy_acc"]), 4),
                            "sampled_macro": round(float(s["macro_mean"]["sampled_reward"]), 4)})
        print(f"  v{ver:3d} @ {t_s:6.2f}s [{time.perf_counter() - te0:.1f}s] | "
              f"macro {s['macro_mean']['greedy_acc']:.3f} | " + " | ".join(f"{k} {s[k]['greedy_acc']:.3f}" for k in tasks),
              flush=True)
    for h in handles:
        h.remove()

    out_obj = {
        "name": ENGINE_NAMES[args.engine],
        "engine": args.engine,
        "device": torch.cuda.get_device_name(),
        "wall_s": round(wall_s, 4),
        "setup_s": round(setup_s, 3),
        "tokens": int(tokens_cum),
        "tokens_per_s": round(tokens_cum / wall_s, 1),
        "groups_gen": int(groups_gen),
        "groups_gen_per_s": round(groups_gen / wall_s, 2),
        "groups_trained": int(groups_trained),
        "groups_trained_per_s": round(groups_trained / wall_s, 2),
        "updates": int(updates),
        "updates_per_s": round(updates / wall_s, 3),
        "gen_time_s": round(gen_time_s, 4),
        "bwd_time_s": round(bwd_time_s, 4),
        "bwd_frac": round(bwd_time_s / wall_s, 4),
        "max_lag0_dlogp": round(max_lag0_dlogp, 5),
        "kernel_launches": int(launches_host),
        "cuda_op_launches": int(launches_cuda_ops),
        "bwd_kernels_per_update": int(bwd_kernels),
        "eager_decode_kernels_per_step": int(eager_decode_kernels),
        "hf_prefill_kernels_per_round": int(hf_prefill_kernels),
        "trained_per_task": [int(x) for x in trained_per_task],
        "reference_baselines": {        # measured separately, not by this tool
            "hf_sdpa_l13_tok_s": 5396.0,
            "repo_grpo_full_torch_tok_s": 3712.0,
            "batched_cudagraph_sdpa_l13_tok_s": 13103.0,
        },
        "eval_store": args.eval_store,
        "eval": eval_points,
        "events_schema": ["t_s", "mode_0gen_1bwd", "tokens_cum", "groups_gen", "groups_trained", "updates",
                          "kernel_launches", "cuda_op_launches"],
        "events": events,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_obj, f)
    print(f"wrote {out_path} ({len(events)} events, {len(eval_points)} eval points)", flush=True)


if __name__ == "__main__":
    main()
