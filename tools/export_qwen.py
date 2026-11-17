#!/usr/bin/env python3
"""Write Qwen/Qwen2.5-0.5B-Instruct in the raw weight format the kernel reads (docs/FORMATS.md section 1).

    python tools/export_qwen.py            # writes data/qwen05b/{weights.bin,weights.idx,config.txt} and checks them
    python tools/export_qwen.py --check    # only checks an existing export

weights.bin holds the tensors back to back, each at a multiple of 64 bytes; weights.idx has one line
`name dtype rows cols offset_bytes nbytes` per tensor; config.txt has one `key value` line per model dimension.
Every value is read from the loaded model and tokenizer, and the export refuses a model that differs from the
geometry src/mrl_common.cuh is compiled for. The check re-reads every tensor through weights.idx and compares it
bit for bit with the model.

The parameters are loaded in float32 (the checkpoint stores bfloat16, so the widening is exact) and written as
bfloat16, except the RMSNorm weights, which stay float32. rope_cos and rope_sin [1024, 32] come from the model's
own rotary embedding:

    inv_freq[i] = 1 / theta ** (arange(0, head_dim, 2) / head_dim)       float32
    cos[p, i], sin[p, i] = cos(float32(p) * inv_freq[i]), sin(...)       float32 cos/sin of the device

torch's CPU and CUDA float32 cos/sin differ in the last bit for about one entry in eight. The tables are computed
on CUDA when a GPU is visible, so the kernel's RoPE equals that of transformers on the GPU bit for bit. Because the
angle is rounded to float32, they differ from the exact cos(p * theta^(-2i/d)) by up to 3.6e-5.
"""
import argparse
import hashlib
import itertools
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ALIGN = 64
NP = {"bf16": np.dtype("<u2"), "f32": np.dtype("<f4")}
# the geometry src/mrl_common.cuh is compiled for; MAX_POS is its MAXPOS, the rows of the RoPE tables
KERNEL = {"hidden": 896, "layers": 24, "heads": 14, "kv_heads": 2, "head_dim": 64, "inter": 4864, "vocab": 151936,
          "rms_eps": 1e-6, "rope_theta": 1e6}
MAX_POS = 1024


def fmt_num(x):
    """1e-06 -> '1e-6', 1000000.0 -> '1000000': plain text that strtod reads back exactly."""
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = re.sub(r"e([+-])0*(\d)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), repr(x))
    assert float(s) == x, (s, x)
    return s


def bf16(t):
    """bfloat16 bit patterns of a float tensor, rounded to nearest even."""
    return t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def f32(t):
    return t.detach().to(torch.float32).numpy().reshape(1, -1)


def read_config(model, tok):
    """The config.txt entries, in file order."""
    core = model.model
    return {
        "hidden": core.embed_tokens.weight.shape[1],
        "layers": len(core.layers),
        "heads": model.config.num_attention_heads,
        "kv_heads": model.config.num_key_value_heads,
        "head_dim": core.layers[0].self_attn.head_dim,
        "inter": core.layers[0].mlp.gate_proj.out_features,
        "vocab": core.embed_tokens.weight.shape[0],
        "rms_eps": core.layers[0].input_layernorm.variance_epsilon,
        "rope_theta": model.config.rope_parameters["rope_theta"],
        "eos_im_end": tok.convert_tokens_to_ids("<|im_end|>"),
        "eos_endoftext": tok.convert_tokens_to_ids("<|endoftext|>"),
        "max_pos": MAX_POS,
    }


def unsupported(model, cfg):
    """What the kernel cannot run in this model; empty if nothing."""
    core = model.model
    a, m = core.layers[0].self_attn, core.layers[0].mlp
    bad = [f"{k} = {cfg[k]}, the kernel has {v}" for k, v in KERNEL.items() if cfg[k] != v]
    if model.lm_head.weight.data_ptr() != core.embed_tokens.weight.data_ptr():
        bad.append("the LM head is not tied to the embedding")
    biased = [lin.bias is not None for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj,
                                               m.down_proj, model.lm_head)]
    if biased != [True] * 3 + [False] * 5:
        bad.append("biases other than on q/k/v_proj")
    if model.config.hidden_act != "silu":
        bad.append(f"activation {model.config.hidden_act}")
    if core.rotary_emb.rope_type != "default" or core.rotary_emb.attention_scaling != 1.0:
        bad.append(f"RoPE type {core.rotary_emb.rope_type}")
    if set(model.config.layer_types) != {"full_attention"}:
        bad.append("layers without full attention")
    if any(not torch.equal(v.to(torch.bfloat16).float(), v) for v in model.state_dict().values()):
        bad.append("parameters that are not exactly bfloat16")
    return bad


def rope_tables(model, device):
    """cos, sin [MAX_POS, head_dim / 2] float32 from the model's rotary embedding, computed on `device`."""
    x = torch.zeros(1, dtype=torch.float32, device=device)
    cos, sin = model.model.rotary_emb(x, torch.arange(MAX_POS, device=device)[None, :])
    cos, sin = cos[0].cpu(), sin[0].cpu()
    half = cos.shape[1] // 2
    assert torch.equal(cos[:, :half], cos[:, half:]) and torch.equal(sin[:, :half], sin[:, half:]), \
        "rotary embedding does not return cat(freqs, freqs)"
    return cos[:, :half].contiguous().numpy(), sin[:, :half].contiguous().numpy()


def tensors(model, cos, sin):
    """(name, dtype, array) in file order."""
    core = model.model
    H = core.embed_tokens.weight.shape[1]
    yield "emb", "bf16", bf16(core.embed_tokens.weight)
    for l, layer in enumerate(core.layers):
        a, m = layer.self_attn, layer.mlp
        w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
        qkv = np.zeros((w.shape[0], H + 32), dtype=np.uint16)          # column H: bias, then zero to H + 32
        qkv[:, :H] = bf16(w)
        qkv[:, H] = bf16(torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]))
        yield f"L{l}.qkv", "bf16", qkv
        yield f"L{l}.o", "bf16", bf16(a.o_proj.weight)
        yield f"L{l}.gu", "bf16", bf16(torch.cat([m.gate_proj.weight, m.up_proj.weight]))
        yield f"L{l}.down", "bf16", bf16(m.down_proj.weight)
        yield f"L{l}.ln1", "f32", f32(layer.input_layernorm.weight)
        yield f"L{l}.ln2", "f32", f32(layer.post_attention_layernorm.weight)
    yield "lnf", "f32", f32(core.norm.weight)
    yield "rope_cos", "f32", cos
    yield "rope_sin", "f32", sin


def write_atomic(path, data):
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)


def config_text(cfg):
    return "".join(f"{k} {fmt_num(v)}\n" for k, v in cfg.items())


def export(out, model, cfg, cos, sin):
    os.makedirs(out, exist_ok=True)
    bin_path = os.path.join(out, "weights.bin")
    sha = hashlib.sha256()
    idx = []
    off = 0
    with open(bin_path + ".tmp", "wb") as fh:
        for name, dt, arr in tensors(model, cos, sin):
            assert arr.dtype == NP[dt], (name, arr.dtype)
            pad = -off % ALIGN
            for buf in (bytes(pad), np.ascontiguousarray(arr).tobytes()):
                fh.write(buf)
                sha.update(buf)
            off += pad
            idx.append(f"{name} {dt} {arr.shape[0]} {arr.shape[1]} {off} {arr.nbytes}\n")
            off += arr.nbytes
    os.replace(bin_path + ".tmp", bin_path)
    write_atomic(os.path.join(out, "weights.idx"), "".join(idx).encode())
    write_atomic(os.path.join(out, "config.txt"), config_text(cfg).encode())
    print(f"wrote {out}: {len(idx)} tensors, {off} bytes, sha256 of weights.bin {sha.hexdigest()}")


def check(out, model, cfg, cos, sin):
    """Problems found in the export in `out`; empty if it matches the model bit for bit."""
    bad = []
    ent = [line.split() for line in open(os.path.join(out, "weights.idx")) if line.strip()]
    mm = np.memmap(os.path.join(out, "weights.bin"), dtype=np.uint8, mode="r")
    end = 0
    for i, (t, e) in enumerate(itertools.zip_longest(tensors(model, cos, sin), ent)):
        if t is None or e is None:
            bad.append(f"weights.idx has {len(ent)} lines, the model " + (f"{i} tensors" if t is None else "more"))
            break
        name, dt, arr = t
        want = [name, dt, str(arr.shape[0]), str(arr.shape[1])]
        if len(e) != 6 or e[:4] != want or int(e[5]) != arr.nbytes:
            bad.append(f"weights.idx line {i + 1} is '{' '.join(e)}', expected '{' '.join(want)} offset {arr.nbytes}'")
            continue
        off = int(e[4])
        if off % ALIGN or off < end or off + arr.nbytes > len(mm):
            bad.append(f"{name}: offset {off} is not a multiple of {ALIGN}, overlaps the previous tensor or runs "
                       f"past the end of weights.bin")
        elif mm[off:off + arr.nbytes].tobytes() != np.ascontiguousarray(arr).tobytes():
            bad.append(f"{name} differs from the model")
        end = off + arr.nbytes
    if not bad and end != len(mm):
        bad.append(f"weights.bin has {len(mm)} bytes, the last tensor ends at {end}")
    if open(os.path.join(out, "config.txt")).read() != config_text(cfg):
        bad.append("config.txt differs from:\n" + config_text(cfg).rstrip())
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face id or local directory")
    ap.add_argument("--out", default="data/qwen05b")
    ap.add_argument("--check", action="store_true", help="do not write; check the existing files")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    cfg = read_config(model, tok)
    bad = unsupported(model, cfg)
    if bad:
        sys.exit(f"{args.model} does not fit the kernel: " + "; ".join(bad))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("no GPU visible: RoPE tables from the CPU, which differ from transformers on CUDA in the last bit")
    cos, sin = rope_tables(model, device)

    if not args.check:
        export(args.out, model, cfg, cos, sin)
    bad = check(args.out, model, cfg, cos, sin)
    for b in bad:
        print("FAIL", b)
    print(f"check of {args.out}: " + ("FAILED" if bad else "every tensor and config.txt match the model"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
