#!/usr/bin/env python3
"""Build the 4-environment prompt stores (math, science, sciq, csqa; docs/FORMATS.md section 2). CPU only.

    python tools/build_multitask4.py [--src data/prompts] [--out data/prompts]

Inputs under --src:
  train, dev_a, dev_a_twin      2-environment stores of math (task 0) and ARC science (task 1) questions; their
                                prompts are reused token for token and their token tables copied. dev_a_twin has
                                the questions and keys of dev_a with the math distractors redrawn
  _multienv_src/sciq.jsonl,     SciQ and CommonsenseQA questions, one JSON object per line with question, options
  _multienv_src/csqa.jsonl      (4), key (0..3) and split ("train" or "heldout")

Outputs under --out:
  multitask4_train      896 prompts per environment (train and the train splits)
  multitask4_dev        400 prompts per environment (dev_a and the heldout splits)
  multitask4_dev_twin   as multitask4_dev, with the math and science prompts of dev_a_twin

The tokenizer is Qwen/Qwen2.5-0.5B-Instruct. The script fails if a train prompt shares a question (sciq, csqa) or
a token hash (all environments) with a dev prompt.
"""
import argparse
import json
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stores

LETTERS = "ABCD"
TASK_NAMES = ["math", "science", "sciq", "csqa"]
MAX_PROMPT = 120


def render_prompt(question, options):
    opts = "\n".join(f"{LETTERS[k]}. {options[k]}" for k in range(4))
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question}\n{opts}\n"
        "Explain briefly (at most 20 words). Then write 'Answer: ' followed by "
        "the letter of the correct option (A, B, C or D).<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def pick_key_uniform(recs, n_total, seed):
    """n_total records, n_total / 4 for each key, in a seeded random order."""
    assert n_total % 4 == 0
    per_key = n_total // 4
    rng = random.Random(seed)
    by_k = {k: [] for k in range(4)}
    for r in recs:
        k = int(r["key"])
        if 0 <= k <= 3 and len(r["options"]) == 4:
            by_k[k].append(r)
    for k in range(4):
        rng.shuffle(by_k[k])
        if len(by_k[k]) < per_key:
            raise RuntimeError(f"not enough records for key {k}: {len(by_k[k])} < {per_key}")
    out = []
    for i in range(per_key):
        for k in range(4):
            out.append(by_k[k][i])
    rng.shuffle(out)
    return out


def write_store(out_dir, task_rows, tok_src_dir):
    """Write one store from task_rows[t] (equal-length lists of rows with tokens), interleaving the environments."""
    os.makedirs(out_dir, exist_ok=True)
    for fn in ("tok_letter.i8", "tok_colon.u8", "tok_answer.u8"):
        shutil.copyfile(os.path.join(tok_src_dir, fn), os.path.join(out_dir, fn))

    n_per = len(task_rows[0])
    assert all(len(x) == n_per for x in task_rows)

    interleaved = []
    for i in range(n_per):
        for t in range(4):
            r = dict(task_rows[t][i])
            r["id"] = len(interleaved)
            r["task"] = t
            r["task_name"] = TASK_NAMES[t]
            r["key_letter"] = LETTERS[int(r["key"])]
            interleaved.append(r)

    all_toks = []
    max_len = max(r["len"] for r in interleaved)
    with open(os.path.join(out_dir, "prompts.idx"), "w") as fidx, \
         open(os.path.join(out_dir, "prompts.jsonl"), "w") as fjl:
        fidx.write(f"{len(interleaved)} {max_len} 0\n")
        off = 0
        for r in interleaved:
            toks = r.pop("tokens")
            fidx.write(f"{r['id']} {r['task']} {r['key']} {r['len']} {off}\n")
            all_toks.extend(toks)
            off += len(toks)
            fjl.write(json.dumps(r, ensure_ascii=False) + "\n")

    np.asarray(all_toks, dtype="<i4").tofile(os.path.join(out_dir, "prompts.i32"))
    with open(os.path.join(out_dir, "envs.json"), "w") as fe:
        json.dump({"names": TASK_NAMES, "ntasks": 4, "n_per_task": n_per}, fe, indent=1)
        fe.write("\n")

    s = stores.Store(out_dir)
    meta = {"n_prompts": s.n, "max_len": s.max_len, "verifier_mode": 0, "tasks": TASK_NAMES, "sha256": s.sha256()}
    with open(os.path.join(out_dir, "build_meta.json"), "w") as fm:
        json.dump(meta, fm, indent=1)
        fm.write("\n")
    print(f"wrote {out_dir}: n={s.n} ({n_per} x 4 tasks), max_len={s.max_len}, sha256={s.sha256()[:16]}")
    return s


def load_existing_split(store_dir):
    """(math rows, science rows) of a 2-environment store, each row with its tokens."""
    s = stores.Store(store_dir)
    with open(os.path.join(store_dir, "prompts.jsonl")) as f:
        jrows = [json.loads(ln) for ln in f]
    by_t = {0: [], 1: []}
    for i, r in enumerate(jrows):
        r = dict(r)
        r["tokens"] = s.prompt(i).astype(int).tolist()
        by_t[int(r["task"])].append(r)
    return by_t[0], by_t[1]


def prepare_ext(jsonl_path, split_name, n_pick, seed, kind_name, tok):
    """n_pick rows of one split of a SciQ/CSQA file: rendered, at most MAX_PROMPT tokens, no question twice
    (ignoring case), keys uniform."""
    with open(jsonl_path) as f:
        raw = [r for r in (json.loads(ln) for ln in f) if r["split"] == split_name]
    valid = []
    seen_q = set()
    for r in raw:
        q = r["question"].strip()
        if q.lower() in seen_q:
            continue
        prompt = render_prompt(q, r["options"])
        ids = tok(prompt, add_special_tokens=False).input_ids
        if len(ids) <= MAX_PROMPT:
            seen_q.add(q.lower())
            valid.append({
                "question": q,
                "options": r["options"],
                "key": int(r["key"]),
                "prompt": prompt,
                "kind": kind_name,
                "source": f"{kind_name}/{split_name}",
                "len": len(ids),
                "tokens": ids,
            })
    return pick_key_uniform(valid, n_pick, seed)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/prompts")
    ap.add_argument("--out", default="data/prompts")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    ext = os.path.join(a.src, "_multienv_src")
    tr_math, tr_sci = load_existing_split(os.path.join(a.src, "train"))
    dv_math, dv_sci = load_existing_split(os.path.join(a.src, "dev_a"))
    tw_math, tw_sci = load_existing_split(os.path.join(a.src, "dev_a_twin"))

    tr_sciq = prepare_ext(os.path.join(ext, "sciq.jsonl"), "train", 896, 20261116, "SciQ", tok)
    dv_sciq = prepare_ext(os.path.join(ext, "sciq.jsonl"), "heldout", 400, 20261117, "SciQ", tok)
    tr_csqa = prepare_ext(os.path.join(ext, "csqa.jsonl"), "train", 896, 20261118, "CSQA", tok)
    dv_csqa = prepare_ext(os.path.join(ext, "csqa.jsonl"), "heldout", 400, 20261119, "CSQA", tok)

    for nm, tr, dv in (("sciq", tr_sciq, dv_sciq), ("csqa", tr_csqa, dv_csqa)):
        q_tr = {r["question"].strip().lower() for r in tr}
        ov = [r for r in dv if r["question"].strip().lower() in q_tr]
        if ov:
            raise RuntimeError(f"{nm}: {len(ov)} dev questions also occur in train")

    s_tr = write_store(os.path.join(a.out, "multitask4_train"), [tr_math, tr_sci, tr_sciq, tr_csqa], os.path.join(a.src, "train"))
    s_dv = write_store(os.path.join(a.out, "multitask4_dev"), [dv_math, dv_sci, dv_sciq, dv_csqa], os.path.join(a.src, "dev_a"))
    s_tw = write_store(os.path.join(a.out, "multitask4_dev_twin"), [tw_math, tw_sci, dv_sciq, dv_csqa], os.path.join(a.src, "dev_a_twin"))

    for name, s in (("multitask4_dev", s_dv), ("multitask4_dev_twin", s_tw)):
        if len(stores.overlap(s_tr, s)):
            raise RuntimeError(f"multitask4_train shares {len(stores.overlap(s_tr, s))} prompts with {name}")
    print("no train prompt occurs in multitask4_dev or multitask4_dev_twin")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
