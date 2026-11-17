# Results

The runs the blog and the demo are built from. Run directories are described in `docs/FORMATS.md`; `tools/rundir.py`
reads them.

| path | what |
|---|---|
| `headline/` | the MegaRL run in the race: 250 updates, 4 tasks, lag limit 8 |
| `headline_eval.json` | held-out accuracy of `headline/` over its archived versions, on `data/prompts/multitask4_dev_twin` |
| `pytorch_hf_trace.json` | the HF transformers PyTorch loop in the race, with its own held-out scores |
| `pytorch_rolling_trace.json` | the hand-tuned PyTorch loop (CUDA graphs, static KV cache, 512 rolling answers) |
| `lag0/`, `lag8/` | the same learner with the lag limit at 0 (synchronous, on-policy) and at 8, for the blog's lag figure |

## headline

Made with the configuration that is now the binary's default, so `build/megarl --out results/run` repeats it. It
finished 250 of 250 updates in 25.6 s of kernel time with 0 audit violations and generated 1.73M tokens.

The repeat will not match it bit for bit. Engines and learner race each other, so which groups each update trains
on depends on timing. The arithmetic of a given update is deterministic, and a run with one engine that alternates
with the learner (`--nroll 1 --ppj 1 --lag 0 --alternate 1`) is reproducible bit for bit.

## lag0 and lag8

These two come from an earlier, smaller configuration than the headline: two tasks (math and ARC science, store
`data/prompts/math_science_train`), 8 groups per update, 2 rollout engines each bound to one task, learning rate
3e-5. They compare time per update with and without a lag limit: 618 ms at lag 0 against 57 ms at lag 8.

`make lagpair` builds the kernel for that configuration (`-DMRL_NTASKS=2 -DMRL_TGRP=8`) and runs both with the
original settings, into `results/run_lag0_2task` and `results/run_lag8_2task`.

## What was made with what

All files here were written by the kernel before this repository's cleanup, which removed unused options and
experiments. Their `run.json` files still carry keys for options that no longer exist (`scale`, `rank`,
`init_adapter`, `ksH`, `tksD`, `poll_relaxed`, `fence_all`). On a deterministic run of the headline build, the
cleaned kernel gives the same final weights, tokens and log-probabilities, bit for bit, as the one that made them.

To keep the repository small, the run directories leave out the per-group tokens and log-probabilities
(`group_tokens.i32`, `group_logp.f32`), the final layer (`mid_cur.bf16`) and the archive of trained versions
(`mid_arch.bf16`, 807 MB for the headline run). Scoring a run with `tools/eval_multitask.py` needs the archive, so
`headline_eval.json` can only be remade from a new run.
