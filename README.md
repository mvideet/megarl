# MegaRL

Online reinforcement learning for a language model in a single CUDA kernel.

One launch on an H100 runs four generation engines and a learner side by side. The engines sample answers from
Qwen2.5-0.5B-Instruct, grade them and queue them; the learner takes batches from the queues, runs a GRPO update on
layer 13 and publishes the new weights, which the engines start using without ever stopping. Nothing goes back to
the CPU until the run is over.

The story behind it, with figures and a replay of the race against PyTorch, is in [`blog/index.html`](blog/index.html)
(one self-contained page: download it and open it in a browser).

## Results

Qwen2.5-0.5B-Instruct on one H100 80GB, all of layer 13 trained (14.9M parameters), four multiple-choice tasks
(arithmetic, ARC, SciQ, CommonsenseQA), 16 groups of 4 answers per update, answers of at most 64 tokens. The three
systems use the same prompts, the same GRPO update (learning rate 4e-5, gradient clip 1.0, lag limit 8) and the same
grader.

| | tokens/s | updates | groups trained/s | kernel launches |
|---|---:|---:|---:|---:|
| MegaRL | 67,700 | 250 in 25.6 s | 156 | 1 |
| PyTorch, hand-tuned (CUDA graphs, static KV cache) | 40,100 | 179 in 26.5 s | 108 | 253,525 |
| PyTorch, HF transformers | 6,100 | 54 in 26.6 s | 32 | 1,327,001 |

Held-out greedy accuracy on 1,600 questions that are never trained on goes from 58.6% to 74.8% (mean over the four
tasks) over the 250 updates. All three systems end at the same accuracy within noise; the difference is how long
they take to get there.

The numbers come from `results/` (see [`results/README.md`](results/README.md)).

## How it works

- **One kernel, many jobs.** The kernel runs 2 blocks of 256 threads on each of the 132 SMs. The work is five jobs
  (four rollout engines and the learner), each an endless series of steps; a step is a fixed list of phases, and a
  phase is a number of independent tasks. Every block loops over the jobs and takes the next task whose
  dependencies are complete. Dependencies are completion counters in global memory, so there is no grid-wide
  barrier and no host involvement (`src/mk_sched.cuh`).
- **Rollout.** Prompts always go through the original weights, so each question's KV cache is built once at
  start-up and reused by every version of the model. During generation the trained layer runs once per cohort of
  rows that share a weight snapshot, and the engines save the hidden state entering layer 13. The head samples with
  Gumbel-max noise; a finished group of four answers is scored, turned into group-relative advantages and queued.
  Groups whose four rewards are equal carry no signal and are dropped.
- **Learner.** It replays layers 13 to 23 from the saved hidden states with exactly the rollout's arithmetic, so at
  lag 0 its log-probabilities equal the generation-time ones bit for bit. It computes the clipped GRPO loss,
  backpropagates through the frozen top layers into layer 13, clips the global gradient norm and applies AdamW with
  fp32 master weights. Every reduction has a fixed order, so the same groups always give the same update.
- **Stale answers are allowed.** Every group pins the snapshot that generated it. The learner accepts a group at
  most 8 versions old, and the clipped importance ratio corrects for the difference. A snapshot is not overwritten
  while any group still holds it. This is what lets the engines keep generating while the learner trains.
- **Checked, not assumed.** Groups move between engines and learner through lock-free queues. At the end of every
  run the host audits the trace and the counters: each group ends exactly once, the learner takes each task's groups
  first-in first-out, no trained group exceeds the lag limit, buffers are conserved. A run prints its violation count, which should be 0.

## Requirements

- An NVIDIA H100 (sm_90a: the GEMM tiles use `wgmma`). Built and tested with CUDA 13.2.
- Python 3 with `torch`, `transformers`, `numpy` and `huggingface_hub` for the tools (tested with torch 2.10 and
  transformers 5.12). The kernel itself needs none of them.

## Running it

```
make                          # build/megarl
python tools/export_qwen.py   # download Qwen2.5-0.5B-Instruct and write data/qwen05b
build/megarl --out results/run
```

With no options the binary runs the headline configuration: 250 updates on `data/prompts/multitask4_train`, about
30 seconds on an H100. `build/megarl --help` lists every option. The run directory it writes is described in
[`docs/FORMATS.md`](docs/FORMATS.md).

Checks:

```
make test    # scheduler stress test: random job graphs, checks ordering and stale reads
make lag0    # synchronous on-policy run: every update must have max_dlogp == 0 (exact log-probability replay)
```

## Layout

| path | contents |
|---|---|
| `src/megarl.cu` | the kernel, the job graphs and the host driver |
| `src/mk_sched.cuh` | the task scheduler (jobs, steps, phases, dependency counters) |
| `src/qwen_fwd.cuh` | the frozen Qwen layers of the rollout and the prompt-cache build |
| `src/mid_train.cuh` | the trained layer: its rollout GEMMs and the whole learner |
| `src/head_sample.cuh`, `src/mrl_lmhead.cuh` | the LM head, sampling, and the arithmetic the head and learner share |
| `src/grpo_loss.cuh`, `src/group_reduce.cuh`, `src/reward.cuh` | GRPO loss, group advantages, reward |
| `src/rl_control.cuh`, `src/rl_state.cuh`, `src/rl_queue.cuh`, `src/policy_versions.cuh`, `src/trace.cuh` | the RL core: admission, selection, publication, queues, snapshots, trace and audit |
| `src/lgemm.cuh`, `src/mrl_wgmma.cuh` | the bf16 GEMM tile |
| `src/mrl_common.cuh`, `src/mrl_host.cuh`, `src/megarl_opts.cuh` | shared constants and layouts, host I/O, options and `run.json` |
| `tests/test_sched.cu` | scheduler stress test |
| `tools/` | weight export, evaluation, the PyTorch baseline, readers, and the demo and blog builders |
| `data/prompts/` | the prompt stores (see `data/README.md`) |
| `results/` | the runs the blog is built from |
| `demo/`, `blog/` | the race replay and the blog post |

## Credits

The design started from Flapping Airplanes' demo of one GPU kernel that trains a model, serves it and plays Doom at
the same time, and from its open rebuild, [open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes)
by Jerry Zhu and Arnold Xu: MegaRL's scheduler follows its job and phase model, and the GEMM tile is derived from
its code (MIT; see `THIRD_PARTY_NOTICES.md`). Thanks also to the Qwen team for Qwen2.5-0.5B-Instruct, the DeepSeekMath
authors for GRPO, Hugging Face for transformers and TRL (whose GRPO loss ours was checked against), and the authors of
ARC, SciQ and CommonsenseQA.

## License

MIT, see `LICENSE`. Third-party code, data and model licences are listed in `THIRD_PARTY_NOTICES.md`.
