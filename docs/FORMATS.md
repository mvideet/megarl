# File formats

The files the Python tools and `build/megarl` exchange. Binary files are little-endian, row-major arrays of packed
records. `bf16` is the raw uint16 of a float rounded to bfloat16 with round-to-nearest-even (what
`tensor.to(torch.bfloat16)` gives).

Model: `Qwen/Qwen2.5-0.5B-Instruct`. Hidden size 896, 24 layers, 14 query heads, 2 key/value heads, head dim 64,
MLP width 4864, vocabulary 151936, RMSNorm eps 1e-6, RoPE theta 1e6, tied embedding and LM head, biases on the
q/k/v projections only.

## 1. Weights: `data/qwen05b/`

Written by `tools/export_qwen.py`. `weights.bin` is the concatenation of the tensors below; `weights.idx` has one
line per tensor:

```
name dtype rows cols offset_bytes nbytes
```

`dtype` is `bf16` or `f32`. Every tensor starts at a multiple of 64 bytes.

| name | dtype | shape | content |
|---|---|---|---|
| `emb` | bf16 | [151936, 896] | `model.embed_tokens.weight` (also the LM head) |
| `L{l}.qkv` | bf16 | [1152, 928] | rows 0..895 `q_proj`, 896..1023 `k_proj`, 1024..1151 `v_proj`; columns 0..895 the weight, column 896 the bias, 897..927 zero |
| `L{l}.o` | bf16 | [896, 896] | `o_proj.weight` |
| `L{l}.gu` | bf16 | [9728, 896] | rows 0..4863 `gate_proj.weight`, 4864..9727 `up_proj.weight` |
| `L{l}.down` | bf16 | [896, 4864] | `down_proj.weight` |
| `L{l}.ln1` | f32 | [1, 896] | `input_layernorm.weight` |
| `L{l}.ln2` | f32 | [1, 896] | `post_attention_layernorm.weight` |
| `lnf` | f32 | [1, 896] | final `norm.weight` |
| `rope_cos` | f32 | [1024, 32] | cos(pos * inv_freq[i]), from the model's own rotary embedding in float32 on CUDA |
| `rope_sin` | f32 | [1024, 32] | sin(pos * inv_freq[i]) |

`l` runs 0..23. Weight matrices keep PyTorch's `[out_features, in_features]` layout.

`config.txt` holds `key value` lines: `hidden 896`, `layers 24`, `heads 14`, `kv_heads 2`, `head_dim 64`,
`inter 4864`, `vocab 151936`, `rms_eps 1e-6`, `rope_theta 1000000`, `eos_im_end 151645`, `eos_endoftext 151643`,
`max_pos 1024`.

## 2. Prompt store: `data/prompts/<set>/`

| file | content |
|---|---|
| `prompts.idx` | first line `n_prompts max_len verifier_mode`, then one line per prompt: `id task key len offset` (`task` indexes `envs.json`; `key` 0..3 is the correct option A..D; `offset` is in tokens into `prompts.i32`) |
| `prompts.i32` | the prompts' token ids, concatenated. Each prompt is rendered with the chat template and ends with the assistant generation prefix; at most 128 tokens |
| `tok_letter.i8` | [151936] int8: 0..3 if the token is the answer letter A..D, else -1 |
| `tok_colon.u8` | [151936] uint8: 1 if the token is a colon |
| `tok_answer.u8` | [151936] uint8: 1 if the token is the word "Answer" (any case, optional leading space) |
| `envs.json` | `{"names": [...]}`, the environment names in task order |
| `prompts.jsonl` | one row per prompt: `id`, `task`, `task_name`, `question`, `options`, `key`, `key_letter`, `prompt` (the rendered text), `kind`, `source`, `len`. Not read by the kernel |
| `build_meta.json` | sizes and task names of the store. Not read by the kernel |

The store digest (`store_sha256` in `src/megarl_opts.cuh`, `Store.sha256()` in `tools/stores.py`) is one SHA-256
over, for each of `prompts.idx`, `prompts.i32`, `tok_letter.i8`, `tok_colon.u8`, `tok_answer.u8`, `prompts.jsonl`
in this order, the file name followed by the file's bytes; a missing file contributes its name only. `run.json`
records it as `config.prompt_set_sha256`.

Reward of one answer `y[0..n-1]` (the generated tokens, end token included):

- `verifier_mode` 0 ("Answer: X"): take the largest `i` with `2 <= i < n`, `tok_letter[y[i]] >= 0`,
  `tok_colon[y[i-1]] == 1` and `tok_answer[y[i-2]] == 1`. The reward is 1 if such an `i` exists and
  `tok_letter[y[i]] == key`, else 0.
- `verifier_mode` 1 (first letter): the reward is 1 if `n >= 1` and `tok_letter[y[0]] == key`, else 0.

With `--require-eos 1` an answer that does not end with `<|im_end|>` or `<|endoftext|>` gets reward 0.

## 3. Run directory: `results/<run>/`

Written by `build/megarl --out DIR`. Times are nanoseconds of the device clock since the kernel started.
`console.log` is opened first and `run.json` written last: a directory with `console.log` and no `run.json` holds
a run that did not finish.

| file | content |
|---|---|
| `run.json` | configuration, command line, counters, sizes and timing (below) |
| `console.log` | everything the run printed, in order |
| `steps.bin` | one record per update |
| `groups.bin` | one record per group that reached a terminal state, in the order they were logged |
| `group_tokens.i32`, `group_logp.f32` | [groups_with_tokens, 4, 64]: generated tokens and generation-time log-probabilities of the first `groups_with_tokens` records of `groups.bin`; entries at `t >= len` are unspecified |
| `trace.bin` | event trace, not sorted by time |
| `mid_cur.bf16` | [14946304] the trained layer after the last update |
| `mid_arch.bf16` | [narch, 14946304] archived versions of the trained layer |
| `mid_arch_ids.i32` | [narch] the version id held by each archive slot (-1: empty) |
| `mid_layer.json` | `mid_layer`, `mid_nw`, `arch_stride`, `narch`, `from_last` (1: the layer acts from the last prompt position on) |

A snapshot of the trained layer (`mid_cur.bf16`, one row of `mid_arch.bf16`) is `W_qkv` [1152, 928], `W_o`
[896, 896], `W_gu` [9728, 896] and `W_down` [896, 4864], concatenated in this order, in the layout of section 1.
Version 0 is the base model. Every version whose id is a multiple of `arch_stride` is archived, plus the last one
in the last slot.

`run.json`:

- `config`: the hyper-parameters (`mrl::Hyper` in `src/mrl_common.cuh`) and the run options (`nroll`, `ppj`,
  `ngrp`, `layout`, `train_sms`, `updates`, `bind`, `use_per_task`, `prompt_seed`, `budget_s`, `count_held`,
  `alternate`, `no_train`), the derived sizes (`npipe`, `pipe_task`, `tgrp`, `gsz`, `maxgen`, `nbuf`, `nslots`),
  `weights`, `prompt_set` and `prompt_set_sha256`;
- `argv`, `run_argv` (the words of this run's `--run` option), `process` (`id`, `runs`, `index`), `build` (`tgrp`,
  `ntasks`, `mid_layer`, `compiled`);
- `prompt_ids`: prompt index of the logs to the `id` column of `prompts.idx`; `prompt_order`: per environment, the
  prompt indices in hand-out order (the n-th group admitted for an environment gets entry `n mod length`);
- `counters`: per environment `admitted`, `finished`, `drop_zerovar`, `drop_stale_inflight`, `drop_stale_queued`,
  `consumed`, `inflight`, `queued`; then `held`, `errors` (16 entries, all 0 in a correct run), `stats`,
  `uid_next`, `tokens_generated`, `stop_reason`, `stop_reason_dev`, `train_ver` and `audit_violations` (violated
  checks of the end-of-run audit, 0 in a correct run);
- `sizes`: `groups`, `groups_total`, `groups_with_tokens`, `trace`, `trace_total`, `steps`, `versions`;
- `timing`: `kernel_ns` (device time of the last update), `exit_ns` (when the kernel stopped), `wall_s`, the host
  set-up times (`pre_main_s`, `pre_load_s`, `load_s`, `prompt_cache_s`, `prep_s`, `setup_s`), the block counts and
  per job (`jobs[]`: the engines, then the learner) `blocks`, `steps`, `busy_ns` and `phases` as
  `[name, kind, tasks per step, ns inside its tasks]`.

64-bit integers are written in full; read them as integers, not doubles.

Stop reasons: 1 all updates done, 2 deadline, 3 learner starved (it took or dropped no group for `--starve-ms`),
4 internal error.

`steps.bin`, 88 bytes per update:

```
[('t_done','<u8'), ('loss','<f4'), ('gnorm','<f4'), ('reserved','<f4'), ('max_dlogp','<f4'), ('info','<i4',16)]
```

`gnorm` is the gradient norm before clipping; `max_dlogp` the largest `|logp_new - logp_behaviour|` over the batch
(0 when every group has lag 0). `info`: tokens, tokens without gradient (clipped or non-finite ratio), version id
after the update, packed tiles, then the uids of the first 12 groups of the batch (-1: none).

`groups.bin`, 104 bytes per group:

```
[('uid','<i4'), ('task','<i4'), ('prompt','<i4'), ('key','<i4'), ('ver_id','<i4'), ('terminal','<i4'),
 ('train_step','<i4'), ('lag','<i4'), ('len','<i4',4), ('reward','<f4',4), ('adv','<f4',4),
 ('t_admit','<u8'), ('t_done','<u8'), ('t_end','<u8')]
```

`terminal`: 5 trained on, 6 dropped because the four rewards were equal, 7 dropped as too old when the learner took
it from the queue, 8 cancelled as too old while generating. `train_step` and `lag` are set for terminal 5 only
(else -1). `ver_id` is the policy version the group was generated with. `len[j]` counts generated tokens including
the end token. `t_done` is when the group was scored (0 if cancelled), `t_end` when it reached its terminal state.
Groups still queued when the kernel stopped have no record.

`trace.bin`, 32 bytes per event: `[('t','<u8'), ('type','<i4'), ('a','<i4'), ('b','<i4'), ('c','<i4'), ('d','<i4'), ('e','<i4')]`

| type | event | a | b | c | d | e |
|---|---|---|---|---|---|---|
| 1 | ADMIT | uid | task | prompt | ver_id | pipeline * 256 + slot |
| 2 | GROUP_DONE | uid | task | answers with reward 1 | longest answer | pipeline * 256 + slot |
| 3 | ENQUEUE | uid | task | buffer | queue length after the push | ver_id |
| 4 | DROP_ZEROVAR | uid | task | buffer | answers with reward 1 | ver_id |
| 5 | CANCEL | uid | task | buffer | `train_ver` seen | ver_id |
| 6 | DROP_STALE | uid | task | buffer | `train_ver` | ver_id |
| 7 | SELECT | update | uid | task | lag | place in the batch |
| 8 | PUBLISH | version id | snapshot slot | update | polls spent waiting for a free slot | 0 |
| 9 | STEP | update | tokens | tokens without gradient | packed tiles | 0 |
| 10 | IDLE | rollout job | 1 began waiting, 0 resumed | 0 | 0 | 0 |
| 11 | SELECT_WAIT | update | us until the batch was complete | groups dropped as stale meanwhile | 1 if the fallback composition was used | 0 |
| 12 | STOP | stop reason | 0 | 0 | 0 | 0 |

PUBLISH is traced after the new snapshot is in place and ADMIT after the group has pinned its snapshot, so a group
admitted after the PUBLISH of version n carries a version of at least n.
