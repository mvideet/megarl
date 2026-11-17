# Data

## Prompt stores

| store | prompts | used for |
|---|---:|---|
| `prompts/multitask4_train` | 3,584 (896 per task) | training |
| `prompts/multitask4_dev_twin` | 1,600 (400 per task) | held-out evaluation |
| `prompts/math_science_train` | 1,792 (896 per task, `math` and `science` only) | the two-task lag comparison in `results/lag0`, `results/lag8` |

Every prompt is a four-option multiple-choice question in Qwen's chat format, followed by the instruction
"Explain briefly (at most 20 words). Then write 'Answer: ' followed by the letter of the correct option (A, B, C or
D)." Within each task the correct letter is balanced over A to D. Prompts are at most 120 tokens.

| task | source |
|---|---|
| `math` | generated: addition, subtraction, multiplication, linear equations and modular arithmetic |
| `science` | ARC-Easy and ARC-Challenge |
| `sciq` | SciQ |
| `csqa` | CommonsenseQA |

`math_science_train` is exactly the math and science half of `multitask4_train`. No training question appears in the
held-out store. In the training stores the wrong math options are near misses of the answer; in
the held-out store they are drawn at random, so a model cannot find the answer from the pattern of the wrong
options. That is what "twin" in the name refers to.

`prompts.jsonl` in each store has one line per prompt: question, options, correct key, the full prompt text, and a
`source` field naming where the question came from (for ARC, the split and the question id). The binary files next
to it are the tokenized prompts and per-token lookup tables the kernel reads; their layout and the reward rule are
in [`../docs/FORMATS.md`](../docs/FORMATS.md). `tools/stores.py` reads and checks a store.

`tools/build_multitask4.py` is the script that assembled the four-task stores. It is kept as a record of how they
were made rather than as something to run: its inputs (earlier two-task stores and extracts of SciQ and
CommonsenseQA) are not part of this repository.

Licences: ARC is CC BY-SA 4.0, SciQ is CC BY-NC 3.0, CommonsenseQA is MIT. The SciQ questions make these stores
non-commercial as a whole.

## Model weights

`qwen05b/` is not checked in. `python tools/export_qwen.py` downloads Qwen2.5-0.5B-Instruct (Apache 2.0) from Hugging
Face and writes it here in the layout the kernel reads (bf16 tensors plus RoPE tables, see `docs/FORMATS.md`).
