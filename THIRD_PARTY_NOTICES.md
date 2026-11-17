# Third-party notices

## open-flapping-airplanes

https://github.com/Bobliuuu/open-flapping-airplanes

`src/mk_sched.cuh` follows the job/step/phase task model of open-flapping-airplanes (`llm/megakernel/ofa_llm.cu`),
and `src/lgemm.cuh` is derived from its `llm/gemm/gemm.cuh`. `blog/assets/ofa_dashboard.png` is a screenshot from
its site.

```
MIT License

Copyright (c) 2026 Jerry Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets in `data/prompts/`

The prompt stores contain questions derived from these datasets, under their own licences:

- ARC (AI2 Reasoning Challenge), Clark et al. 2018: CC BY-SA 4.0.
- SciQ, Welbl et al. 2017: CC BY-NC 3.0 (non-commercial).
- CommonsenseQA, Talmor et al. 2019: MIT.

The arithmetic questions are generated.

## Model

Qwen2.5-0.5B-Instruct (Qwen team, Alibaba Cloud) is downloaded by `tools/export_qwen.py` and is not included. It is
released under the Apache 2.0 licence.
