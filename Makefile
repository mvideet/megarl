NVCC   ?= nvcc
PYTHON ?= python3
ARCH   ?= -gencode arch=compute_90a,code=sm_90a
NVFLAGS = $(ARCH) -O3 -std=c++17 -Isrc

HEADERS = $(wildcard src/*.cuh)

.PHONY: all test weights headline lag0 lagpair demo blog clean

all: build/megarl

build/megarl: src/megarl.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -o $@ src/megarl.cu

build/test_sched: tests/test_sched.cu src/mk_sched.cuh
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -o $@ tests/test_sched.cu

test: build/test_sched
	build/test_sched

# Qwen2.5-0.5B-Instruct in the kernel's weight format (docs/FORMATS.md section 1).
weights:
	$(PYTHON) tools/export_qwen.py --out data/qwen05b

# The headline run: 250 GRPO updates with up to 8 versions of staleness (the binary's defaults).
headline: build/megarl
	build/megarl --out results/run --overwrite 1

# Synchronous on-policy loop: every update replays the rollout's log-probabilities bit for bit (max_dlogp == 0).
lag0: build/megarl
	build/megarl --out results/run_lag0 --overwrite 1 --lag 0 --target 4 --count-held 1 --updates 30

# The blog's lag-0 vs lag-8 comparison (results/lag0, results/lag8) used a smaller two-task build.
build/megarl2: src/megarl.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -DMRL_NTASKS=2 -DMRL_TGRP=8 -o $@ src/megarl.cu

lagpair: build/megarl2
	build/megarl2 --prompts data/prompts/math_science_train --nroll 2 --bind 1 --pair-mode 0 --lr 3e-5 \
	  --seed 1234 --prompt-seed 0 --overwrite 1 \
	  --run "--out results/run_lag0_2task --lag 0 --target 4 --count-held 1 --updates 20" \
	  --run "--out results/run_lag8_2task --lag 8 --updates 100"

demo:
	$(PYTHON) tools/demo_bundle.py --out demo/bundle.json
	$(PYTHON) tools/build_demo.py

blog:
	$(PYTHON) tools/blog_build.py

clean:
	rm -rf build
