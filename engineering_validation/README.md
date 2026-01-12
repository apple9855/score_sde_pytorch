# Engineering Validation Pipeline

This folder contains a **minimal, non-invasive engineering validation pipeline**
for the original `score_sde_pytorch` repository.

This document is intended for readers who want to understand the engineering execution and reproducibility of the original implementation, rather than the theoretical derivation of score-based SDEs.

The goal is **not** to retrain models or reimplement algorithms,
but to verify that:

- the original codebase can be executed end-to-end
- pretrained checkpoints can be loaded correctly
- sampling + evaluation (IS / FID / KID) runs successfully
- results are reproducible in a clean GPU environment


---

## Scope and Principles

- ✅ Use **official pretrained checkpoints**
- ✅ Do **not modify core training / sampling logic**
- ✅ Keep validation **lightweight** (CIFAR-10 only)
- ❌ No retraining
- ❌ No architectural changes

This validation is designed as an **engineering sanity check**, not a benchmark.

---

## Environment

- GPU: NVIDIA RTX 4090 (RunPod)
- OS: Ubuntu (containerized)
- Python: 3.11
- Frameworks:
  - PyTorch (sampling)
  - TensorFlow + TF-GAN (evaluation metrics)
  - JAX (dataset / evaluation utilities, CPU-only)

---

## Pipeline Overview

The validation pipeline consists of **four steps**:

```text
01_bootstrap.sh       # Environment preparation
02_download_ckpt.sh   # Download checkpoints + dataset stats
03_run_sampling.sh    # Run sampling + evaluation
04_make_report.py     # Collect results and generate figures
```
Each step is isolated and executable independently.

---

## Step-by-Step Execution

### Step 1 — Environment Bootstrap

```bash
bash engineering_validation/01_bootstrap.sh
```

Purpose:
- Prepare a clean Python environment
- Install all required dependencies
- Enforce TensorFlow CPU-only execution (to avoid PyTorch/TensorFlow GPU conflicts)

Expected outcome:
- PyTorch detects CUDA correctly
- TensorFlow imports successfully on CPU
- No missing dependency errors

---

### Step 2 — Download Checkpoints and Dataset Statistics
```bash
bash engineering_validation/02_download_ckpt.sh
```
Purpose:
- Download official pretrained checkpoints:
    - VE-SDE (NCSN++) — CIFAR-10
	- VP-SDE (DDPM++) — CIFAR-10
- Download CIFAR-10 dataset statistics used for FID / KID computation

Expected outcome:
- Checkpoints available under:
    - exp/ve/cifar10_ncsnpp_continuous/checkpoints/
	- exp/vp/cifar10_ddpmpp_continuous/checkpoints/
- Dataset statistics present at:
    - assets/stats/cifar10_stats.npz
 
---

### Step 3 - Sampling and Evaluation
```bash
bash engineering_validation/03_run_sampling.sh
```
Execution characteristics:
- Sampling: PyTorch + CUDA (GPU)
- Evaluation: TensorFlow / TFGAN (CPU)
- JAX: CPU-only utilities for dataset handling

Artifacts generated in this step are stored under:
- exp/*/eval/
- engineering_validation/logs/
- engineering_validation/artifacts/

---

### Step 4 - Result Aggregation
```python
python engineering_validation/04_make_report.py
```
Expected outcome:
- engineering_validation/results/ containing:
- REPORT.md
- summary.csv
- sample visualization images
- environment snapshot

Runtime artifacts under `engineering_validation/results/` and `logs/` are excluded from version control by design.
The repository focuses on reproducible execution, not storing one-off outputs.

---

## Notes on Evaluation Metrics

- Inception Score (IS) and FID are reported as reference indicators.
- KID may appear as NaN under this mini-evaluation setting due to limited sample count.
- The focus of this validation is pipeline correctness, not metric optimization.

---

## What This Enables Next

This validation serves as a stable baseline for:
- sampler comparison
- solver ablation
- controlled modifications without altering the original codebase