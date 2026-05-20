# Experiment Log

## Run 001 — First GRPO Training Run

**Date:** 2026-05-20
**Branch:** `feature/first-training-run`
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Hardware:** Colab T4 (15.8 GB VRAM)

### Config

| Parameter | Value |
|---|---|
| `learning_rate` | `1e-5` |
| `batch_size` | `2` |
| `grad_accum` | `4` |
| `epochs` | `1` |
| `num_generations` | `4` |
| `max_completion_length` | `256` |
| `load_in_4bit` | `True` |

### Reward Curve

| Step | Reward |
|------|--------|
| 1 | 0.6025 |
| 2 | 0.5588 |
| 3 | 0.5588 |
| 4 | 0.6102 |
| 5 | 0.6750 |
| 6 | 0.7005 |

### Observations

- Reward trends upward from ~0.56 to ~0.70 across 6 steps ✓
- Brief dip at steps 2–3 is expected early-training noise — GRPO rollouts are noisy in the first few steps
  because the policy hasn't yet differentiated good from bad completions within each generation group
- Recovery from step 3 onwards is clean with no oscillation
- No OOM errors with 4-bit quantisation on T4
- Checkpoint saved to Drive: `interruptbench/checkpoints/grpo-qwen`

### Failure modes / things to watch

- Run was short (6 steps) — unclear if reward will plateau or continue rising
- `num_generations=4` is the minimum for GRPO to compute a meaningful advantage estimate;
  below 4 the variance in group rewards is too high to get a reliable gradient signal
- LR of `1e-5` may be too aggressive for a 0.5B model — if run-002 shows oscillation, drop it

### Next
→ Run 002: tune LR to 5e-6, run 2 epochs, compare reward curve