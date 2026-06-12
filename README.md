# InterruptBench

InterruptBench is a lightweight benchmark and training environment for testing how language-model agents handle interruptions during long-horizon work.

The core environment presents an agent with a multi-step task, then injects mid-trajectory interruptions such as clarifications, scope changes, priority shifts, or contradictions. The agent is rewarded for addressing the current subtask while explicitly acknowledging and incorporating the latest interruption.

## Features

* Gymnasium-compatible `InterruptEnv`
* Synthetic multi-step tasks across writing, engineering, research, and product domains
* Configurable interruption frequency and difficulty
* Structured reward components for subtask completion, interruption adherence, efficiency, and completion bonus
* Random rollout and reward-diagnostic scripts
* GRPO training script using TRL and Hugging Face Transformers
* Docker-based local workflow
* Constrained Docker sandbox wrapper for executing generated code safely

## Repository layout

```text
interruptbench/
  configs/
    default.yaml          # Default environment and training config
    env_config.py         # Typed environment config
  envs/
    interrupt_env.py      # Gymnasium environment
    interruption.py       # Interruption scheduler and templates
    reward.py             # Reward function
    task_generator.py     # Synthetic task templates

infra/
  sandbox.py              # DockerSandbox wrapper

scripts/
  random_rollout.py       # Run sample episodes with a simple policy
  diagnose_reward.py      # Inspect reward signal quality
  train_grpo.py           # GRPO training entrypoint
  test_sandbox.py         # Sandbox checks
  smoke_test_container.sh # End-to-end Docker smoke test

docs/
  colab.md                # Google Colab setup notes
  experiments.md          # Experiment log
```

## Requirements

For local development:

* Python 3.11+
* Docker and Docker Compose
* `make`, optional but recommended

Python dependencies are pinned in `requirements.txt`.

For GRPO training, a CUDA-capable GPU is recommended. The included Colab notes assume a T4 GPU.

## Quick start

Clone the repository:

```bash
git clone https://github.com/PrabhavD/interruptbench.git
cd interruptbench
```

Build the Docker image:

```bash
make build
```

Run a random-policy rollout:

```bash
make rollout
```

Run the container smoke test:

```bash
make smoke
```

You can also run Docker Compose commands directly:

```bash
docker compose build
docker compose run --rm interruptbench python scripts/random_rollout.py
```

## Local Python usage

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a short rollout:

```bash
python scripts/random_rollout.py
```

Use the environment directly:

```python
from interruptbench.envs import InterruptEnv

env = InterruptEnv(
    config={
        "difficulty": "medium",
        "domain": "any",
        "interrupt_frequency": "high",
        "interrupt_difficulty": "any",
        "seed": 42,
    }
)

obs, _ = env.reset()

print(env.full_prompt())

obs, reward, terminated, truncated, info = env.step(
    "Noted. I will address the current subtask while incorporating the latest update."
)

print(reward)
print(info["reward_components"])
```

## Configuration

The default configuration lives at:

```text
interruptbench/configs/default.yaml
```

Environment options include:

```yaml
env:
  difficulty: "any"              # easy | medium | hard | any
  domain: "any"                  # writing | engineering | research | product | any
  interrupt_frequency: "medium"  # low | medium | high
  interrupt_difficulty: "any"    # easy | medium | hard | any
  max_steps: 20
  seed: 42
```

The training section controls model name, batch size, learning rate, generations, epochs, logging, and checkpointing.

## Reward model

Each environment step returns a scalar reward plus named reward components:

* `subtask`: whether the response engages with the current subtask
* `interruption_adherence`: whether the response acknowledges and incorporates the active interruption
* `efficiency`: a step-based efficiency score
* `completion_bonus`: bonus for reaching the final subtask
* `total`: weighted aggregate reward

The current reward implementation uses heuristic keyword-based scoring. It is intended as a simple baseline and can be replaced with a learned evaluator or LLM judge while preserving the same reward-function interface.

## Reward diagnostics

Before training, run the diagnostic script to inspect whether the reward signal has useful variance:

```bash
python scripts/diagnose_reward.py
```

This runs episodes across difficulty settings and prints reward distribution statistics.

## GRPO training

Run a dry run first:

```bash
python scripts/train_grpo.py --dry-run --episodes 10
```

Start training with the default config:

```bash
python scripts/train_grpo.py \
  --config interruptbench/configs/default.yaml \
  --output-dir checkpoints/grpo-qwen
```

To enable Weights & Biases logging, set `WANDB_PROJECT` before training:

```bash
export WANDB_PROJECT=interruptbench
python scripts/train_grpo.py
```

Final checkpoints are saved under the configured output directory.

## Running on Google Colab

See:

```text
docs/colab.md
```

The Colab workflow covers GPU verification, cloning the repo, installing pinned TRL/Transformers versions, and running a dry training check.

## Docker sandbox

InterruptBench includes a small Docker-based sandbox wrapper in `infra/sandbox.py`. It runs generated Python code with constrained defaults:

* no network access
* CPU and memory limits
* process limit
* read-only root filesystem
* temporary writable `/tmp`

Build the base image first:

```bash
make build
```

Then run the sandbox tests:

```bash
python scripts/test_sandbox.py
```

## Development commands

```bash
make build        # Build Docker image
make up           # Start service
make down         # Stop service
make shell        # Open a shell in the container
make rollout      # Run random rollout
make test-sandbox # Run sandbox tests
make smoke        # Run end-to-end smoke test
make logs         # Follow Docker logs
make clean        # Remove container resources and image
```

## Project status - In Active Development

InterruptBench is an experimental benchmark/training scaffold. The current implementation is useful for prototyping interruption-aware agent training, reward diagnostics, and GRPO experiments, but the default reward function and task templates should be treated as baseline components rather than production-grade evaluators.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
