from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interruptbench.envs import InterruptEnv
from interruptbench.envs.reward import compute_reward


SYSTEM_PROMPT = """You are an agent operating in InterruptBench.
At each step, respond to the CURRENT SUBTASK.
If there is an INTERRUPTION, explicitly acknowledge it and incorporate it.
Be concise, action-oriented, and faithful to the latest instruction state.
"""


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_prompt(full_prompt_text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def scripted_policy(obs: dict[str, Any]) -> str:
    subtask = obs["current_subtask"]
    interruption = obs.get("interruption")

    parts = []
    if interruption:
        parts.append(f"Noted the update: {interruption}")
    parts.append(f"I will now address this step: {subtask}.")
    parts.append("I will keep the response aligned with the latest constraints.")
    return " ".join(parts)


def generate_dataset(
    env_config: dict[str, Any],
    tokenizer,
    n_episodes: int,
    seed: int,
) -> Dataset:
    rows = []
    env = InterruptEnv(config=env_config)

    for episode_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        done = False

        while not done:
            prompt = build_prompt(env.full_prompt(), tokenizer)

            rows.append(
                {
                    "prompt": prompt,
                    "current_subtask": obs["current_subtask"],
                    "active_interruption": obs.get("interruption"),
                    "step": int(obs["subtask_index"]),
                    "n_subtasks": len(env._task.subtasks),
                    "task_description": obs["task_description"],
                }
            )

            action = scripted_policy(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return Dataset.from_list(rows)


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        if "text" in completion:
            return str(completion["text"])

    return str(completion)


def make_reward_fn():
    def reward_fn(
        prompts,
        completions,
        current_subtask,
        active_interruption,
        step,
        n_subtasks,
        **kwargs,
    ):
        rewards = []

        for completion, subtask, interruption, step_i, n_subtasks_i in zip(
            completions,
            current_subtask,
            active_interruption,
            step,
            n_subtasks,
        ):
            response = _completion_to_text(completion)

            interruption_obj = (
                SimpleNamespace(message=interruption)
                if interruption is not None
                else None
            )

            reward = compute_reward(
                response=response,
                current_subtask=subtask,
                active_interruption=interruption_obj,
                step=int(step_i),
                n_subtasks=int(n_subtasks_i),
                interruptions_acknowledged=0,
                n_interruptions=0,
                done=int(step_i) >= int(n_subtasks_i) - 1,
            )
            rewards.append(float(reward["total"]))

        return rewards

    return reward_fn


def build_training_args(cfg: dict[str, Any], output_dir: str) -> GRPOConfig:
    training = cfg["training"]

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_ok = torch.cuda.is_available() and not bf16_ok
    report_to = ["wandb"] if os.getenv("WANDB_PROJECT") else []

    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=training.get("learning_rate", 1e-5),
        per_device_train_batch_size=training.get("batch_size", 4),
        gradient_accumulation_steps=training.get("grad_accum", 4),
        num_train_epochs=training.get("epochs", 3),
        num_generations=training.get("num_generations", 4),
        max_completion_length=training.get("max_completion_length", 128),
        logging_steps=training.get("logging_steps", 1),
        save_steps=training.get("save_steps", 50),
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=report_to,
        bf16=bf16_ok,
        fp16=fp16_ok,
        gradient_checkpointing=True,
        log_completions=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="interruptbench/configs/default.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/grpo-qwen",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build tokenizer, dataset, reward fn, and trainer config without training.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override training.n_episodes from YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["training"]["model"]
    tokenizer = make_tokenizer(model_name)

    n_episodes = args.episodes or cfg["training"].get("n_episodes", 200)
    dataset = generate_dataset(
        env_config=cfg["env"],
        tokenizer=tokenizer,
        n_episodes=n_episodes,
        seed=cfg["env"].get("seed", 42),
    )

    reward_fn = make_reward_fn()
    training_args = build_training_args(cfg, args.output_dir)

    print(f"Model: {model_name}")
    print(f"Dataset rows: {len(dataset)}")
    print(f"Sample columns: {dataset.column_names}")
    print()
    print("Sample prompt:")
    print(dataset[0]["prompt"][:1000])
    print()

    if args.dry_run:
        dummy_rewards = reward_fn(
            prompts=[dataset[0]["prompt"]],
            completions=["Noted the update. I will address the current subtask directly."],
            current_subtask=[dataset[0]["current_subtask"]],
            active_interruption=[dataset[0]["active_interruption"]],
            step=[dataset[0]["step"]],
            n_subtasks=[dataset[0]["n_subtasks"]],
        )
        print(f"Dry run reward sample: {dummy_rewards[0]:.3f}")
        print("Dry run succeeded.")
        return

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final model to: {final_dir}")


if __name__ == "__main__":
    main()