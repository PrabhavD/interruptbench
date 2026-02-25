"""
scripts/diagnose_reward.py

Runs 100 episodes per difficulty tier with a random policy and prints
reward distribution statistics. Use this to verify the reward signal
has meaningful variance before starting RL training.

A good reward signal should have:
  - Mean well below maximum (headroom for improvement)
  - Non-zero std (variance for GRPO to exploit)
  - Interruption adherence NOT consistently 1.0 (would mean no interruptions firing)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from collections import defaultdict
from interruptbench.envs import InterruptEnv

RANDOM_RESPONSES = [
    "Understood. I will now address this subtask as requested.",
    "Noted the update. Adjusting my approach accordingly and proceeding.",
    "Here is my response to the current subtask with all requirements in mind.",
    "Acknowledged. Incorporating the new constraint into this section.",
    "Per the revised instructions, I am now focusing on this component.",
]

def run_diagnostic(difficulty, interrupt_difficulty, interrupt_frequency, n_episodes=100):
    env = InterruptEnv(config={
        "difficulty": difficulty,
        "interrupt_difficulty": interrupt_difficulty,
        "interrupt_frequency": interrupt_frequency,
        "seed": 42,
    })

    all_rewards = []
    component_totals = defaultdict(float)
    n_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done, step = False, 0
        while not done:
            action = RANDOM_RESPONSES[step % len(RANDOM_RESPONSES)]
            obs, reward, terminated, truncated, info = env.step(action)
            all_rewards.append(reward)
            for k, v in info["reward_components"].items():
                component_totals[k] += v
            n_steps += 1
            done = terminated or truncated
            step += 1

    mean_r = sum(all_rewards) / len(all_rewards)
    variance = sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)
    std_r = variance ** 0.5
    min_r = min(all_rewards)
    max_r = max(all_rewards)

    print(f"\n  difficulty={difficulty}, interrupt_difficulty={interrupt_difficulty}, "
          f"frequency={interrupt_frequency}")
    print(f"  Steps: {n_steps} across {n_episodes} episodes")
    print(f"  Reward — mean: {mean_r:.3f}  std: {std_r:.3f}  "
          f"min: {min_r:.3f}  max: {max_r:.3f}")
    print("  Component averages:")
    for k, v in component_totals.items():
        if k != "total":
            print(f"    {k:28s}: {v/n_steps:.3f}")

    return mean_r, std_r

if __name__ == "__main__":
    print("=" * 60)
    print("Reward Distribution Diagnostic")
    print("=" * 60)

    configs = [
        ("easy",   "easy",   "low"),
        ("medium", "medium", "medium"),
        ("hard",   "hard",   "high"),
        ("any",    "any",    "medium"),
    ]

    results = []
    for diff, int_diff, freq in configs:
        mean, std = run_diagnostic(diff, int_diff, freq)
        results.append((diff, mean, std))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for diff, mean, std in results:
        headroom = ((1.0 - mean) / 1.0) * 100
        print(f"  {diff:8s}  mean={mean:.3f}  std={std:.3f}  "
              f"headroom={headroom:.1f}%")

    print("\nHealthy signals: mean 0.4–0.8, std > 0.05, headroom > 20%")
