import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interruptbench.envs import InterruptEnv

RANDOM_RESPONSES = [
    "Understood. I will now address this subtask as requested.",
    "Noted the update. Adjusting my approach accordingly and proceeding.",
    "Here is my response to the current subtask with all requirements in mind.",
    "Acknowledged. Incorporating the new constraint into this section.",
    "Per the revised instructions, I am now focusing on this component.",
]

def run_episode(env, verbose=True):
    obs, _ = env.reset()
    total_reward, step = 0.0, 0
    if verbose:
        print(f"\n{chr(61)*60}")
        print(f"TASK: {obs['task_description']}")
        print(chr(61)*60)
    while True:
        action = RANDOM_RESPONSES[step % len(RANDOM_RESPONSES)]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        if verbose:
            env.render()
        if terminated or truncated:
            break
    if verbose:
        print(f"\nEpisode done — steps: {step}, total reward: {total_reward:.3f}")
    return total_reward, step

if __name__ == "__main__":
    env = InterruptEnv(config={"difficulty": "medium", "interrupt_frequency": "high", "seed": 42})
    print("Running 3 random-policy episodes...")
    rewards = []
    for i in range(3):
        r, s = run_episode(env, verbose=(i == 0))
        rewards.append(r)
        print(f"Episode {i+1}: reward={r:.3f}, steps={s}")
    print(f"\nMean reward: {sum(rewards)/len(rewards):.3f}")
    print("Environment is working correctly")
