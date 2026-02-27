from __future__ import annotations
import gymnasium as gym
from typing import Any, Dict, Optional

from interruptbench.configs.env_config import EnvConfig
from .task_generator import Task, TaskGenerator
from .interruption import Interruption, InterruptionScheduler
from .reward import compute_reward

class InterruptEnv(gym.Env):
    metadata = {"render_modes": ["text"]}

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        # Accept either a dict (backwards compatible) or EnvConfig dataclass (preferred)
        if isinstance(config, dict):
            cfg = EnvConfig.from_dict(config)
        elif isinstance(config, EnvConfig):
            cfg = config
        else:
            cfg = EnvConfig()  # Use defaults
        
        self.task_generator = TaskGenerator(
            difficulty=cfg.difficulty,
            domain=cfg.domain,
            seed=cfg.seed,
        )
        self.interrupt_scheduler = InterruptionScheduler(
            frequency=cfg.interrupt_frequency,
            difficulty=cfg.interrupt_difficulty,
            seed=cfg.seed,
        )
        self.max_steps = cfg.max_steps
        self.cfg = cfg
        self.action_space = gym.spaces.Text(min_length=1, max_length=2048)
        self.observation_space = gym.spaces.Dict({
            "subtask_index": gym.spaces.Discrete(20),
        })
        self._task = None
        self._interruptions = []
        self._step = 0
        self._subtask_idx = 0
        self._history = []
        self._interruptions_acknowledged = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._task = self.task_generator.sample()
        self._interruptions = self.interrupt_scheduler.schedule(len(self._task.subtasks))
        self._step = 0
        self._subtask_idx = 0
        self._history = []
        self._interruptions_acknowledged = 0
        return self._obs(), {}

    def step(self, action: str):
        assert self._task is not None, "Call reset() before step()"
        active_interrupt = self._get_active_interruption()
        current_subtask = self._task.subtasks[self._subtask_idx]
        reward_components = compute_reward(
            response=action,
            current_subtask=current_subtask,
            active_interruption=active_interrupt,
            step=self._step,
            n_subtasks=len(self._task.subtasks),
            interruptions_acknowledged=self._interruptions_acknowledged,
            n_interruptions=len(self._interruptions),
            done=self._subtask_idx >= len(self._task.subtasks) - 1,
        )
        if active_interrupt and reward_components["interruption_adherence"] > 0.5:
            self._interruptions_acknowledged += 1
        self._history.append({
            "subtask": current_subtask,
            "interruption": active_interrupt.message if active_interrupt else None,
            "response": action,
            "reward": reward_components,
        })
        self._subtask_idx += 1
        self._step += 1
        terminated = self._subtask_idx >= len(self._task.subtasks)
        truncated = self._step >= self.max_steps
        return self._obs(), reward_components["total"], terminated, truncated,                {"reward_components": reward_components}

    def render(self):
        if not self._history:
            print(f"Task: {self._task.title}")
            return
        last = self._history[-1]
        print(f"[Step {self._step}] Subtask: {last['subtask']}")
        if last["interruption"]:
            print(f"  Interruption: {last['interruption']}")
        print(f"  Response: {last['response'][:80]}...")
        print(f"  Reward: {last['reward']['total']:.3f}")

    def _obs(self) -> Dict:
        if self._task is None or self._subtask_idx >= len(self._task.subtasks):
            return {"task_description": "", "current_subtask": "",
                    "subtask_index": self._subtask_idx, "history": self._history,
                    "interruption": None}
        active = self._get_active_interruption()
        return {
            "task_description": self._task.description,
            "current_subtask": self._task.subtasks[self._subtask_idx],
            "subtask_index": self._subtask_idx,
            "history": list(self._history),
            "interruption": active.message if active else None,
        }

    def _get_active_interruption(self):
        for intr in self._interruptions:
            if intr.step == self._subtask_idx:
                return intr
        return None

    def full_prompt(self) -> str:
        obs = self._obs()
        lines = [
            f"TASK: {obs['task_description']}",
            f"CURRENT SUBTASK ({obs['subtask_index'] + 1}/{len(self._task.subtasks)}): {obs['current_subtask']}",
        ]
        if obs["interruption"]:
            lines.append(f"INTERRUPTION: {obs['interruption']}")
        if self._history:
            lines.append("PREVIOUS STEPS:")
            for h in self._history[-3:]:
                lines.append(f"  [{h['subtask']}] {h['response'][:60]}...")
        lines.append("Your response:")
        return "\n".join(lines)
