import random
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Interruption:
    type: str
    message: str
    step: int
    target_subtask: Optional[int] = None

INTERRUPTION_TEMPLATES = {
    "scope_change": [
        "New requirement: the final output must be under 500 words.",
        "Update: focus only on open-source solutions.",
        "Change of scope: include a comparison with the previous year data.",
        "New constraint: assume a budget of under 10,000 USD.",
    ],
    "priority_shift": [
        "Reprioritise: the last subtask is now most urgent — complete it next.",
        "The team lead wants the summary section done before anything else.",
        "Deprioritise the background section — skip it for now.",
        "Focus on the recommendations section first; other sections can wait.",
    ],
    "contradiction": [
        "Disregard the previous instruction about word count.",
        "Actually, do not include open-source solutions.",
        "Ignore the earlier reprioritisation — resume the original order.",
        "The format requirement has changed: use bullet points, not prose.",
    ],
    "clarification": [
        "Clarification: recent means published after 2023.",
        "To clarify: the target audience is non-technical stakeholders.",
        "Clarification: the system handles 10M events per hour, not per day.",
        "Note: focus on EU regulatory requirements only.",
    ],
}

class InterruptionScheduler:
    def __init__(self, frequency="medium", seed=None):
        self.frequency = frequency
        self.rng = random.Random(seed)
        self._count_map = {"low": 1, "medium": 2, "high": 3}

    def schedule(self, n_subtasks: int) -> List[Interruption]:
        n = self._count_map.get(self.frequency, 2)
        n = min(n, n_subtasks - 1)
        steps = sorted(self.rng.sample(range(1, n_subtasks), n))
        interruptions = []
        for step in steps:
            itype = self.rng.choice(list(INTERRUPTION_TEMPLATES.keys()))
            msg = self.rng.choice(INTERRUPTION_TEMPLATES[itype])
            interruptions.append(Interruption(type=itype, message=msg, step=step))
        return interruptions
