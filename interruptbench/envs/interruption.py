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
        # existing 4
        "New requirement: the final output must be under 500 words.",
        "Update: focus only on open-source solutions.",
        "Change of scope: include a comparison with the previous year data.",
        "New constraint: assume a budget of under 10,000 USD.",
        # new 4
        "Revised scope: the deliverable is now a slide deck, not a document.",
        "New requirement: all recommendations must cite a source.",
        "Scope update: exclude any solutions requiring cloud infrastructure.",
        "Additional constraint: the output must be accessible to a non-English audience.",
    ],
    "priority_shift": [
        # existing 4
        "Reprioritise: the last subtask is now most urgent — complete it next.",
        "The team lead wants the summary section done before anything else.",
        "Deprioritise the background section — skip it for now.",
        "Focus on the recommendations section first; other sections can wait.",
        # new 4
        "New priority: the risk analysis section must be completed immediately.",
        "Shift focus: deliver the executive summary first, then resume normal order.",
        "De-scope the implementation details — focus on the high-level design only.",
        "Urgent: complete the testing section before anything else.",
    ],
    "contradiction": [
        # existing 4
        "Disregard the previous instruction about word count.",
        "Actually, do not include open-source solutions.",
        "Ignore the earlier reprioritisation — resume the original order.",
        "The format requirement has changed: use bullet points, not prose.",
        # new 4
        "Correction: the previous scope change is no longer applicable.",
        "Disregard the budget constraint mentioned earlier.",
        "Revert to the original task order — the priority shift is cancelled.",
        "The audience requirement has changed back to technical stakeholders.",
    ],
    "clarification": [
        # existing 4
        "Clarification: recent means published after 2023.",
        "To clarify: the target audience is non-technical stakeholders.",
        "Clarification: the system handles 10M events per hour, not per day.",
        "Note: focus on EU regulatory requirements only.",
        # new 4
        "Clarification: performance benchmarks should use P99 latency, not mean.",
        "Note: all cost estimates should be in GBP, not USD.",
        "Clarification: the word limit applies per section, not to the total document.",
        "To clarify: open-source means permissively licensed (MIT/Apache), not copyleft.",
    ],
}

# Interruption types ranked by cognitive difficulty for the agent
DIFFICULTY_PROFILE = {
    "easy":   ["clarification"],
    "medium": ["clarification", "scope_change", "priority_shift"],
    "hard":   ["scope_change", "priority_shift", "contradiction"],
    "any":    ["scope_change", "priority_shift", "contradiction", "clarification"],
}

class InterruptionScheduler:
    """
    Schedules interruptions across the steps of a task episode.

    Given a task with n subtasks, selects a subset of steps at which
    interruptions will fire, and assigns each an interruption type drawn
    from the allowed types for the configured difficulty level.

    Interruption frequency controls how many interruptions occur per episode:
        - low:    1 interruption
        - medium: 2 interruptions
        - high:   3 interruptions

    Interruption difficulty controls which types are allowed:
        - easy:   clarification only (additive, no conflict)
        - medium: clarification, scope_change, priority_shift
        - hard:   scope_change, priority_shift, contradiction
        - any:    all 4 types

    Args:
        frequency:  str — how many interruptions per episode ("low" | "medium" | "high")
        difficulty: str — which interruption types are allowed ("easy" | "medium" | "hard" | "any")
        seed:       int | None — RNG seed for reproducible scheduling
    """
    def __init__(self, frequency="medium", difficulty="any", seed=None):
        self.frequency = frequency
        self.difficulty = difficulty          # ← new parameter
        self.rng = random.Random(seed)
        self._count_map = {"low": 1, "medium": 2, "high": 3}

    def schedule(self, n_subtasks: int) -> List[Interruption]:
        """
        Generate a list of interruptions to inject during a task episode.

        Selects interruption steps by sampling without replacement from
        steps 1..n_subtasks-1 (never fires on step 0 so the agent always
        sees the task before the first interruption). Each selected step
        is assigned one interruption type drawn uniformly from the
        allowed types for the configured difficulty level.

        Args:
            n_subtasks: int — total number of subtasks in the episode

        Returns:
            List[Interruption] sorted by step index
        """
        n = self._count_map.get(self.frequency, 2)
        n = min(n, n_subtasks - 1)
        steps = sorted(self.rng.sample(range(1, n_subtasks), n))
        
        allowed_types = DIFFICULTY_PROFILE.get(self.difficulty, DIFFICULTY_PROFILE["any"])
        
        interruptions = []
        for step in steps:
            itype = self.rng.choice(allowed_types)
            msg = self.rng.choice(INTERRUPTION_TEMPLATES[itype])
            interruptions.append(Interruption(type=itype, message=msg, step=step))
        return interruptions

