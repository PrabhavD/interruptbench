from typing import Dict
from .interruption import Interruption

def compute_reward(
    response: str,
    current_subtask: str,
    active_interruption,
    step: int,
    n_subtasks: int,
    interruptions_acknowledged: int,
    n_interruptions: int,
    done: bool,
) -> Dict[str, float]:
    """
    Compute the weighted reward for a single agent response.

    Returns a dict of named components and a "total" scalar. Breaking the
    reward into named components makes it easy to log each separately in
    W&B and diagnose which signal is driving or blocking learning.

    Components and weights:
        subtask (50%)               — does the response engage with the current
                                      subtask? Scored by keyword overlap with
                                      the subtask description. Floor of 0.2
                                      ensures the random policy receives partial
                                      credit; ceiling of 1.0 for full coverage.

        interruption_adherence (30%) — if an interruption was active this step,
                                       did the agent explicitly acknowledge it?
                                       Scored by presence of acknowledgement
                                       phrases or interruption keywords in the
                                       response. Defaults to 1.0 when no
                                       interruption is active (no penalty).

        efficiency (10%)            — decays linearly from 1.0 to 0.0 as the
                                      episode approaches 2× the expected length.
                                      Encourages the agent to complete tasks
                                      without unnecessary steps.

        completion_bonus (10%)      — flat 1.0 on the final subtask, 0.0
                                      otherwise. Rewards episode completion
                                      rather than early termination.

    Note: subtask and interruption_adherence use heuristic keyword scorers.
    Replace _subtask_score() and _interruption_score() with LLM judge calls
    for production training — the function signature is unchanged.

    Args:
        response:                  str   — the agent's text response this step
        current_subtask:           str   — the subtask the agent should address
        active_interruption:       Interruption | None — interruption fired this
                                   step, or None if no interruption is active
        step:                      int   — current step index (0-based)
        n_subtasks:                int   — total subtasks in the episode
        interruptions_acknowledged: int  — running count of acknowledged
                                   interruptions (reserved for future use)
        n_interruptions:           int   — total interruptions scheduled this
                                   episode (reserved for future use)
        done:                      bool  — True if this is the final subtask

    Returns:
        Dict[str, float] with keys: "subtask", "interruption_adherence",
        "efficiency", "completion_bonus", "total"
    """
    r = {}
    r["subtask"] = _subtask_score(response, current_subtask)
    if active_interruption is not None:
        r["interruption_adherence"] = _interruption_score(response, active_interruption)
    else:
        r["interruption_adherence"] = 1.0
    r["efficiency"] = max(0.0, 1.0 - (step / (n_subtasks * 2)))
    r["completion_bonus"] = 1.0 if done else 0.0
    r["total"] = (
        0.50 * r["subtask"]
        + 0.30 * r["interruption_adherence"]
        + 0.10 * r["efficiency"]
        + 0.10 * r["completion_bonus"]
    )
    return r

def _subtask_score(response: str, subtask: str) -> float:
    if not response.strip():
        return 0.0
    keywords = [w.lower() for w in subtask.split() if len(w) > 4]
    if not keywords:
        return 0.5
    hits = sum(1 for kw in keywords if kw in response.lower())
    # Give partial credit even for zero hits - random policy gets 0.2 baseline
    # so a trained model that actually addresses the subtask scores meaningfully higher
    base = 0.2
    return base + (1.0 - base) *min(1.0, hits / len(keywords))

def _interruption_score(response: str, interruption) -> float:
    acknowledgement_phrases = [
        "noted", "understood", "acknowledged", "updated", "revised",
        "adjusting", "changing", "incorporating", "as requested",
        "per the update", "following the new", "given the change",
    ]
    r = response.lower()
    if any(phrase in r for phrase in acknowledgement_phrases):
        return 1.0
    interrupt_keywords = [w.lower() for w in interruption.message.split() if len(w) > 4]
    hits = sum(1 for kw in interrupt_keywords if kw in r)
    return min(0.6, hits / max(len(interrupt_keywords), 1))
