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
