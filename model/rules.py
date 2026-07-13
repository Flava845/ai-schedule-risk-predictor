"""
Rule-based (heuristic) layer of the hybrid risk engine.

This mirrors the "simplified to rule-based + light ML" pivot described in
the project's Challenges & Risks slide: transparent, explainable rules
that (a) produce a standalone heuristic score and (b) can hard-escalate
the ML prediction when a task hits an unambiguous danger condition,
so the tool never quietly under-reports an obvious risk.
"""

from dataclasses import dataclass


RISK_LEVELS = ["Low", "Medium", "High"]


@dataclass
class RuleResult:
    score: float          # 0..1 heuristic risk score
    level: str            # Low / Medium / High from the rules alone
    triggered: list        # list of (reason, weight) tuples that fired
    escalate_to: str | None  # forced minimum level, or None


def evaluate_rules(task: dict) -> RuleResult:
    """
    task expects keys:
      days_until_deadline, percent_complete, expected_percent_complete,
      num_dependencies, num_dependencies_incomplete,
      resource_availability_pct, team_workload_score, past_delays_count
    """
    progress_gap = task["expected_percent_complete"] - task["percent_complete"]
    dep_ratio = (
        task["num_dependencies_incomplete"] / task["num_dependencies"]
        if task["num_dependencies"] > 0
        else 0
    )

    triggered = []
    score = 0.0

    w = 0.35 * max(progress_gap, 0) / 100
    score += w
    if progress_gap > 15:
        triggered.append((f"Behind schedule by {progress_gap:.0f} pts vs expected progress", w))

    w = 0.20 * dep_ratio
    score += w
    if dep_ratio >= 0.5 and task["num_dependencies"] > 0:
        triggered.append((f"{task['num_dependencies_incomplete']}/{task['num_dependencies']} dependencies still incomplete", w))

    w = 0.15 * (1 - task["resource_availability_pct"] / 100)
    score += w
    if task["resource_availability_pct"] < 50:
        triggered.append((f"Low resource availability ({task['resource_availability_pct']:.0f}%)", w))

    w = 0.15 * (task["team_workload_score"] / 10)
    score += w
    if task["team_workload_score"] >= 7:
        triggered.append((f"Team workload high ({task['team_workload_score']:.1f}/10)", w))

    w = 0.10 * max(min((5 - task["days_until_deadline"]) / 10, 1), 0)
    score += w
    if task["days_until_deadline"] <= 3:
        triggered.append((f"Only {task['days_until_deadline']} day(s) until deadline", w))

    w = 0.05 * min(task["past_delays_count"], 3) / 3
    score += w
    if task["past_delays_count"] >= 2:
        triggered.append((f"Task has slipped {task['past_delays_count']} times before", w))

    score = max(0.0, min(1.0, score))
    level = "Low" if score < 0.35 else "Medium" if score < 0.65 else "High"

    # Hard escalation guardrails: conditions a PM would never let slide
    # to "Low" or "Medium" no matter what the ML model says.
    escalate_to = None
    if task["days_until_deadline"] <= 0 and task["percent_complete"] < 100:
        escalate_to = "High"
        triggered.append(("Deadline has passed and task is not complete", 1.0))
    elif task["days_until_deadline"] <= 2 and progress_gap > 30:
        escalate_to = "High"
        triggered.append(("Deadline in <=2 days with a large progress gap", 1.0))
    elif dep_ratio >= 0.75 and task["days_until_deadline"] <= 5:
        escalate_to = escalate_to or "Medium"

    return RuleResult(score=score, level=level, triggered=triggered, escalate_to=escalate_to)


def top_mitigation_suggestions(task: dict, rule_result: RuleResult, final_level: str) -> list:
    """Map the strongest risk drivers to concrete mitigation suggestions."""
    suggestions = []
    reasons = " ".join(r for r, _ in rule_result.triggered).lower()

    if final_level == "Low":
        return ["No action needed — continue monitoring progress at the next check-in."]

    if "behind schedule" in reasons:
        suggestions.append("Reprioritise this task or reassign extra effort to close the progress gap.")
    if "dependencies" in reasons:
        suggestions.append("Escalate blocked dependencies to their owners; consider re-sequencing work that doesn't need them.")
    if "resource availability" in reasons:
        suggestions.append("Free up or reallocate team capacity, or negotiate a short deadline extension.")
    if "workload" in reasons:
        suggestions.append("Redistribute tasks across the team to relieve overloaded members.")
    if "deadline" in reasons:
        suggestions.append("Flag to the project manager now — prepare a contingency or fallback plan before the deadline.")
    if "slipped" in reasons:
        suggestions.append("Review why this task type keeps slipping and add buffer time in future planning.")

    if not suggestions:
        suggestions.append("Monitor closely and re-check progress before the next milestone.")

    return suggestions
