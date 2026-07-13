"""
Hybrid risk predictor: combines the ML classifier's prediction with the
rule engine's guardrails, and attaches human-readable explanations and
mitigation suggestions.

Final level = ML prediction, unless the rule engine's escalate_to
guardrail requires a higher minimum level (rules never downgrade,
only escalate) — this is the "hybrid" behaviour described on the
project's Change Management slide.
"""

import os

import joblib
import pandas as pd

from .rules import RISK_LEVELS, evaluate_rules, top_mitigation_suggestions

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "risk_model.joblib")

_bundle = None


def _load_model():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _derive_features(task: dict) -> dict:
    task = dict(task)
    task["progress_gap"] = task["expected_percent_complete"] - task["percent_complete"]
    return task


def predict_task(task: dict) -> dict:
    """
    task: dict with raw input fields (see rules.evaluate_rules docstring,
    minus progress_gap which is derived here).
    Returns a dict with ml_level, rule_level, final_level, probabilities,
    triggered reasons, and mitigation suggestions.
    """
    task = _derive_features(task)
    bundle = _load_model()
    model, features = bundle["model"], bundle["features"]

    X = pd.DataFrame([{f: task[f] for f in features}])
    ml_level = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0].round(3)))

    rule_result = evaluate_rules(task)

    final_level = ml_level
    if rule_result.escalate_to and RISK_LEVELS.index(rule_result.escalate_to) > RISK_LEVELS.index(final_level):
        final_level = rule_result.escalate_to

    suggestions = top_mitigation_suggestions(task, rule_result, final_level)

    return {
        "ml_level": ml_level,
        "ml_probabilities": proba,
        "rule_level": rule_result.level,
        "rule_score": round(rule_result.score, 3),
        "escalated": final_level != ml_level,
        "final_level": final_level,
        "triggered_reasons": rule_result.triggered,
        "mitigation_suggestions": suggestions,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized version of predict_task: scores the whole DataFrame with a
    single model call instead of one call per row, which is what makes
    scoring an entire project portfolio fast."""
    bundle = _load_model()
    model, features = bundle["model"], bundle["features"]

    work = df.copy()
    work["progress_gap"] = work["expected_percent_complete"] - work["percent_complete"]

    X = work[features]
    ml_levels = model.predict(X)

    rule_results = [evaluate_rules(row.to_dict()) for _, row in work.iterrows()]

    final_levels = []
    escalated = []
    for ml_level, rule_result in zip(ml_levels, rule_results):
        final_level = ml_level
        if rule_result.escalate_to and RISK_LEVELS.index(rule_result.escalate_to) > RISK_LEVELS.index(final_level):
            final_level = rule_result.escalate_to
        final_levels.append(final_level)
        escalated.append(final_level != ml_level)

    out = df.copy()
    out["ml_level"] = ml_levels
    out["rule_level"] = [r.level for r in rule_results]
    out["final_level"] = final_levels
    out["escalated"] = escalated
    out["mitigation_suggestions"] = [
        "; ".join(top_mitigation_suggestions(work.iloc[i].to_dict(), rule_results[i], final_levels[i]))
        for i in range(len(work))
    ]
    return out
