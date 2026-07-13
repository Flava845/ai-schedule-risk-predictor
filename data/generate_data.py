"""
Synthetic dataset generator for the AI Schedule Risk Predictor.

Per the project's Assumptions & Constraints slide, no real student
assignment database is available, so a synthetic dataset with
realistic value ranges is used to train and demo the model.

Each row represents one project/assignment task with schedule and
workload features, plus a simulated ground-truth risk label.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_TASKS = 1200

PRIORITIES = ["Low", "Medium", "High"]
CATEGORIES = ["Research", "Design", "Development", "Testing", "Documentation", "Presentation"]


def generate_tasks(n=N_TASKS, seed=RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    task_duration_days = rng.integers(1, 15, size=n)
    days_until_deadline = rng.integers(-2, 21, size=n)  # negative = already overdue
    percent_complete = np.clip(rng.normal(55, 25, size=n), 0, 100)
    expected_percent_complete = np.clip(
        100 - (days_until_deadline / task_duration_days.clip(min=1)) * 100, 0, 100
    )
    num_dependencies = rng.integers(0, 6, size=n)
    num_dependencies_incomplete = np.array(
        [rng.integers(0, d + 1) for d in num_dependencies]
    )
    resource_availability_pct = np.clip(rng.normal(70, 20, size=n), 0, 100)
    team_workload_score = np.clip(rng.normal(5, 2.5, size=n), 0, 10)  # 0 = idle, 10 = overloaded
    priority = rng.choice(PRIORITIES, size=n, p=[0.3, 0.45, 0.25])
    category = rng.choice(CATEGORIES, size=n)
    past_delays_count = rng.poisson(0.6, size=n)

    df = pd.DataFrame(
        {
            "task_id": [f"T{i+1:04d}" for i in range(n)],
            "task_name": [f"{c} task {i+1}" for i, c in enumerate(category)],
            "category": category,
            "priority": priority,
            "task_duration_days": task_duration_days,
            "days_until_deadline": days_until_deadline,
            "percent_complete": percent_complete.round(1),
            "expected_percent_complete": expected_percent_complete.round(1),
            "num_dependencies": num_dependencies,
            "num_dependencies_incomplete": num_dependencies_incomplete,
            "resource_availability_pct": resource_availability_pct.round(1),
            "team_workload_score": team_workload_score.round(1),
            "past_delays_count": past_delays_count,
        }
    )

    df["progress_gap"] = df["expected_percent_complete"] - df["percent_complete"]

    # --- simulate a ground-truth risk score to derive training labels ---
    # Weighted combination of the strongest delay indicators, plus noise,
    # loosely modelling how a real project manager would judge risk.
    score = (
        0.35 * (df["progress_gap"].clip(lower=0) / 100)
        + 0.20 * (df["num_dependencies_incomplete"] / (df["num_dependencies"] + 1))
        + 0.15 * (1 - df["resource_availability_pct"] / 100)
        + 0.15 * (df["team_workload_score"] / 10)
        + 0.10 * np.clip((5 - df["days_until_deadline"]) / 10, 0, 1)
        + 0.05 * (df["past_delays_count"].clip(upper=3) / 3)
    )
    noise = rng.normal(0, 0.05, size=n)
    df["risk_score_raw"] = np.clip(score + noise, 0, 1)

    # Bin by rank (quantiles) rather than fixed cutoffs so the synthetic
    # dataset always yields a usable, roughly balanced spread of classes
    # (~45% Low / 35% Medium / 20% High) regardless of the raw score scale.
    df["risk_label"] = pd.qcut(
        df["risk_score_raw"].rank(method="first"),
        q=[0, 0.45, 0.80, 1.0],
        labels=["Low", "Medium", "High"],
    )

    return df


if __name__ == "__main__":
    data = generate_tasks()
    out_path = __file__.replace("generate_data.py", "tasks_synthetic.csv")
    data.to_csv(out_path, index=False)
    print(f"Generated {len(data)} synthetic tasks -> {out_path}")
    print(data["risk_label"].value_counts())
