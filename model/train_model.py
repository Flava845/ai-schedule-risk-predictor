"""
Trains the lightweight ML layer of the hybrid risk engine.

A RandomForest is used over the engineered features (progress gap,
dependency ratio, resource availability, workload, days to deadline,
past delays) to predict Low/Medium/High risk. This is intentionally
"light ML" per the project's Assumptions slide (70-80% accuracy is
acceptable for the MVP) rather than a heavy deep-learning model.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "tasks_synthetic.csv")
MODEL_PATH = os.path.join(HERE, "risk_model.joblib")
METRICS_PATH = os.path.join(HERE, "metrics.json")

FEATURE_COLUMNS = [
    "days_until_deadline",
    "percent_complete",
    "expected_percent_complete",
    "progress_gap",
    "num_dependencies",
    "num_dependencies_incomplete",
    "resource_availability_pct",
    "team_workload_score",
    "past_delays_count",
]


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.round(4).tolist()))

    joblib.dump({"model": clf, "features": FEATURE_COLUMNS}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {"accuracy": acc, "report": report, "feature_importances": importances},
            f,
            indent=2,
        )

    print(f"Test accuracy: {acc:.3f}")
    print(f"Model saved -> {MODEL_PATH}")
    print(f"Metrics saved -> {METRICS_PATH}")
    print("\nFeature importances:")
    for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {feat:32s} {imp:.3f}")


if __name__ == "__main__":
    main()
