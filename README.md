# AI Schedule Risk Predictor

MVP prototype for the "AI Schedule Risk Predictor" project: predicts
whether a task is likely to be delayed (Low / Medium / High risk) from
schedule, dependency, resource and workload inputs, and suggests
mitigation actions — built to match the project's SMART objectives and
Scope.

## How it works (hybrid rule-based + ML)

1. **Rule engine** (`model/rules.py`) — transparent, weighted heuristic
   score plus hard-escalation guardrails (e.g. a task with a passed
   deadline that isn't complete is always High risk, regardless of
   what the model predicts).
2. **ML model** (`model/train_model.py`) — a Random Forest trained on
   a synthetic dataset (`data/generate_data.py`), since real student
   assignment data wasn't available. Test accuracy is ~77%, within the
   70-80% range the project's Assumptions slide considers acceptable
   for an MVP.
3. **Hybrid combiner** (`model/predictor.py`) — takes the ML
   prediction as the default, then escalates it if the rule engine's
   guardrails require a higher minimum level. Never downgrades.

## Setup

```bash
pip install -r requirements.txt
python data/generate_data.py    # regenerate the synthetic dataset
python model/train_model.py     # retrain the model (writes risk_model.joblib)
streamlit run app.py            # launch the dashboard
```

## Using the dashboard

- **Assess a Task** — enter one task's details and get an instant risk
  level, the model's confidence, the specific reasons it was flagged,
  and mitigation suggestions.
- **Project Portfolio** — upload a CSV of tasks (or use the bundled
  sample dataset) to score an entire project at once, filter by risk
  level, and export the results.
- **About** — explains the hybrid methodology and shows the model's
  measured accuracy, for use in the project write-up / presentation.

## Required CSV columns for batch scoring

```
task_id, task_name, days_until_deadline, percent_complete,
expected_percent_complete, num_dependencies, num_dependencies_incomplete,
resource_availability_pct, team_workload_score, past_delays_count
```

## Project scope reminder

This MVP is for demonstration only: it uses synthetic/manually entered
data, a simplified hybrid model rather than a heavily trained
production model, and is not integrated with a live task-tracking
system — consistent with the project's stated Scope and Constraints.
