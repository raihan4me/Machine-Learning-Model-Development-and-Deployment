"""Compare a production batch with the training reference using Evidently."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

FEATURES = ["age", "income", "loan_amount", "credit_score",
            "employment_years", "num_open_accounts", "home_ownership"]


def main(batch_path: str) -> int:
    ref = pd.read_csv("data/raw/loans.csv")[FEATURES]
    cur = pd.read_csv(batch_path)[FEATURES]

    report = Report(metrics=[DataDriftPreset(drift_share=0.5)])
    report.run(reference_data=ref, current_data=cur)

    out = Path("reports"); out.mkdir(exist_ok=True)
    stem = Path(batch_path).stem
    report.save_html(str(out / f"drift_{stem}.html"))
    result = report.as_dict()

    summary = next(m["result"] for m in result["metrics"] if m["metric"] == "DatasetDriftMetric")
    table = next(m["result"] for m in result["metrics"] if m["metric"] == "DataDriftTable")
    drifted = sorted(c for c, v in table["drift_by_columns"].items() if v["drift_detected"])

    model = joblib.load("app/model/model.joblib")
    ref_pd = float(model.predict_proba(ref)[:, 1].mean())
    cur_pd = float(model.predict_proba(cur)[:, 1].mean())

    record = {
        "batch": stem,
        "dataset_drift": summary["dataset_drift"],
        "drifted_columns": drifted,
        "share_drifted": round(summary["share_of_drifted_columns"], 3),
        "mean_pred_default_ref": round(ref_pd, 3),
        "mean_pred_default_cur": round(cur_pd, 3),
    }
    (out / f"drift_{stem}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))

    if summary["dataset_drift"]:
        print("ACTION: RETRAIN RECOMMENDED - collect labels for the new population")
        return 1
    print("ACTION: no retraining needed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
