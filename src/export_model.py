"""Export models:/loan-default-classifier@champion to app/model for serving."""
import json

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
NAME, ALIAS = "loan-default-classifier", "champion"

mv = MlflowClient().get_model_version_by_alias(NAME, ALIAS)
model = mlflow.sklearn.load_model(f"models:/{NAME}@{ALIAS}")
joblib.dump(model, "app/model/model.joblib")
run = mlflow.get_run(mv.run_id)
meta = {
    "model_name": NAME,
    "model_version": str(mv.version),
    "alias": ALIAS,
    "run_id": mv.run_id,
    "model_family": run.data.tags.get("model_family"),
    "data_md5": run.data.tags.get("data_md5"),
    "roc_auc": round(run.data.metrics["roc_auc"], 4),
}
with open("app/model/metadata.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print(json.dumps(meta, indent=2))
