"""Register the best run (by ROC AUC) and point the 'champion' alias at it."""
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
NAME = "loan-default-classifier"

best = mlflow.search_runs(experiment_names=["loan-default"],
                          order_by=["metrics.roc_auc DESC"], max_results=1).iloc[0]
print(f"best run: {best['tags.mlflow.runName']} roc_auc={best['metrics.roc_auc']:.4f}")

mv = mlflow.register_model(f"runs:/{best.run_id}/model", NAME)
client = MlflowClient()
client.set_registered_model_alias(NAME, "champion", mv.version)
client.update_model_version(NAME, mv.version,
                            description=f"Auto-promoted: roc_auc={best['metrics.roc_auc']:.4f}")
client.set_model_version_tag(NAME, mv.version, "validation_status", "approved")
print(f"registered {NAME} version {mv.version} as @champion")
