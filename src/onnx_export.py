"""Convert the logreg pipeline to ONNX, check parity and latency, log to MLflow."""
import time

import mlflow
import mlflow.sklearn
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("http://127.0.0.1:5000")
NUMERIC = ["age", "income", "loan_amount", "credit_score", "employment_years", "num_open_accounts"]
CATEGORICAL = ["home_ownership"]

run = mlflow.search_runs(experiment_names=["loan-default"],
                         filter_string="tags.model_family = 'logreg'",
                         order_by=["start_time DESC"], max_results=1).iloc[0]
pipe = mlflow.sklearn.load_model(f"runs:/{run.run_id}/model")

initial_types = [(c, FloatTensorType([None, 1])) for c in NUMERIC] + \
                [(c, StringTensorType([None, 1])) for c in CATEGORICAL]
onx = convert_sklearn(pipe, initial_types=initial_types,
                      options={LogisticRegression: {"zipmap": False}}, target_opset=17)
path = "artifacts/loan_logreg.onnx"
with open(path, "wb") as fh:
    fh.write(onx.SerializeToString())

df = pd.read_csv("data/raw/loans.csv").sample(1000, random_state=0)
X = df[NUMERIC + CATEGORICAL]
feed = {c: X[[c]].to_numpy().astype(np.float32) for c in NUMERIC}
feed.update({c: X[[c]].to_numpy().astype(str).astype(object) for c in CATEGORICAL})

sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("onnx outputs:", [o.name for o in sess.get_outputs()])
onnx_proba = sess.run(["probabilities"], feed)[0][:, 1]
skl_proba = pipe.predict_proba(X)[:, 1]
max_diff = float(np.max(np.abs(onnx_proba - skl_proba)))
print(f"max |sklearn - onnx| = {max_diff:.2e}")
assert max_diff < 1e-4, "parity check failed"

t0 = time.perf_counter(); [pipe.predict_proba(X) for _ in range(20)]; t_skl = time.perf_counter() - t0
t0 = time.perf_counter(); [sess.run(["probabilities"], feed) for _ in range(20)]; t_onnx = time.perf_counter() - t0
print(f"20 x 1000 rows: sklearn={t_skl*1000:.0f} ms  onnxruntime={t_onnx*1000:.0f} ms")

with mlflow.start_run(run_id=run.run_id):
    mlflow.log_artifact(path, artifact_path="onnx")
    mlflow.log_metric("onnx_max_abs_diff", max_diff)
print("PARITY OK - logged to run", run.run_id[:8])
