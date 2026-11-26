"""Train candidate models and log them to MLflow."""
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "loan-default"
DATA = "data/raw/loans.csv"
NUMERIC = ["age", "income", "loan_amount", "credit_score", "employment_years", "num_open_accounts"]
CATEGORICAL = ["home_ownership"]
TARGET = "default"


def data_md5(dvc_file: str = DATA + ".dvc") -> str:
    with open(dvc_file) as fh:
        return yaml.safe_load(fh)["outs"][0]["md5"]


def build_pipeline(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


CANDIDATES = {
    "logreg": (LogisticRegression(max_iter=1000, C=1.0), {"C": 1.0, "max_iter": 1000}),
    "random_forest": (
        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        {"n_estimators": 200, "max_depth": 8},
    ),
    "xgboost": (
        XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9, eval_metric="logloss", random_state=42, n_jobs=2),
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    ),
}


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_csv(DATA)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    md5 = data_md5()

    for name, (clf, params) in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(clf).fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = {
                "roc_auc": roc_auc_score(y_test, proba),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
            }
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"model_family": name, "data_md5": md5, "train_rows": len(X_train)})
            signature = infer_signature(X_test, pipe.predict_proba(X_test))
            mlflow.sklearn.log_model(pipe, artifact_path="model",
                                     signature=signature, input_example=X_test.head(3))
            print(f"{name:14s} run={run.info.run_id[:8]} "
                  + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
