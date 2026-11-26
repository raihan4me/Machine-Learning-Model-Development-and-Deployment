#!/usr/bin/env bash
cd ~/loan-mlops || exit 1
PASS=0; FAIL=0
check() { if eval "$2" >/dev/null 2>&1; then echo "[PASS] $1"; PASS=$((PASS+1)); else echo "[FAIL] $1"; FAIL=$((FAIL+1)); fi; }

echo "== Task 1: data versioning"
check "loans.csv.dvc tracked in git"      "git ls-files --error-unmatch data/raw/loans.csv.dvc"
check "DVC cache and remote in sync"      "dvc status -c | grep -q 'in sync'"

echo "== Task 2: experiment tracking"
check "MLflow server healthy"             "curl -sf http://127.0.0.1:5000/health"
check ">=3 runs in loan-default"          "[ \$(curl -s -X POST http://127.0.0.1:5000/api/2.0/mlflow/runs/search -H 'Content-Type: application/json' -d \"{\\\"experiment_ids\\\":[\\\"\$(curl -s 'http://127.0.0.1:5000/api/2.0/mlflow/experiments/get-by-name?experiment_name=loan-default' | jq -r .experiment.experiment_id)\\\"]}\" | jq '.runs|length') -ge 3 ]"

echo "== Task 3: registry and artifacts"
check "champion alias exists"             "curl -sf 'http://127.0.0.1:5000/api/2.0/mlflow/registered-models/alias?name=loan-default-classifier&alias=champion' | jq -e .model_version.version"
check "joblib model + metadata exported"  "test -s app/model/model.joblib && jq -e .model_version app/model/metadata.json"
check "ONNX model present"                "test -s artifacts/loan_logreg.onnx"

echo "== Task 4: API and tests"
check "pytest suite passes"               "python -m pytest -q"

echo "== Task 5: container"
check "image loan-api:1.0.0 exists"       "docker image inspect loan-api:1.0.0"
check "image runs as non-root"            "[ \"\$(docker image inspect loan-api:1.0.0 --format '{{.Config.User}}')\" = 10001 ]"
check "compose file valid"                "docker compose config -q"

echo "== Task 6: Kubernetes"
check "deployment fully ready"            "[ \"\$(kubectl -n ml-serving get deploy loan-api -o jsonpath='{.status.readyReplicas}')\" = \"\$(kubectl -n ml-serving get deploy loan-api -o jsonpath='{.spec.replicas}')\" ]"
check "running image is 1.1.0"            "kubectl -n ml-serving get deploy loan-api -o jsonpath='{.spec.template.spec.containers[0].image}' | grep -q 'loan-api:1.1.0'"
check "ingress /health returns 1.1.0"     "curl -sf localhost:8080/health | jq -e '.app_version==\"1.1.0\"'"
check "ingress /predict works"            "curl -sf -X POST localhost:8080/predict -H 'Content-Type: application/json' -d @sample.json | jq -e '.predictions|length==2'"

echo "== Task 7: drift"
check "week01 no drift"                   "jq -e '.dataset_drift==false' reports/drift_batch_week01.json"
check "week04 drift detected"             "jq -e '.dataset_drift==true' reports/drift_batch_week04.json"
check "HTML reports saved"                "ls reports/drift_batch_week04.html"

echo; echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
