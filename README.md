Scenario. You are the ML engineer at NorthBridge Lending, a small consumer-loan company. The risk team wants a model that estimates the probability that a loan applicant will default. The model must be served to the loan-origination system as a REST API. Management has set these requirements:

The training data must be versioned so that any model can be traced back to the exact data it was trained on.
Every experiment must be tracked, and the production model must be selected from a model registry, not copied by hand.
A portable ONNX artifact must be produced for a future edge or partner integration.
The API must validate its input, expose health and Prometheus metrics, and be covered by automated tests.
The service must run as a container, first under docker compose with Prometheus scraping it, then on Kubernetes with probes, resource limits, rolling updates and rollback.
Production input data must be checked for drift so the team knows when to retrain.

By the end of this lab you will be able to:

Version datasets with DVC and link a data version to an experiment run.

Run a local MLflow tracking server with a SQL backend and proxied artifact store, log parameters, metrics and models, and query runs through the MLflow REST API.

Compare scikit-learn and XGBoost candidates and promote the best one using MLflow registry aliases.

Serialise models with joblib, convert a scikit-learn pipeline to ONNX and verify numerical parity with onnxruntime.

Build a FastAPI inference service with pydantic validation, health checks and Prometheus metrics, and test it with pytest.

Containerise the service securely (non-root, health check, pinned dependencies) and run it with docker compose alongside Prometheus.

Deploy to a local Kubernetes cluster (k3d) with probes, resource limits and configuration, then perform a rolling update and recover from a failed rollout.

Detect data drift with Evidently and turn the result into a retraining decision.
