FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first: it is cached until requirements change
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

RUN useradd --create-home --uid 10001 appuser
COPY app/ ./app/

ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}

USER 10001
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
