#!/usr/bin/env bash
set -euo pipefail

echo "==> Building container"
docker compose build

echo "==> Running random rollout"
docker compose run --rm interruptbench python scripts/random_rollout.py

echo "==> Running sandbox test from host"
python scripts/test_sandbox.py

echo "==> All smoke tests passed"