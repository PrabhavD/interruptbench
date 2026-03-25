IMAGE_NAME=interruptbench:base
SERVICE_NAME=interruptbench

.PHONY: build up down shell rollout test-sandbox smoke logs clean

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

shell:
	docker compose run --rm $(SERVICE_NAME) /bin/bash

rollout:
	docker compose run --rm $(SERVICE_NAME) python scripts/random_rollout.py

test-sandbox:
	python scripts/test_sandbox.py

smoke:
	./scripts/smoke_test_container.sh

logs:
	docker compose logs -f

clean:
	docker compose down --remove-orphans
	docker image rm -f $(IMAGE_NAME) || true