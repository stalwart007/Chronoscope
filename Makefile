# Chronoscope, developer entry points.
SHELL := /bin/bash
PY    := backend/.venv/bin/python
PIP   := uv pip install --python backend/.venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help setup setup-ml sample dev api web test lint fmt typecheck build up up-llm down logs clean pull-models

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv, install core deps + frontend packages
	@command -v uv >/dev/null || { echo "install uv: https://docs.astral.sh/uv/"; exit 1; }
	uv venv --python 3.12 backend/.venv
	$(PIP) -e "backend[dev]"
	cd frontend && npm install
	@echo "ok ready, run 'make sample' then 'make dev'"

setup-ml: ## Add the full-fidelity models (CLIP, sentence-transformers, Whisper)
	$(PIP) -e "backend[ml]"
	@echo "ok full-fidelity encoders installed"

sample: ## Generate the synthetic demo video + sidecar transcript
	$(PY) scripts/make_sample.py

dev: ## Run API and web dev servers together
	@trap 'kill 0' EXIT INT TERM; \
	$(MAKE) api & \
	$(MAKE) web & \
	wait

api: ## Run the API with autoreload
	cd backend && .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

web: ## Run the Vite dev server
	cd frontend && npm run dev

test: ## Run the backend test suite
	cd backend && .venv/bin/python -m pytest -q

lint: ## Ruff + mypy + tsc
	cd backend && .venv/bin/python -m ruff check app tests
	cd backend && .venv/bin/python -m mypy app || true
	cd frontend && npx tsc --noEmit

fmt: ## Auto-format and auto-fix
	cd backend && .venv/bin/python -m ruff check --fix app tests && .venv/bin/python -m ruff format app tests

typecheck: ## Frontend types only
	cd frontend && npx tsc --noEmit

build: ## Build the production frontend bundle
	cd frontend && npm run build

up: ## Start the full stack in Docker
	docker compose up -d --build
	@echo "-> http://localhost:8080"

up-llm: ## Start the stack including a local Ollama
	docker compose --profile llm up -d --build
	@echo "-> http://localhost:8080  (then: make pull-models)"

pull-models: ## Pull free local models into the Ollama container
	docker compose exec ollama ollama pull qwen2.5:7b-instruct
	docker compose exec ollama ollama pull llava:7b

down: ## Stop the stack
	docker compose --profile llm down

logs: ## Tail the API logs
	docker compose logs -f api

clean: ## Remove generated data and caches
	rm -rf data/artifacts data/chronoscope.db* backend/.pytest_cache backend/.ruff_cache backend/.mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
