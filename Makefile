# Common tasks. Run `make help` to see everything.
.DEFAULT_GOAL := help
.PHONY: help setup verify test lint format links links-external examples datasets check clean modules up down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create a virtual environment and install dependencies
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Now activate it:  source .venv/bin/activate"
	@echo "Windows:          .\\.venv\\Scripts\\Activate.ps1"

verify: ## Check that your learning environment is set up correctly
	python scripts/verify_setup.py

test: ## Run the test suite
	pytest -q

lint: ## Lint the codebase
	ruff check .

format: ## Auto-format the codebase
	ruff format .
	ruff check --fix .

links: ## Validate internal Markdown links (fast, offline)
	python scripts/check_links.py

links-external: ## Also HTTP-check every external URL (slow, needs network)
	python scripts/check_links.py --external

examples: ## Run every documented code example and check its stated output
	python scripts/check_examples.py --strict

datasets: ## Verify the committed sample datasets still match their generator
	python scripts/make_sample_datasets.py --check

modules: ## Regenerate module backlog READMEs (skips authored modules)
	python scripts/generate_module_readmes.py

check: lint test links examples datasets ## Run everything CI runs

up: ## Start local services (vector store, PostgreSQL, MLflow)
	docker compose up -d

down: ## Stop local services
	docker compose down

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist *.egg-info
