SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help setup setup-pyenv setup-venv setup-venv-pyenv setup-poetry setup-poetry-pyenv doctor serve jobs

help:
	@echo "Common targets:"
	@echo "  make setup             # Auto manager (poetry if installed, else venv)"
	@echo "  make setup-pyenv       # Auto manager + pyenv install from .python-version"
	@echo "  make setup-venv        # Force venv + pip install"
	@echo "  make setup-venv-pyenv  # Force venv + pyenv install"
	@echo "  make setup-poetry      # Force Poetry install flow"
	@echo "  make setup-poetry-pyenv# Force Poetry + pyenv install"
	@echo "  make doctor            # Run transcribe-local doctor"
	@echo "  make jobs              # Start job runner"
	@echo "  make serve             # Start web server"

setup:
	./scripts/setup-env.sh

setup-pyenv:
	./scripts/setup-env.sh --install-python

setup-venv:
	./scripts/setup-env.sh --manager venv

setup-venv-pyenv:
	./scripts/setup-env.sh --manager venv --install-python

setup-poetry:
	./scripts/setup-env.sh --manager poetry

setup-poetry-pyenv:
	./scripts/setup-env.sh --manager poetry --install-python

doctor:
	@if [[ -x .venv/bin/transcribe-local ]]; then \
		.venv/bin/transcribe-local doctor; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run transcribe-local doctor; \
	else \
		echo "No environment detected. Run 'make setup' first."; \
		exit 1; \
	fi

jobs:
	@if [[ -x .venv/bin/transcribe-local ]]; then \
		.venv/bin/transcribe-local jobs start; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run transcribe-local jobs start; \
	else \
		echo "No environment detected. Run 'make setup' first."; \
		exit 1; \
	fi

serve:
	@if [[ -x .venv/bin/transcribe-local ]]; then \
		.venv/bin/transcribe-local serve --host 0.0.0.0 --port 8000; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run transcribe-local serve --host 0.0.0.0 --port 8000; \
	else \
		echo "No environment detected. Run 'make setup' first."; \
		exit 1; \
	fi
