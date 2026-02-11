# Minimal Makefile for llm-adapter demo

.PHONY: help venv install start start-bg stop kill logs

VENV := .venv
PYTHON := $(VENV)/bin/python
LOG_DIR := logs
LOG_FILE := $(LOG_DIR)/llm_adapter_demo.log
PORT := 8100

help:
	@echo "Available targets:"
	@echo "  make venv      - create Python virtualenv in .venv"
	@echo "  make install   - install llm-adapter in editable mode into .venv"
	@echo "  make start     - run FastAPI demo in foreground (reload, logs to console)"
	@echo "  make start-bg  - run FastAPI demo in background (logs to $(LOG_FILE))"
	@echo "  make stop      - stop demo server gracefully (SIGTERM)"
	@echo "  make kill      - force-kill demo server (SIGKILL)"
	@echo "  make logs      - tail the background log file"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
		echo "Created $(VENV)/"; \
	else \
		echo "$(VENV)/ already exists (reusing)"; \
	fi

install: venv
	. $(VENV)/bin/activate && \
		python -m pip install --upgrade pip && \
		pip install -e .

start: venv
	@echo "Starting llm-adapter FastAPI demo in foreground on port $(PORT) ..."
	$(VENV)/bin/python -m uvicorn llm_adapter_demo.api:app --reload --port $(PORT)

start-bg: venv
	@mkdir -p $(LOG_DIR)
	@echo "Starting llm-adapter FastAPI demo in background on port $(PORT) ..."
	nohup $(VENV)/bin/python -m uvicorn llm_adapter_demo.api:app --reload --port $(PORT) \
		> $(LOG_FILE) 2>&1 & echo $$! > .uvicorn_pid
	@echo "Background PID written to .uvicorn_pid"

stop:
	@echo "Stopping llm-adapter demo (SIGTERM) ..."
	@if [ -f .uvicorn_pid ]; then \
		rm -f .uvicorn_pid; \
	fi
	@PIDS=$$(lsof -ti :$(PORT) 2>/dev/null); \
	if [ -n "$$PIDS" ]; then echo "$$PIDS" | xargs kill -15 2>/dev/null || true; fi

kill:
	@echo "Force-killing llm-adapter demo (SIGKILL) ..."
	@if [ -f .uvicorn_pid ]; then \
		rm -f .uvicorn_pid; \
	fi
	@PIDS=$$(lsof -ti :$(PORT) 2>/dev/null); \
	if [ -n "$$PIDS" ]; then echo "$$PIDS" | xargs kill -9 2>/dev/null || true; fi

logs:
	@if [ -f $(LOG_FILE) ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "No log file at $(LOG_FILE) yet. Start the app with 'make start-bg' first."; \
	fi
