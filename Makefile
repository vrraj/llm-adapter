# Minimal Makefile for llm-adapter demo

.PHONY: help venv install start start-bg fg stop kill logs

VENV := .venv
PYTHON := $(VENV)/bin/python
LOG_DIR := logs
LOG_FILE := $(LOG_DIR)/llm_adapter_demo.log
PORT := 7100
HOST := localhost
UI_URL := http://$(HOST):$(PORT)/ui/

help:
	@echo "Available targets:"
	@echo "  make venv      - create Python virtualenv in .venv"
	@echo "  make install   - install llm-adapter in editable mode into .venv"
	@echo "  make start     - run FastAPI demo in background (logs to $(LOG_FILE))"
	@echo "  make fg        - run FastAPI demo in foreground (reload, logs to console)"
	@echo "  make start-bg  - alias for make start"
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
	@mkdir -p $(LOG_DIR)
	@echo "Starting llm-adapter FastAPI demo in background on port $(PORT) ..."
	nohup $(VENV)/bin/python -m uvicorn llm_adapter_demo.api:app --reload --port $(PORT) \
		> $(LOG_FILE) 2>&1 & echo $$! > .uvicorn_pid
	@echo "Waiting for server to come up ..."
	@for i in $$(seq 1 30); do \
		curl -s -o /dev/null http://$(HOST):$(PORT)/ && break; \
		sleep 1; \
	done
	@if curl -s -o /dev/null http://$(HOST):$(PORT)/; then \
		echo ""; \
		echo "=========================================================="; \
		echo " llm-adapter demo is running"; \
		echo ""; \
		echo "   Interactive Playground UI : $(UI_URL)"; \
		echo "   API root                  : http://$(HOST):$(PORT)/"; \
		echo "   Logs                      : $(LOG_FILE)  (make logs)"; \
		echo "   Stop                      : make stop"; \
		echo "=========================================================="; \
	else \
		echo ""; \
		echo "Server did not come up within 30s. Check logs: make logs"; \
	fi

start-bg: start
	@echo "(make start-bg is an alias for make start)"

fg: venv
	@echo "Starting llm-adapter FastAPI demo in foreground on port $(PORT) ..."
	@echo "Interactive Playground UI (once started): $(UI_URL)"
	$(VENV)/bin/python -m uvicorn llm_adapter_demo.api:app --reload --port $(PORT)

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
		echo "No log file at $(LOG_FILE) yet. Start the app with 'make start' first."; \
	fi
