# GoodBoy

GoodBoy is a **local autonomous agent harness** for your machine. It runs an interactive loop that:

- prompts a model for structured “agent steps” (JSON)
- executes allowed local tools (shell and Python)
- handles clarifications / retries
- supports per-session model selection

## Features

- **Interactive CLI** (conversation-style): run tasks, then keep iterating.
- **Structured agent protocol**: model responses are parsed into typed steps.
- **Local tool execution**: shell commands and Python code (configurable timeout).
- **Model allowlist + interactive model picker**.
- **Default reasoning-effort picker** for reasoning models.
- **Session logging** (optional) to a local log directory.
- **TLS/proxy-friendly OpenAI client options** (custom CA bundle, disable verify).

## Requirements

- Python **3.10+**

## Install

### From source (editable)

```bash
pip install -e .
```

### Install into a fresh venv (helper)

```bash
python dev.py
```

## Configure OpenAI credentials

GoodBoy stores configuration in a local `.env` file at the project root.

```bash
goodboy setup
```

This prompts for:
- your **OpenAI API key**
- the **default model**
- the optional **default reasoning effort**

To check current saved settings:

```bash
goodboy status
```

## Usage

### Run the agent

```bash
goodboy
```

or equivalently:

```bash
python main.py
```

You’ll be prompted to enter a task. During the session you can also type special commands (see below).

### CLI flags

```bash
goodboy [OPTIONS]
```

Common options:
- `-f` / `--show-thoughts`: show model thoughts on each step
- `-v` / `--verbose`: show thoughts, tool/model switches, and task status lines
- `-m` / `--show-model`: print the model used on each agent response
- `-c` / `--show-commands`: print shell/Python commands the agent runs (not their output)
- `/commands`: toggle command visibility during a session (same as `-c`)
- `/autoswitch`: toggle automatic model switching (agent may escalate models between turns)
- `/reasoning`: set the default reasoning effort for reasoning models
- `-s` / `--stream-output` or `/stream`: stream each model response to the console as it is generated
- `-d` / `--debug`: show commands and stdout/stderr from shell and Python tool runs
- `-i` / `--debug-input`: print the full prompt sent to the model each turn
- `-o` / `--debug-output`: print the model’s raw response in full each turn

## Environment variables (.env)

GoodBoy reads configuration from `.env` (and also supports related exported env vars such as `SSL_CERT_FILE`).

Key variables:
- `OPENAI_API_KEY` – your OpenAI API key
- `OPENAI_MODEL` – default model id (used when selecting the model for a new run)
- `GOODBOY_SHOW_COMMANDS` – when `true`, show shell/Python commands (no output); updated by `/commands`
- `GOODBOY_AUTO_MODEL_SWITCH` – when `true`, allow proactive model escalation; updated by `/autoswitch`
- `GOODBOY_REASONING_EFFORT` – default reasoning effort for reasoning models; updated by `/reasoning`

Agent behavior:
- `GOODBOY_MAX_TURNS` – maximum agent turns per task (default: `40`)
- `GOODBOY_TOOL_TIMEOUT_SEC` – timeout for tool execution (default: `120.0`)
- `GOODBOY_MAX_CLARIFICATIONS` – max clarification turns (default: `3`)
- `GOODBOY_PLAN_MODE` – `auto` (plan required for complex tasks), `always`, or `off` (default: `auto`)
- `GOODBOY_VERIFY_BEFORE_COMPLETE` – block `task_complete` until tests pass after edits (default: `true`)
- `GOODBOY_WORKING_MEMORY_MAX` – cap on durable memory lines (default: `30`)

Project memory: add `AGENTS.md`, `.goodboy/memory.md`, or `GOODBOY.md` in the repo root with test commands and conventions (see [`examples/AGENTS.md`](examples/AGENTS.md)). GoodBoy injects the first file found into every task.

Session logging:
- `GOODBOY_SESSION_LOG` – enable/disable session logs (default: enabled)
- `GOODBOY_LOG_DIR` – optional log output directory

TLS / HTTPS proxy configuration:
- `GOODBOY_SSL_CA_BUNDLE` – path to a `.pem` CA bundle file
- `GOODBOY_SSL_VERIFY` – set to `false` (case-insensitive) to disable TLS verification

## Harness commands

During an interactive session, GoodBoy supports built-in REPL commands (e.g. exit/clear/model/reasoning).

For a full list, see:
- `agent/repl_commands.py`

## Tests

Run the test suite with:

```bash
pytest
```

## Project scripts

This repository defines console scripts via `pyproject.toml`:
- `goodboy` → `main:cli`
- `llm` → `llm:main`
