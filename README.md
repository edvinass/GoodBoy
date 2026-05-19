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
- **In-process local LLMs** (optional): download GGUF weights during setup and run them inside GoodBoy via llama-cpp-python — no Ollama or other daemon.

## Requirements

- Python **3.10+**
- For local models: **~8–16GB RAM** recommended for the default 7B Q4 catalog model; disk space for the download (~4–5GB)

## Install

One-time setup installs GoodBoy on your machine. After that, run `goodboy` from **any** project directory—the agent uses that repo as its workspace (git root when you are inside a repo).

### 1. Clone GoodBoy somewhere permanent

Pick a fixed location (you will not need to `cd` here for daily use):

```bash
git clone https://github.com/edvinass/GoodBoy.git ~/.local/share/goodboy
cd ~/.local/share/goodboy
```

Replace the URL with your fork or copy of the repo if needed.

### 2. Create a venv and install the CLI

```bash
python3 dev.py
```

This creates `.venv` in the clone and installs the `goodboy` command into `.venv/bin`.

**Cloud-only (OpenAI API):** the steps above are enough.

**Local GGUF models (optional):** install extra dependencies, then re-run setup:

```bash
.venv/bin/pip install -e ".[local]"
```

This adds `llama-cpp-python` and `huggingface-hub` for on-device models.

### 3. Put `goodboy` on your PATH

Add the venv’s `bin` directory to your shell profile so `goodboy` works in every terminal and from any repo.

**macOS / Linux (zsh — default on macOS):**

```bash
echo 'export PATH="$HOME/.local/share/goodboy/.venv/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**macOS / Linux (bash):**

```bash
echo 'export PATH="$HOME/.local/share/goodboy/.venv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):** add the `Scripts` folder to your user `Path`, for example:

`%USERPROFILE%\.local\share\goodboy\.venv\Scripts`

Adjust the path if you cloned somewhere other than `~/.local/share/goodboy`. After editing your profile, open a **new** terminal and check:

```bash
which goodboy   # macOS/Linux
goodboy --help
```

### 4. Configure API key and default model

GoodBoy stores credentials in `.env` inside the **GoodBoy install directory** (not in each project you work on).

```bash
goodboy setup
```

Run this once after install (and again whenever you want to change keys or models).

### 5. Use it in any repository

```bash
cd ~/projects/my-app
goodboy
```

The harness runs tools in that project’s git root (or the current directory if it is not a git repo).

### Alternative: pipx (global CLI without editing PATH)

If you use [pipx](https://pipx.pypa.io/):

```bash
git clone https://github.com/edvinass/GoodBoy.git ~/.local/share/goodboy
cd ~/.local/share/goodboy
pipx install -e .
# optional local models:
pipx install -e ".[local]" --force
goodboy setup
```

`pipx` installs `goodboy` into an isolated environment and links it on your PATH automatically.

### Manual install (editable, existing venv)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # or: pip install -e ".[local]"
```

Then add `.venv/bin` (or `Scripts` on Windows) to your PATH as in step 3.

## Configure credentials and models

Configuration lives in `.env` at the GoodBoy install root (see step 4 above). Project-specific notes belong in each repo (`AGENTS.md`, `.goodboy/memory.md`, or `GOODBOY.md`).

```bash
goodboy setup
```

This walks you through:
- **Cloud (OpenAI)**, **Local (on this machine)**, or **Both**
- OpenAI API key when using cloud (optional if your default model is local)
- Downloading a local GGUF model (when local or both is selected)
- Choosing the default model from the combined picker (cloud + installed local models)

Local model IDs use the `local:` prefix (for example `local:qwen2.5-coder-7b-q4`). During setup you can pick from several GGUF options (Llama 3.2 3B, Phi-3.5 Mini, Qwen2.5 Coder 7B/14B, Mistral 7B, Gemma 2 9B, Granite 8B, DeepSeek Coder V2 Lite, and more). Weights are stored under `~/.goodboy/models/` (override with `GOODBOY_MODELS_DIR`).

You can also drop any `.gguf` file directly into that folder (top level); it will show up in `/model` automatically—no rename required unless you want it to match a catalog download name.

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
- `OPENAI_API_KEY` – your OpenAI API key (optional when default model is `local:…`)
- `OPENAI_MODEL` – default model id (OpenAI id or `local:…` for on-device inference)
- `GOODBOY_MODELS_DIR` – directory for downloaded GGUF weights (default: `~/.goodboy/models`)
- `GOODBOY_SHOW_COMMANDS` – when `true`, show shell/Python commands (no output); updated by `/commands`
- `GOODBOY_AUTO_MODEL_SWITCH` – when `true`, allow proactive model escalation; updated by `/autoswitch`
- `GOODBOY_REASONING_EFFORT` – default reasoning effort for reasoning models; updated by `/reasoning`

Agent behavior:
- `GOODBOY_MAX_TURNS` – maximum agent turns per task (default: `40`)
- `GOODBOY_TOOL_TIMEOUT_SEC` – timeout for tool execution (default: `180.0`, 3 minutes)
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
