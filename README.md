# GoodBoy

GoodBoy is a **local autonomous agent harness** for your machine. It runs an interactive loop that:

- prompts a model for structured “agent steps” (JSON)
- executes allowed local tools (shell and Python)
- handles clarifications / retries
- supports per-session model selection

## Repository layout

| Directory | Purpose |
|-----------|---------|
| [`python/`](python/) | CLI, agent harness, and tests (`pyproject.toml` lives here) |
| [`web/`](web/) | Vue install landing page (Railway) |
| [`install.sh`](install.sh) | One-line installer (repo root) |

`.env` and the install venv (`.venv`) stay at the **repo/install root**, not inside `python/`.

## Features

- **Interactive CLI** (conversation-style): run tasks, then keep iterating.
- **Structured agent protocol**: model responses are parsed into typed steps.
- **Local tool execution**: shell commands and Python code (configurable timeout).
- **Multiple model providers**: OpenAI Responses API, **DeepSeek** (OpenAI-compatible chat completions), and in-process local GGUF models — all selectable from one model picker.
- **Default reasoning-effort picker** for reasoning models.
- **Session logging** (optional) to a local log directory.
- **TLS/proxy-friendly HTTP client options** (custom CA bundle, disable verify) shared by all cloud providers.
- **In-process local LLMs** (optional): download GGUF weights during setup and run them inside GoodBoy via llama-cpp-python — no Ollama or other daemon.

## Requirements

- Python **3.10+**
- For local models: **~8–16GB RAM** recommended for the default 7B Q4 catalog model; disk space for the download (~4–5GB)

## Install

One-time setup installs GoodBoy on your machine. After that, run `goodboy` from **any** project directory—the agent uses that repo as its workspace (git root when you are inside a repo).

### Quick install (macOS / Linux)

```bash
curl -fsSL https://YOUR-DOMAIN/install.sh | bash
```

(Replace `YOUR-DOMAIN` with your [Railway-hosted install page](web/) domain. The installer downloads the app tarball from the same host — no GitHub account or `git clone` required.)

Or install from GitHub (needs a public repo and `git`):

```bash
curl -fsSL https://raw.githubusercontent.com/edvinass/GoodBoy/main/install.sh | bash
```

This installs GoodBoy to `~/.local/share/goodboy`, creates a venv, installs the CLI, and appends the venv’s `bin` directory to your shell profile. Open a new terminal, then run `goodboy` once (first run walks through API key and model setup).

Optional: install local GGUF model dependencies during install:

```bash
GOODBOY_LOCAL=1 bash -c "$(curl -fsSL https://raw.githubusercontent.com/edvinass/GoodBoy/main/install.sh)"
```

No GitHub at all: use `GOODBOY_INSTALL_BASE_URL` (as on the Railway landing page) or run from a checkout with `GOODBOY_SOURCE_DIR="$PWD" bash install.sh`. For a fork via git, set `GOODBOY_REPO_URL`.

### Manual install

#### 1. Clone GoodBoy somewhere permanent

Pick a fixed location (you will not need to `cd` here for daily use):

```bash
git clone https://github.com/edvinass/GoodBoy.git ~/.local/share/goodboy
cd ~/.local/share/goodboy
```

Replace the URL with your fork or copy of the repo if needed.

#### 2. Create a venv and install the CLI

From the clone directory:

```bash
GOODBOY_SOURCE_DIR="$PWD" bash install.sh
```

Or run the installer after cloning (same as the curl one-liner, without re-cloning):

```bash
bash install.sh
```

This creates `.venv` in the install directory and installs the `goodboy` command into `.venv/bin`.

**Cloud-only (OpenAI API):** the steps above are enough.

**Local GGUF models (optional):** install extra dependencies, then re-run setup:

```bash
.venv/bin/pip install -e "python[local]"
```

This adds `llama-cpp-python` and `huggingface-hub` for on-device models.

#### 3. Put `goodboy` on your PATH

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

#### 4. Configure API key and default model

GoodBoy stores credentials in `.env` inside the **GoodBoy install directory** (not in each project you work on).

```bash
goodboy setup
```

Run this once after install (and again whenever you want to change keys or models).

#### 5. Use it in any repository

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
pipx install -e python
# optional local models:
pipx install -e "python[local]" --force
goodboy setup
```

`pipx` installs `goodboy` into an isolated environment and links it on your PATH automatically.

### Manual install (editable, existing venv)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e python       # or: pip install -e "python[local]"
```

Then add `.venv/bin` (or `Scripts` on Windows) to your PATH as in step 3.

## Configure credentials and models

Configuration lives in `.env` at the GoodBoy install root (see step 4 above). Project-specific notes belong in each repo (`AGENTS.md`, `.goodboy/memory.md`, or `GOODBOY.md`).

```bash
goodboy setup
```

This walks you through:
- **Cloud (OpenAI)**, **DeepSeek**, **Local (on this machine)**, or **Multiple providers**
- OpenAI API key when using OpenAI (optional if your default model is local or DeepSeek)
- DeepSeek API key when using DeepSeek (separate from your OpenAI key — get one at https://platform.deepseek.com/api_keys)
- Downloading a local GGUF model (when local or multi-provider is selected)
- Choosing the default model from the combined picker (OpenAI + DeepSeek + installed local models)

Local model IDs use the `local:` prefix (for example `local:qwen2.5-coder-7b-q4`). During setup you can pick from several GGUF options (Llama 3.2 3B, Phi-3.5 Mini, Qwen2.5 Coder 7B/14B, Mistral 7B, Gemma 2 9B, Granite 8B, DeepSeek Coder V2 Lite, and more). Weights are stored under `~/.goodboy/models/` (override with `GOODBOY_MODELS_DIR`).

DeepSeek cloud model IDs use their native names: `deepseek-v4-pro`, `deepseek-v4-flash`, plus legacy aliases `deepseek-chat` and `deepseek-reasoner`. GoodBoy talks to `https://api.deepseek.com` via the OpenAI-compatible Chat Completions endpoint; you only need a `DEEPSEEK_API_KEY` in `.env` to use them. DeepSeek thinking-mode reasoning is enabled automatically for the reasoning-capable models. See [`api-docs.deepseek.com`](https://api-docs.deepseek.com/) for the full DeepSeek API reference and current pricing.

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
cd python && python main.py
```

You’ll be prompted to enter a task. During the session you can also type special commands (see below).

### CLI flags

```bash
goodboy [OPTIONS]
```

Common options:
- `-f` / `--show-thoughts`: show model thoughts on each step (enabled by default)
- `--hide-thoughts`: hide model thoughts on each step
- `-v` / `--verbose`: show thoughts, tool/model switches, and task status lines
- `-m` / `--show-model`: print the model used on each agent response
- `-c` / `--show-commands`: print shell/Python commands the agent runs (not their output)
- `/commands`: toggle command visibility during a session (same as `-c`)
- `/reasoning`: set the default reasoning effort for reasoning models
- `-s` / `--stream-output` or `/stream`: stream each model response to the console as it is generated
- `-d` / `--debug`: show commands and stdout/stderr from shell and Python tool runs
- `-i` / `--debug-input`: print the full prompt sent to the model each turn
- `-o` / `--debug-output`: print the model’s raw response in full each turn

## Environment variables (.env)

GoodBoy reads configuration from `.env` (and also supports related exported env vars such as `SSL_CERT_FILE`).

Key variables:
- `OPENAI_API_KEY` – your OpenAI API key (optional when default model is `local:…` or `deepseek-…`)
- `DEEPSEEK_API_KEY` – your DeepSeek API key (required to use any `deepseek-…` model)
- `OPENAI_MODEL` – default model id (OpenAI id, `deepseek-…`, or `local:…` for on-device inference)
- `GOODBOY_MODELS_DIR` – directory for downloaded GGUF weights (default: `~/.goodboy/models`)
- `GOODBOY_SHOW_COMMANDS` – when `true`, show shell/Python commands (no output); updated by `/commands`
- `GOODBOY_REASONING_EFFORT` – default reasoning effort for reasoning models; updated by `/reasoning`

Agent behavior:
- `GOODBOY_MAX_TURNS` – maximum agent turns per task (default: `40`)
- `GOODBOY_TOOL_TIMEOUT_SEC` – timeout for tool execution (default: `180.0`, 3 minutes)
- `GOODBOY_MAX_CLARIFICATIONS` – max clarification turns (default: `3`)
- `GOODBOY_PLAN_MODE` – `auto` (plan required for complex tasks), `always`, or `off` (default: `auto`)
- `GOODBOY_VERIFY_BEFORE_COMPLETE` – block `task_complete` until tests pass after edits (default: `true`)
- `GOODBOY_WORKING_MEMORY_MAX` – cap on durable memory lines (default: `30`)
- `GOODBOY_CONTEXT_RECENT_FULL_TURNS` – how many recent agent turns keep full tool output in the prompt (default: `15`; local models use `GOODBOY_LOCAL_RECENT_FULL_TURNS`, default `3`)
- `GOODBOY_CONTEXT_TOKEN_BUDGET` – optional max estimated tokens for the per-turn transcript; when set, older turns are compacted further until under budget
- `GOODBOY_COMPLEX_WINDOW_MULTIPLIER` – multiply the recent-full window for complex tasks (keyword heuristic; default: `2.0`, capped at 30 turns)

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
- `python/agent/repl_commands.py`

## Tests

Run the test suite with:

```bash
cd python && pytest
```

## Project scripts

Console scripts are defined in [`python/pyproject.toml`](python/pyproject.toml):
- `goodboy` → `main:cli`
- `llm` → `llm:main`
