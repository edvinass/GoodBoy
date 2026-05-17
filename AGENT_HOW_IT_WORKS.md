# How the GoodBoy Agent Works

This document explains how GoodBoy works internally: how a user task becomes model output, how that output is validated, how tools are executed, and how the harness decides whether to continue, ask for clarification, or finish.

## High-level flow

GoodBoy is a local autonomous coding-agent harness. A typical task follows this loop:

1. The user enters a task in the CLI.
2. GoodBoy builds a system prompt that defines the agent's behavior.
3. The model returns one JSON object representing a single action.
4. The harness validates that JSON against a strict schema.
5. If the action is a tool call, GoodBoy executes it in the local workspace.
6. The result is added back into the task context.
7. The loop repeats until the model emits `task_complete`, `need_user_input`, or `failed`.

The core idea is that the model does not produce free-form operational text on normal turns. It produces a structured step that the harness can parse, validate, and execute.

## Main components

Key files in the project:

- `main.py`: CLI entrypoint.
- `agent/harness.py`: interactive session and slash-command handling.
- `agent/loop.py`: core turn-by-turn orchestration.
- `agent/prompt.py`: system-prompt construction.
- `agent/types.py`: structured schema for agent responses.
- `agent/tools.py`: local shell and Python execution.
- `agent/file_tools.py`: structured file read/edit helpers.
- `llm.py`: structured LLM calls.
- `settings.py`: environment and runtime configuration.

A simple mental model:

- `harness.py` manages the user session.
- `loop.py` manages the execution state machine.
- `types.py` defines the contract between model and harness.
- `tools.py` and `file_tools.py` perform the actual work.

## Interactive session lifecycle

When you run `goodboy`, the CLI creates an `AgentHarness`.

The harness is responsible for:

- showing the startup UI,
- reading user input,
- handling slash commands like `/model`, `/reasoning`, `/clear`, `/commands`, `/autoswitch`, and `/stream`,
- opening the optional session log,
- calling `AgentLoop.run(...)` for real tasks,
- keeping conversation history and paused context.

This makes GoodBoy an interactive task session rather than a one-shot command wrapper.

## The structured agent protocol

The model must emit exactly one JSON object per turn. In `agent/types.py`, that object is parsed into `AgentStep`.

Important fields include:

- `action`
- `status`
- `thought`
- `command`
- `code`
- `path`
- `patch`
- `old_string`
- `new_string`
- `start_line`
- `end_line`
- `message`
- `tools`
- `model`
- `reasoning_effort`

Supported actions are:

- `run_shell`
- `run_python`
- `read_file`
- `apply_patch`
- `str_replace`
- `switch_model`
- `switch_tools`
- `switch_api` (deprecated alias)
- `need_user_input`
- `task_complete`
- `failed`

Validation rules enforce required fields for each action. Examples:

- `run_shell` requires `command`.
- `run_python` requires `code`.
- `read_file` requires `path`.
- `apply_patch` requires `path` and `patch`.
- `str_replace` requires `path`, `old_string`, and `new_string`.
- terminal actions require `message`.

Additional cross-field rules also apply, such as:

- `tools` is only valid on `switch_tools`.
- `model` is not allowed on `switch_tools`.
- `reasoning_effort` must match allowed values.

This strict schema is one of the main reliability features of GoodBoy.

## Prompt construction

GoodBoy builds its prompt in `agent/prompt.py`.

The prompt has two parts:

### Stable instructions

These include:

- agent identity,
- coding methodology,
- available tools,
- strict harness rules,
- hosted tool reference,
- cost/model guidance,
- optional model/reasoning guidance when auto-switching is enabled.

This section is intended to stay stable for prompt caching.

### Session suffix

This contains session-specific visibility instructions, such as whether the user can see:

- thoughts,
- commands,
- debug output.

That separation prevents small UI toggles from invalidating the full stable prompt prefix.

## What happens on each loop turn

`AgentLoop` in `agent/loop.py` is the core execution engine.

On each turn it roughly:

1. builds the current input from the task, prior turns, tool results, user replies, and parse errors,
2. chooses the current model and reasoning settings,
3. calls the LLM through structured-output helpers in `llm.py`,
4. parses the raw response into an `AgentStep`,
5. validates routing and tool/model constraints,
6. executes the action or returns a terminal result,
7. records the turn in context,
8. repeats until completion or failure.

The loop returns a `LoopResult` with outcomes like:

- `TASK_COMPLETE`
- `NEED_USER_INPUT`
- `FAILED`
- `MAX_TURNS`
- `STOPPED`

## Context and memory

GoodBoy stores task state in structured context objects.

Each `TurnRecord` can include:

- the parsed `AgentStep`,
- the model used,
- the reasoning effort used,
- the tool result,
- any parse error.

The context also tracks user replies, active hosted tools, and other state needed for iterative execution.

## Model calls and parsing

The LLM call path goes through `llm.py`, which requests structured JSON output matching the `agent_step` schema.

After the model replies, GoodBoy:

1. strips accidental markdown fences if present,
2. decodes the first JSON object,
3. validates it as an `AgentStep`,
4. applies additional routing checks.

If parsing fails, the error is fed back into context and the model gets another turn. If invalid JSON happens twice in a row, the harness stops with failure rather than looping forever.

## Tool execution

### Local tools

The primary local tools are implemented in `agent/tools.py`:

#### `run_shell`

- runs with `subprocess.run(..., shell=True)`,
- executes in the workspace,
- captures stdout and stderr,
- enforces a timeout,
- returns a structured `ToolResult`.

#### `run_python`

- runs a Python subprocess using the same interpreter as the harness,
- executes `python -c <code>`,
- captures stdout and stderr,
- enforces a timeout,
- returns a structured `ToolResult`.

Tool output is truncated with a head-and-tail strategy so both early exploration output and late failure output survive.

### Structured file tools

`agent/file_tools.py` implements safer file-oriented actions:

- `read_file`
- `str_replace`
- `apply_patch`

These tools resolve paths relative to the workspace and reject path escapes outside it.

Notable behavior:

- `read_file` can return numbered line ranges,
- `str_replace` fails if the target string is missing or ambiguous,
- `apply_patch` uses `patch -p0` when available and falls back to a Python single-hunk patcher if needed,
- successful edits include a unified diff in the tool output.

## Hosted tools

GoodBoy can also enable OpenAI hosted tools via `switch_tools`.

Examples include:

- `web_search`
- `file_search`
- `image_generation`
- `code_interpreter`
- `hosted_shell`
- `apply_patch`
- `skills`
- `computer_use`
- `mcp`
- `tool_search`

Important constraints:

- hosted tools are separate from local repo operations,
- `switch_tools` only changes tool availability for later model calls,
- the active model must support the requested hosted tools,
- changing hosted tools resets the response chain.

## Model routing and reasoning

GoodBoy supports two modes.

### Fixed session mode

When automatic model switching is off:

- the model is fixed for the session unless the user changes it,
- default reasoning is session-controlled,
- the agent may not set `model` or `reasoning_effort` in JSON,
- `switch_model` is disallowed.

### Automatic model switching mode

When auto-switching is on:

- the agent may choose allowlisted models between turns,
- it may also adjust reasoning effort,
- routing turns may use a cheaper capable model to decide the next model,
- chosen models are validated against the allowlist and active hosted tools.

## Clarification handling

If the model emits `need_user_input`, the harness treats it as a structured clarification request.

It will:

- track how many clarification turns have occurred,
- prevent repeated questions,
- fail after too many clarification turns,
- collect the user reply and add it back into context,
- add a proceed-style directive when the reply means “go ahead”.

This helps prevent the model from getting stuck in repetitive clarification loops.

## Terminal outcomes

Model-level terminal actions are:

- `task_complete`
- `need_user_input`
- `failed`

The harness itself can also stop because of:

- max turns reached,
- user interruption,
- repeated invalid JSON,
- empty clarification reply,
- API connection failure.

## Logging, UI, and visibility

GoodBoy includes optional session logging and a conversational terminal UI.

Depending on settings, the UI can show:

- status lines,
- thoughts,
- model labels,
- command previews,
- streamed model output,
- token usage,
- session notices.

The logging layer can record user input, LLM requests and responses, parse errors, agent steps, replies, and session state changes.

## Safety model and tradeoffs

GoodBoy is intentionally powerful and not sandboxed.

Key tradeoffs:

- shell commands run with full user privileges,
- Python code runs locally in a subprocess,
- strict schema validation constrains format, not capability,
- bounded retries and max-turn limits reduce runaway loops,
- secrets should not be printed or exfiltrated.

So GoodBoy is best understood as a strict local harness around a powerful model, not as a sandbox or permission system.

## Summary

In one sentence: GoodBoy works by forcing the model to act as a structured step generator inside a strict local harness, where every step is validated, optionally executed, recorded, and fed into the next turn until the task is completed or the system stops safely.
