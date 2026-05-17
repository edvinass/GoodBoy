# Project memory for GoodBoy

GoodBoy loads this file (or `.goodboy/memory.md` / `GOODBOY.md`) into every task.

## Verification

Run tests before `task_complete`:

```bash
pytest tests/ -q
```

## Module map

- `agent/loop.py` — turn orchestration
- `agent/context.py` — session state and prompts
- `agent/types.py` — JSON agent protocol

## Conventions

- Match existing test style in `tests/`
- Prefer `str_replace` for small edits; `apply_patch` for multi-line diffs

## Complex tasks

For refactors or multi-file work, call `update_plan` with at least two items before editing files.
