# Neo

A local coding agent for your terminal. Give it a task, it gets to work.

## Install

One command on macOS / Linux:

```bash
curl -fsSL https://goodboy.codes/install.sh | bash
```

Then open a new terminal and run:

```bash
neo setup
```

This will ask for your API key and let you pick a model (OpenAI, DeepSeek, or a local model that runs on your machine).

Full instructions and other install options are on [goodboy.codes](https://goodboy.codes/).

## Use it

From any project folder:

```bash
neo
```

Tell it what you want done. It runs shell and Python commands locally to get the job done, and asks before doing anything risky.

## Useful commands

- `neo setup` — change API key or default model
- `neo status` — show current settings
- `neo --help` — list all flags
- Inside a session: type `/model`, `/reasoning`, `/commands`, or `exit`

## Requirements

- Python 3.10+
- For local models: ~8–16 GB RAM and ~5 GB disk

## Links

- Website: [goodboy.codes](https://goodboy.codes/)
- Issues & source: this repo
