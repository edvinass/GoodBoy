"""Application settings loaded from and saved to .env."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"

OPENAI_API_KEY_VAR = "OPENAI_API_KEY"
OPENAI_MODEL_VAR = "OPENAI_MODEL"
DEFAULT_MODEL = "gpt-4o-mini"

_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_env() -> None:
    load_dotenv(ENV_FILE, override=True)


def _parse_env_lines(lines: list[str]) -> tuple[list[str], dict[str, str]]:
    """Return preserved lines (comments/blanks) and key→value map from assignments."""
    preserved: list[str] = []
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            preserved.append(line)
            continue
        match = _ENV_LINE.match(stripped)
        if match:
            values[match.group(1)] = match.group(2)
        else:
            preserved.append(line)
    return preserved, values


def save_env(updates: dict[str, str]) -> None:
    """Write or update keys in .env; other keys and comments are kept."""
    lines: list[str] = []
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        text = ENV_FILE.read_text(encoding="utf-8")
        lines = text.splitlines()
        _, values = _parse_env_lines(lines)

    values.update(updates)

    out: list[str] = []
    written: set[str] = set()
    for line in lines:
        stripped = line.strip()
        match = _ENV_LINE.match(stripped) if stripped and not stripped.startswith("#") else None
        if match:
            key = match.group(1)
            if key in updates:
                out.append(f"{key}={updates[key]}")
                written.add(key)
            else:
                out.append(line)
        else:
            out.append(line)

    for key, value in updates.items():
        if key not in written:
            if out and out[-1].strip():
                out.append("")
            out.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    get_settings.cache_clear()
    load_env()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str | None

    @classmethod
    def from_env(cls) -> Settings:
        load_env()
        return cls(
            openai_api_key=os.getenv(OPENAI_API_KEY_VAR),
            openai_model=os.getenv(OPENAI_MODEL_VAR),
        )

    @property
    def default_model(self) -> str:
        return self.openai_model or DEFAULT_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
