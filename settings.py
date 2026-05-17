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
GOODBOY_MAX_TURNS_VAR = "GOODBOY_MAX_TURNS"
GOODBOY_TOOL_TIMEOUT_SEC_VAR = "GOODBOY_TOOL_TIMEOUT_SEC"
GOODBOY_MAX_CLARIFICATIONS_VAR = "GOODBOY_MAX_CLARIFICATIONS"
GOODBOY_LOG_DIR_VAR = "GOODBOY_LOG_DIR"
GOODBOY_SESSION_LOG_VAR = "GOODBOY_SESSION_LOG"
GOODBOY_SSL_CA_BUNDLE_VAR = "GOODBOY_SSL_CA_BUNDLE"
GOODBOY_SSL_VERIFY_VAR = "GOODBOY_SSL_VERIFY"
GOODBOY_SHOW_COMMANDS_VAR = "GOODBOY_SHOW_COMMANDS"
GOODBOY_AUTO_MODEL_SWITCH_VAR = "GOODBOY_AUTO_MODEL_SWITCH"
GOODBOY_REASONING_EFFORT_VAR = "GOODBOY_REASONING_EFFORT"
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_MAX_TURNS = 40
DEFAULT_TOOL_TIMEOUT_SEC = 120.0
DEFAULT_MAX_CLARIFICATIONS = 3

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
    try:
        from llm import clear_openai_client_cache

        clear_openai_client_cache()
    except ImportError:
        pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str | None
    max_turns: int = DEFAULT_MAX_TURNS
    tool_timeout_sec: float = DEFAULT_TOOL_TIMEOUT_SEC
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS
    session_log_enabled: bool = True
    session_log_dir: Path | None = None
    ssl_ca_bundle: str | None = None
    ssl_verify: bool = True
    show_commands: bool = False
    auto_model_switch: bool = False
    default_reasoning_effort: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        load_env()
        log_dir_raw = os.getenv(GOODBOY_LOG_DIR_VAR)
        log_dir = Path(log_dir_raw).expanduser() if log_dir_raw and log_dir_raw.strip() else None
        ssl_ca = (
            os.getenv(GOODBOY_SSL_CA_BUNDLE_VAR)
            or os.getenv("SSL_CERT_FILE")
            or os.getenv("REQUESTS_CA_BUNDLE")
        )
        ssl_ca_bundle = ssl_ca.strip() if ssl_ca and ssl_ca.strip() else None
        reasoning_raw = os.getenv(GOODBOY_REASONING_EFFORT_VAR)
        default_reasoning: str | None = None
        if reasoning_raw is not None and reasoning_raw.strip():
            default_reasoning = reasoning_raw.strip()
        return cls(
            openai_api_key=os.getenv(OPENAI_API_KEY_VAR),
            openai_model=os.getenv(OPENAI_MODEL_VAR),
            max_turns=_env_int(GOODBOY_MAX_TURNS_VAR, DEFAULT_MAX_TURNS),
            tool_timeout_sec=_env_float(
                GOODBOY_TOOL_TIMEOUT_SEC_VAR, DEFAULT_TOOL_TIMEOUT_SEC
            ),
            max_clarifications=_env_int(
                GOODBOY_MAX_CLARIFICATIONS_VAR, DEFAULT_MAX_CLARIFICATIONS
            ),
            session_log_enabled=_env_bool(GOODBOY_SESSION_LOG_VAR, True),
            session_log_dir=log_dir,
            ssl_ca_bundle=ssl_ca_bundle,
            ssl_verify=_env_bool(GOODBOY_SSL_VERIFY_VAR, True),
            show_commands=_env_bool(GOODBOY_SHOW_COMMANDS_VAR, False),
            auto_model_switch=_env_bool(GOODBOY_AUTO_MODEL_SWITCH_VAR, False),
            default_reasoning_effort=default_reasoning,
        )

    @property
    def default_model(self) -> str:
        return self.openai_model or DEFAULT_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
