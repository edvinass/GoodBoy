"""Application settings loaded from and saved to .env."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
ENV_FILE = ROOT_DIR / ".env"

OPENAI_API_KEY_VAR = "OPENAI_API_KEY"
OPENAI_MODEL_VAR = "OPENAI_MODEL"
DEEPSEEK_API_KEY_VAR = "DEEPSEEK_API_KEY"
GOODBOY_MAX_TURNS_VAR = "GOODBOY_MAX_TURNS"
GOODBOY_TOOL_TIMEOUT_SEC_VAR = "GOODBOY_TOOL_TIMEOUT_SEC"
GOODBOY_MAX_CLARIFICATIONS_VAR = "GOODBOY_MAX_CLARIFICATIONS"
GOODBOY_LOG_DIR_VAR = "GOODBOY_LOG_DIR"
GOODBOY_SESSION_LOG_VAR = "GOODBOY_SESSION_LOG"
GOODBOY_SSL_CA_BUNDLE_VAR = "GOODBOY_SSL_CA_BUNDLE"
GOODBOY_SSL_VERIFY_VAR = "GOODBOY_SSL_VERIFY"
GOODBOY_SHOW_COMMANDS_VAR = "GOODBOY_SHOW_COMMANDS"
GOODBOY_REASONING_EFFORT_VAR = "GOODBOY_REASONING_EFFORT"
GOODBOY_CONTEXT_RECENT_FULL_TURNS_VAR = "GOODBOY_CONTEXT_RECENT_FULL_TURNS"
GOODBOY_LOCAL_RECENT_FULL_TURNS_VAR = "GOODBOY_LOCAL_RECENT_FULL_TURNS"
GOODBOY_RESPONSE_CHAIN_VAR = "GOODBOY_RESPONSE_CHAIN"
GOODBOY_STRICT_JSON_VAR = "GOODBOY_STRICT_JSON"
GOODBOY_PLAN_MODE_VAR = "GOODBOY_PLAN_MODE"
GOODBOY_WORKING_MEMORY_MAX_VAR = "GOODBOY_WORKING_MEMORY_MAX"
GOODBOY_VERIFY_BEFORE_COMPLETE_VAR = "GOODBOY_VERIFY_BEFORE_COMPLETE"
GOODBOY_CONTEXT_TOKEN_BUDGET_VAR = "GOODBOY_CONTEXT_TOKEN_BUDGET"
GOODBOY_COMPLEX_WINDOW_MULTIPLIER_VAR = "GOODBOY_COMPLEX_WINDOW_MULTIPLIER"
GOODBOY_MODELS_DIR_VAR = "GOODBOY_MODELS_DIR"
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_MODELS_DIR = Path.home() / ".goodboy" / "models"
DEFAULT_PLAN_MODE = "auto"
DEFAULT_WORKING_MEMORY_MAX = 30
DEFAULT_MAX_TURNS = 500
DEFAULT_TOOL_TIMEOUT_SEC = 180.0
DEFAULT_MAX_CLARIFICATIONS = 3
DEFAULT_CONTEXT_RECENT_FULL_TURNS = 15
DEFAULT_LOCAL_RECENT_FULL_TURNS = 3
DEFAULT_COMPLEX_WINDOW_MULTIPLIER = 2.0

_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_env(*, override: bool = False) -> None:
    """Populate ``os.environ`` from ``.env``.

    By default, existing environment variables (shell exports, test
    monkeypatches, CI-provided values) take precedence over ``.env``. Pass
    ``override=True`` to force ``.env`` to win, e.g. after :func:`save_env`
    just rewrote a key and the running process should pick up the new value.
    """
    load_dotenv(ENV_FILE, override=override)


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
    load_env(override=True)
    try:
        from llm import clear_openai_client_cache

        clear_openai_client_cache()
    except ImportError:
        pass
    try:
        from agent.deepseek_llm import clear_deepseek_client_cache

        clear_deepseek_client_cache()
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


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str | None
    deepseek_api_key: str | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    tool_timeout_sec: float = DEFAULT_TOOL_TIMEOUT_SEC
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS
    session_log_enabled: bool = True
    session_log_dir: Path | None = None
    ssl_ca_bundle: str | None = None
    ssl_verify: bool = True
    show_commands: bool = False
    default_reasoning_effort: str | None = None
    context_recent_full_turns: int = DEFAULT_CONTEXT_RECENT_FULL_TURNS
    local_recent_full_turns: int = DEFAULT_LOCAL_RECENT_FULL_TURNS
    response_chain_enabled: bool = True
    strict_json_schema: bool = True
    plan_mode: str = DEFAULT_PLAN_MODE
    working_memory_max: int = DEFAULT_WORKING_MEMORY_MAX
    verify_before_complete: bool = True
    context_token_budget: int | None = None
    complex_window_multiplier: float = DEFAULT_COMPLEX_WINDOW_MULTIPLIER
    models_dir: Path = DEFAULT_MODELS_DIR

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
        models_dir_raw = os.getenv(GOODBOY_MODELS_DIR_VAR)
        models_dir = (
            Path(models_dir_raw).expanduser()
            if models_dir_raw and models_dir_raw.strip()
            else DEFAULT_MODELS_DIR
        )
        deepseek_key_raw = os.getenv(DEEPSEEK_API_KEY_VAR)
        deepseek_api_key = (
            deepseek_key_raw.strip()
            if deepseek_key_raw and deepseek_key_raw.strip()
            else None
        )
        return cls(
            openai_api_key=os.getenv(OPENAI_API_KEY_VAR),
            openai_model=os.getenv(OPENAI_MODEL_VAR),
            deepseek_api_key=deepseek_api_key,
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
            default_reasoning_effort=default_reasoning,
            context_recent_full_turns=max(
                1,
                _env_int(
                    GOODBOY_CONTEXT_RECENT_FULL_TURNS_VAR,
                    DEFAULT_CONTEXT_RECENT_FULL_TURNS,
                ),
            ),
            local_recent_full_turns=max(
                1,
                _env_int(
                    GOODBOY_LOCAL_RECENT_FULL_TURNS_VAR,
                    DEFAULT_LOCAL_RECENT_FULL_TURNS,
                ),
            ),
            response_chain_enabled=_env_bool(GOODBOY_RESPONSE_CHAIN_VAR, True),
            strict_json_schema=_env_bool(GOODBOY_STRICT_JSON_VAR, True),
            plan_mode=(
                os.getenv(GOODBOY_PLAN_MODE_VAR) or DEFAULT_PLAN_MODE
            ).strip().lower(),
            working_memory_max=max(
                1,
                _env_int(
                    GOODBOY_WORKING_MEMORY_MAX_VAR,
                    DEFAULT_WORKING_MEMORY_MAX,
                ),
            ),
            verify_before_complete=_env_bool(
                GOODBOY_VERIFY_BEFORE_COMPLETE_VAR, True
            ),
            context_token_budget=_env_optional_int(
                GOODBOY_CONTEXT_TOKEN_BUDGET_VAR
            ),
            complex_window_multiplier=_env_float(
                GOODBOY_COMPLEX_WINDOW_MULTIPLIER_VAR,
                DEFAULT_COMPLEX_WINDOW_MULTIPLIER,
            ),
            models_dir=models_dir,
        )

    @property
    def default_model(self) -> str:
        return self.openai_model or DEFAULT_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


def is_configured(cfg: Settings | None = None) -> bool:
    """True when any provider key is set, or the default model is an installed local model."""
    resolved = cfg or get_settings()
    if resolved.openai_api_key:
        return True
    if resolved.deepseek_api_key:
        return True
    try:
        from agent.local_llm import has_installed_local_model

        return has_installed_local_model(resolved.default_model)
    except ImportError:
        return False
