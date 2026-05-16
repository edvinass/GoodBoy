"""Application settings loaded from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"


def load_env() -> None:
    load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None

    @classmethod
    def from_env(cls) -> Settings:
        load_env()
        return cls(openai_api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
