"""Tests for install.sh."""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"


def _run_install(env: dict[str, str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, **env}
    return subprocess.run(
        ["bash", str(INSTALL_SH)],
        cwd=cwd or REPO_ROOT,
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )


def test_install_sh_syntax():
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_install_from_source_dir(tmp_path: Path):
    result = _run_install(
        {
            "GOODBOY_SOURCE_DIR": str(REPO_ROOT),
            "GOODBOY_INSTALL_DIR": str(tmp_path / "ignored"),
            "GOODBOY_NO_PATH": "1",
            "HOME": str(tmp_path),
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    assert venv_python.is_file()
    assert "GoodBoy is installed." in result.stdout
    assert "goodboy setup" in result.stdout


def test_install_from_tarball(tmp_path: Path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    with tarfile.open(serve_dir / "goodboy.tar.gz", "w:gz") as tar:
        tar.add(REPO_ROOT / "goodboy", arcname="goodboy")
    install_root = tmp_path / "install"
    result = _run_install(
        {
            "GOODBOY_INSTALL_BASE_URL": serve_dir.as_uri(),
            "GOODBOY_INSTALL_DIR": str(install_root),
            "GOODBOY_NO_PATH": "1",
            "HOME": str(tmp_path),
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (install_root / "goodboy" / "pyproject.toml").is_file()
    assert "Downloading GoodBoy from" in result.stdout


def test_install_idempotent_with_existing_venv(tmp_path: Path):
    env = {
        "GOODBOY_SOURCE_DIR": str(REPO_ROOT),
        "GOODBOY_NO_PATH": "1",
        "HOME": str(tmp_path),
    }
    first = _run_install(env)
    assert first.returncode == 0, first.stderr + first.stdout
    second = _run_install(env)
    assert second.returncode == 0, second.stderr + second.stdout
    assert "Using existing venv" in second.stdout
