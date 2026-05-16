"""Tests for the local development environment bootstrap script."""

import importlib.util
from pathlib import Path


def _load_dev_module():
    spec = importlib.util.spec_from_file_location("dev", Path(__file__).resolve().parents[1] / "dev.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dev = _load_dev_module()


def test_venv_python_posix(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(dev.sys, "platform", "darwin")
    monkeypatch.setattr(dev, "VENV_DIR", tmp_path / ".venv")

    assert dev.venv_python() == tmp_path / ".venv" / "bin" / "python"


def test_venv_python_windows(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(dev.sys, "platform", "win32")
    monkeypatch.setattr(dev, "VENV_DIR", tmp_path / ".venv")

    assert dev.venv_python() == tmp_path / ".venv" / "Scripts" / "python.exe"


def test_create_venv_uses_existing_directory(monkeypatch, tmp_path: Path, capsys):
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    monkeypatch.setattr(dev, "VENV_DIR", venv_dir)

    def fail_create(*_args, **_kwargs):
        raise AssertionError("venv.create should not be called for an existing venv")

    monkeypatch.setattr(dev.venv, "create", fail_create)

    dev.create_venv()

    assert f"Using existing venv: {venv_dir}" in capsys.readouterr().out


def test_create_venv_creates_missing_directory(monkeypatch, tmp_path: Path, capsys):
    venv_dir = tmp_path / ".venv"
    monkeypatch.setattr(dev, "VENV_DIR", venv_dir)
    calls = []

    def fake_create(path: Path, *, with_pip: bool):
        calls.append((path, with_pip))

    monkeypatch.setattr(dev.venv, "create", fake_create)

    dev.create_venv()

    assert calls == [(venv_dir, True)]
    assert f"Creating venv: {venv_dir}" in capsys.readouterr().out


def test_install_editable_runs_pip_install_from_project_root(monkeypatch, tmp_path: Path):
    root = tmp_path / "project"
    python = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(dev, "ROOT", root)
    monkeypatch.setattr(dev, "venv_python", lambda: python)
    calls = []

    def fake_run(args, *, cwd: Path, check: bool):
        calls.append((args, cwd, check))

    monkeypatch.setattr(dev.subprocess, "run", fake_run)

    dev.install_editable()

    assert calls == [([str(python), "-m", "pip", "install", "-e", "."], root, True)]


def test_main_bootstraps_and_prints_posix_activation(monkeypatch, tmp_path: Path, capsys):
    venv_dir = tmp_path / ".venv"
    monkeypatch.setattr(dev, "VENV_DIR", venv_dir)
    monkeypatch.setattr(dev.sys, "platform", "linux")
    calls = []
    monkeypatch.setattr(dev, "create_venv", lambda: calls.append("create_venv"))
    monkeypatch.setattr(dev, "install_editable", lambda: calls.append("install_editable"))

    dev.main()

    assert calls == ["create_venv", "install_editable"]
    out = capsys.readouterr().out
    assert "Installing package in editable mode..." in out
    assert f"source {venv_dir}/bin/activate" in out
    assert "Then run: python main.py  or  goodboy" in out


def test_main_prints_windows_activation(monkeypatch, tmp_path: Path, capsys):
    venv_dir = tmp_path / ".venv"
    monkeypatch.setattr(dev, "VENV_DIR", venv_dir)
    monkeypatch.setattr(dev.sys, "platform", "win32")
    monkeypatch.setattr(dev, "create_venv", lambda: None)
    monkeypatch.setattr(dev, "install_editable", lambda: None)

    dev.main()

    assert rf"  {venv_dir}\Scripts\activate" in capsys.readouterr().out
