import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepecohab.app import services, updater

#: Imports the updater by path, as the helper runs it, without the deepecohab package.
_IMPORT_UPDATER = (
	f"import sys; sys.path.insert(0, {str(Path(updater.__file__).parent)!r}); import updater\n"
)


@pytest.mark.parametrize(
	("installed", "on_pypi", "offered"),
	[
		("0.6.0rc2", "0.5.3", None),  # an rc above the last stable release
		("0.6.0rc2", "0.6.0", "0.6.0"),
		("0.5.3", "0.5.3", None),
		("0.5.2", "0.5.3", "0.5.3"),
	],
)
def test_newer_release(monkeypatch, installed, on_pypi, offered):
	body = json.dumps({"info": {"version": on_pypi}}).encode()
	monkeypatch.setattr(services.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
	monkeypatch.setattr(services, "version", lambda _: installed)
	services.newer_release.cache_clear()
	assert services.newer_release() == offered


def test_newer_release_offline(monkeypatch):
	def unreachable(*_a, **_k):
		raise OSError("no network")

	monkeypatch.setattr(services.urllib.request, "urlopen", unreachable)
	services.newer_release.cache_clear()
	assert services.newer_release() is None
	services.newer_release.cache_clear()


@pytest.mark.parametrize(
	("uv", "tool_dir", "offered"),
	[
		(None, "", False),  # no uv on PATH
		("uv", "{tools}", True),  # a uv tool install
		("uv", "{elsewhere}", False),  # a venv or a plain pip install
	],
)
def test_can_update_only_from_a_uv_tool_install(monkeypatch, tmp_path, uv, tool_dir, offered):
	tools = tmp_path / "tools"
	dirs = {"tools": tools, "elsewhere": tmp_path / "venvs"}
	monkeypatch.setattr(updater.shutil, "which", lambda _: uv)
	monkeypatch.setattr(
		updater.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=tool_dir.format(**dirs))
	)
	monkeypatch.setattr(updater.sys, "prefix", str(tools / "deepecohab"))
	assert updater.can_update() is offered


def test_commands_upgrade_the_app_extra_and_restart_with_the_same_options(monkeypatch):
	monkeypatch.setattr(updater.shutil, "which", lambda name: f"/bin/{name}")
	monkeypatch.setattr(updater.sys, "argv", ["/bin/deepecohab-app", "--port", "8060"])
	assert updater.commands() == (
		["/bin/uv", "tool", "upgrade", "deepecohab[app]"],
		["/bin/deepecohab-app", "--port", "8060"],
	)


def _append(log: Path, line: str) -> list[str]:
	"""A command appending ``line`` to ``log``, ``{pid}`` filled in: stands in for uv or the app."""
	code = (
		"import os, sys\n"
		"open(sys.argv[1], 'a').write(sys.argv[2].format(pid=os.getpid()) + chr(10))"
	)
	return [sys.executable, "-c", code, str(log), line]


def _lines(log: Path, count: int, timeout: float = 60) -> list[str]:
	"""``log``'s lines once it has ``count`` of them."""
	deadline = time.monotonic() + timeout
	lines = []
	while time.monotonic() < deadline:
		lines = log.read_text().splitlines() if log.exists() else []
		if len(lines) >= count:
			return lines
		time.sleep(0.2)
	pytest.fail(f"{log} holds {lines} after {timeout}s")


def test_helper_upgrades_once_the_app_exits_then_restarts_it(tmp_path):
	# The app outlives spawning the helper by a second: an upgrade logged before "app
	# exited" would mean the helper did not wait for it.
	log = tmp_path / "log"
	app = _IMPORT_UPDATER + (
		"import gc, json, os, time\n"
		"updater.spawn_helper(*json.loads(sys.argv[1]))\n"
		"gc.collect()\n"
		"time.sleep(1)\n"
		"open(sys.argv[2], 'a').write('app exited' + chr(10))\n"
		"os._exit(0)\n"
	)
	commands = [_append(log, "upgraded"), _append(log, "restarted")]
	subprocess.run([sys.executable, "-c", app, json.dumps(commands), str(log)], timeout=60)

	assert _lines(log, 3) == ["app exited", "upgraded", "restarted"]


def test_upgrade_retries_a_failed_attempt(monkeypatch, tmp_path):
	# Fails until its marker exists, as an upgrade blocked by a file not yet released would.
	marker = tmp_path / "released"
	code = "import pathlib, sys; m = pathlib.Path(sys.argv[1]); ok = m.exists(); m.touch()"
	flaky = [sys.executable, "-c", f"{code}; sys.exit(0 if ok else 1)", str(marker)]
	monkeypatch.setattr(updater.time, "sleep", lambda _: None)

	assert updater.upgrade(flaky)
	assert not updater.upgrade([sys.executable, "-c", "raise SystemExit(1)"])


@pytest.mark.skipif(sys.platform == "win32", reason="Windows updates through the helper")
def test_restart_in_place_upgrades_then_execs_the_app_in_the_same_process(tmp_path):
	log = tmp_path / "log"
	code = _IMPORT_UPDATER + "import json\nupdater.restart_in_place(*json.loads(sys.argv[1]))\n"
	commands = [_append(log, "upgraded"), _append(log, "restarted in {pid}")]
	process = subprocess.Popen([sys.executable, "-c", code, json.dumps(commands)])

	assert process.wait(timeout=60) == 0
	# The app took over the process that started the update, rather than a child of it.
	assert _lines(log, 2) == ["upgraded", f"restarted in {process.pid}"]
