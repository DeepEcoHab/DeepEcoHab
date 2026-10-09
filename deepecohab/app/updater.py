"""Upgrade DeepEcoHab and start it again.

Windows cannot replace a file a running process has loaded, so there the app hands the
upgrade to a helper and exits: :func:`spawn_helper` runs this file as a script, which
waits for the app to exit, upgrades and starts the app again in its own console. That
is why this module is stdlib only and the helper runs on the base interpreter - importing
the deepecohab package would load polars and dash from the environment being upgraded.
POSIX replaces files by rename, so there the app upgrades in place and execs itself.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

# Keep each pipe's write end alive until this process exits. On Windows, Popen
# is not retained by subprocess's active-child list when the caller discards it.
_helpers: list[subprocess.Popen] = []


def can_update() -> bool:
	"""Whether this app runs from a ``uv tool install``, which ``uv tool upgrade`` updates."""
	uv = shutil.which("uv")
	if uv is None:
		return False
	tools = subprocess.run([uv, "tool", "dir"], capture_output=True, text=True).stdout.strip()
	return bool(tools) and Path(sys.prefix).resolve().parent == Path(tools).resolve()


def commands() -> tuple[list[str], list[str]]:
	"""The upgrade command, and the command that starts the app with its current options."""
	app = shutil.which("deepecohab-app")
	start = [app] if app else [sys.executable, "-m", "deepecohab.app"]
	return [shutil.which("uv") or "uv", "tool", "upgrade", "deepecohab[app]"], [
		*start,
		*sys.argv[1:],
	]


def upgrade(command: list[str], output: TextIO | None = None, attempts: int = 3) -> bool:
	"""Run ``command`` until it succeeds, a few times at most, writing to ``output``.

	Windows can release an exited process's files a moment after it is gone.
	"""
	for attempt in range(attempts):
		if attempt:
			time.sleep(2)
		if subprocess.run(command, stdout=output, stderr=output).returncode == 0:
			return True
	print("Update failed; starting the installed version.", file=output, flush=True)
	return False


def spawn_helper(upgrade_command: list[str], app_command: list[str]) -> subprocess.Popen:
	"""Start the helper that upgrades and restarts the app once this process exits.

	The helper learns of the exit from its stdin: the OS closes this end of the pipe when
	the process goes, whichever way it goes.
	"""
	helper = subprocess.Popen(
		[
			getattr(sys, "_base_executable", sys.executable),
			__file__,
			json.dumps([upgrade_command, app_command]),
		],
		stdin=subprocess.PIPE,
		creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
	)
	_helpers.append(helper)
	return helper


def restart_in_place(upgrade_command: list[str], app_command: list[str]) -> None:
	"""Upgrade, then replace this process with the app: same PID, same terminal."""
	upgrade(upgrade_command)
	sys.stdout.flush()
	os.execv(app_command[0], app_command)


def update() -> None:
	"""Upgrade DeepEcoHab and start it again; this process does not return."""
	upgrade_command, app_command = commands()
	if sys.platform == "win32":
		spawn_helper(upgrade_command, app_command)
		os._exit(0)
	restart_in_place(upgrade_command, app_command)


def _helper() -> None:
	upgrade_command, app_command = json.loads(sys.argv[1])
	# Popen gave this process the app's stdout, whose console closes with the app, so
	# on Windows everything writes to the console the helper was given instead. The app
	# started here keeps it, as the console it runs in from now on.
	output = open("CONOUT$", "w") if sys.platform == "win32" else sys.stdout  # noqa: SIM115
	print("Waiting for DeepEcoHab to close...", file=output, flush=True)
	sys.stdin.read()
	upgrade(upgrade_command, output)
	sys.exit(subprocess.run(app_command, stdout=output, stderr=output).returncode)


if __name__ == "__main__":
	_helper()
