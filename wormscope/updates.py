"""Is there a newer wormscope, and how to get it.

The version on GitHub is the newest release (or, without releases, the newest
tag) of REPO; ours is ``wormscope.__version__``. To publish an update: bump
``__version__`` in wormscope/__init__.py, push, and tag/release it as e.g.
``v3.2.0``.

Upgrading is only automated for the recommended install
(``uv tool install …``), where it is one safe command. A git checkout is a
working copy that may hold your own edits, so for that we only say what to run.
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__

REPO = "LaerkeRaaschou/wormscope"
API = f"https://api.github.com/repos/{REPO}"


def _get_json(url, timeout):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"wormscope/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def latest_version(timeout=4):
    """Newest published version as a string like "3.2.0". Raises on network
    problems (or a private repo, which GitHub answers with 404)."""
    try:
        tag = _get_json(f"{API}/releases/latest", timeout).get("tag_name", "")
    except urllib.error.HTTPError as err:
        if err.code != 404:                  # 404 = no releases: try tags
            raise
        tags = _get_json(f"{API}/tags", timeout)
        tag = tags[0]["name"] if tags else ""
    return tag.lstrip("vV")


def _as_tuple(version):
    parts = []
    for piece in str(version).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(remote, local=__version__):
    return bool(remote) and _as_tuple(remote) > _as_tuple(local)


def install_kind():
    """"uv-tool" (installed with `uv tool install`), "git" (a checkout run with
    `python main.py`) or "other"."""
    prefix = Path(sys.prefix)
    if prefix.parent.name == "tools" and "uv" in {p.lower() for p in prefix.parts}:
        return "uv-tool"
    if (Path(__file__).resolve().parents[1] / ".git").exists():
        return "git"
    return "other"


def find_uv():
    """The uv executable. A desktop shortcut may start us without the user's
    PATH, so uv's default install folders are checked too."""
    found = shutil.which("uv")
    if found:
        return found
    exe = "uv.exe" if os.name == "nt" else "uv"
    for folder in (Path.home() / ".local" / "bin", Path.home() / ".cargo" / "bin"):
        if (folder / exe).is_file():
            return str(folder / exe)
    return None


def how_to_update():
    """A sentence telling the user how to update this particular install."""
    kind = install_kind()
    if kind == "uv-tool":
        return "Run:  uv tool upgrade wormscope"
    if kind == "git":
        root = Path(__file__).resolve().parents[1]
        return (f"In {root} run:  git pull\n"
                "then reinstall the requirements if they changed.")
    return "Reinstall wormscope the way you originally installed it."


def can_self_update():
    return install_kind() == "uv-tool" and find_uv() is not None


def self_update(timeout=900):
    """Run `uv tool upgrade wormscope`. Returns (ok, output)."""
    uv = find_uv()
    if install_kind() != "uv-tool" or uv is None:
        return False, how_to_update()
    proc = subprocess.run([uv, "tool", "upgrade", "wormscope"],
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def update_after_exit():
    """Windows can't replace the files of a program that is running, so there
    the upgrade runs in its own console window that waits for wormscope to
    close first. The caller must quit the app right after this."""
    uv = find_uv()
    if install_kind() != "uv-tool" or uv is None:
        raise RuntimeError(how_to_update())
    steps = (f'timeout /t 3 /nobreak >nul & "{uv}" tool upgrade wormscope & '
             'echo. & echo Finished. You can start wormscope again. & pause')
    subprocess.Popen(f'cmd /c "{steps}"',
                     creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
