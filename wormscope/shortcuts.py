"""Create a desktop shortcut that starts wormscope — no terminal needed.

* Windows: a .lnk on the Desktop (made by PowerShell, so no extra packages).
* macOS:   a tiny wormscope.app on the Desktop.
* Linux:   a .desktop entry in the applications menu (and on the Desktop).

The shortcut runs the ``wormscope`` launcher that installing the package
creates (on Windows a windowed .exe, so no console pops up); without one it
falls back to ``python -m wormscope`` with this same interpreter.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

NAME = "wormscope"


def launcher_command():
    """The command (list of strings) that starts wormscope with this install."""
    here = Path(sys.executable).parent
    if os.name == "nt":
        for exe in (here / "wormscope.exe", here / "Scripts" / "wormscope.exe"):
            if exe.is_file():
                return [str(exe)]
        pythonw = here / "pythonw.exe"
        return [str(pythonw if pythonw.is_file() else sys.executable),
                "-m", "wormscope"]
    exe = here / "wormscope"
    if exe.is_file():
        return [str(exe)]
    return [sys.executable, "-m", "wormscope"]


def _desktop():
    return Path.home() / "Desktop"


def _windows_shortcut(cmd):
    def ps(s):                               # PowerShell single-quoted literal
        return "'" + s.replace("'", "''") + "'"
    script = (
        "$d = [Environment]::GetFolderPath('Desktop'); "
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
        f"(Join-Path $d '{NAME}.lnk')); "
        f"$s.TargetPath = {ps(cmd[0])}; "
        f"$s.Arguments = {ps(subprocess.list2cmdline(cmd[1:]))}; "
        f"$s.WorkingDirectory = {ps(str(Path.home()))}; "
        "$s.Save(); Write-Output (Join-Path $d '" + NAME + ".lnk')")
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-Command", script],
        capture_output=True, text=True, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return Path(out.stdout.strip().splitlines()[-1])


def _mac_app(cmd):
    app = _desktop() / f"{NAME}.app"
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    (app / "Contents" / "Info.plist").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>\n'
        f'  <key>CFBundleName</key><string>{NAME}</string>\n'
        f'  <key>CFBundleExecutable</key><string>{NAME}</string>\n'
        '  <key>CFBundleIdentifier</key><string>dk.aau.wormscope</string>\n'
        '  <key>CFBundlePackageType</key><string>APPL</string>\n'
        '</dict></plist>\n')
    launcher = macos / NAME
    launcher.write_text("#!/bin/bash\nexec " + shlex.join(cmd) + ' "$@"\n')
    launcher.chmod(0o755)
    return app


def _linux_desktop_entry(cmd):
    entry = ("[Desktop Entry]\nType=Application\n"
             f"Name={NAME}\nComment=Annotate and measure worms in videos\n"
             f"Exec={shlex.join(cmd)}\nTerminal=false\n"
             "Categories=Science;Graphics;\n")
    apps = Path.home() / ".local" / "share" / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    target = apps / f"{NAME}.desktop"
    target.write_text(entry)
    target.chmod(0o755)
    if _desktop().is_dir():
        on_desktop = _desktop() / f"{NAME}.desktop"
        on_desktop.write_text(entry)
        on_desktop.chmod(0o755)
        return on_desktop
    return target


def create_desktop_shortcut():
    """Create (or overwrite) the shortcut and return where it was put."""
    cmd = launcher_command()
    if os.name == "nt":
        return _windows_shortcut(cmd)
    if sys.platform == "darwin":
        return _mac_app(cmd)
    return _linux_desktop_entry(cmd)
