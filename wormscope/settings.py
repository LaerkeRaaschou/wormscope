"""
    Creates a JSON that remembers sam paths

    {
      "sam2_cfg":  "/Users/you/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
      "sam2_ckpt": "/Users/you/sam2/checkpoints/sam2.1_hiera_large.pt",
      "sam2_device": ""
    }
"""

import json
import os
from pathlib import Path

from . import config

DEFAULTS = {
    "sam2_cfg": "/Users/laerkesteinraaschou/Desktop/job/wormscope/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_ckpt": "/Users/laerkesteinraaschou/Desktop/job/wormscope/sam2/checkpoints/sam2.1_hiera_large.pt",
    "sam2_device": "",        # "" = auto: cuda -> mps -> cpu
}


def settings_path() -> Path:
    """Where the settings file lives."""
    #override = os.environ.get("WORMSCOPE_SETTINGS")
    #if override:
    #    return Path(override).expanduser()
    return Path.home() / ".wormscope" / "settings.json"


def load() -> dict:
    """The saved settings, merged over the defaults. Never raises."""
    values = dict(DEFAULTS)
    p = settings_path()
    try:
        if p.is_file():
            stored = json.loads(p.read_text())
            if isinstance(stored, dict):
                values.update({k: v for k, v in stored.items() if v is not None})
    except (OSError, ValueError) as err:
        print(f"[settings] could not read {p}: {err}")
    return values


def save(values: dict) -> Path:
    """Merge `values` into the settings file and return its path."""
    p = settings_path()
    current = load()
    current.update(values)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(current, indent=2) + "\n")
        print(f"[settings] saved {sorted(values)} -> {p}")
    except OSError as err:
        print(f"[settings] could not write {p}: {err}")
    return p


def get(key, default=None):
    return load().get(key) or default


def _looks_like_config_name(value: str) -> bool:
    p = Path(value)
    return not p.is_absolute() and p.suffix.lower() in (".yaml", ".yml")


def _first_usable(candidates, allow_config_name=False):
    for value, source in candidates:
        if not value:
            continue
        expanded = Path(value).expanduser()
        if expanded.exists():
            return str(expanded), source
        if allow_config_name and _looks_like_config_name(value):
            return str(value), f"{source} (Hydra config name)"
        print(f"[settings] {source} path does not exist, skipping: {value}")
    return "", ""


def sam2_paths(cfg=None, ckpt=None):
    saved = load()
    cfg_value, cfg_src = _first_usable([
        (cfg, "argument"),
        (saved.get("sam2_cfg"), "settings file"),
        (os.environ.get("SAM2_CFG"), "SAM2_CFG env var"),
        (getattr(config, "SAM2_CFG", ""), "config.py"),
    ], allow_config_name=True)
    ckpt_value, ckpt_src = _first_usable([
        (ckpt, "argument"),
        (saved.get("sam2_ckpt"), "settings file"),
        (os.environ.get("SAM2_CKPT"), "SAM2_CKPT env var"),
        (getattr(config, "SAM2_CKPT", ""), "config.py"),
    ])

    for env_key, chosen, src in (("SAM2_CFG", cfg_value, cfg_src),
                                 ("SAM2_CKPT", ckpt_value, ckpt_src)):
        env_val = os.environ.get(env_key)
        if env_val and src.startswith("settings file") and env_val != chosen:
            print(f"[settings] NOTE: ${env_key}={env_val} is ignored — the "
                  f"settings file ({settings_path()}) sets {chosen}. "
                  f"Use “SAM2 files…” to change it.")
    note = f"cfg from {cfg_src or 'nowhere'}, ckpt from {ckpt_src or 'nowhere'}"
    return cfg_value, ckpt_value, note


def sam2_device():
    """Preferred torch device, or "" for auto-detect."""
    return (get("sam2_device") or os.environ.get("SAM2_DEVICE")
            or getattr(config, "SAM2_DEVICE", "") or "")


def save_sam2_paths(cfg, ckpt, device=None):
    """Remember the SAM2 config + checkpoint (and optionally the device)."""
    values = {"sam2_cfg": str(cfg), "sam2_ckpt": str(ckpt)}
    if device is not None:
        values["sam2_device"] = str(device)
    return save(values)


# A plain string, not _main's docstring: the usage message is output the user
# reads, so it must not disappear when docstrings are trimmed (or when Python
# runs with -OO, which strips them all).
USAGE = """Inspect or set the saved settings without opening the GUI:

    python -m wormscope.settings
    python -m wormscope.settings --set-sam2 <config> <checkpoint>
    python -m wormscope.settings --clear-sam2
"""


def _main(argv):
    if len(argv) >= 2 and argv[0] == "--set-sam2":
        save_sam2_paths(argv[1], argv[2] if len(argv) > 2 else "")
    elif argv and argv[0] == "--clear-sam2":
        save({"sam2_cfg": "", "sam2_ckpt": ""})
    elif argv:
        print(USAGE)
        return 2
    print(f"settings file : {settings_path()}"
          f"{'' if settings_path().is_file() else '  (not created yet)'}")
    for k, v in load().items():
        print(f"  {k:12s}= {v or '(unset)'}")
    cfg, ckpt, source = sam2_paths()
    print(f"resolved      : {source}")
    print(f"  config      = {cfg or '(none)'}")
    print(f"  checkpoint  = {ckpt or '(none)'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
