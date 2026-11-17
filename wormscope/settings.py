"""
    Creates a JSON that remembers the SAM2 choices

    {
      "sam2_model": "small",
      "sam2_cfg":  "",
      "sam2_ckpt": "",
      "sam2_device": ""
    }

Normally only "sam2_model" matters: the config comes from the installed sam2
package and the checkpoint is downloaded once (segmentation/sam2_models.py).
"sam2_cfg"/"sam2_ckpt" are only set when files are chosen by hand, and then win.
"""

import json
import os
from pathlib import Path

from . import config

DEFAULTS = {
    "sam2_model": "",         # "" = default size (see segmentation/sam2_models.py)
    "sam2_cfg": "",
    "sam2_ckpt": "",
    "sam2_device": "",        # "" = auto: cuda -> mps -> cpu
    "check_updates": "on",    # "off" = never look for a newer version
    "last_update_check": "",  # date of the last look (once a day at most)
    "annotation_mode": "",    # last Automatic / Manual choice ("" = automatic)
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


def _is_stub_config(path: Path) -> bool:
    """True for a .yaml that isn't really a SAM2 config.

    The SAM2 repo has symlinked configs at its top level (sam2/sam2_hiera_l.yaml
    -> configs/sam2/sam2_hiera_l.yaml). A git clone on Windows usually turns
    each symlink into a one-line text file holding the target path; Hydra then
    loads an empty model and SAM2Base fails with "missing 3 required
    positional arguments". A real config always has `_target_` entries."""
    if path.suffix.lower() not in (".yaml", ".yml") or not path.is_file():
        return False
    try:
        return "_target_" not in path.read_text(errors="ignore")[:20000]
    except OSError:
        return False


def _first_usable(candidates, allow_config_name=False):
    for value, source in candidates:
        if not value:
            continue
        expanded = Path(value).expanduser()
        if expanded.exists():
            if _is_stub_config(expanded):
                print(f"[settings] {source} config is not a real SAM2 config "
                      f"(a symlink stub?), skipping: {value}")
                continue
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

    # Nothing (usable) chosen by hand: fall back to the automatic model. A
    # checkpoint we recognise by name gets its matching config, so an existing
    # .pt is reused instead of downloaded again.
    if not (cfg_value and ckpt_value):
        from .segmentation import sam2_models
        if ckpt_value and not cfg_value:
            derived = sam2_models.config_for_checkpoint(ckpt_value)
            if derived:
                cfg_value, cfg_src = derived, "matched to the checkpoint"
        if not (cfg_value and ckpt_value):
            key = sam2_models.current_model()
            cfg_value, ckpt_value = sam2_models.auto_pair(key)
            cfg_src = f"automatic (model '{key}')"
            ckpt_src = (f"automatic (model '{key}')" if ckpt_value
                        else f"automatic (model '{key}', not downloaded yet)")

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
    python -m wormscope.settings --model tiny|small|base_plus|large
    python -m wormscope.settings --download        # fetch the chosen model now
"""


def _main(argv):
    if len(argv) >= 2 and argv[0] == "--set-sam2":
        save_sam2_paths(argv[1], argv[2] if len(argv) > 2 else "")
    elif argv and argv[0] == "--clear-sam2":
        save({"sam2_cfg": "", "sam2_ckpt": ""})
    elif len(argv) >= 2 and argv[0] == "--model":
        from .segmentation import sam2_models
        sam2_models.set_model(argv[1])
    elif argv and argv[0] == "--download":
        from .segmentation import sam2_models
        sam2_models.download()
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
