"""Which SAM2 model to use, and getting its weights onto this machine.

Nobody should have to hunt for SAM2's files:

* the **config** ships inside the installed ``sam2`` package and is handed to
  Hydra by NAME (``configs/sam2.1/sam2.1_hiera_s.yaml``), so there is no file to
  find — and no risk of picking the one-line symlink stub a Windows git clone
  leaves behind;
* the **checkpoint** (.pt) is downloaded once from Meta into
  ``~/.wormscope/checkpoints`` and reused from then on.

Only the model *size* is a choice (``settings.json`` -> ``"sam2_model"``).
Manually chosen files (“Choose SAM2 files manually…”) still win over all of
this — see :func:`wormscope.settings.sam2_paths`.
"""

import os
import urllib.request
from pathlib import Path

from .. import config, settings

BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"

# key -> (menu label, Hydra config name, checkpoint file, approx. download MB)
MODELS = {
    "tiny": ("Tiny — fastest", "configs/sam2.1/sam2.1_hiera_t.yaml",
             "sam2.1_hiera_tiny.pt", 160),
    "small": ("Small — good on a laptop", "configs/sam2.1/sam2.1_hiera_s.yaml",
              "sam2.1_hiera_small.pt", 185),
    "base_plus": ("Base+", "configs/sam2.1/sam2.1_hiera_b+.yaml",
                  "sam2.1_hiera_base_plus.pt", 325),
    "large": ("Large — most accurate, slow on CPU",
              "configs/sam2.1/sam2.1_hiera_l.yaml",
              "sam2.1_hiera_large.pt", 900),
}
DEFAULT_MODEL = "small"


class DownloadCancelled(Exception):
    """The user pressed Cancel while a checkpoint was downloading."""


def current_model():
    """The chosen model size: settings file -> SAM2_MODEL env var -> config.py
    -> "small". Unknown values fall back to the default."""
    for value in (settings.get("sam2_model"), os.environ.get("SAM2_MODEL"),
                  getattr(config, "SAM2_MODEL", "")):
        if value in MODELS:
            return value
    return DEFAULT_MODEL


def set_model(key):
    """Remember `key` as the model to use, and forget manually chosen files
    (they would otherwise keep winning over the new choice)."""
    if key not in MODELS:
        raise ValueError(f"unknown SAM2 model '{key}' (choose from "
                         f"{', '.join(MODELS)})")
    settings.save({"sam2_model": key, "sam2_cfg": "", "sam2_ckpt": ""})


def checkpoint_dir():
    """Where downloaded checkpoints go (WORMSCOPE_CHECKPOINTS overrides)."""
    override = os.environ.get("WORMSCOPE_CHECKPOINTS")
    if override:
        return Path(override).expanduser()
    return settings.settings_path().parent / "checkpoints"


def _search_dirs():
    """Folders checked for an already-present checkpoint, in order: the
    download folder, then wormscope/models/ (handy in a development checkout —
    *.pt files there are git-ignored)."""
    return [checkpoint_dir(), Path(__file__).resolve().parents[1] / "models"]


def find_checkpoint(key):
    """Path of this model's checkpoint if it is on disk already, else None."""
    name = MODELS[key][2]
    for folder in _search_dirs():
        p = folder / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def config_for_checkpoint(ckpt_path):
    """The Hydra config name that matches a known checkpoint filename, or ""."""
    name = Path(str(ckpt_path)).name
    for _label, cfg, ckpt, _mb in MODELS.values():
        if ckpt == name:
            return cfg
    return ""


def auto_pair(key=None):
    """(config name, checkpoint path or "") for the chosen model. An empty
    checkpoint means it still has to be downloaded."""
    key = key or current_model()
    ckpt = find_checkpoint(key)
    return MODELS[key][1], (str(ckpt) if ckpt else "")


def download(key=None, progress=None, timeout=30):
    """Download the model's checkpoint into checkpoint_dir() and return its path.

    `progress(done_bytes, total_bytes)` is called as data arrives; returning
    True cancels (raises DownloadCancelled). The file is written as *.part and
    only renamed when complete, so an interrupted download never looks finished.
    """
    key = key or current_model()
    _label, _cfg, name, _mb = MODELS[key]
    dest_dir = checkpoint_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[sam2] downloading {BASE_URL + name} -> {dest}")
    try:
        with urllib.request.urlopen(BASE_URL + name, timeout=timeout) as resp, \
                open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress is not None and progress(done, total):
                    raise DownloadCancelled()
        if total and done != total:
            raise OSError(f"download incomplete ({done} of {total} bytes)")
        os.replace(part, dest)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    print(f"[sam2] saved {dest}")
    return dest
