"""Where the SAM2 config and checkpoint come from, and why.

Four sources, in this order: an argument, the settings file, the environment,
config.py. The order is the whole feature — picking the files once in the GUI
has to beat an ``export SAM2_CFG=…`` left over in somebody's shell profile, or
the paths would not really be remembered.

conftest.py points WORMSCOPE_SETTINGS at a temporary file for the whole
session, so none of this can touch your real ~/.wormscope/settings.json.
"""

import os

import pytest

from wormscope import settings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """A settings file and an environment with nothing left in them."""
    monkeypatch.setenv("WORMSCOPE_SETTINGS", str(tmp_path / "settings.json"))
    for var in ("SAM2_CFG", "SAM2_CKPT", "SAM2_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(settings.config, "SAM2_CFG", "", raising=False)
    monkeypatch.setattr(settings.config, "SAM2_CKPT", "", raising=False)
    return tmp_path


@pytest.fixture
def sam2_files(clean_env):
    cfg = clean_env / "sam2.1_hiera_l.yaml"
    ckpt = clean_env / "sam2.1_hiera_large.pt"
    cfg.write_text("stub")
    ckpt.write_bytes(b"stub")
    return cfg, ckpt


def test_the_settings_file_lives_where_the_env_var_says(clean_env):
    assert settings.settings_path() == clean_env / "settings.json"


def test_nothing_configured_resolves_to_nothing(clean_env):
    cfg, ckpt, _note = settings.sam2_paths()
    assert (cfg, ckpt) == ("", ""), \
        "no guessing — the GUI answers this by opening the file picker"


def test_environment_variables_are_used(clean_env, sam2_files, monkeypatch):
    cfg_file, ckpt_file = sam2_files
    monkeypatch.setenv("SAM2_CFG", str(cfg_file))
    monkeypatch.setenv("SAM2_CKPT", str(ckpt_file))
    cfg, ckpt, note = settings.sam2_paths()
    assert cfg == str(cfg_file) and ckpt == str(ckpt_file)
    assert "env var" in note, "the note says where each value came from"


def test_the_settings_file_outranks_the_environment(clean_env, sam2_files,
                                                    monkeypatch):
    cfg_file, ckpt_file = sam2_files
    other = clean_env / "other.yaml"
    other.write_text("stub")
    monkeypatch.setenv("SAM2_CFG", str(cfg_file))
    settings.save_sam2_paths(other, ckpt_file)
    cfg, _ckpt, note = settings.sam2_paths()
    assert cfg == str(other), "what you picked in the GUI wins"
    assert "settings file" in note


def test_an_explicit_argument_outranks_everything(clean_env, sam2_files):
    cfg_file, ckpt_file = sam2_files
    settings.save_sam2_paths(cfg_file, ckpt_file)
    other = clean_env / "arg.yaml"
    other.write_text("stub")
    cfg, _ckpt, note = settings.sam2_paths(cfg=other)
    assert cfg == str(other) and "argument" in note


def test_settings_survive_a_fresh_load(clean_env, sam2_files):
    cfg_file, ckpt_file = sam2_files
    settings.save_sam2_paths(cfg_file, ckpt_file)
    assert settings.load()["sam2_ckpt"] == str(ckpt_file)
    assert "sam2_ckpt" in (clean_env / "settings.json").read_text(), \
        "plain JSON you can edit by hand"


def test_a_path_that_no_longer_exists_is_skipped(clean_env, sam2_files,
                                                 monkeypatch):
    """Moving the checkpoint should fall through to the next source, not fail
    with a confusing model-loading error."""
    cfg_file, ckpt_file = sam2_files
    settings.save_sam2_paths(clean_env / "moved-away.yaml", ckpt_file)
    monkeypatch.setenv("SAM2_CFG", str(cfg_file))
    cfg, _ckpt, note = settings.sam2_paths()
    assert cfg == str(cfg_file) and "env var" in note


def test_a_hydra_config_name_is_accepted_even_though_no_such_file_exists(
        clean_env, sam2_files):
    """`configs/sam2.1/sam2.1_hiera_l.yaml` is resolved inside the installed
    sam2 package at load time, so it never exists relative to the cwd. An
    existence check that rejected it broke `export SAM2_CFG=<hydra name>`."""
    _cfg_file, ckpt_file = sam2_files
    name = "configs/sam2.1/sam2.1_hiera_l.yaml"
    settings.save_sam2_paths(name, ckpt_file)
    cfg, _ckpt, note = settings.sam2_paths()
    assert cfg == name
    assert "Hydra config name" in note, "and the note says that is what it is"


def test_a_checkpoint_is_never_treated_as_a_config_name(clean_env):
    """The leniency above is for .yaml only; a missing .pt really is missing."""
    settings.save_sam2_paths("configs/x.yaml", "checkpoints/missing.pt")
    _cfg, ckpt, _note = settings.sam2_paths()
    assert ckpt == ""


def test_clearing_the_saved_paths_falls_back_to_the_environment(
        clean_env, sam2_files, monkeypatch):
    cfg_file, ckpt_file = sam2_files
    settings.save_sam2_paths(clean_env / "other.yaml", ckpt_file)
    settings.save({"sam2_cfg": "", "sam2_ckpt": ""})
    monkeypatch.setenv("SAM2_CFG", str(cfg_file))
    monkeypatch.setenv("SAM2_CKPT", str(ckpt_file))
    cfg, ckpt, _note = settings.sam2_paths()
    assert (cfg, ckpt) == (str(cfg_file), str(ckpt_file))


def test_a_shadowed_env_var_is_reported_out_loud(clean_env, sam2_files,
                                                 monkeypatch, capsys):
    """Silently ignoring $SAM2_CFG is baffling from the outside, so the app
    prints a NOTE explaining which value won."""
    cfg_file, ckpt_file = sam2_files
    other = clean_env / "other.yaml"
    other.write_text("stub")
    settings.save_sam2_paths(other, ckpt_file)
    monkeypatch.setenv("SAM2_CFG", str(cfg_file))
    settings.sam2_paths()
    assert "NOTE" in capsys.readouterr().out


def test_a_corrupt_settings_file_does_not_crash_the_app(clean_env):
    (clean_env / "settings.json").write_text("{not json")
    assert settings.load() == settings.DEFAULTS


def test_the_device_is_auto_unless_asked(clean_env, monkeypatch):
    assert settings.sam2_device() == "", "empty means auto: cuda -> mps -> cpu"
    settings.save({"sam2_device": "cpu"})
    assert settings.sam2_device() == "cpu"


def test_saving_merges_rather_than_replaces(clean_env):
    settings.save({"sam2_device": "cpu"})
    settings.save({"sam2_cfg": "x.yaml"})
    values = settings.load()
    assert values["sam2_device"] == "cpu" and values["sam2_cfg"] == "x.yaml"


def test_the_real_user_settings_file_is_never_touched():
    """A guard on the guard: if this ever fails, the suite is writing into
    somebody's home directory."""
    assert "wormscope-tests-" in os.environ["WORMSCOPE_SETTINGS"]


# --- the command-line front end ---------------------------------------------
def test_the_cli_prints_what_is_set_and_what_wins(clean_env, sam2_files, capsys):
    """`python -m wormscope.settings` — for checking the paths without opening
    the GUI, which is how you debug a checkpoint that will not load."""
    cfg_file, ckpt_file = sam2_files
    settings.save_sam2_paths(cfg_file, ckpt_file)
    assert settings._main([]) == 0
    out = capsys.readouterr().out
    assert str(cfg_file) in out and "resolved" in out


def test_the_cli_can_set_the_paths(clean_env, sam2_files):
    cfg_file, ckpt_file = sam2_files
    assert settings._main(["--set-sam2", str(cfg_file), str(ckpt_file)]) == 0
    assert settings.load()["sam2_cfg"] == str(cfg_file)


def test_the_cli_can_clear_them(clean_env, sam2_files):
    cfg_file, ckpt_file = sam2_files
    settings.save_sam2_paths(cfg_file, ckpt_file)
    assert settings._main(["--clear-sam2"]) == 0
    assert settings.load()["sam2_cfg"] == ""


def test_an_unknown_flag_prints_the_usage(clean_env, capsys):
    assert settings._main(["--what"]) == 2
    assert "--set-sam2" in capsys.readouterr().out
