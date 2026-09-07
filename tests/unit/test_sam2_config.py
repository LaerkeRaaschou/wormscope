"""Turning the .yaml you picked in a file dialog into something Hydra accepts.

SAM2 composes its config with Hydra, initialised with the *sam2 package itself*
as the search path. Hydra therefore wants a package-relative name like
``configs/sam2.1/sam2.1_hiera_l.yaml``; handed an absolute path it strips the
leading slash and fails with ``Cannot find primary config 'Users/…'``.

A stand-in sam2 package stands in for the real (multi-GB) one.
"""

import sys
import types

import pytest

from wormscope.segmentation import (hydra_config_name, pair_warning,
                                    sam2_config_dir, sam2_package_dir)


@pytest.fixture
def fake_sam2(tmp_path, monkeypatch):
    """A directory laid out like an installed sam2 package."""
    pkg = tmp_path / "site-packages" / "sam2"
    (pkg / "configs" / "sam2").mkdir(parents=True)
    (pkg / "configs" / "sam2.1").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "configs" / "sam2" / "sam2_hiera_l.yaml").write_text("stub")
    (pkg / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml").write_text("stub")
    module = types.ModuleType("sam2")
    module.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "sam2", module)
    return pkg


@pytest.fixture
def loose_config(tmp_path):
    p = tmp_path / "my_configs" / "custom.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("stub")
    return p


def test_the_installed_package_is_found(fake_sam2):
    assert sam2_package_dir() == fake_sam2
    assert sam2_config_dir() == fake_sam2 / "configs"


def test_an_absolute_path_inside_the_package_becomes_a_relative_name(fake_sam2):
    """This is the exact failure: an absolute path from the file dialog."""
    cfg = fake_sam2 / "configs" / "sam2" / "sam2_hiera_l.yaml"
    name, overrides = hydra_config_name(cfg)
    assert name == "configs/sam2/sam2_hiera_l.yaml"
    assert not name.startswith("/"), "a leading slash is what Hydra chokes on"
    assert overrides == [], "no search path needed — it is inside the package"


def test_a_sam2_1_config_too(fake_sam2):
    cfg = fake_sam2 / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    assert hydra_config_name(cfg)[0] == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_a_name_that_is_already_a_hydra_name_is_passed_through(fake_sam2):
    name, overrides = hydra_config_name("configs/sam2.1/sam2.1_hiera_l.yaml")
    assert name == "configs/sam2.1/sam2.1_hiera_l.yaml" and overrides == []


def test_a_config_stored_elsewhere_keeps_its_name_and_adds_a_search_path(
        loose_config):
    """You are allowed to keep your configs wherever you like."""
    name, overrides = hydra_config_name(loose_config)
    assert name == "custom.yaml"
    assert overrides == [f"hydra.searchpath=[file://{loose_config.parent}]"]


def test_that_holds_even_when_sam2_is_installed(fake_sam2, loose_config):
    name, overrides = hydra_config_name(loose_config)
    assert name == "custom.yaml" and overrides


def test_a_home_relative_path_is_expanded():
    name, _overrides = hydra_config_name("~/nonexistent-cfg.yaml")
    assert "~" not in name, "Hydra would take the tilde literally"


# --- the config / checkpoint pair -------------------------------------------
def test_a_mismatched_pair_is_flagged_before_loading():
    """Otherwise it fails much later with a wall of missing state-dict keys."""
    warning = pair_warning("configs/sam2.1/sam2.1_hiera_l.yaml",
                           "/models/sam2.1_hiera_tiny.pt")
    assert warning, "large config + tiny checkpoint should be called out"


def test_a_matching_pair_is_not_flagged():
    assert not pair_warning("configs/sam2.1/sam2.1_hiera_l.yaml",
                            "/models/sam2.1_hiera_large.pt")
