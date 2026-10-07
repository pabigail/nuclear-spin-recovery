"""Configuration of site memory, site-scaled offsets and tempering's inner moves.

The rule the rest of the configuration lives by applies here with more force:
a key that is silently ignored produces a completed run that sampled a model
nobody chose.  A relaxation width that was typed but not applied would give a
constrained fit labelled as a relaxed one.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from nuclear_spin_recovery import (
    RJMCMC,
    RWMH,
    GaussianOffset,
    ParallelTempering,
    RunConfig,
    SiteScaledOffset,
)

SITE_OFFSETS = {"algorithm": "offsets", "n_steps": 10, "radius": 1.0,
                "width": "site", "fraction_par": 0.1, "fraction_perp": 0.05,
                "floor": 0.5}


@pytest.fixture
def raw(nv2_path):
    return {
        "table": {"path": str(nv2_path), "strong_thresh": 750.0,
                  "weak_thresh": 5.0, "coupling_cutoff": 100.0},
        "experiment": [{"n_pulses": 16, "b_z": 311.0, "tau_start": 3.2e-5,
                        "tau_stop": 8.0e-3, "tau_count": 40}],
        "data": {"mode": "simulate", "truth_k": 4, "truth_seed": 7,
                 "noise": 0.002},
        "likelihood": {"sigma": 0.02},
        "state": {"lam": 3.0e-3, "n_stretch": 1.0, "k_max": 16},
        "schedule": [{"algorithm": "sites", "n_steps": 20, "radius": 6.0}],
        "ensemble": {"n_ensembles": 2, "n_steps": 60, "n_burn": 20,
                     "root_seed": 2026, "init_k": [2, 5]},
    }


def relaxed(raw, memory="site", **offsets):
    out = copy.deepcopy(raw)
    out["state"]["offset_memory"] = memory
    out["schedule"].append({**SITE_OFFSETS, **offsets})
    return out


# ---------------------------------------------------------------- defaults

def test_an_old_config_still_means_what_it_meant(raw):
    """No site memory, and an offsets block with an absolute Gaussian width."""
    raw["schedule"].append({"algorithm": "offsets", "n_steps": 5})
    config = RunConfig.from_dict(raw)
    assert config.state["offset_memory"] == "spin"
    block = config.schedule[1]
    assert block["width"] == "absolute"
    assert (block["fraction_par"], block["fraction_perp"], block["floor"]) == (0, 0, 0)
    assert block["prior"] == "gaussian"
    assert block["redraw_unoccupied"] is False
    table = config.build_table()
    proposal = config.build_schedule(table).steps[1].algorithm.proposal
    assert isinstance(proposal, GaussianOffset)
    assert not config._state(table, [0, 1]).has_site_memory


def test_resolved_record_round_trips(raw):
    first = RunConfig.from_dict(relaxed(raw))
    assert RunConfig.from_dict(first.to_dict()).to_dict() == first.to_dict()


# ------------------------------------------------------------ site scaling

def test_site_width_builds_the_site_scaled_kernel(raw):
    config = RunConfig.from_dict(relaxed(raw, prior="flat", redraw_unoccupied=True))
    table = config.build_table()
    proposal = config.build_schedule(table).steps[1].algorithm.proposal
    assert isinstance(proposal, SiteScaledOffset)
    assert proposal.fraction == (0.1, 0.05)
    assert proposal.floor == 0.5
    assert proposal.prior == "flat"
    assert proposal.redraw_unoccupied is True
    assert proposal.width(0, 0) == pytest.approx(
        max(0.5, 0.1 * abs(table.a_par[0])))


def test_site_memory_reaches_the_state(raw):
    config = RunConfig.from_dict(relaxed(raw))
    table = config.build_table()
    state = config._state(table, [0, 1])
    assert state.has_site_memory
    assert state.site_dA_par.shape == (1, len(table))


def test_site_width_defaults_to_no_relaxation(raw):
    """Fraction and floor are the author's to set; unset, nothing relaxes."""
    block = {"algorithm": "offsets", "n_steps": 5, "width": "site"}
    raw["state"]["offset_memory"] = "site"
    raw["schedule"].append(block)
    config = RunConfig.from_dict(raw)
    table = config.build_table()
    proposal = config.build_schedule(table).steps[1].algorithm.proposal
    assert proposal.width(0, 0) == 0.0 and proposal.width(0, 1) == 0.0
    assert proposal.prior == "gaussian"


def test_site_width_without_site_memory_raises(raw):
    with pytest.raises(ValueError, match="offset_memory"):
        RunConfig.from_dict(relaxed(raw, memory="spin"))


def test_site_only_keys_with_an_absolute_width_raise(raw):
    """They would be ignored, and the run would not be the one described."""
    with pytest.raises(ValueError, match="width = 'site'"):
        RunConfig.from_dict(relaxed(raw, width="absolute"))


@pytest.mark.parametrize("change, match", [
    ({"prior": "laplace"}, "prior"),
    ({"width": "relative"}, "width"),
    ({"fraction_par": -0.1}, "fraction_par"),
    ({"floor": -1.0}, "floor"),
])
def test_meaningless_offset_settings_raise(raw, change, match):
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict(relaxed(raw, **change))


def test_unknown_offset_memory_raises_naming_the_choices(raw):
    raw["state"]["offset_memory"] = "lattice"
    with pytest.raises(ValueError) as err:
        RunConfig.from_dict(raw)
    assert "spin" in str(err.value) and "site" in str(err.value)


def test_a_misspelt_offset_key_raises_naming_it(raw):
    with pytest.raises(ValueError, match="fraction_parr"):
        RunConfig.from_dict(relaxed(raw, fraction_parr=0.1))


# ------------------------------------------------------- tempering's inner

def tempered(raw, **tempering):
    out = relaxed(raw)
    out["schedule"].append({"algorithm": "rjmcmc", "n_steps": 5, "k_max": 16})
    out["schedule"].append({"algorithm": "tempering", "n_steps": 3,
                            "n_replicas": 4, **tempering})
    return out


def test_tempering_still_defaults_to_a_site_walk(raw):
    config = RunConfig.from_dict(tempered(raw))
    ladder = config.build_schedule(config.build_table()).steps[-1].algorithm
    assert isinstance(ladder, ParallelTempering)
    assert [s.algorithm.block.name for s in ladder.inner] == ["sites"]
    assert [s.n_steps for s in ladder.inner] == [1]


def test_tempering_can_run_every_move_on_every_rung(raw):
    """Escaping a wrong configuration needs the hot rungs to change the
    number of spins and the couplings, not only the sites."""
    config = RunConfig.from_dict(tempered(
        raw, inner=["rjmcmc", "sites", "offsets"], inner_steps=[1, 1, 4]))
    ladder = config.build_schedule(config.build_table()).steps[-1].algorithm
    inner = list(ladder.inner)
    assert isinstance(inner[0].algorithm, RJMCMC)
    assert isinstance(inner[1].algorithm, RWMH)
    assert isinstance(inner[2].algorithm.proposal, SiteScaledOffset)
    assert [s.n_steps for s in inner] == [1, 1, 4]
    # the rungs move as the single chain does
    assert inner[2].algorithm.proposal.fraction == (0.1, 0.05)


def test_inner_naming_an_undeclared_algorithm_raises(raw):
    with pytest.raises(ValueError, match="lam"):
        RunConfig.from_dict(tempered(raw, inner=["sites", "lam"]))


def test_tempering_cannot_temper_itself(raw):
    with pytest.raises(ValueError, match="itself"):
        RunConfig.from_dict(tempered(raw, inner=["tempering"]))


@pytest.mark.parametrize("steps", [[1], [1, 0], [1, 1, 1]])
def test_inner_steps_must_match_the_inner_schedule(raw, steps):
    with pytest.raises(ValueError, match="inner_steps"):
        RunConfig.from_dict(tempered(raw, inner=["sites", "offsets"],
                                     inner_steps=steps))


# ------------------------------------------------------------------- a run

def test_a_relaxed_run_is_reproducible_and_keeps_its_memory(raw, tmp_path):
    from nuclear_spin_recovery import run_ensemble

    config = RunConfig.from_dict(tempered(
        raw, inner=["rjmcmc", "sites", "offsets"]))
    first = run_ensemble(config, 0, out_dir=tmp_path / "a")
    second = run_ensemble(config, 0, out_dir=tmp_path / "b")
    for name in ("site_idx", "k", "dA_par", "dA_perp", "log_prob"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert np.any(first.dA_par != 0.0)
