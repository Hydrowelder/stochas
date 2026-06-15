"""Tests for StochasBase."""

import numpy as np

from stochas import (
    DesignFloat,
    DistName,
    NamedValue,
    NormalDistribution,
    StochasBase,
)
from stochas.named_value import ValueName


def test_sample_dist_registers_distribution_and_named_value():
    """Ensure sample_dist registers both the distribution and the sampled value."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    nv = sb.sample_dist(dist)

    assert "x" in sb.dists
    assert "x" in sb.named
    assert sb.named["x"] is nv


def test_sample_dist_returns_existing_named_value():
    """Ensure repeated sampling without force returns the already-registered value."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    nv1 = sb.sample_dist(dist)
    nv2 = sb.sample_dist(dist)

    assert nv2 is nv1
    assert sb.named["x"] is nv1


def test_sample_dist_force_overwrites_named_value():
    """Ensure force=True overwrites both the distribution and named value registries."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    nv1 = sb.sample_dist(dist)
    nv2 = sb.sample_dist(dist, force=True)

    assert nv2 is not nv1
    assert sb.named["x"] is nv2
    assert sb.dists["x"] is dist


def test_sample_design_registers_value():
    """Ensure sample_design registers the design value and returns its stored value."""
    sb = StochasBase()
    dv = DesignFloat(name=ValueName("p"), low=0, high=10, stored_value=5.0)

    result = sb.sample_design(dv)

    assert result == 5.0
    assert "p" in sb.design
    assert "p" in sb.named


def test_sample_design_uses_existing_override():
    """Ensure an existing entry in `named` is used instead of the design value's own value."""
    sb = StochasBase()
    sb.named.update(
        NamedValue[np.ndarray](name=ValueName("q"), stored_value=np.array([3.5]))
    )
    dv = DesignFloat(name=ValueName("q"), low=0, high=10, stored_value=5.0)

    result = sb.sample_design(dv)

    assert result == 3.5
    assert "q" not in sb.design or sb.design["q"] is dv


def test_with_overrides_and_with_override():
    """Ensure with_overrides and with_override both set the named dict and return self."""
    sb = StochasBase()
    overrides = sb.named.__class__()
    overrides.update(
        NamedValue[np.ndarray](name=ValueName("r"), stored_value=np.array([1.0]))
    )

    result = sb.with_overrides(overrides)
    assert result is sb
    assert sb.named is overrides

    other = sb.named.__class__()
    result = sb.with_override(other)
    assert result is sb
    assert sb.named is other


def test_with_seed_and_with_trial_num_propagate_to_dists():
    """Ensure with_seed and with_trial_num update the model and all registered distributions."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("z"), mu=0, sigma=1)
    sb.dists.update(dist)

    result = sb.with_seed(42)
    assert result is sb
    assert sb.seed == 42
    assert sb.dists["z"].seed == 42

    result = sb.with_trial_num(3)
    assert result is sb
    assert sb.trial_num == 3
    assert sb.dists["z"].trial_num == 3


def test_is_nominal():
    """Ensure is_nominal reflects whether trial_num matches NOMINAL_TRIAL_NUM."""
    sb = StochasBase()
    assert sb.is_nominal is True

    sb.with_trial_num(1)
    assert sb.is_nominal is False
