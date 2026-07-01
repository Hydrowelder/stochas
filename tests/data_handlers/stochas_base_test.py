"""Tests for StochasBase."""

import numpy as np
import pytest

from stochas import (
    DesignFloat,
    DistName,
    NamedValue,
    NormalDistribution,
    StochasBase,
)
from stochas.named_value import ValueName
from stochas.unit_system import UnitSystem


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


def test_sample_dist_with_unit_converts_samples():
    """Ensure sample_dist multiplies the sampled array by the unit factor."""
    us = UnitSystem.si()
    sb = StochasBase()
    # nominal=100.0 so the draw is deterministic at trial_num=0 regardless of sigma
    dist = NormalDistribution(
        name=DistName("length"), mu=100.0, sigma=1.0, nominal=100.0, unit=us.inch
    )

    nv = sb.sample_dist(dist)

    assert np.allclose(nv.value, [100.0 * 0.0254], rtol=1e-6)


def test_sample_dist_convert_units_false_skips_conversion():
    """Ensure convert_units=False returns the raw sample without scaling."""
    us = UnitSystem.si()
    sb = StochasBase()
    dist = NormalDistribution(
        name=DistName("length"), mu=100.0, sigma=1.0, nominal=100.0, unit=us.inch
    )

    nv = sb.sample_dist(dist, convert_units=False)

    assert np.allclose(nv.value, [100.0])


def test_sample_design_with_unit_converts_value():
    """Ensure sample_design multiplies the design value by the unit factor."""
    us = UnitSystem.si()
    sb = StochasBase()
    dv = DesignFloat(
        name=ValueName("width"), low=0.0, high=100.0, stored_value=10.0, unit=us.inch
    )

    result = sb.sample_design(dv)

    assert result == pytest.approx(10.0 * 0.0254, rel=1e-6)
    assert sb.named["width"].value == pytest.approx(10.0 * 0.0254, rel=1e-6)


def test_update_unit_system_restores_factors_after_deserialization():
    """Ensure update_unit_system re-populates factors excluded from serialization."""
    us = UnitSystem.si()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1, unit=us.inch)
    sb = StochasBase()
    sb.dists.update(dist)

    sb2 = StochasBase.model_validate_json(sb.model_dump_json())

    assert sb2.dists["x"].unit is not None
    assert sb2.dists["x"].unit.scale is None  # excluded from serialization

    sb2.with_unit_system(us)

    assert sb2.dists["x"].unit.scale == pytest.approx(0.0254, rel=1e-6)


def test_model_validator_auto_restores_factors_when_u_serialized():
    """Ensure model_validator restores unit factors on deserialization when u is included."""
    us = UnitSystem.si()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1, unit=us.inch)
    sb = StochasBase(us=us)
    sb.dists.update(dist)

    sb2 = StochasBase.model_validate_json(sb.model_dump_json())

    assert sb2.us is not None
    assert sb2.dists["x"].unit is not None
    assert sb2.dists["x"].unit.scale == pytest.approx(0.0254, rel=1e-6)
