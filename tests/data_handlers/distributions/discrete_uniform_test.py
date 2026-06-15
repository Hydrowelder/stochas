"""Tests for DiscreteUniformDistribution."""

import numpy as np
import pytest

from stochas import DistName
from stochas.distribution import DiscreteUniformDistribution


def test_discrete_uniform_validation():
    """Ensure high must be greater than low."""
    with pytest.raises(ValueError, match="high must be greater than low"):
        DiscreteUniformDistribution(name=DistName("bad"), low=5, high=1)


def test_discrete_uniform_sampling_bounds():
    """Verify samples are integers within the inclusive [low, high] range."""
    dist = DiscreteUniformDistribution(name=DistName("d"), low=1, high=5, seed=42)

    samples = dist.sample(1000)

    assert np.all(samples >= 1)
    assert np.all(samples <= 5)
    assert np.all(samples % 1 == 0)


def test_discrete_uniform_is_continuous():
    """Ensure DiscreteUniformDistribution reports as discrete."""
    dist = DiscreteUniformDistribution(name=DistName("d"), low=1, high=5)

    assert dist.is_continuous is False


def test_discrete_uniform_pmf_pdf_cdf_ppf():
    """Verify pmf/pdf/cdf/ppf are consistent for a uniform integer distribution."""
    dist = DiscreteUniformDistribution(name=DistName("d"), low=1, high=4)

    # Each of the 4 outcomes (1, 2, 3, 4) is equally likely.
    assert np.isclose(dist.pmf(1), 0.25)
    assert np.isclose(dist.pdf(1), dist.pmf(1))

    assert np.isclose(dist.cdf(1), 0.25)
    assert np.isclose(dist.cdf(4), 1.0)

    assert dist.ppf(0.25) == 1
    assert dist.ppf(1.0) == 4
