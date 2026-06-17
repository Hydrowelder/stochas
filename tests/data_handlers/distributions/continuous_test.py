"""Tests for continuous distributions."""

import numpy as np
import pytest

from stochas import (
    BetaDistribution,
    DistName,
    ExponentialDistribution,
    GammaDistribution,
    LogNormalDistribution,
    NormalDistribution,
    RayleighDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
    WeibullDistribution,
)


def test_normal_distribution_properties():
    mu, sigma = 100, 15
    dist = NormalDistribution(name=DistName("iq"), mu=mu, sigma=sigma)

    # Statistical properties
    assert dist.pdf(mu) > 0
    assert np.isclose(dist.cdf(mu), 0.5)
    assert np.isclose(dist.ppf(0.5), mu)
    assert dist.is_continuous is True

    # Sampling
    samples = dist.sample(1000)
    assert np.isclose(np.mean(samples), mu, atol=2.0)
    assert np.isclose(np.std(samples), sigma, atol=2.0)


def test_normal_distribution_invalid_sigma():
    """Ensure sigma must be positive."""
    with pytest.raises(ValueError, match="negative standard deviation"):
        NormalDistribution(name=DistName("bad"), mu=0, sigma=0)


def test_normal_distribution_nominal_sample():
    """Ensure sample() returns the nominal value when trial_num is nominal."""
    dist = NormalDistribution(name=DistName("iq"), mu=100, sigma=15, nominal=110)

    samples = dist.sample(3)

    assert np.array_equal(samples, np.full(3, 110))


def test_uniform_distribution_bounds():
    dist = UniformDistribution(name=DistName("u"), low=10, high=20)
    samples = dist.sample(100)

    assert np.all(samples >= 10)
    assert np.all(samples <= 20)
    assert np.isclose(dist.cdf(15), 0.5)
    assert dist.pdf(15) > 0
    assert np.isclose(dist.ppf(0.5), 15)
    assert dist.is_continuous is True


def test_uniform_distribution_validation():
    """Ensure high must be greater than low."""
    with pytest.raises(ValueError, match="high must be greater than low"):
        UniformDistribution(name=DistName("bad"), low=20, high=10)


def test_triangular_validation():
    with pytest.raises(ValueError, match="Must satisfy low <= mode <= high"):
        TriangularDistribution(name=DistName("bad"), low=10, mode=5, high=20)


def test_triangular_properties():
    """Verify Triangular sampling, PDF, CDF, PPF, and is_continuous."""
    dist = TriangularDistribution(name=DistName("tri"), low=0, mode=5, high=10)

    samples = dist.sample(100)
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)

    assert dist.pdf(5) > 0
    assert np.isclose(dist.cdf(0), 0.0)
    assert np.isclose(dist.cdf(10), 1.0)
    assert np.isclose(dist.ppf(0.5), dist.mode, atol=1.0)
    assert dist.is_continuous is True


def test_truncated_normal_bounds():
    """Verify samples never exceed the specified lower and upper bounds."""
    mu, sigma = 100, 10
    lower, upper = 95, 105
    dist = TruncatedNormalDistribution(
        name=DistName("clamped"), mu=mu, sigma=sigma, low=lower, high=upper
    )

    samples = dist.sample(1000)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

    # CDF at lower bound should be 0, upper should be 1
    assert np.isclose(dist.cdf(lower), 0.0, atol=1e-7)
    assert np.isclose(dist.cdf(upper), 1.0, atol=1e-7)

    assert dist.pdf(mu) > 0
    assert dist.is_continuous is True


def test_truncated_normal_extreme_bounds():
    """Verify it handles a mean that is outside the bounds."""
    # Mean is 0, but we only allow samples between 10 and 20
    dist = TruncatedNormalDistribution(
        name=DistName("offset"), mu=0, sigma=1, low=10, high=20
    )
    samples = dist.sample(100)
    assert np.all(samples >= 10)


def test_log_normal_properties():
    """Verify LogNormal is positive and respects the shape parameter."""
    s = 0.5  # sigma of the log
    scale = np.exp(2)  # mu of the log is 2
    dist = LogNormalDistribution(name=DistName("skewed"), s=s, scale=scale)

    samples = dist.sample(1000)
    assert np.all(samples > 0)

    # Median of LogNormal is the 'scale' parameter (exp(mu))
    # So CDF(scale) should be 0.5
    assert np.isclose(dist.cdf(scale), 0.5)
    assert np.isclose(dist.ppf(0.5), scale)

    assert dist.pdf(scale) > 0
    assert dist.is_continuous is True


def test_exponential_properties():
    """Verify Exponential distribution follows the rate lambda."""
    lam = 0.5
    dist = ExponentialDistribution(name=DistName("decay"), lam=lam)

    samples = dist.sample(1000)
    assert np.all(samples >= 0)

    # Theoretical mean of Exponential is 1/lambda
    expected_mean = 1 / lam
    assert np.isclose(np.mean(samples), expected_mean, atol=0.2)

    # CDF of Exponential is 1 - exp(-lam * x)
    # At x = 1/lam, CDF is 1 - exp(-1) ≈ 0.632
    assert np.isclose(dist.cdf(expected_mean), 1 - np.exp(-1))

    assert dist.pdf(expected_mean) > 0
    assert dist.is_continuous is True


def test_rayleigh_properties():
    """Verify Rayleigh distribution follows the scale (sigma) parameter."""
    scale = 2.0
    dist = RayleighDistribution(name=DistName("error"), scale=scale)

    samples = dist.sample(1000)
    assert np.all(samples >= 0)

    # Theoretical mean of Rayleigh is scale * sqrt(pi / 2)
    expected_mean = scale * np.sqrt(np.pi / 2)
    assert np.isclose(np.mean(samples), expected_mean, atol=0.2)

    # Median of Rayleigh is scale * sqrt(ln(4))
    median = scale * np.sqrt(np.log(4))
    assert np.isclose(dist.cdf(median), 0.5)
    assert np.isclose(dist.ppf(0.5), median)

    assert dist.pdf(median) > 0
    assert dist.is_continuous is True


def test_rayleigh_validation():
    with pytest.raises(ValueError, match="non-positive scale"):
        RayleighDistribution(name=DistName("bad"), scale=0)


@pytest.mark.parametrize(
    "dist_instance",
    [
        TruncatedNormalDistribution(
            name=DistName("tn"), mu=50, sigma=5, low=40, high=60
        ),
        LogNormalDistribution(name=DistName("ln"), s=0.5, scale=10),
        ExponentialDistribution(name=DistName("ex"), lam=2.0),
        RayleighDistribution(name=DistName("ry"), scale=2.0),
    ],
)
def test_continuous_ppf_consistency(dist_instance):
    """Generic check that PPF and CDF are true inverses."""
    qs = [0.01, 0.5, 0.99]
    for q in qs:
        val = dist_instance.ppf(q)
        assert np.isclose(dist_instance.cdf(val), q, atol=1e-7)


def test_gamma_properties():
    """Verify Gamma sampling is positive and mean tracks alpha*beta."""
    alpha, beta = 3.0, 2.0
    dist = GammaDistribution(name=DistName("g"), alpha=alpha, beta=beta)

    samples = dist.sample(1000)
    assert np.all(samples > 0)
    assert np.isclose(np.mean(samples), alpha * beta, atol=0.3)

    assert dist.pdf(alpha * beta) > 0
    assert np.isclose(dist.cdf(0), 0.0, atol=1e-6)
    assert np.isclose(dist.cdf(dist.ppf(0.5)), 0.5, atol=1e-6)
    assert dist.is_continuous is True


def test_gamma_validation():
    with pytest.raises(ValueError, match="alpha"):
        GammaDistribution(name=DistName("bad"), alpha=0.0, beta=1.0)
    with pytest.raises(ValueError, match="beta"):
        GammaDistribution(name=DistName("bad"), alpha=1.0, beta=-1.0)


def test_beta_properties():
    """Verify Beta samples are in (0,1) and CDF at mode is consistent."""
    alpha, beta = 2.0, 5.0
    dist = BetaDistribution(name=DistName("b"), alpha=alpha, beta=beta)

    samples = dist.sample(1000)
    assert np.all(samples > 0)
    assert np.all(samples < 1)

    # mean of Beta(alpha, beta) = alpha / (alpha + beta)
    expected_mean = alpha / (alpha + beta)
    assert np.isclose(np.mean(samples), expected_mean, atol=0.05)

    assert dist.pdf(0.5) > 0
    assert np.isclose(dist.cdf(0.0), 0.0, atol=1e-6)
    assert np.isclose(dist.cdf(1.0), 1.0, atol=1e-6)
    assert dist.is_continuous is True


def test_beta_validation():
    with pytest.raises(ValueError, match="alpha"):
        BetaDistribution(name=DistName("bad"), alpha=0.0, beta=1.0)
    with pytest.raises(ValueError, match="beta"):
        BetaDistribution(name=DistName("bad"), alpha=1.0, beta=0.0)


def test_weibull_properties():
    """Verify Weibull samples are positive and PPF/CDF are consistent."""
    shape, scale = 2.0, 100.0
    dist = WeibullDistribution(name=DistName("w"), shape=shape, scale=scale)

    samples = dist.sample(1000)
    assert np.all(samples > 0)

    # median of Weibull = scale * ln(2)^(1/shape)
    median = scale * (np.log(2) ** (1 / shape))
    assert np.isclose(dist.cdf(median), 0.5, atol=0.01)
    assert np.isclose(dist.ppf(0.5), median, atol=0.5)

    assert dist.pdf(scale) > 0
    assert dist.is_continuous is True


def test_weibull_validation():
    with pytest.raises(ValueError, match="shape"):
        WeibullDistribution(name=DistName("bad"), shape=0.0, scale=1.0)
    with pytest.raises(ValueError, match="scale"):
        WeibullDistribution(name=DistName("bad"), shape=1.0, scale=-1.0)


@pytest.mark.parametrize(
    "dist, expected",
    [
        (
            NormalDistribution(name=DistName("n"), mu=1.0, sigma=2.0),
            {"mu": 1.0, "sigma": 2.0},
        ),
        (
            UniformDistribution(name=DistName("u"), low=0.5, high=1.5),
            {"low": 0.5, "high": 1.5},
        ),
        (
            TriangularDistribution(name=DistName("t"), low=0.0, mode=0.5, high=1.0),
            {"low": 0.0, "mode": 0.5, "high": 1.0},
        ),
        (
            TruncatedNormalDistribution(
                name=DistName("tn"), mu=0.0, sigma=1.0, low=-2.0, high=2.0
            ),
            {"mu": 0.0, "sigma": 1.0, "low": -2.0, "high": 2.0},
        ),
        (
            LogNormalDistribution(name=DistName("ln"), s=0.5, scale=2.0),
            {"s": 0.5, "scale": 2.0},
        ),
        (
            ExponentialDistribution(name=DistName("ex"), lam=1.5),
            {"lam": 1.5},
        ),
        (
            RayleighDistribution(name=DistName("ry"), scale=3.0),
            {"scale": 3.0},
        ),
        (
            GammaDistribution(name=DistName("ga"), alpha=2.0, beta=1.5),
            {"alpha": 2.0, "beta": 1.5},
        ),
        (
            BetaDistribution(name=DistName("be"), alpha=2.0, beta=5.0),
            {"alpha": 2.0, "beta": 5.0},
        ),
        (
            WeibullDistribution(name=DistName("wb"), shape=2.0, scale=100.0),
            {"shape": 2.0, "scale": 100.0},
        ),
    ],
)
def test_continuous_table_params(dist, expected):
    assert dist.table_params == expected
