import numpy as np
import pytest

from process_manager import (
    BernoulliDistribution,
    CategoricalDistribution,
    DistName,
    PoissonDistribution,
)
from process_manager.distribution import DistributionDict, PermutationDistribution
from process_manager.named_value import NamedValueDict


def test_categorical_pmf_cdf():
    """Verify Categorical logic for non-numeric types."""
    choices = {"Low": 0.2, "Medium": 0.5, "High": 0.3}
    dist = CategoricalDistribution(name=DistName("risk"), choices=choices)

    assert dist.pmf("Low") == 0.2
    assert dist.pmf("Medium") == 0.5
    assert dist.pmf("None") == 0.0

    # CDF follows order of choices list: 0.2, 0.2+0.5, 0.2+0.5+0.3
    assert np.isclose(dist.cdf("Low"), 0.2)
    assert np.isclose(dist.cdf("Medium"), 0.7)
    assert np.isclose(dist.cdf("High"), 1.0)


def test_poisson_properties():
    """Verify Poisson PMF and step-function CDF."""
    lam = 2.0
    dist = PoissonDistribution(name=DistName("calls"), lam=lam)

    # PMF at k=0 is e^-lambda
    expected_pmf_0 = np.exp(-lam)
    assert np.isclose(dist.pmf(0), expected_pmf_0)

    # CDF at 0 should be same as PMF at 0
    assert np.isclose(dist.cdf(0), expected_pmf_0)
    # CDF should be a step function (cdf(0.5) == cdf(0))
    assert dist.cdf(0.5) == dist.cdf(0)

    # PPF
    assert dist.ppf(0.01) >= 0


def test_poisson_sampling():
    """Verify Poisson samples are non-negative integers."""
    dist = PoissonDistribution(name=DistName("test"), lam=5.0, seed=42)
    samples = dist.sample(100)

    assert np.all(samples >= 0)
    assert np.all(samples % 1 == 0)  # Check if integers
    assert np.isclose(np.mean(samples), 5.0, atol=1.0)


def test_bernoulli_initialization():
    """Verify Bernoulli validates probability bounds."""
    # Valid
    dist = BernoulliDistribution(name=DistName("test"), p=0.7, seed=42)
    assert dist.p == 0.7

    # Invalid
    with pytest.raises(ValueError, match="between 0 and 1"):
        BernoulliDistribution(name=DistName("bad"), p=1.2)


def test_bernoulli_math_properties():
    """Verify PMF, CDF, and PPF logic for Bernoulli."""
    p = 0.7
    dist = BernoulliDistribution(name=DistName("coin"), p=p)

    # PMF: P(X=1) = p, P(X=0) = 1-p
    assert np.isclose(dist.pmf(1), p)
    assert np.isclose(dist.pmf(0), 1 - p)
    assert dist.pmf(5) == 0.0

    # CDF: P(X <= x)
    assert np.isclose(dist.cdf(0), 1 - p)
    assert np.isclose(dist.cdf(1), 1.0)
    assert dist.cdf(-1) == 0.0

    # PPF: Inverse CDF
    assert dist.ppf(0.1) == 0  # Since P(X<=0) = 0.3, 0.1 quantile is 0
    assert dist.ppf(0.8) == 1  # Since P(X<=0) = 0.3, 0.8 quantile must be 1


def test_bernoulli_sampling():
    """Verify samples are binary and follow the distribution."""
    dist = BernoulliDistribution(name=DistName("test"), p=0.5, seed=123)
    samples = np.asarray(dist.sample(1000))

    assert set(samples).issubset({0, 1})
    # Mean of Bernoulli is p
    assert np.isclose(np.mean(samples), 0.5, atol=0.05)


def test_discrete_seeding_consistency():
    """Verify that different discrete classes respect the salted seed."""
    seed = 99
    # Same name + same seed = Same results
    p1 = PoissonDistribution(name=DistName("var"), lam=2, seed=seed)
    p2 = PoissonDistribution(name=DistName("var"), lam=2, seed=seed)
    assert np.array_equal(p1.sample(10), p2.sample(10))

    # Different name + same seed = Different results
    b1 = BernoulliDistribution(name=DistName("x"), p=0.5, seed=seed)
    b2 = BernoulliDistribution(name=DistName("y"), p=0.5, seed=seed)
    assert not np.array_equal(b1.sample(20), b2.sample(20))


def test_permutation_integrity():
    """Verify a single draw contains all original items exactly once and is squeezable."""
    items = ["Alpha", "Beta", "Gamma"]
    dist = PermutationDistribution[str](name=DistName("task_order"), items=items)

    # Draw size=1
    sample = dist.sample(size=1)

    # Verify 2D shape (size, items) -> (1, 3)
    assert sample.shape == (1, 3)

    # Verify Squeeze works
    squeezed = sample.squeeze()
    assert squeezed.shape == (3,)
    assert set(squeezed) == set(items)


def test_permutation_multiple_size():
    """Verify drawing multiple permutations returns a (size, N) array."""
    items = [1, 2, 3, 4]
    dist = PermutationDistribution[int](name=DistName("multi_test"), items=items)

    size = 5
    # Move past nominal trial to use draw()
    dist.trial_num = 1
    samples = dist.sample(size=size)

    assert samples.shape == (size, 4)
    # Ensure every row is a valid shuffle
    for row in samples:
        assert set(row) == set(items)


def test_permutation_nominal_fallback():
    """Verify trial_num=0 returns the nominal value inside a 2D array."""
    items = [1, 2, 3]
    nominal_order = [3, 2, 1]

    dist = PermutationDistribution[int](
        name=DistName("nominal_test"), items=items, nominal=nominal_order
    )

    dist.trial_num = 0
    sample = dist.sample(size=1)

    # Check that it's wrapped in a list/array for 2D structure
    assert sample.shape == (1, 3)
    assert np.array_equal(sample[0], nominal_order)


def test_permutation_serialization_and_typing():
    """
    Verify that sample_and_update_dicts resolves the TypeVar 'T'
    even with the nested array structure.
    """
    items = [1.1, 2.2, 3.3]  # Float items
    dist = PermutationDistribution[float](name=DistName("ser_test"), items=items)

    dist_dict = DistributionDict()
    named_dict = NamedValueDict()

    # This call relies on: concrete_type = samples.dtype.type().item().__class__
    # Since samples is now a 2D array of floats, .item() on a float64
    # should still resolve to <class 'float'>.
    nv = dist.sample_and_update_dicts(
        dist_dict=dist_dict, named_value_dict=named_dict, size=1
    )

    # 1. Check runtime type resolution
    # nv.stored_value is the 2D array
    assert nv.stored_value.dtype.kind == "f"  # pyright: ignore[reportAttributeAccessIssue]

    # 2. Check Serialization
    dumped = nv.model_dump()

    # Should be a nested list: [[1.1, 2.2, 3.3]]
    assert isinstance(dumped["stored_value"], np.ndarray)
    assert isinstance(dumped["stored_value"][0], np.ndarray)
    assert isinstance(dumped["stored_value"][0][0], float)


def test_permutation_pmf_logic():
    """Verify probability math for a single permutation outcome."""
    items = ["A", "B", "C"]  # 3! = 6
    dist = PermutationDistribution[str](name=DistName("pmf"), items=items)

    expected_prob = 1.0 / 6.0
    # Valid permutation
    assert np.isclose(dist.pmf(np.array(["C", "A", "B"])), expected_prob)
    # Invalid permutation
    assert dist.pmf(np.array(["A", "B", "D"])) == 0.0


def test_permutation_errors():
    """Ensure unimplemented methods raise errors."""
    dist = PermutationDistribution(name=DistName("test"), items=[1, 2])
    with pytest.raises(NotImplementedError):
        dist.cdf([1, 2])


if __name__ == "__main__":
    test_permutation_integrity()
