"""Tests for the Distribution base class and its dict/list registries."""

import numpy as np
from numpydantic import NDArray

from stochas import DistName, NormalDistribution
from stochas.distribution import DistributionDict, DistributionList
from stochas.named_value import NamedValueDict


def test_distribution_seeding_and_salting():
    """Verify that names act as unique salts for the same seed."""
    seed = 42

    # Same name, same seed -> Identical results
    dist_a1 = NormalDistribution(name=DistName("x"), mu=0, sigma=1, seed=seed)
    dist_a2 = NormalDistribution(name=DistName("x"), mu=0, sigma=1, seed=seed)

    # Different name, same seed -> Different results
    dist_b = NormalDistribution(name=DistName("y"), mu=0, sigma=1, seed=seed)

    samples_a1 = dist_a1.sample(5)
    samples_a2 = dist_a2.sample(5)
    samples_b = dist_b.sample(5)

    assert np.array_equal(samples_a1, samples_a2)
    assert not np.array_equal(samples_a1, samples_b)


def test_serialization_roundtrip():
    """Verify Pydantic serialization preserves state."""
    dist = NormalDistribution(name=DistName("test"), mu=10, sigma=2, seed=123)
    json_data = dist.model_dump_json()

    new_dist = NormalDistribution.model_validate_json(json_data)

    assert new_dist.name == dist.name
    assert new_dist.seed == dist.seed
    assert new_dist.mu == dist.mu
    assert new_dist.nominal == dist.nominal
    assert np.array_equal(dist.sample(5), new_dist.sample(5))


def test_serialization_roundtrip_dict():
    """Verify Pydantic serialization preserves state."""
    named_dict = NamedValueDict[NDArray]()
    dist_dict = DistributionDict()
    NormalDistribution(
        name=DistName("test"), mu=10, sigma=2, seed=123
    ).sample_and_update_dicts(dist_dict=dist_dict, named_value_dict=named_dict, size=2)

    json_data = named_dict.model_dump_json()
    _new_dist_dict = NamedValueDict[NDArray].model_validate_json(json_data)


def test_sample_and_update_dicts_returns_existing_without_force():
    """Ensure a second call without force returns the already-registered NamedValue."""
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)
    dist_dict = DistributionDict()
    named_dict = NamedValueDict()

    nv1 = dist.sample_and_update_dicts(dist_dict=dist_dict, named_value_dict=named_dict)
    nv2 = dist.sample_and_update_dicts(dist_dict=dist_dict, named_value_dict=named_dict)

    assert nv2 is nv1


def test_sample_and_update_dicts_force_overwrites_and_warns(caplog):
    """Ensure force=True overwrites the existing NamedValue and logs a warning."""
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)
    dist_dict = DistributionDict()
    named_dict = NamedValueDict()

    nv1 = dist.sample_and_update_dicts(dist_dict=dist_dict, named_value_dict=named_dict)
    nv2 = dist.sample_and_update_dicts(
        dist_dict=dist_dict, named_value_dict=named_dict, force=True
    )

    assert nv2 is not nv1
    assert named_dict["x"] is nv2
    assert "already exists in named_value_list" in caplog.text


def test_distribution_dict_helpers():
    """Verify DistributionDict's conversion and trial-number propagation."""
    dist_dict = DistributionDict()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)
    dist_dict.update(dist)

    dist_list = dist_dict.distribution_list
    assert isinstance(dist_list, DistributionList)
    assert list(dist_list) == [dist]

    dist_dict.set_trial_nums(3)
    assert dist_dict["x"].trial_num == 3


def test_distribution_list_helpers():
    """Verify DistributionList's conversion and trial-number propagation."""
    dist_list = DistributionList()
    dist = NormalDistribution(name=DistName("y"), mu=0, sigma=1)
    dist_list.append(dist)

    dist_dict = dist_list.to_distribution_dict
    assert isinstance(dist_dict, DistributionDict)
    assert dist_dict["y"] is dist

    dist_list.set_trial_nums(2)
    assert dist_list[0].trial_num == 2


if __name__ == "__main__":
    test_serialization_roundtrip_dict()
