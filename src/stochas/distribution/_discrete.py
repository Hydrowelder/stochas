"""Discrete and non-numeric probability distributions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import scipy.stats as stats
from numpydantic import NDArray
from pydantic import Field, PrivateAttr, model_validator

from stochas.distribution._base import (
    DISCRETE_MSG,
    UNDEFINED,
    Distribution,
    DistType,
    SerializableUndefined,
    logger,
)

if TYPE_CHECKING:
    from scipy.stats.distributions import rv_discrete
else:
    rv_discrete = Any


class DiscreteUniformDistribution(Distribution[int]):
    """
    Represent a distribution where every integer in a range is equally likely.

    Use this for discrete counts like number of objects, gear teeth, or array indices where you need equal probability across the whole range.

    Examples:
        ```python
        dist = DiscreteUniformDistribution(name=name, low=-1, high=2)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.integers.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/discrete_uniform.png" width="600" />

    """

    low: int
    high: int

    _scipy: rv_discrete = PrivateAttr()

    dist_type: Literal[DistType.DISCRETE_UNIFORM] = DistType.DISCRETE_UNIFORM

    @model_validator(mode="after")
    def validate_low_high(self) -> Self:
        if self.high <= self.low:
            msg = f"Distribution {self.name}: high must be greater than low"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.randint(low=self.low, high=self.high + 1)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.integers(low=self.low, high=self.high, size=size, endpoint=True)

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def pdf(self, x: Any) -> float | NDArray[Any, float]:
        logger.warning(DISCRETE_MSG)
        return self.pmf(k=x)

    def pmf(self, k: int | np.ndarray) -> float | np.ndarray:
        """Probability Mass Function."""
        return self._scipy.pmf(k)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high}


class CategoricalDistribution[T](Distribution[T]):
    """
    Represent a discrete distribution over a fixed set of named choices.

    Ideal for switching between discrete simulation states, such as different materials, object types, or floor textures.

    Examples:
        ```python
        class Material(StrEnum):
            STEEL = "steel"
            WOOD = "wood"

        dist = CategoricalDistribution[Material](
            name="floor_material",
            choices={Material.STEEL: 0.8, Material.WOOD: 0.2}
        )
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Categorical_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/categorical.png" width="600" />

    """

    choices: dict[T, float]
    """Choices for the categorical distribution. Tuples have the format (category, probability). This guarantees each category has an associated probability."""

    _scipy: rv_discrete = PrivateAttr()

    dist_type: Literal[DistType.CATEGORICAL] = DistType.CATEGORICAL

    @property
    def categories(self) -> list[T]:
        return list(self.choices.keys())

    @property
    def probabilities(self) -> list[float]:
        return list(self.choices.values())

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        s = sum(self.probabilities)
        if not np.isclose(s, 1, atol=1e-8):
            msg = f"Distribution {self.name} sum of probabilities ({s:.2f}) do not sum to 1"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        indices = np.arange(len(self.choices))
        self._scipy = stats.rv_discrete(
            name=self.name, values=(indices, self.probabilities), seed=self.rng
        )
        return self

    def draw(self, size: int = 1) -> NDArray[Any, T]:
        return self.rng.choice(a=self.categories, size=size, p=self.probabilities)  # type: ignore

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def pdf(self, x: Any) -> float | NDArray[Any, float]:
        """Categorical distributions have no PDF. Did you mean to use pmf?"""
        logger.warning(DISCRETE_MSG)
        return self.pmf(x=x)

    def pmf(self, x: T) -> float:
        """Probability Mass Function. Returns the probability of a specific category 'x'."""
        try:
            return self.choices[x]
        except KeyError:
            return 0.0

    def cdf(self, x: Any) -> float:
        """
        Cumulative Distribution Function.

        Note:
            This follows the order of the 'choices' list.

        """
        try:
            idx = self.categories.index(x)
            return self._scipy.cdf(idx)
        except ValueError:
            return 0.0

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"choices": ", ".join(f"{k}: {v}" for k, v in self.choices.items())}


class PermutationDistribution[T](Distribution[T]):
    """
    Takes a master list and returns either the original or a shuffled version.

    Trial 0 (Nominal) returns the items in their provided order.
    Trial > 0 returns a unique random permutation of those items.
    """

    items: list[T]
    """The master list to be shuffled."""

    nominal: list[T] | list[None] | SerializableUndefined = Field(default=UNDEFINED)  # pyright: ignore[reportIncompatibleVariableOverride]

    dist_type: Literal[DistType.PERMUTATION] = DistType.PERMUTATION

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def sample(self, size: int = 1) -> np.ndarray:
        """The core sampling logic for the distribution."""
        if self.is_nominal and self.nominal is not UNDEFINED:
            return np.asarray([self.nominal])
        else:
            return self.draw(size=size)

    def draw(self, size: int = 1) -> np.ndarray:
        """Returns a shuffled version of the items."""
        return np.array([self.rng.permutation(self.items) for _ in range(size)])  # pyright: ignore[reportArgumentType, reportCallIssue]

    def pdf(self, x: Any) -> float:
        return self.pmf(x)

    def pmf(self, x: list[T] | np.ndarray) -> float:
        """Uniform probability of 1/n! if x is a valid permutation."""
        n = len(self.items)
        if len(x) != n or set(x) != set(self.items):
            return 0.0
        return 1.0 / math.factorial(n)

    def cdf(self, x: Any) -> float:
        raise NotImplementedError("CDF not defined for ShuffledDistribution")

    def ppf(self, q: Any) -> Any:
        raise NotImplementedError("PPF not defined for ShuffledDistribution")

    @property
    def table_params(self) -> dict[str, Any]:
        return {"items": str(self.items)}


class PoissonDistribution(Distribution[int]):
    """
    Represent the probability of a number of events occurring in a fixed interval.

    Useful for counting discrete occurrences, like how many distractor objects should appear in a scene.

    Examples:
        ```python
        # Sampling the number of random noise impulses to apply during a trial
        dist = PoissonDistribution(name="impulse_count", lam=3.5)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Poisson_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/poisson.png" width="600" />

    """

    lam: float
    """Lambda: Average rate of occurrences"""

    _scipy: rv_discrete = PrivateAttr()

    dist_type: Literal[DistType.POISSON] = DistType.POISSON

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.poisson(mu=self.lam)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.poisson(lam=self.lam, size=size)

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        logger.warning(DISCRETE_MSG)
        return self.pmf(k=x)

    def pmf(self, k: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pmf(k)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"lam": self.lam}


class BernoulliDistribution(Distribution[bool]):
    """
    Represent a single binary trial (Success/Failure).

    The simplest distribution, used for "on/off" flags, such as enabling or disabling a specific sensor or randomized model feature.

    Examples:
        ```python
        # 50% chance to enable gravity compensation for this trial
        dist = BernoulliDistribution(name="use_grav_comp", p=0.5)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Bernoulli_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/bernoulli.png" width="600" />

    """

    p: float
    """Probability of success (0.0 to 1.0)."""

    _scipy: rv_discrete = PrivateAttr()

    dist_type: Literal[DistType.BERNOULLI] = DistType.BERNOULLI

    @model_validator(mode="after")
    def validate_probability(self) -> Self:
        if not (0 <= self.p <= 1):
            msg = f"Probability p must be between 0 and 1. Got {self.p}"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.bernoulli(self.p)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        # numpy doesn't have a 'bernoulli', so we use binomial with n=1
        return self.rng.binomial(n=1, p=self.p, size=size)

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def pdf(self, x: int | np.ndarray) -> float | np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        logger.warning(DISCRETE_MSG)
        return self.pmf(x=x)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"p": self.p}


class BinomialDistribution(Distribution[int]):
    """
    Represent the number of successes in a fixed number of independent trials.

    Use this when modeling counts with a known upper bound, such as the number of sensors that report an anomaly out of a fixed-size sensor array.

    Examples:
        ```python
        # Number of faulty actuators out of 12 in a trial
        dist = BinomialDistribution(name="faulty_actuators", n=12, p=0.05)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Binomial_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/binomial.png" width="600" />

    """

    n: int
    """Number of trials. Must be at least 1."""

    p: float
    """Probability of success on each trial (0.0 to 1.0)."""

    _scipy: rv_discrete = PrivateAttr()

    dist_type: Literal[DistType.BINOMIAL] = DistType.BINOMIAL

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.n < 1:
            msg = f"Distribution {self.name}: n must be at least 1."
            logger.error(msg)
            raise ValueError(msg)
        if not (0 <= self.p <= 1):
            msg = f"Distribution {self.name}: p must be between 0 and 1. Got {self.p}"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.binom(n=self.n, p=self.p)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.binomial(n=self.n, p=self.p, size=size)

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def pdf(self, x: int | np.ndarray) -> float | np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        logger.warning(DISCRETE_MSG)
        return self.pmf(k=x)

    def pmf(self, k: int | np.ndarray) -> float | np.ndarray:
        """Probability Mass Function."""
        return self._scipy.pmf(k)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"n": self.n, "p": self.p}
