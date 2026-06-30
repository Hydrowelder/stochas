"""Base classes, enums, and helpers shared by all distributions."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NewType,
    Self,
)

import numpy as np
from numpydantic import NDArray
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    field_validator,
    model_validator,
)

from stochas.mixins import MetadataMixin
from stochas.named_value import NamedValue, ValueName

if TYPE_CHECKING:
    from stochas.distribution import DistributionDict
    from stochas.named_value import NamedValueDict

logger = logging.getLogger(__name__)

DistName = NewType("DistName", str)
"""Alias of string. Used to type hint a distribution."""

NOMINAL_TRIAL_NUM = 0
"""Trial number definition where nominal case will be used."""

DISCRETE_MSG = "Discrete distributions use pmf, not pdf. Using pmf method instead."

_INVALID_CATEGORY_CHARS: frozenset[str] = frozenset(r'\/:*?"<>|' + "\x00")


class DistType(StrEnum):
    NORMAL = "normal"
    """Normal distribution."""

    UNIFORM = "uniform"
    """Uniform distribution."""

    DISCRETE_UNIFORM = "discrete_uniform"
    """Discrete uniform distribution."""

    CATEGORICAL = "categorical"
    """Categorical distribution."""

    PERMUTATION = "permutation"
    """Permutation distribution."""

    TRIANGULAR = "triangular"
    """Triangular distribution."""

    TRUNCATED_NORMAL = "truncated_normal"
    """Truncated Normal distribution."""

    LOG_NORMAL = "log_normal"
    """Log Normal distribution."""

    POISSON = "poisson"
    """Poisson distribution."""

    EXPONENTIAL = "exponential"
    """Exponential distribution."""

    RAYLEIGH = "rayleigh"
    """Rayleigh distribution."""

    BERNOULLI = "bernoulli"
    """Bernoulli distribution."""

    GAMMA = "gamma"
    """Gamma distribution."""

    BETA = "beta"
    """Beta distribution."""

    WEIBULL = "weibull"
    """Weibull distribution."""

    BINOMIAL = "binomial"
    """Binomial distribution."""

    NEGATIVE_BINOMIAL = "negative_binomial"
    """Negative Binomial distribution."""

    GEOMETRIC = "geometric"
    """Geometric distribution."""

    LOGISTIC = "logistic"
    """Logistic distribution."""

    PARETO = "pareto"
    """Pareto distribution."""

    STUDENT_T = "student_t"
    """Student's t distribution."""

    HYPERGEOMETRIC = "hypergeometric"
    """Hypergeometric distribution."""

    BETA_BINOMIAL = "beta_binomial"
    """Beta-Binomial distribution."""

    CAUCHY = "cauchy"
    """Cauchy distribution."""

    CHI_SQUARED = "chi_squared"
    """Chi-squared distribution."""

    LAPLACE = "laplace"
    """Laplace distribution."""

    F = "f"
    """F distribution."""

    @property
    def table_header(self) -> str:
        return self.replace("_", " ").title()


class Undefined(StrEnum):
    """Undefined value."""

    UNDEFINED = "__UNDEFINED__"


UNDEFINED = Undefined.UNDEFINED
"""Sentinel to differentiate between None and unset."""


def validate_undefined(v: Any) -> Any:
    if v == "__UNDEFINED__":
        return UNDEFINED
    return v


SerializableUndefined = Annotated[
    Undefined,
    BeforeValidator(validate_undefined),
    PlainSerializer(lambda _: "__UNDEFINED__", return_type=str),
]


class Distribution[T](ABC, MetadataMixin):
    model_config = ConfigDict(arbitrary_types_allowed=False)

    name: DistName
    """Name of the distribution."""

    seed: int | None = None
    """Seed of the distribution.

    Leave as None or omit to use a random seed. If the seed is not None, the specified seed will be salted with the name and trial_num attributes to add randomness. This allows you to use the same seed for multiple distributions while also being able to simply serialize and deserialize the distribution. See the `validate_seed` validator method for how the seed is hashed.

    Note:
        | Salt        | Functional Purpose       | Description                                                                                                                                                                           |
        |:------------|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        | `seed`      | **Campaign Entropy**     | Controls the broad "global" cycle. Change this to generate a completely new set of results for the same analysis (e.g., using a date-based seed for a daily batch).                   |
        | `name`      | **Parameter Decoupling** | Ensures unique draws for different parameters. Without this, two distributions with the same config (e.g., an `x` and `y` center of gravity) would produce identical, coupled values. |
        | `trial_num` | **Iteration Variance**   | Provides a unique draw for each Monte Carlo trial. This ensures that every individual simulation trial receives a unique value from the dispersion.                                   |

    Clear as mud?
    """

    nominal: T | None | SerializableUndefined = Field(default=UNDEFINED)
    """Value the distribution should take if the trial_number attribute is equal to 0."""

    trial_num: int = NOMINAL_TRIAL_NUM
    """Run number for sampling from the distribution. This is used to salt the seed (if specified)."""

    _rng: np.random.Generator = PrivateAttr()
    """Random number generator."""

    def with_seed(self, seed: int | None) -> Self:
        """Sets the seed for the distribution to the provided and resets the pseudorandom number generator."""
        self.seed = seed
        self.refresh_seed()
        return self

    def with_trial_num(self, trial_num: int) -> Self:
        """Sets the trial number for the distribution to the provided and resets the pseudorandom number generator."""
        self.trial_num = trial_num
        self.refresh_seed()
        return self

    def refresh_seed(self) -> None:
        """Resets the pseudorandom number generator."""
        if self.seed is not None:
            # combine name and run number to salt
            name_to_salt = f"{self.name}_{self.trial_num}"

            # generate a repeatable salt for the seed, name, and trial_number
            salt = int(hashlib.md5(name_to_salt.encode()).hexdigest(), 16)
            local_seed = (self.seed + salt) % (2**32)

            self._rng = np.random.default_rng(seed=local_seed)
        else:
            # use pure random value
            self._rng = np.random.default_rng(seed=self.seed)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Rejects characters that are illegal in directory names on any major OS."""
        bad = _INVALID_CATEGORY_CHARS & set(v)
        if bad:
            raise ValueError(
                f"category contains characters invalid in directory names: {sorted(bad)}"
            )
        return v

    @model_validator(mode="after")
    def validate_seed(self) -> Self:
        """Validates that random number generators have been set for the distribution."""
        self.refresh_seed()
        return self

    @property
    def rng(self) -> np.random.Generator:
        """Provides a localized random number generator."""
        return self._rng

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Flag indicating if the distribution is continuous (True) or discrete (False)."""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def is_discrete(self) -> bool:
        """Flag indicating if the distribution is discrete (True) or continuous (False)."""
        return not self.is_continuous

    @abstractmethod
    def draw(self, size: int = 1) -> NDArray[Any, T]:
        """Perform a random draw"""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    def sample(self, size: int = 1) -> NDArray[Any, T]:
        """The core sampling logic for the distribution."""
        if self.is_nominal and self.nominal is not UNDEFINED:
            return np.full(size, self.nominal)
        else:
            return self.draw(size=size)

    @abstractmethod
    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Probability Density Function."""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    @abstractmethod
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Cumulative Distribution Function."""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    def sample_to_named_value(self, size: int = 1) -> NamedValue[NDArray[Any, T]]:
        """Samples the distribution and returns the NamedValue it makes, inheriting this distribution's metadata."""
        samples = self.sample(size=size)
        concrete_type = samples.dtype.type().item().__class__  # pyright: ignore[reportAttributeAccessIssue]
        return NamedValue[NDArray[Any, concrete_type]](
            name=ValueName(self.name),
            stored_value=samples,
            **self.metadata_dict(),
        )

    def sample_and_update_dicts(
        self,
        dist_dict: DistributionDict,
        named_value_dict: NamedValueDict,
        size: int = 1,
        force: bool = False,
        warn: bool = True,
    ) -> NamedValue[NDArray[Any, T]]:
        """Samples from the distribution and registers the result."""
        if self.name in named_value_dict and not force:
            return named_value_dict[self.name]
        elif self.name in named_value_dict and force and warn:
            logger.warning(
                f"NamedValue {self.name} already exists in named_value_list. Overwriting!"
            )

        nv = self.sample_to_named_value(size=size)

        if self.name not in dist_dict:
            dist_dict.update(self)  # pyright: ignore[reportArgumentType]
        elif force:
            dist_dict.force_update(self)  # pyright: ignore[reportArgumentType]

        if force:
            named_value_dict.force_update(nv, warn=warn)
        else:
            named_value_dict.update(nv)
        return nv

    @abstractmethod
    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        """
        Percent Point Function (Inverse of CDF). Used to find the value at a specific quantile (e.g., 0.95).

        Args:
            q: Probability (0.0 to 1.0).

        """
        msg = f"Method 'ppf' not implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def has_nominal(self) -> bool:
        """Returns whether or not the distribution has a nominal value."""
        return self.nominal is not UNDEFINED

    @property
    def is_nominal(self) -> bool:
        """Returns whether or not the distribution will be treated as a nominal (when sampled it will always return the nominal if set)."""
        return self.trial_num == NOMINAL_TRIAL_NUM and self.has_nominal

    @property
    @abstractmethod
    def table_params(self) -> dict[str, Any]:
        """Distribution-specific parameters to include as columns in a report table row."""


class DiscreteDistribution[T](Distribution[T], ABC):
    """Base class for distributions over a countable set of values, exposed through a probability mass function (pmf) rather than a pdf."""

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    @abstractmethod
    def pmf(self, x: Any) -> float | np.ndarray:
        """Probability Mass Function."""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    def pdf(self, x: Any) -> float | np.ndarray:
        """Discrete distributions have no pdf; warns and delegates to pmf."""
        logger.warning(DISCRETE_MSG)
        return self.pmf(x)


class ContinuousDistribution[T](Distribution[T], ABC):
    """Base class for distributions over a continuous range, exposed through a probability density function (pdf)."""

    @property
    def is_continuous(self) -> Literal[True]:
        return True
