"""StochasBase, the top-level model for registering distributions and named/design values."""

from __future__ import annotations

import logging
from typing import Any, Self, TypeVar, cast, overload

from numpydantic import NDArray
from pydantic import BaseModel, Field, SerializeAsAny

from stochas.design_variable import (
    AnyDesignValue,
    DesignBool,
    DesignCategorical,
    DesignFloat,
    DesignInt,
    DesignValueDict,
)
from stochas.distribution import (
    NOMINAL_TRIAL_NUM,
    AnyDist,
    Distribution,
    DistributionDict,
)
from stochas.named_value import NamedValue, NamedValueDict

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StochasBase(BaseModel):
    trial_num: int = NOMINAL_TRIAL_NUM
    """Trial number identified for this instance of the model."""

    seed: int | None = None
    """Campaign seed for calculating random numbers."""

    design: DesignValueDict = Field(default_factory=DesignValueDict)
    """Registry of parameters for hyperparameter tuning."""

    dists: DistributionDict = Field(default_factory=DistributionDict)
    """Random distributions used to generate the model."""

    named: SerializeAsAny[NamedValueDict[NDArray]] = Field(
        default_factory=NamedValueDict[NDArray]
    )
    """Final 'baked' values from a random draw, global override, or design study."""

    def sample_dist[T](
        self,
        dist: Distribution[T],
        size: int = 1,
        force: bool = False,
        warn: bool = True,
        reset_rng: bool = True,
    ) -> NamedValue[NDArray[Any, T]]:
        """
        Sets the seed and trial number of the distribution, sample, registers it and the sampled value to the model, and returns the named value.

        If the NamedValue is already registered, the registered named value is returned.

        Args:
            dist (Dist): Distribution to sample and register.
            size (int, optional): Number of samples to take. Will be embedded in the returned NamedValue. Defaults to 1.
            force (bool, optional): Force the sampled value into the NamedValueDict if it already exists. Defaults to False.
            warn (bool, optional): Whether or not to warn if there is a conflict while forcing. Defaults to True.
            reset_rng (bool, optional): Whether or not to reset the seed and trial number for the distribution. Setting the seed and trial number will reset the random number cycle. This will not skip registering the distribution or NamedValues to the MojoModel. If you want pseudorandom number generation, setting to False will require you to manually set the seed and trial number before passing the distribution into `sample_dist`. Defaults to True.

        Returns:
            NamedValue[NDArray]: NamedValue containing the random draw.

        """
        if reset_rng:
            dist.with_seed(self.seed).with_trial_num(self.trial_num)

        # Distribution is abstract; any instance is necessarily one of AnyDist's
        # concrete members, so this cast just bridges the generic call-site type
        # to the closed discriminated union DistributionDict stores.
        concrete_dist = cast(AnyDist, dist)
        if dist.name not in self.dists:
            self.dists.update(concrete_dist)
        elif force:
            self.dists.force_update(concrete_dist)

        nv = dist.sample_to_named_value(size=size)

        if nv in self.named and not force:
            # defined with global override or already sampled
            if warn:
                logger.warning(
                    f"NamedValue {nv.name} already registered. Returning it instead of the sampled value."
                )
            return self.named[nv.name]
        elif nv in self.named and force:
            # defined with global override but forced to update
            if warn:
                logger.warning(
                    f"NamedValue {nv.name} already registered. Force setting it to the new value in the registry.",
                )
            self.named.force_update(nv, warn=False)
        else:
            # standard random draw
            self.named.update(nv)
        return nv

    @overload
    def sample_design(
        self, dv: DesignFloat, force: bool = False, warn: bool = True
    ) -> float: ...

    @overload
    def sample_design(
        self, dv: DesignInt, force: bool = False, warn: bool = True
    ) -> int: ...

    @overload
    def sample_design(
        self, dv: DesignBool, force: bool = False, warn: bool = True
    ) -> bool: ...

    @overload
    def sample_design(
        self, dv: DesignCategorical[T], force: bool = False, warn: bool = True
    ) -> T: ...

    def sample_design(
        self, dv: AnyDesignValue, force: bool = False, warn: bool = True
    ) -> Any:
        self.design[dv.name] = dv

        if dv.name in self.named and not force:
            if warn:
                logger.info(f"Using override for design variable: {dv.name}")
            val = self.named[dv.name].value
            return val.item() if hasattr(val, "item") else val

        self.named.update(dv)
        return dv.value

    def with_overrides(self, overrides: NamedValueDict[NDArray]) -> Self:
        """
        Sets the NamedValueDict to the provided override.

        This is useful for manually setting some named values to be used.
        """
        self.named = overrides
        return self

    def with_seed(self, seed: int | None = None) -> Self:
        """
        Sets the seed to the provided value.

        This is useful for initializaing the model.
        """
        self.seed = seed
        for _, dist in self.dists.items():
            dist.with_seed(seed)
        return self

    def with_trial_num(self, trial_num: int) -> Self:
        """
        Sets the trial_num to the provided value.

        This is useful for initializaing the model.
        """
        self.trial_num = trial_num
        for _, dist in self.dists.items():
            dist.with_trial_num(trial_num)
        return self

    @property
    def is_nominal(self) -> bool:
        return self.trial_num == NOMINAL_TRIAL_NUM
