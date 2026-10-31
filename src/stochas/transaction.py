"""Retryable, all-or-nothing blocks of correlated random draws for `StochasBase`. Reach this module through `StochasBase.transaction()` rather than constructing a `Transaction` directly."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast, overload

from stochas.design_variable import (
    AnyDesignValue,
    DesignBool,
    DesignCategorical,
    DesignFloat,
    DesignInt,
)
from stochas.distribution import AnyDist, Distribution, DistributionDict
from stochas.distribution.base import NDArray
from stochas.named_value import NamedValue

if TYPE_CHECKING:
    from stochas.base import StochasBase

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Transaction:
    """
    Iterable of retry attempts returned by `StochasBase.transaction()`.

    Each iteration yields a `TransactionAttempt` to use as `with attempt:`. Construct via `Transaction.begin()`, which snapshots `base.dists`, `base.design`, and `base.named`. A failed attempt (retryable or not) rolls back to that snapshot before the exception is either suppressed (to retry) or allowed to propagate, so a transaction only ever commits the registrations made by the attempt that succeeds.

    Iteration ends as soon as an attempt finishes without raising. If `max_retries` attempts are rejected without one succeeding, the next iteration raises `RuntimeError`, chained to the last rejected exception.
    """

    base: StochasBase = field(repr=False)
    """The model whose `dists`, `design`, and `named` registries are snapshotted and rolled back."""

    retry_on: type[Exception] | tuple[type[Exception], ...]
    """Exception type(s) that mean "bad draw, retry" rather than a real error. Any other exception propagates after rollback."""

    max_retries: int
    """Maximum number of attempts before giving up with `RuntimeError`."""

    _attempts: int = field(default=0, init=False)
    _done: bool = field(default=False, init=False)
    _last_exception: BaseException | None = field(default=None, init=False)
    _seeded: DistributionDict = field(
        default_factory=DistributionDict, init=False, repr=False
    )
    _warned_nominal_names: set[str] = field(default_factory=set, init=False)
    _dists_snapshot: dict[str, Any] = field(default_factory=dict, init=False)
    _design_snapshot: dict[str, Any] = field(default_factory=dict, init=False)
    _named_snapshot: dict[str, Any] = field(default_factory=dict, init=False)

    @overload
    @classmethod
    def begin(
        cls, base: StochasBase, retry_on: type[Exception], max_retries: int = 10
    ) -> Self: ...
    @overload
    @classmethod
    def begin(
        cls,
        base: StochasBase,
        retry_on: tuple[type[Exception], ...],
        max_retries: int = 10,
    ) -> Self: ...
    @classmethod
    def begin(
        cls,
        base: StochasBase,
        retry_on: type[Exception] | tuple[type[Exception], ...],
        max_retries: int = 10,
    ) -> Self:
        """
        Starts a transaction, snapshotting `base.dists`, `base.design`, and `base.named` for rollback.

        The snapshots are shallow copies of each registry's underlying dict, so a rollback restores which names are registered, not the internal state of the registered objects. A distribution's RNG stream therefore keeps advancing across attempts, which is what lets a retry draw a new value.

        Args:
            base (StochasBase): Model to run the transaction against.
            retry_on (type[Exception] | tuple[type[Exception], ...]): Exception type(s) that mean "bad draw, retry" rather than a real error.
            max_retries (int, optional): Maximum number of attempts before raising `RuntimeError`. Defaults to 10.

        Returns:
            Transaction: The new transaction, ready to iterate over.

        """
        transaction = cls(base=base, retry_on=retry_on, max_retries=max_retries)
        transaction._dists_snapshot = dict(base.dists.root)
        transaction._design_snapshot = dict(base.design.root)
        transaction._named_snapshot = dict(base.named.root)
        return transaction

    def __iter__(self) -> Self:
        """Returns the transaction itself; each `next()` yields the next `TransactionAttempt`."""
        return self

    def __next__(self) -> TransactionAttempt:
        """
        Returns the next attempt, or ends the loop once an attempt has succeeded.

        Returns:
            TransactionAttempt: A fresh attempt to wrap in `with attempt:`.

        Raises:
            StopIteration: If an attempt has already finished without raising.
            RuntimeError: If `max_retries` attempts were rejected without one succeeding. Chained to the last rejected exception.

        """
        if self._done:
            raise StopIteration
        if self._attempts >= self.max_retries:
            msg = f"Failed to complete transaction after {self.max_retries} attempt(s)."
            logger.error(msg)
            raise RuntimeError(msg) from self._last_exception
        return TransactionAttempt(self)

    def _handle_attempt_result(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None
    ) -> bool:
        """
        Finishes one attempt: marks the transaction done on success, otherwise rolls back and decides whether to retry.

        Called from `TransactionAttempt.__exit__`. A failed attempt is rolled back to the snapshot taken in `begin` whatever the exception type. A retryable exception is counted and suppressed so the loop runs another attempt, and once more than half of `max_retries` have been used each rejection also logs a high-rejection-rate warning.

        Args:
            exc_type (type[BaseException] | None): Type of the exception raised in the `with` block, or None if it exited cleanly.
            exc_value (BaseException | None): The raised exception, or None if the block exited cleanly.

        Returns:
            bool: True to suppress the exception because it is one of `retry_on`. False if the attempt succeeded or the exception should propagate.

        """
        if exc_type is None:
            self._done = True
            return False

        # roll back this attempt's partial registrations before deciding what to do next
        self.base.dists.root = dict(self._dists_snapshot)
        self.base.design.root = dict(self._design_snapshot)
        self.base.named.root = dict(self._named_snapshot)

        if not issubclass(exc_type, self.retry_on):
            return False

        self._attempts += 1
        self._last_exception = exc_value
        if self._attempts > self.max_retries // 2:
            logger.warning(
                f"High rejection rate detected for transaction. Attempt {self._attempts}/{self.max_retries}. Latest error: {exc_value}"
            )
        return True


@dataclass
class TransactionAttempt:
    """
    One attempt of a `StochasBase.transaction()`; use as `with attempt:`.

    Sample through `attempt.sample_dist` and `attempt.sample_design` instead of the model's own methods, so retries draw new values and everything registered during a rejected attempt is rolled back. Anything else on the model is reachable through `attempt.base`. Raise one of the transaction's `retry_on` exceptions inside the `with` block to reject the attempt.
    """

    transaction: Transaction
    """The transaction this attempt belongs to."""

    @property
    def base(self) -> StochasBase:
        """The `StochasBase` model this transaction is retrying against, which is the one `.transaction()` was called on. Use it for anything not wrapped on `attempt`, e.g. `attempt.base.with_seed(...)` or `attempt.base.model_dump()`."""
        return self.transaction.base

    @property
    def attempt_number(self) -> int:
        """1-indexed number of this attempt, which is the number of attempts rejected so far plus one."""
        return self.transaction._attempts + 1

    def sample_dist[T](
        self,
        dist: Distribution[T],
        size: int = 1,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> NamedValue[NDArray[Any, T]]:
        """
        Samples `dist` through `attempt.base`, reseeding it only the first time this transaction sees `dist.name`.

        Reseeding on every attempt would reset the distribution's RNG to the same state each time and draw the exact same rejected value forever, so `reset_rng` is not exposed here. It is derived instead from whether this transaction has already seeded a distribution with this name. Reusing one instance across attempts and building a fresh instance with the same name on every attempt both work: the previous instance's seed, trial number, and live RNG stream are carried over to the new one (`Distribution.adopt_prior_state`), so both patterns draw identical values.

        If the carried-over state leaves `dist.is_nominal` true, every remaining attempt for that name will return the same nominal value that was already rejected. A warning is logged once per name per transaction, not once per retry.

        Args:
            dist (Distribution): Distribution to sample and register.
            size (int, optional): Number of samples to take. Defaults to 1.
            force (bool, optional): Force the sampled value into the model's registries if the name already exists. Defaults to False.
            warn (bool, optional): Whether or not to warn if there is a conflict while forcing. Defaults to True.
            convert_units (bool, optional): Whether to convert the samples into the model's base unit when `dist.unit` is a `UnitDescriptor`. Defaults to True.

        Returns:
            NamedValue[NDArray]: NamedValue containing the random draw.

        """
        # Distribution is abstract; any instance is necessarily one of AnyDist's
        # concrete members, so this cast just bridges the generic call-site type
        # to the closed discriminated union DistributionDict stores.
        concrete_dist = cast(AnyDist, dist)

        prior = self.transaction._seeded.get(dist.name)
        reset_rng = prior is None
        if prior is not None:
            dist.adopt_prior_state(prior)
        self.transaction._seeded.force_update(concrete_dist, warn=False)

        if (
            not reset_rng
            and dist.is_nominal
            and dist.name not in self.transaction._warned_nominal_names
        ):
            self.transaction._warned_nominal_names.add(dist.name)
            logger.warning(
                f"Distribution {dist.name} is nominal on retry attempt "
                f"{self.attempt_number} of a transaction; every remaining attempt "
                "will keep returning the same nominal value instead of a new draw."
            )

        return self.transaction.base.sample_dist(
            dist,
            size=size,
            force=force,
            warn=warn,
            reset_rng=reset_rng,
            convert_units=convert_units,
        )

    @overload
    def sample_design(
        self,
        dv: DesignFloat,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> float: ...
    @overload
    def sample_design(
        self,
        dv: DesignInt,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> int: ...
    @overload
    def sample_design(
        self,
        dv: DesignBool,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> bool: ...
    @overload
    def sample_design(
        self,
        dv: DesignCategorical[T],
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> T: ...
    def sample_design(
        self,
        dv: AnyDesignValue,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> Any:
        """
        Samples `dv` through `attempt.base`.

        Design variables are not RNG-backed (an optimizer, not stochas, picks their value), so there is no reseeding to manage. This is a plain passthrough to `StochasBase.sample_design`, kept on the attempt so a design variable stays fixed for the duration of one attempt and rolls back with everything else on a rejected one, exactly like `sample_dist`.

        Args:
            dv (AnyDesignValue): Design variable to register and sample.
            force (bool, optional): Overwrite an existing entry in the model's `named` registry. Defaults to False.
            warn (bool, optional): Whether or not to log a message when using an override or forcing. Defaults to True.
            convert_units (bool, optional): Whether to convert the value into the model's base unit when `dv.unit` is a `UnitDescriptor`. Defaults to True.

        Returns:
            The design variable's value: `float` for `DesignFloat`, `int` for `DesignInt`, `bool` for `DesignBool`, and the selected choice for `DesignCategorical`.

        """
        return self.transaction.base.sample_design(
            dv, force=force, warn=warn, convert_units=convert_units
        )

    def __enter__(self) -> Self:
        """Starts the attempt. Nothing is set up here; rollback and retry decisions are made in `__exit__`."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        """
        Ends the attempt: commits it on success, otherwise rolls back and decides whether to retry.

        Args:
            exc_type (type[BaseException] | None): Type of the exception raised in the `with` block, or None if it exited cleanly.
            exc_value (BaseException | None): The raised exception, or None if the block exited cleanly.
            traceback (object): Traceback of the raised exception. Unused.

        Returns:
            bool: True if the exception is one of the transaction's `retry_on` types, which suppresses it so the loop runs another attempt. False otherwise, so any other exception propagates after rollback.

        """
        return self.transaction._handle_attempt_result(exc_type, exc_value)
