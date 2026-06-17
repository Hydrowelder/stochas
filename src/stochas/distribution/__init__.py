"""Probability distributions used to sample design and named values."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Annotated

from pydantic import Field

from stochas.base_collections import BaseDict, BaseList
from stochas.distribution._base import (
    _INVALID_CATEGORY_CHARS,
    DISCRETE_MSG,
    NOMINAL_TRIAL_NUM,
    UNDEFINED,
    DistName,
    Distribution,
    DistType,
    SerializableUndefined,
    Undefined,
    logger,
    validate_undefined,
)
from stochas.distribution._continuous import (
    BetaDistribution,
    CauchyDistribution,
    ChiSquaredDistribution,
    ExponentialDistribution,
    FDistribution,
    GammaDistribution,
    LaplaceDistribution,
    LogisticDistribution,
    LogNormalDistribution,
    NormalDistribution,
    ParetoDistribution,
    RayleighDistribution,
    StudentTDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
    WeibullDistribution,
)
from stochas.distribution._discrete import (
    BernoulliDistribution,
    BetaBinomialDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    GeometricDistribution,
    HypergeometricDistribution,
    NegativeBinomialDistribution,
    PermutationDistribution,
    PoissonDistribution,
)

__all__ = [
    "DISCRETE_MSG",
    "NOMINAL_TRIAL_NUM",
    "UNDEFINED",
    "_INVALID_CATEGORY_CHARS",
    "AnyDist",
    "BernoulliDistribution",
    "BetaBinomialDistribution",
    "BetaDistribution",
    "BinomialDistribution",
    "CategoricalDistribution",
    "CauchyDistribution",
    "ChiSquaredDistribution",
    "DistName",
    "DistType",
    "Distribution",
    "DistributionDict",
    "DistributionList",
    "ExponentialDistribution",
    "FDistribution",
    "GammaDistribution",
    "GeometricDistribution",
    "HypergeometricDistribution",
    "LaplaceDistribution",
    "LogNormalDistribution",
    "LogisticDistribution",
    "NegativeBinomialDistribution",
    "NormalDistribution",
    "ParetoDistribution",
    "PermutationDistribution",
    "PoissonDistribution",
    "RayleighDistribution",
    "SerializableUndefined",
    "StudentTDistribution",
    "TriangularDistribution",
    "TruncatedNormalDistribution",
    "Undefined",
    "UniformDistribution",
    "WeibullDistribution",
    "logger",
    "validate_undefined",
]

AnyDist = Annotated[
    NormalDistribution
    | UniformDistribution
    | CategoricalDistribution
    | PermutationDistribution
    | DiscreteUniformDistribution
    | TriangularDistribution
    | TruncatedNormalDistribution
    | LogNormalDistribution
    | PoissonDistribution
    | ExponentialDistribution
    | RayleighDistribution
    | BernoulliDistribution
    | GammaDistribution
    | BetaDistribution
    | WeibullDistribution
    | BinomialDistribution
    | NegativeBinomialDistribution
    | GeometricDistribution
    | LogisticDistribution
    | ParetoDistribution
    | StudentTDistribution
    | HypergeometricDistribution
    | BetaBinomialDistribution
    | CauchyDistribution
    | ChiSquaredDistribution
    | LaplaceDistribution
    | FDistribution,
    Field(discriminator="dist_type"),
]


class DistributionDict(BaseDict[AnyDist]):
    """Dictionary specifically for sampled results."""

    @property
    def distribution_list(self) -> DistributionList:
        """Converts the DistributionDict to a DistributionList."""
        return DistributionList(list(self.values()))

    def set_trial_nums(self, trial_num: int) -> None:
        for dist in self.values():
            if dist.trial_num != trial_num:
                dist.trial_num = trial_num

    def to_tables(self, directory: Path) -> None:
        """Writes one CSV per dist type, organized into per-category subdirectories."""
        by_category: defaultdict[str, list[AnyDist]] = defaultdict(list)
        for dist in self.values():
            by_category[dist.category].append(dist)

        for category, dists in by_category.items():
            category_dir = directory / category
            category_dir.mkdir(parents=True, exist_ok=True)

            by_type: defaultdict[str, list[AnyDist]] = defaultdict(list)
            for dist in dists:
                by_type[dist.dist_type].append(dist)

            for dist_type_key, type_dists in by_type.items():
                # columns are inferred from the first dist; all same-type dists share identical keys
                buf = io.StringIO()
                fieldnames = ["Name", "Units", *type_dists[0].table_params.keys()]
                writer = csv.DictWriter(buf, fieldnames=fieldnames)
                writer.writeheader()
                for d in type_dists:
                    writer.writerow(
                        {"Name": d.name, "Units": d.units, **d.table_params}
                    )
                (category_dir / f"{dist_type_key}.csv").write_text(
                    buf.getvalue(), newline=""
                )


class DistributionList(BaseList[AnyDist]):
    """List specifically for distributions."""

    @property
    def to_distribution_dict(self) -> DistributionDict:
        """Converts the DistributionList to a DistributionDict."""
        d = DistributionDict()
        d.update_many(self.root)
        return d

    def set_trial_nums(self, trial_num: int) -> None:
        for dist in self:
            if dist.trial_num != trial_num:
                dist.trial_num = trial_num
