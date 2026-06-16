"""Generates figures to be included as previews on docstrings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure

from stochas import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    Dist,
    DistName,
    DistType,
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    PoissonDistribution,
    RayleighDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
)

ASSET_DIR = Path(__file__).parent
ROSE_500 = "#f43f5e"
SKY_500 = "#0ea5e9"

DPI = 200
ASPECT_RATIO = 12 / 5

WIDTH = 6  # inches
HEIGHT = WIDTH / ASPECT_RATIO


@dataclass
class DistSweep:
    """A family of same-typed distributions, varying one "critical" parameter."""

    distributions: Sequence[Dist]
    labels: list[str]
    sup_title: str


DiscreteDist = BernoulliDistribution | PoissonDistribution | DiscreteUniformDistribution


def get_dist_sweep(dist_type: DistType) -> DistSweep | None:
    name = DistName(dist_type)
    sup_title = f"{name.replace('_', ' ').title()} Distribution"

    match dist_type:
        case DistType.NORMAL:
            sigmas = [0.5, 1.0, 2.0]
            dists = [
                NormalDistribution(name=name, mu=0, sigma=sigma) for sigma in sigmas
            ]
            labels = [f"sigma={sigma}" for sigma in sigmas]
            sup_title += " (sweeping sigma)"
        case DistType.UNIFORM:
            highs = [1.0, 2.0, 4.0]
            dists = [UniformDistribution(name=name, low=0, high=high) for high in highs]
            labels = [f"[0, {high}]" for high in highs]
            sup_title += " (sweeping high)"
        case DistType.DISCRETE_UNIFORM:
            highs = [2, 5, 10]
            dists = [
                DiscreteUniformDistribution(name=name, low=0, high=high)
                for high in highs
            ]
            labels = [f"[0, {high}]" for high in highs]
            sup_title += " (sweeping high)"
        case DistType.CATEGORICAL:

            class Blood(StrEnum):
                O_P = "O+"
                O_N = "O-"
                A_P = "A+"
                A_N = "A-"
                B_P = "B+"
                B_N = "B-"
                AB_P = "AB+"
                AB_N = "AB-"

            dists = [
                CategoricalDistribution[Blood](
                    name=name,
                    choices={
                        Blood.O_P: 0.36,
                        Blood.O_N: 0.14,
                        Blood.A_P: 0.28,
                        Blood.A_N: 0.08,
                        Blood.B_P: 0.08,
                        Blood.B_N: 0.03,
                        Blood.AB_P: 0.02,
                        Blood.AB_N: 0.01,
                    },
                    nominal=Blood.O_P,
                )
            ]
            labels = ["blood types"]
            sup_title += " (blood types)"
        case DistType.TRIANGULAR:
            modes = [0.25, 0.5, 0.75]
            dists = [
                TriangularDistribution(name=name, low=0, high=1, mode=mode)
                for mode in modes
            ]
            labels = [f"mode={mode}" for mode in modes]
            sup_title += " (sweeping mode)"
        case DistType.TRUNCATED_NORMAL:
            lows = [-2.0, -1.0, 0.0]
            dists = [
                TruncatedNormalDistribution(name=name, mu=0, sigma=1, low=low)
                for low in lows
            ]
            labels = [f"low={low}" for low in lows]
            sup_title += " (sweeping low)"
        case DistType.LOG_NORMAL:
            shapes = [0.25, 0.5, 1.0]
            dists = [LogNormalDistribution(name=name, s=s, scale=1) for s in shapes]
            labels = [f"s={s}" for s in shapes]
            sup_title += " (sweeping s)"
        case DistType.POISSON:
            lams = [1.0, 4.0, 10.0]
            dists = [PoissonDistribution(name=name, lam=lam) for lam in lams]
            labels = [f"λ={lam}" for lam in lams]
            sup_title += " (sweeping λ)"
        case DistType.EXPONENTIAL:
            lams = [0.5, 1.0, 2.0]
            dists = [ExponentialDistribution(name=name, lam=lam) for lam in lams]
            labels = [f"λ={lam}" for lam in lams]
            sup_title += " (sweeping λ)"
        case DistType.RAYLEIGH:
            scales = [0.5, 1.0, 2.0]
            dists = [RayleighDistribution(name=name, scale=scale) for scale in scales]
            labels = [f"scale={scale}" for scale in scales]
            sup_title += " (sweeping scale)"
        case DistType.BERNOULLI:
            ps = [0.25, 0.5, 0.75]
            dists = [BernoulliDistribution(name=name, p=p) for p in ps]
            labels = [f"p={p}" for p in ps]
            sup_title += " (sweeping p)"
        case _:
            print(f"Distribution of type {dist_type} is not implemented")
            return None

    return DistSweep(distributions=dists, labels=labels, sup_title=sup_title)


def shades(base_color: str, n: int) -> list[tuple[float, float, float]]:
    """Returns n shades of base_color, from a light tint to the full color."""
    base = np.array(to_rgb(base_color))
    white = np.array([1.0, 1.0, 1.0])
    if n == 1:
        return [tuple(base)]
    fracs = np.linspace(0.7, 0.0, n)
    return [tuple((1 - f) * base + f * white) for f in fracs]


def plot_and_save_sweep(sweep: DistSweep):
    """Generates a dual-panel plot for a family of distributions and saves it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
    fig: Figure
    ax1: Axes
    ax2: Axes

    n = len(sweep.distributions)
    pdf_colors = shades(ROSE_500, n)
    cdf_colors = shades(SKY_500, n)
    first = sweep.distributions[0]

    if first.is_continuous:
        # use ppf to find a shared support across the whole sweep (0.1% to 99.9%)
        try:
            bounds = [
                (dist.ppf(0.001), dist.ppf(0.999)) for dist in sweep.distributions
            ]
            x_min = min(lo for lo, _ in bounds)
            x_max = max(hi for _, hi in bounds)
        except Exception:  # fallback for edge cases
            x_min, x_max = -5, 5
        x = np.linspace(x_min, x_max, 500)

        for dist, label, pdf_color, cdf_color in zip(
            sweep.distributions, sweep.labels, pdf_colors, cdf_colors
        ):
            y_pdf = dist.pdf(x)
            y_cdf = dist.cdf(x)

            ax1.plot(x, y_pdf, lw=2, color=pdf_color, label=label)
            ax1.fill_between(x, y_pdf, alpha=0.1, color=pdf_color)

            ax2.plot(x, y_cdf, lw=2, color=cdf_color, label=label)

        ax1.set_ylabel("Density (PDF)")
        ax2.set_ylabel("Cumulative Prob (CDF)")

    elif first.is_discrete:
        if isinstance(first, CategoricalDistribution):
            # categorical x-axis uses named labels rather than a numeric range
            categorical_dists = cast(
                list[CategoricalDistribution[Any]], sweep.distributions
            )
            x = np.arange(len(first.choices))
            eval_at = first.categories
            tick_labels = [str(c) for c in first.choices]
            ax1.set_xticks(x, tick_labels, rotation=45)
            ax2.set_xticks(x, tick_labels, rotation=45)

            for dist, label, pdf_color, cdf_color in zip(
                categorical_dists, sweep.labels, pdf_colors, cdf_colors
            ):
                y_pmf = [dist.pmf(val) for val in eval_at]
                y_cdf = [dist.cdf(val) for val in eval_at]

                ax1.stem(
                    x,
                    y_pmf,
                    basefmt=" ",
                    linefmt=to_hex(pdf_color),
                    markerfmt="o",
                    label=label,
                )
                ax2.step(x, y_cdf, where="post", color=cdf_color, lw=2, label=label)

        else:
            discrete_dists = cast("list[DiscreteDist]", sweep.distributions)

            # use ppf to find a shared integer support across the sweep
            bounds = [(dist.ppf(0.001), dist.ppf(0.999)) for dist in discrete_dists]
            x_min = int(np.floor(min(lo for lo, _ in bounds)))
            x_max = int(np.ceil(max(hi for _, hi in bounds)))
            x = list(range(x_min, x_max + 1))

            for dist, label, pdf_color, cdf_color in zip(
                discrete_dists, sweep.labels, pdf_colors, cdf_colors
            ):
                y_pmf = np.asarray([dist.pmf(val) for val in x], dtype=float)
                y_cdf = np.asarray([dist.cdf(val) for val in x], dtype=float)

                ax1.plot(x, y_pmf, "o-", color=pdf_color, label=label)
                ax2.step(x, y_cdf, where="post", color=cdf_color, lw=2, label=label)

        ax1.set_ylabel("Probability (PMF)")
        ax2.set_ylabel("Cumulative Prob (CDF)")

    # cleanup and labeling
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    fig.suptitle(sweep.sup_title, fontsize=14)

    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Value")

    if n > 1:
        ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(ASSET_DIR / f"{first.dist_type.lower()}.png", dpi=DPI)
    plt.close(fig)


def main():
    for dist_type in DistType:
        print(f"{dist_type=}")
        sweep = get_dist_sweep(dist_type)
        if sweep:
            plot_and_save_sweep(sweep)


if __name__ == "__main__":
    main()
