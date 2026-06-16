"""MkDocs hooks that generate documentation assets before each build."""

import csv
import shutil
from pathlib import Path

import tabulate as tabulate_mod

_EXAMPLE_TABLES_DIR = Path(__file__).parent / "assets" / "example_tables"


def on_pre_build(config: dict) -> None:
    """Regenerate example report tables so docs always reflect the current to_tables output."""
    _generate_example_tables(_EXAMPLE_TABLES_DIR)


def _csv_to_md(csv_path: Path) -> None:
    """Write a GitHub-flavored markdown table alongside a CSV file."""
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    md = tabulate_mod.tabulate(rows, headers="keys", tablefmt="github")
    csv_path.with_suffix(".md").write_text(md + "\n")


def _generate_example_tables(output_dir: Path = Path("report_tables")) -> None:
    # ---8<--- [start: report_tables]
    from pathlib import Path

    import stochas

    dist_dict = stochas.DistributionDict()
    dist_dict.update(
        stochas.NormalDistribution(
            name=stochas.DistName("link_mass"),
            mu=1.5,
            sigma=0.15,
            category="link_properties",
            units="kg",
        )
    )
    dist_dict.update(
        stochas.NormalDistribution(
            name=stochas.DistName("link_inertia"),
            mu=0.02,
            sigma=0.002,
            category="link_properties",
            units="kg·m²",
        )
    )
    dist_dict.update(
        stochas.TruncatedNormalDistribution(
            name=stochas.DistName("link_length"),
            mu=0.25,
            sigma=0.01,
            low=0.0,
            category="link_properties",
            units="m",
        )
    )
    dist_dict.update(
        stochas.UniformDistribution(
            name=stochas.DistName("init_joint_ang"),
            low=-0.5,
            high=0.5,
            category="initial_conditions",
            units="rad",
        )
    )
    dist_dict.to_tables(Path("report_tables"))
    # ---8<--- [end: report_tables]
    shutil.rmtree(Path("report_tables"))
    dist_dict.to_tables(output_dir)
    for csv_path in output_dir.rglob("*.csv"):
        _csv_to_md(csv_path)
