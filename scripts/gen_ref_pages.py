"""
Generate API reference pages in docs/reference/.

Run this script before building the docs site:

    python scripts/gen_ref_pages.py

The output is written to docs/reference/ and is consumed by zensical (or
mkdocs) via the ::: autodoc directives. The directory is gitignored; this
script must be run as part of the CI pipeline before zensical build.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS_REF = ROOT / "docs" / "reference"

# ---------------------------------------------------------------------------
# Modules to document: (dotted identifier, output path relative to DOCS_REF)
# ---------------------------------------------------------------------------
MODULES: list[tuple[str, str]] = [
    ("stochas", "stochas/index.md"),
    ("stochas.base", "stochas/base.md"),
    ("stochas.base_collections", "stochas/base_collections.md"),
    ("stochas.design_variable", "stochas/design_variable.md"),
    ("stochas.distribution", "stochas/distribution.md"),
    ("stochas.mixins", "stochas/mixins.md"),
    ("stochas.named_value", "stochas/named_value.md"),
    ("stochas.utils", "stochas/utils.md"),
]


def main() -> None:
    DOCS_REF.mkdir(parents=True, exist_ok=True)

    for module, rel_path in MODULES:
        dest = DOCS_REF / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"::: {module}\n", encoding="utf-8")
        print(f"  {dest.relative_to(ROOT)}")

    print(f"\nWrote {len(MODULES)} pages to {DOCS_REF.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
