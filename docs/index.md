---
hide:
  - navigation
#   - toc
---
<style>
  .md-typeset h1,
  .md-content__button {
    display: none;
  }
  a.badge-link::after {
    content: none !important;
  }
</style>

<p align="center" class="splash">
    <img alt="Stochas" src="assets/read_me_banner.svg" style="width: 100%">
</p>

# stochas: Smart Data Orchestration

<p align="center">
  <a href="https://pypi.org/project/stochas/" class="badge-link">
    <img src="https://img.shields.io/pypi/v/stochas.svg?cacheSeconds=300" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/stochas/" class="badge-link">
    <img src="https://img.shields.io/pypi/pyversions/stochas.svg?cacheSeconds=86400" alt="Python versions">
  </a>
  <a href="https://github.com/Hydrowelder/stochas/actions/workflows/workflow.yml" class="badge-link">
    <img src="https://github.com/Hydrowelder/stochas/actions/workflows/workflow.yml/badge.svg?branch=main" alt="Tests & Release Status">
  </a>
  <a href="https://docs.pydantic.dev/latest/" class="badge-link">
    <img src="https://img.shields.io/badge/Pydantic-v2-FF43A1?logo=pydantic&logoColor=white" alt="Pydantic v2">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0" class="badge-link">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a>
  <a href="https://hydrowelder.github.io/stochas/" class="badge-link">
    <img src="https://img.shields.io/badge/docs-GitHub_Pages-blue.svg" alt="Documentation">
  </a>
  <a href="https://github.com/Hydrowelder/stochas/discussions" class="badge-link" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/discussions-GitHub-blue?logo=github&logoColor=white" alt="GitHub Discussions">
  </a>
  <a href="https://pypistats.org/packages/stochas" class="badge-link">
    <img src="https://img.shields.io/pypi/dm/stochas.svg?cacheSeconds=86400" alt="Downloads">
  </a>
</p>

`stochas` is a Pythons framework built to handle the complexity of **Monte Carlo simulations**, **parametric studies**, and **probabilistic modeling**.

It provides a robust bridge between abstract statistical rules and concrete simulation data, ensuring your experiments are repeatable, traceable, and easy to manage.

---

## Installation

Install the package via your preferred manager:

=== "`uv`"

    ```bash linenums="0"
    uv add stochas
    ```

=== "`pip`"

    ```bash linenums="0"
    pip install stochas
    ```

---

## Core Features

* **Salted Seeding**: Combines Global Seeds, Parameter Names, and Trial Numbers for unique but deterministic draws.
* **Numeric Mixins**: Use your data containers directly in math operations (`container * 5.0`) without manually extracting values.
* **Nominal Support**: Easily toggle between "Perfect World" (Trial 0) and "Probabilistic World" (Monte Carlo) results.
* **Pydantic Foundation**: Every component is a Pydantic model, providing out-of-the-box validation and effortless JSON serialization.

---

## Why use `stochas`?

Managing hundreds of simulation trials can quickly become a mess of manual seeds and inconsistent data. `stochas` solves this by providing:

* **Repeatable Randomness**: Our "Salted Seed" logic ensures that any specific trial can be perfectly recreated, even years later, by tying randomness to simple to set and store values.
* **Smart Containers**: `NamedValue` objects behave like numbers or arrays but protect your data from accidental overwrites using a state-machine logic.
* **Physics-Ready Distributions**: A wide range of built-in distributions (Normal, Truncated Normal, Log-Normal, etc.) that handle their own random number generators internally.
* **Serialized Registries**: Automatically track exactly which "rules" (`Distributions`) and "results" (`NamedValues`) were used in every trial for easy export to JSON or databases.
