# stochas: Smart Data Orchestration

> [Read the Docs](https://hydrowelder.github.io/stochas/)

> [View on PyPi](https://pypi.org/project/process_manager/)

`stochas` is a Python framework built to handle the complexity of **Monte Carlo simulations**, **parametric studies**, and **probabilistic modeling**.

It provides a robust bridge between abstract statistical rules and concrete simulation data, ensuring your experiments are repeatable, traceable, and easy to manage.

---

## Installation

Install the package via your preferred manager:

=== "`uv`"

    ```bash
    uv add stochas
    ```

=== "`pip`"

    ```bash
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
