# Distributions: Defining the Rules of Chance

!!! abstract

    **Distributions** serve as the probabilistic blueprints for your simulation's uncertainty. By defining the "shape" of possible variations, they enable rigorous Monte Carlo analysis. This guide details the mechanics of **Salted Seeds** for deterministic reproducibility, the distinction between configuration registries (`DistributionDict`) and result registries (`NamedValueDict`), and the core workflows for both locked and repeated sampling.

---

A **Distribution** is a mathematical recipe for generating random numbers. While a `NamedValue` represents a single point in time, a `Distribution` represents the "shape" of all possible values.

### The "What" and the "Why"

Simulations are often used for **[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method)** analysis where the same task is run hundreds of times with slight variations to see how often it fails. Distributions define those variations.

Instead of saying "the floor is slippery," you define a `UniformDistribution` for friction between 0.1 and 0.4. The system will then pick a new, valid number for every trial.

### Advanced Features

#### Repeatable Randomness (The "Salted" Seed)

To make sure your results are repeatable (great for debugging), we use a "salted" seeding method. If you provide a single global `seed`, the system automatically mixes it with other parameters to create a unique local seed for every draw.

The three ingredients in the "Salt" are:

1. **Global Seed**: Controls the broad "campaign." Change this to get a totally different set of results.
2. **Distribution Name**: Ensures unique draws for different parameters. Without this, two parameters with the same config (like `x_offset` and `y_offset`, if using the same [dispersion](https://en.wikipedia.org/wiki/Statistical_dispersion)) would produce identical, coupled values.
3. **Trial Number**: Ensures that every iteration in your Monte Carlo run gets a unique value from the dispersion.

#### The Registry: DistributionDict

A `DistributionDict` acts as a centralized record of the "rules" used during a simulation. While a `NamedValueDict` stores the **results** (the numbers), the `DistributionDict` stores the **config** (the math).

This is critical for:

- **Serialization**: Saving exactly what settings were used so a colleague can recreate the simulation.
- **Bulk Updates**: Changing the `trial_num` for every distribution at once as the simulation progresses.

### Supported Distribution Types

!!! info

    [Click here to see the technical API for all supported types](../reference/stochas/distribution.md).

- **Normal**: The classic Bell Curve for natural variation.
- **Uniform**: For strict ranges where any value is equally likely.
- **Discrete Uniform**: Similar to a uniform distribution, but only allows for integers to be returned.
- **Categorical**: To pick from a fixed set of named choices (e.g., Materials).
- **Bernoulli**: A simple True/False coin flip.
- **Truncated Normal**: A Bell Curve with hard physical limits (e.g., mass cannot be negative).
- **Log Normal**: For positive values with "long-tail" outliers (e.g., contact forces).
- **Triangular**: A simpler alternative to Normal when you only know min, max, and peak.
- **Poisson / Exponential**: For modeling the frequency or time between random events.
- **Rayleigh**: For modeling the magnitude of a 2D vector with independent Normal components (e.g., radial positioning error).
- **Permutation**: To return a shuffled version of a master list.

---

### Example: Sampling into Registries

When using `sample_and_update_dicts`, the system checks if a value with that name already exists in your `NamedValueDict`. If it does, it **returns the existing value** instead of drawing a new one. This ensures all parts of your simulation use the same "random" choice for a single trial.

```python
--8<-- "docs/user-guides/distributions.py:basics"
```

### Example: Repeated Sampling

If you need to pull many different random numbers from the same distribution without locking them into a registry (e.g., for noise injection or redraws to meet some constraint), use the `.sample()` or `.draw()` methods directly. These move the Random Number Generator forward with every call.

```python
--8<-- "docs/user-guides/distributions.py:repeat_draw"
```

---

### Generating Report Tables

When a simulation uses many distributions, it is useful to export a human-readable summary for inclusion in a design document or report. `DistributionDict` supports this with the `to_tables()` method.

#### Categorizing Distributions

Each distribution has a `category` field. Setting it to a meaningful label groups related distributions together. One subdirectory is created per unique category value, and each distribution type within it gets its own CSV file.

```python
--8<-- "docs/hooks.py:report_tables"
```

!!! tip

    Any distribution that does not have `category` set will land in an `uncategorized/` subdirectory.

#### Output Format

`to_tables(directory)` creates one subdirectory per category, then writes one `.csv` file per distribution type within it. The filename is the distribution type name (e.g. `normal.csv`, `truncated_normal.csv`). Each file is a flat table whose columns are `Name` plus every parameter specific to that type.

The example above produces the following layout:

``` linenums="0" title=""
report_tables/
├── link_properties/
│   ├── normal.csv
│   └── truncated_normal.csv
└── initial_conditions/
    └── uniform.csv
```

???+ example "link_properties/normal.csv"

    === "Table"

        --8<-- "docs/assets/example_tables/link_properties/normal.md"

    === "Plaintext"

        ``` linenums="0"
        --8<-- "docs/assets/example_tables/link_properties/normal.csv"
        ```


???+ example "link_properties/truncated_normal.csv"

    === "Table"

        --8<-- "docs/assets/example_tables/link_properties/truncated_normal.md"

    === "Plaintext"

        ``` linenums="0"
        --8<-- "docs/assets/example_tables/link_properties/truncated_normal.csv"
        ```
