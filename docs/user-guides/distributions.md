# Distributions: Defining the Rules of Chance

---

A **Distribution** is a mathematical recipe for generating random numbers. While a `NamedValue` represents a single point in time, a `Distribution` represents the "shape" of all possible values.

### The "What" and the "Why"
Simulations are often used for **[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method)** analysis—running the same task hundreds of times with slight variations to see how often it fails. Distributions define those variations.

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

* **Serialization**: Saving exactly what settings were used so a colleague can recreate the simulation.
* **Bulk Updates**: Changing the `trial_num` for every distribution at once as the simulation progresses.

### Supported Distribution Types

!!! info
    [Click here to see the technical API for all supported types][process_manager.distribution].

* **Normal**: The classic Bell Curve for natural variation.
* **Uniform**: For strict ranges where any value is equally likely.
* **Categorical**: To pick from a fixed set of named choices (e.g., Materials).
* **Bernoulli**: A simple True/False coin flip.
* **Truncated Normal**: A Bell Curve with hard physical limits (e.g., mass cannot be negative).
* **Log Normal**: For positive values with "long-tail" outliers (e.g., contact forces).
* **Triangular**: A simpler alternative to Normal when you only know min, max, and peak.
* **Poisson / Exponential**: For modeling the frequency or time between random events.
* **Permutation**: To return a shuffled version of a master list.

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
