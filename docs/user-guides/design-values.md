# Design Values: Evolving Your System

---

While Distributions represent the uncertainty you can't control (like wind or manufacturing tolerances), Design Values represent the parameters you can control. These are the knobs you turn to find the absolute best version of your system.

## Optimization vs. Randomness

In a generic design workflow, we distinguish between two types of variables:

| Feature     | Monte Carlo (Distributions)          | Optimization (Design Values)             |
|:------------|:-------------------------------------|:-----------------------------------------|
| **Goal**    | Assess reliability and error bounds. | Find peak performance or trade-offs.     |
| **Logic**   | Values are drawn from a PDF/PMF.     | Values are chosen by a search algorithm. |
| **Tooling** | Standard statistical sampling.       | Optuna (Tuning) or pymoo (Pareto).       |

---

## The Search Space API

Design values define a "Search Space"—a mathematical boundary that an optimizer is allowed to explore. Every design variable in your registry must provide three things:

1. A Name: To track the variable across trials.
2. A Default: The initial "best guess" before the optimizer starts.
3. Bounds: The constraints (low/high or choices) that the solver must respect.

### Supported Variable Types

| Type                | Data                       | Best Used For...                                 |
|:--------------------|:---------------------------|:-------------------------------------------------|
| `DesignFloat`       | $x \in [low, high]$        | Continuous values like stiffness or damping.     |
| `DesignInt`         | $i \in \{low, ..., high\}$ | Discrete counts like solver iterations or parts. |
| `DesignBool`        |$\{True, False\}$           | Enabling/disabling specific logic branches.      |
| `DesignCategorical` | $\{'A', 'B', 'C'\}$        | Picking between distinct materials or styles.    |

---

## Integrating with Solvers

One of the primary advantages of this toolkit is its Solver-Agnostic nature. Your model defines the variables once, and the system can automatically translate them into the specific language of different optimization libraries.

### Optuna Integration (`to_optuna`)

For single-objective tuning, the variable suggests a value to a trial.

* **Continuous:** Uses Bayesian TPE or CMA-ES sampling.
* **Refinement:** The refine method allows the search space to "shrink" around the best known solution for more precision.

### pymoo Integration (`to_pymoo`)

For multi-objective trade-offs, the variable generates a pymoo.core.variable object.

* **Population-based:** Allows for genetic operators (Crossover and Mutation).
* **Pareto Front:** Enables the discovery of the frontier of efficiency.

---

## Usage Example

The following script demonstrates how to define a generic "System" and sample design variables into a registry for optimization.

```python
--8<-- "docs/user-guides/design_values.py"
```

---

## Search Space Refinement

When you reach the end of an optimization study, you may find that the "best" trial is settled in a specific neighborhood. The `refine` method automates the "zoom-in" process.

Given a refinement factor $F \in (0, 1)$ and the best known value $x_{best}$, the new bounds are calculated as:

$$low_{new} = \max(low_{old}, x_{best} - \frac{(high_{old} - low_{old}) \cdot F}{2})$$

$$high_{new} = \min(high_{old}, x_{best} + \frac{(high_{old} - low_{old}) \cdot F}{2})$$

---

!!! success
    By using the `sample_design` pattern, you have decoupled your **Design Intent** from the **Optimization Math**. You can switch from Optuna's Bayesian search to pymoo's Genetic algorithms without changing a single line of your model generation logic.
