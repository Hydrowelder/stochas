# StochasBase: The Simulation Conductor

!!! abstract
    **StochasBase** serves as the central conductor for your simulation lifecycle, synchronizing **Aleatory** randomness with **Epistemic** design choices into a single, deterministic stream. By leveraging "Salted Seed" logic, it ensures that complex Monte Carlo campaigns remain perfectly repeatable across different machines and collaborators. This guide explores the orchestration of model inputs, the mechanics of the "Baked" registry, and the use of manual overrides to isolate variables for high-fidelity debugging and "Golden Case" testing.

---

`StochasBase` is the central "brain" for a simulation trial. It orchestrates the lifecycle of a model by bridging the gap between mathematical uncertainty and deterministic execution.

### Centralized Trial Orchestration

A single instance of `StochasBase` manages [two distinct pillars](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic) of simulation inputs:

* **Aleatory Uncertainty (`sample_dist`):** Represents "luck" or noise. These are random draws from probability distributions (Normal, Uniform, etc.) that you cannot control but must account for.
* **Epistemic Uncertainty (`sample_design`):** Represents "choices." These are tunable parameters (Design Variables) used by optimizers like Optuna or pymoo to find peak performance.

### The Pillars of Repeatability

To ensure that a simulation can be perfectly recreated by a colleague, the orchestrator utilizes a "Salted Seed" logic. Every random draw is a deterministic function of the trial's metadata:

$$Seed_{local} = \mathcal{H}(Seed_{global}, Trial_{num}, Variable_{name})$$

* **Global Seed:** Controls the entire campaign.
* **Trial Number:** Ensures trial 10 is different from trial 11.
* **Variable Name:** Ensures "stiffness" doesn't get the same random value as "damping."

---

## The "Baked" Registry (named)

While `StochasBase` is designed to automate randomness and optimization, you often need to bypass the math to test a specific "Golden Case" or debug a known failure point. Overrides allow you to inject fixed values into the Baked Registry before the simulation begins.

The orchestrator follows a strict priority sequence when a variable is requested:

1. **Registry Check:** It looks in the `named` dictionary. If a value exists, it returns it immediately.
2. **Logic Execution:** If the registry is empty, it proceeds to draw from a distribution or ask an optimizer for a suggestion.

By using `with_overrides()`, you populate the registry early, effectively "locking" those variables for the duration of the trial.

---

## Quick Implementation

```python
--8<-- "docs/user-guides/orchestration.py"
```
