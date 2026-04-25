# NamedValues: Smart Containers for Simulation Data

!!! abstract
    **NamedValues** serve as the deterministic "Source of Truth" for your simulation parameters. By implementing a strict `UNSET`/`SET` lifecycle and **numeric mixins**, these containers ensure data integrity across complex physics models while behaving like native Python numbers in mathematical operations. This guide explores how to use NamedValues and their protected collections to prevent accidental data overrides and maintain a traceable, frozen state for every simulation trial.

---

## NamedValue
A **NamedValue** is more than just a variable; it is a smart container designed specifically for the needs of simulation and data tracking. It ensures that once a value is decided for a trial, it stays that way.

### What is it?
In complex simulations, you often need to share a single piece of data (like the mass of a robot arm) across many different functions. A `NamedValue` acts as the "Source of Truth" for that specific parameter.

It tracks its own lifecycle using a simple state machine:

* **`UNSET`**: The container is waiting for data. If you try to read it, the system will raise an error to prevent you from using uninitialized data.
* **`SET`**: The container has been populated. At this point, it becomes **frozen**. Any accidental attempt to overwrite it will be blocked unless you explicitly use a "force" command.

### Why use it?
The primary advantage of `NamedValue` is its **Numeric Mixin** behavior. Even though it is a complex object that tracks names and states, you can use it in your code as if it were a simple float or a NumPy array.

You can do math directly with the container:

* `result = my_named_value * 5.0`
* `total = value_a + value_b`

---

## Working with Collections
In a real project, you'll likely have dozens of parameters. We use specialized collections to manage them:

* **NamedValueDict**: Best for quick lookups. You can ask for `my_dict["motor_friction"]` and get the container immediately.
* **NamedValueList**: Useful when you need to iterate through all sampled values to save them to a file or a database.

These collections protect their entries from wayward data updates. If a `NamedValue` with a specific name already exists in the the collection, the value will not be allowed in (without using a force).

### Simple Example
```python
--8<-- "docs/user-guides/named_values.py:basics"
```

### Protected Entry Example

Continuing from the previous example

```python
--8<-- "docs/user-guides/named_values.py:protection"
```
