# Changelog

## Version 2.2.8 (2026-09-10)

- Added changelog
- Fixed bug where `Transaction` would lose distributions seeded states
    - Now tracks by distribution name instead of by instance
    - Retrying with a freshly constructed distribution on every attempt (as the docs demonstrate) now advances the RNG the same way reusing one instance does, instead of silently drawing unseeded values or getting stuck on a nominal value.

## Version 2.2.7 (2026-09-06)

- Fixed a security issue with `mkdocs-material` by auditing existing packages
    - Dependabot alerted to a developer-only dependency that was unused

## Version 2.2.6 (2026-09-06)

- Cut cold boot time in half by lazy-loading `scipy`
- Fixed workflow to use SSH key signing since this repo requires it

## Version 2.2.5 (2026-08-31)

- Added a transaction system to make retrying random draws easier

## Version 2.2.4 (2026-08-11)

- Fixed many Dependabot issues by bumping versions and removing unneeded packages

## Version 2.2.3 (2026-07-03)

- Made compound units easier to define with `__getitem__` so you can write:

    ```py
    US = stochas.UnitSystem.si()
    velocity = 4 * US["ft/s"]
    ```

- Fixed registering new units with `UnitSystem`

## Version 2.2.2 (2026-07-03)

- Sampled units now come with their actual unit instead of `None`

## Version 2.2.0 (2026-07-01)

- Added `UnitSystem`
    - Allows you to multiply a value by a unit and easily express that value in a declared unit system
    - Helpful for converting from IPS to SI
    - Collections, distributions, and named values now track their units using a more formal system
- Added an example to the docs on how to use `UnitSystem`

## Version 2.1.8 (2026-06-29)

- Sampled named values now inherit metadata from distributions when sampled

## Version 2.1.7 (2026-06-29)

- Added metadata mixin, which adds metadata fields to parameters
- Removed `StochasBase.with_override()` since it was a duplicate of `StochasBase.with_overrides()`

## Version 2.1.6 (2026-06-20)

- Added `reset_rng` argument to `sample_dist` since `with_seed` and `with_trial_num` always reset RNG
- Broke up `Distribution` into `Continuous` and `Discrete`

## Version 2.1.5 (2026-06-16)

- Added support for a bunch of new distributions
- Fixed a Windows bug where newlines were added to the inputs table

## Version 2.1.4 (2026-06-15)

- Fixed serialization of distributions

## Version 2.1.2 (2026-06-15)

- Added new metadata fields for categorizing and declaring the units of a distribution
- Added the `AnyDist.to_tables()` method to easily record a distribution to a table
    - Tables are broken up by distribution type and category

## Version 2.1.1 (2026-06-14)

- Expanded unit tests
- Changed docs site to `zensical` instead of `mkdocs`

## Version 2.1.0 (2026-06-13)

- Added Rayleigh distribution
- Added validation for design variables

## Version 2.0.4 (2026-05-27)

- Fixed the `sample_dist` bug where repeated calls would crash the code rather than return the existing distribution

## Version 2.0.3 (2026-05-06)

- Fixes for multiprocessing

## Version 2.0.2 (2026-05-01)

- Documentation page improvements

## Version 2.0.1 (2026-04-24)

- Updated site CSS and logo

## Version 2.0.0 (2026-04-20)

- Added design variable system for `mujoco-mojo`. Expanded this system to support:
    - Boolean suggestors
    - `pymoo` suggestions
    - Value refinement when rerunning a study
- Expanded documentation

## Version 1.0.4 (2026-04-18)

- Fixes to named value serialization with generic types
- Switched to using a sentinel value instead of overloading `NamedValueState`

## Version 1.0.3 (2026-04-17)

- Added support for Python 3.14

## Version 1.0.2 (2026-04-04)

- Updated branding
- CI/CD fixes and improvements

## Version 1.0.0 (2026-04-03)

- Initial release of `stochas` (based on work by Talbot Knighton)
    - Full rewrite of `process_manager`, renamed to `stochas`
- Changed license
- Updated README and documentation site
- Added distributions to public API
- `NamedValue` objects now work like numerical values via `NumericMixin`
