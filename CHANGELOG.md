# Changelog

## v2.2.8 (2026-09-10)


- Added changelog

## v2.2.7 (2026-09-06)

- Fixed a security issue with `mkdocs-material` by auditing existing packages.
    - Dependabot warned to a developer only dependency which was unused.

## v2.2.6 (2026-09-06)

- Cut cold boot time in half by lazy loading `scipy`.
- Fixed workflow to use ssh-key signing since this repo requires that.

## v2.2.5 (2026-08-31)

- Added transaction system to make retrying random draws easier

## v2.2.4 (2026-08-11)

- Fixed many dependabot issues by bumping versions and removing unneeded packages.

## v2.2.3 (2026-07-03)

- Make compound units easier to define with __getitem__ so you can use something more like

    ```py
    US = stochas.UnitSystem.si()
    velocity = 4 * US["ft/s"]
    ```

- Fixed registering new units with `UnitSystem`

## v2.2.2 (2026-07-03)

- Sampled units now come with their actual unit instead of None

## v2.2.0 (2026-07-01)

- Added `Unitsystem`
    - Feature allows you to multipy a value by a unit and easily express that value in a declared unit system.
    - This is very helpful for going from IPS to SI.
    - Collections, distributions, and named values now track their units using a more formal system.
- Added an example to the docs on how to use `UnitSystem`

## v2.1.8 (2026-06-29)

- Sampled named values now inherit metadata from distributions when sampled.

## v2.1.7 (2026-06-29)

- Added metadata mixinwhich adds metadata fields to parameters.
- Removed `StochasBase.with_override()`.

## v2.1.6 (2026-06-20)

- Added `reset_rng` argument to `sample_dist` since `with_seed` and `with_trial_num` always reset RNG.
- Broke up Distribution into Continuous and Discrete.

## v2.1.5 (2026-06-16)

- Added support for a **bunch** of new distributions
- Fix windows bug where newlines got added to inputs table

## v2.1.4 (2026-06-15)

- Fix serialization of distribution

## v2.1.2 (2026-06-15)

- Added new metadata fields for categorizing and declaring the units of a distribution.
- Added the `AnyDist.to_tables()` method to easily record a distribution to a table.
    - Tables are broken up by distribution type and category.

## v2.1.1 (2026-06-14)

- Expanded unit tests
- Changed docs site to `zensical` instead of `mkdocs`

## v2.1.0 (2026-06-13)

- Added Rayleigh distribution
- Validations for design variables

## v2.0.4 (2026-05-27)

- Fixed the `sample_dist` bug where repeated calls would crash the code rather than return the existing

## v2.0.3 (2026-05-06)

- Fixes for multiprocessing

## v2.0.2 (2026-05-01)

- Docs pages improvements

## v2.0.1 (2026-04-24)

- Updated site CSS and logo

## v2.0.0 (2026-04-20)

- Added design variable system `mujoco-mojo`. Also expanded this system to support:
    - Boolean suggestors
    - `pymoo` suggestions
    - Value refinement when rerunning a study
- Expanded documentation.

## v1.0.4 (2026-04-18)

- Fixes to named value serialization with generic types
- Switching to use a sentinel value instead of overloading the NamedValueState

## v1.0.3 (2026-04-17)

- Added support for python 3.14

## v1.0.2 (2026-04-04)

- Branding updated
- CI/CD fixes and improvements

## v1.0.0 (2026-04-03)

- Initial release of `stochas`. This package is based on work by Talbot Knighton.
    - This update was effectively a full rewrite of `process_manager` and was renamed to `stochas`
- Change license
- Update README, docs site
- Added distributions to public API
- NamedValue work like numerical values via NumericMixin
