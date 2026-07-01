"""
Physical unit system for MuJoCo Mojo models.

Declare the unit system used in a model so that values in the generate step can be
expressed in any unit and converted automatically, and so that telemetry metadata can
report concrete units instead of abstract Pint dimensions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload

import pint
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import numpy as np

__all__ = ["UnitDescriptor", "UnitSystem", "ureg"]

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()


class UnitDescriptor(BaseModel):
    """
    A named unit with an affine conversion (scale + offset) to the model's base unit for that dimension.

    The full conversion is `base_value = scale * source_value + offset`. For purely multiplicative units (everything except temperature offsets like degF/degC), `offset` is 0 and `float(descriptor)` returns the scale — multiply by it to convert FROM this unit INTO the model's base unit for the same physical dimension. `str(descriptor)` returns the unit name (e.g. `"inch"`, `"meter"`).

    The `scale` and `offset` are computed by `UnitSystem.__getattr__` and excluded from serialization. After deserializing a model that contains `UnitDescriptor` fields, call `StochasBase.with_unit_system(us)` to re-populate them before sampling.

    Example usage::

        mojo_model.u = UnitSystem.si()  # base units: meter, kilogram, second

        body.inertial = mj.Inertial(
            mass=2.5,
            pos=Pos(np.array([0, 1, 2]) * mojo_model.u.inch),  # converts inches to meters
        )
    """

    model_config = ConfigDict(frozen=True)

    name: str
    """Unit name (e.g. `"inch"`, `"pound"`). Serialized; used to re-resolve the conversion parameters via `StochasBase.with_unit_system()`."""

    scale: float | None = Field(default=None, exclude=True)
    """Multiplicative scale from this unit to the model base unit. Excluded from serialization; re-populated by `StochasBase.with_unit_system()`."""

    offset: float = Field(default=0.0, exclude=True)
    """Additive offset applied after scaling: `base = scale * source + offset`. Zero for all non-offset units (everything except absolute temperature conversions like degF/degC). Excluded from serialization."""

    def __float__(self) -> float:
        if self.scale is None:
            msg = (
                f"UnitDescriptor '{self.name}' has no conversion scale. "
                "Call with_unit_system(us) on the model to resolve it against a UnitSystem."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        return self.scale

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"UnitDescriptor({self.name!r}, scale={self.scale!r}, offset={self.offset!r})"

    @overload
    def __mul__(self, other: int | float) -> float: ...
    @overload
    def __mul__(self, other: np.ndarray) -> np.ndarray: ...
    @overload
    def __mul__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any:
        return other * float(self) + self.offset

    @overload
    def __rmul__(self, other: int | float) -> float: ...
    @overload
    def __rmul__(self, other: np.ndarray) -> np.ndarray: ...
    @overload
    def __rmul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any:
        return other * float(self) + self.offset

    @overload
    def __truediv__(self, other: int | float) -> float: ...
    @overload
    def __truediv__(self, other: np.ndarray) -> np.ndarray: ...
    @overload
    def __truediv__(self, other: Any) -> Any: ...
    def __truediv__(self, other: Any) -> Any:
        return float(self) / other

    @overload
    def __rtruediv__(self, other: int | float) -> float: ...
    @overload
    def __rtruediv__(self, other: np.ndarray) -> np.ndarray: ...
    @overload
    def __rtruediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any:
        return other / float(self)


class UnitSystem(BaseModel):
    """
    Declares the physical unit system for a MuJoCo model.

    Each field names the base unit for one Pint dimension. Any Pint-recognized unit -- base or compound -- can be accessed as an attribute and returns a `UnitDescriptor` whose `scale` and `offset` describe the affine conversion `base = scale * source + offset` FROM that unit INTO the model's equivalent unit for the same dimensionality. For all non-temperature units `offset` is 0 and `float(descriptor)` returns `scale`. A dimension must be configured for every component that appears in the target unit's dimensionality; otherwise `AttributeError` is raised.

    Built-in coherent factory methods -- in each, the natural force unit is the product of the mass and length base units divided by time squared:

    - `UnitSystem.si()`: meter / kilogram / second -- force = newton
    - `UnitSystem.cgs()`: centimeter / gram / second -- force = dyne
    - `UnitSystem.fps()`: foot / slug / second -- force = lbf
    - `UnitSystem.ips()`: inch / slinch / second -- force = lbf  (slinch = 12 slugs = lbf*s^2/in)

    Example::

        mojo_model.u = UnitSystem.si()

        float(mojo_model.u.meter)        # -> 1.0 (scale; offset is 0.0)
        float(mojo_model.u.inch)         # -> 0.0254
        float(mojo_model.u.newton)       # -> 1.0  (kg*m/s^2 in SI)
        float(mojo_model.u.lbf)          # -> 4.448...
        float(mojo_model.u.horsepower)   # -> 745.7...
        float(mojo_model.u.volt)         # -> 1.0  (requires current="ampere")
        mojo_model.u.degF.scale          # -> 5/9; .offset -> 255.372... K
        str(mojo_model.u.inch)           # -> "inch"
        mojo_model.u.length              # -> "m" (for UnitSystem.si())
    """

    # --- mechanical base dimensions ---
    length: str
    """Base length unit (e.g. `"meter"`, `"inch"`, `"foot"`)."""
    mass: str
    """Base mass unit (e.g. `"kilogram"`, `"slug"`, `"slinch"`, `"gram"`). Use a coherent mass unit for the chosen length scale so that derived force units come out naturally -- see the factory methods for the standard combinations."""
    time: str = "s"
    """Base time unit. Defaults to `"second"`, which is the conventional choice, but MuJoCo has no intrinsic time scale -- it treats time as whatever unit the user treats it as."""

    # --- other SI base dimensions (all optional; only needed when resolving units in those dimensions) ---
    temperature: str | None = None
    """Base temperature unit (e.g. `"kelvin"`, `"degC"`). Required for thermal units like `joule_per_kelvin`."""
    current: str | None = None
    """Base electric current unit (e.g. `"ampere"`). Required for electromagnetic units like `volt`, `ohm`, `farad`."""
    amount: str | None = None
    """Base amount-of-substance unit (e.g. `"mole"`). Required for molar quantities."""
    luminosity: str | None = None
    """Base luminous intensity unit (e.g. `"candela"`). Required for photometric units like `lux`, `lumen`."""

    @classmethod
    def si(cls) -> UnitSystem:
        """SI base units: meter, kilogram, second, kelvin, ampere, mole, candela."""
        return cls(
            length="m",
            mass="kg",
            temperature="K",
            current="A",
            amount="mol",
            luminosity="cd",
        )

    @classmethod
    def cgs(cls) -> UnitSystem:
        """Centimeter-gram-second system (mechanical dimensions only). Coherent force unit: dyne (1 dyne = 1 g*cm/s^2 = 1e-5 N). Extend with `model_copy` to add thermal or electromagnetic base units."""
        return cls(length="cm", mass="g")

    @classmethod
    def fps(cls) -> UnitSystem:
        """Foot-slug-second system (mechanical dimensions only). Coherent force unit: lbf (1 lbf = 1 slug*ft/s^2). Extend with `model_copy` to add thermal or electromagnetic base units."""
        return cls(length="ft", mass="slug")

    @classmethod
    def ips(cls) -> UnitSystem:
        """Inch-slinch-second system (mechanical dimensions only). Coherent force unit: lbf (1 lbf = 1 slinch*in/s^2; 1 slinch = 12 slugs). Extend with `model_copy` to add thermal or electromagnetic base units."""
        return cls(length="in", mass="slinch")

    def __getattr__(self, name: str) -> UnitDescriptor:
        if name.startswith("_"):
            msg = (
                f"{name!r} is not a recognized since it starts with an underscore ('_')"
            )
            logger.error(msg)
            raise AttributeError(msg)

        try:
            unit = ureg.parse_units(name)
        except Exception:
            msg = f"{name!r} is not a recognized Pint unit and is not an attribute of UnitSystem"
            logger.error(msg)
            raise AttributeError(msg) from None

        dim_dict = dict(ureg.get_dimensionality(unit))

        # map Pint dimension strings to the configured base unit name (None entries are skipped)
        base_unit_for_dim: dict[str, str] = {
            k: v
            for k, v in {
                "[length]": self.length,
                "[mass]": self.mass,
                "[time]": self.time,
                "[temperature]": self.temperature,
                "[current]": self.current,
                "[substance]": self.amount,
                "[luminosity]": self.luminosity,
            }.items()
            if v is not None
        }

        # build the compound model unit for this dimensionality by multiplying base units raised
        # to their exponents, e.g. {"[mass]": 1, "[length]": 1, "[time]": -2} (force) ->
        # kilogram * meter * second^-2 for SI.  dimensionless units (empty dim_dict) pass through
        # unchanged since the loop never executes.
        model_unit = ureg.dimensionless
        for dim_key, exp in dim_dict.items():
            base_name = base_unit_for_dim.get(dim_key)
            if base_name is None:
                configured = ", ".join(
                    f"{k}={v!r}" for k, v in base_unit_for_dim.items()
                )
                msg = (
                    f"UnitSystem cannot resolve a model unit for {name!r}: "
                    f"no base unit configured for Pint dimension {dim_key!r} "
                    f"(configured: {configured})"
                )
                logger.error(msg)
                raise AttributeError(msg)
            model_unit = model_unit * ureg.parse_units(base_name) ** exp

        offset = float(ureg.Quantity(0, unit).to(model_unit).magnitude)
        scale = float(ureg.Quantity(1, unit).to(model_unit).magnitude) - offset
        return UnitDescriptor(name=name, scale=scale, offset=offset)


if __name__ == "__main__":
    breakpoint()
