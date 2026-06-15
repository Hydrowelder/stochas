"""Tests for design variables (DesignBool, DesignCategorical, DesignFloat, DesignInt) and their collections."""

import numpy as np
import optuna
import pytest
from pydantic import ValidationError
from pymoo.core.variable import Binary, Choice, Integer, Real

from stochas.design_variable import (
    DesignBool,
    DesignCategorical,
    DesignFloat,
    DesignInt,
    DesignValueDict,
    DesignValueList,
)
from stochas.named_value import ValueName


@pytest.fixture
def trial() -> optuna.Trial:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    return study.ask()


def test_design_bool():
    """Ensure DesignBool converts to pymoo and ignores refine."""
    db = DesignBool(name=ValueName("flag"))

    assert db.type == "binary"
    assert isinstance(db.to_pymoo(), Binary)

    db.refine(factor=0.5, best_params={"flag": True})


def test_design_bool_to_optuna(trial: optuna.Trial):
    """Ensure DesignBool.to_optuna returns a value from optuna's categorical sampler."""
    db = DesignBool(name=ValueName("flag"))

    assert db.to_optuna(trial) in (True, False)


def test_design_categorical(trial: optuna.Trial):
    """Ensure DesignCategorical converts to optuna and pymoo, and ignores refine."""
    dc = DesignCategorical[str](name=ValueName("mode"), choices=["stiff", "soft"])

    assert dc.type == "categorical"
    assert dc.to_optuna(trial) in ("stiff", "soft")

    choice = dc.to_pymoo()
    assert isinstance(choice, Choice)
    np.testing.assert_array_equal(
        choice.options, np.array(["stiff", "soft"], dtype=object)
    )

    dc.refine(factor=0.5, best_params={"mode": "stiff"})


def test_design_float_validate_bounds():
    """Ensure DesignFloat enforces high > low, log requires low > 0, and step > 0."""
    DesignFloat(name=ValueName("p"), low=0.0, high=10.0)

    with pytest.raises(ValidationError, match="high must be greater than low"):
        DesignFloat(name=ValueName("p"), low=10.0, high=0.0)

    with pytest.raises(
        ValidationError, match="low must be greater than 0 when log=True"
    ):
        DesignFloat(name=ValueName("p"), low=0.0, high=10.0, log=True)

    with pytest.raises(ValidationError, match="step must be greater than 0"):
        DesignFloat(name=ValueName("p"), low=0.0, high=10.0, step=0.0)


def test_design_float_to_optuna_and_pymoo(trial: optuna.Trial):
    """Ensure DesignFloat converts to optuna and pymoo representations."""
    df = DesignFloat(name=ValueName("p"), low=0.0, high=10.0)

    value = df.to_optuna(trial)
    assert 0.0 <= value <= 10.0

    real = df.to_pymoo()
    assert isinstance(real, Real)
    assert real.bounds == (0.0, 10.0)


def test_design_float_refine():
    """Ensure refine shrinks the bounds around the best value without exceeding the originals."""
    df = DesignFloat(name=ValueName("p"), low=0.0, high=10.0)

    df.refine(factor=0.5, best_params={"p": 5.0})

    assert df.low == 2.5
    assert df.high == 7.5


def test_design_int_validate_bounds():
    """Ensure DesignInt enforces high > low, log requires low > 0, and step > 0."""
    DesignInt(name=ValueName("n"), low=0, high=10)

    with pytest.raises(ValidationError, match="high must be greater than low"):
        DesignInt(name=ValueName("n"), low=10, high=0)

    with pytest.raises(
        ValidationError, match="low must be greater than 0 when log=True"
    ):
        DesignInt(name=ValueName("n"), low=0, high=10, log=True)

    with pytest.raises(ValidationError, match="step must be greater than 0"):
        DesignInt(name=ValueName("n"), low=0, high=10, step=0)


def test_design_int_to_optuna_and_pymoo(trial: optuna.Trial):
    """Ensure DesignInt converts to optuna and pymoo representations."""
    di = DesignInt(name=ValueName("n"), low=0, high=10)

    value = di.to_optuna(trial)
    assert 0 <= value <= 10

    integer = di.to_pymoo()
    assert isinstance(integer, Integer)
    assert integer.bounds == (0, 10)


def test_design_int_refine():
    """Ensure refine shrinks the integer bounds around the best value, rounding outward."""
    di = DesignInt(name=ValueName("n"), low=0, high=10)

    di.refine(factor=0.5, best_params={"n": 5})

    assert di.low == 2
    assert di.high == 8


def test_design_value_dict():
    """Ensure DesignValueDict supports NamedValue-keyed lookups and conversions."""
    d = DesignValueDict()
    df = DesignFloat(name=ValueName("p"), low=0.0, high=10.0, stored_value=5.0)
    d.update(df)

    assert df in d
    assert d.get_value("p") == 5.0
    assert d.get_raw_value("p") == 5.0

    nv_list = d.named_value_list
    assert isinstance(nv_list, DesignValueList)
    assert list(nv_list) == [df]


def test_design_value_list_to_dict():
    """Ensure DesignValueList converts to a DesignValueDict."""
    nv_list = DesignValueList()
    df = DesignFloat(name=ValueName("p"), low=0.0, high=10.0, stored_value=5.0)
    nv_list.append(df)

    d = nv_list.to_named_value_dict
    assert isinstance(d, DesignValueDict)
    assert d.get_value("p") == 5.0
