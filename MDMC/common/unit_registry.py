"""Contains the Pint unit registry for MDMC, and functions that interact with it."""
from typing import Any
import warnings

import pint

UREG = pint.UnitRegistry()
UREG.setup_matplotlib()
UREG.load_definitions('pint_defs.txt')
UREG.default_system = "MDMC"
UREG.default_format = '~D'

print((1 * UREG.Unit('kcal / (angstrom * mol)')).to_base_units())

# we currently have to define translations for literally every unit
# (including interaction functions) because Pint doesn't do piecewise conversion yet
# and also angles have no dimension
MDMC_UNITS = {
    'LENGTH': UREG.angstrom,
    'TIME': UREG.fs,
    'MASS': UREG.amu,
    'CHARGE': UREG.e,
    'ANGLE': UREG.deg,
    'TEMPERATURE': UREG.kelvin,
    'ENERGY': UREG.kJ / UREG.mol,
    'FORCE': UREG.kJ / (UREG.angstrom * UREG.mol),
    'PRESSURE': UREG.Pa,
    'ENERGY_TRANSFER': UREG.meV,
    'POTENTIAL_STRENGTH': UREG.kJ / (UREG.mol * (UREG.ang ** 2)),
    'BUCK_B': 1 / UREG.angstrom,
    'BUCK_C': (UREG.angstrom ** 6) * UREG.kJ / UREG.mol
}

LAMMPS_UNITS = {
    'LENGTH': UREG.angstrom,
    'TIME': UREG.fs,
    'MASS': UREG.g / UREG.mol,
    'CHARGE': UREG.e,
    'ANGLE': UREG.deg,
    'TEMPERATURE': UREG.kelvin,
    'ENERGY': UREG.kcal / UREG.mol,
    'FORCE': UREG.kcal / (UREG.angstrom * UREG.mol),
    'PRESSURE': UREG.atm,
    'POTENTIAL_STRENGTH': UREG.kcal / (UREG.mol * (UREG.ang ** 2)),
    'BUCK_B': 1 / UREG.angstrom,
    'BUCK_C': (UREG.angstrom ** 6) * UREG.kcal / UREG.mol
}


def convert_unit(quantity: pint.Quantity, target: str, **settings) -> pint.Quantity:
    """
    Converts units between the unit system used by MDMC and unit systems used by MD engines.
    Currently accepted systems:
    'MDMC'
    'LAMMPS'

    Parameters
    ----------
    quantity: pint.Quantity
        The Pint quantity to be converted.
    target: str
        The target system.
    **settings:
        strip: bool
            if True, converts the quantity to just its magnitude after unit conversion.

    Returns
    -------
    pint.Quantity
        The quantity converted to the target system's units.
    """

    systems = {
        'MDMC': MDMC_UNITS,
        'LAMMPS': LAMMPS_UNITS
    }

    if not isinstance(quantity, pint.Quantity):
        warnings.warn("convert_unit was passed a quantity with no units. Was this intentional?")
        return quantity

    try:
        target_system = systems[target]
    except KeyError as error:
        raise KeyError("The system you are attempting to convert to is not recognised by MDMC. "
                       f"Recognised systems are: {systems.keys()}") from error

    try:
        result = quantity.to(target_system[_get_quantity_type(quantity)])
    except KeyError as error:
        raise KeyError(f"The quantity type {_get_quantity_type(quantity)} is not recognised by "
                       f"target system {target}") from error
    except pint.DimensionalityError:
        # amu <-> g/mol conversion is not currently supported by pint
        if _get_quantity_type(quantity) == 'MASS':
            if dict(quantity.dimensionality) == {'[mass]': 1}:
                quantity = quantity.to(
                    UREG.amu).magnitude * (UREG.g / UREG.mol)
            elif dict(quantity.dimensionality) == {'[mass]': 1, '[substance]': -1}:
                quantity = quantity.to(
                    (UREG.g / UREG.mol)).magnitude * UREG.amu
            result = quantity.to(target_system[_get_quantity_type(quantity)])

    if settings.get('strip', False):
        return strip_unit(result)
    return result


def _get_quantity_type(quantity):

    quantity_types = {
        '[length]': 'LENGTH',
        '[time]': 'TIME',
        '[mass]': 'MASS',
        '[mass] / [substance]': 'MASS',
        '[current] * [time]': 'CHARGE',
        'dimensionless': 'ANGLE',  # angles have no dimensionality
        '[temperature]': 'TEMPERATURE',
        '[length] ** 2 * [mass] / [substance] / [time] ** 2': 'ENERGY',
        '[length] * [mass] / [substance] / [time] ** 2': 'FORCE',
        '[mass] / [length] / [time] ** 2': 'PRESSURE',
        '[length] ** 2 * [mass] / [time] ** 2': 'ENERGY_TRANSFER',
        '[mass] / [substance] / [time] ** 2': 'POTENTIAL_STRENGTH',
        '1 / [length]': 'BUCK_B',
        '[length] ** 8 * [mass] / [substance] / [time] ** 2': 'BUCK_C'
        
    }

    try:
        return quantity_types[str(quantity.dimensionality)]
    except KeyError as error:
        raise KeyError("The quantity you're attempting to convert is not of a type "
                       "recognised by MDMC. Recognised quantity types are: "
                       f"{quantity_types.values()}") from error

def strip_unit(obj: Any) -> Any:
    """
    Remove Pint units from a quantity if it has them, else do nothing.
    Useful if an object can be a Quantity but can also be another data type
    (as accessing .magnitude will then raise an AttributeError)

    Parameters
    ----------
    obj: Any
        The object that may or may not be a Quantity.
    """

    if isinstance(obj, pint.Quantity):
        return obj.magnitude
    return obj
