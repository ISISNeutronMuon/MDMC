"""Contains the Pint unit registry for MDMC, and functions that interact with it."""
from typing import Any

import pint

UREG = pint.UnitRegistry()
UREG.setup_matplotlib()
UREG.define('@alias angstrom = ang = Ang = AA')
UREG.default_format = '~D'

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
    'ENERGY_TRANSFER': UREG.meV
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
    'PRESSURE': UREG.atm
}


def convert_unit(quantity: pint.Quantity, target: str) -> pint.Quantity:
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

    Returns
    -------
    pint.Quantity
        The quantity converted to the target system's units.
    """

    systems = {
        'MDMC': MDMC_UNITS,
        'LAMMPS': LAMMPS_UNITS
    }

    # dict to convert pint dimensionality to string quantity names
    quantity_types = {
        {'[length': 1}: 'LENGTH',
        {'[time]': 1}: 'TIME',
        {'[mass]': 1}: 'MASS',
        {'[mass]': 1, '[substance]': -1}: 'MASS',
        {'[current]': 1, '[time]': 1}: 'CHARGE',
        {}: 'ANGLE',  # angles have no dimensionality
        {'[temperature]': 1}: 'TEMPERATURE',
        {'[mass]': 1, '[length]': 2, '[time]': -2, '[substance]': -1}: 'ENERGY',
        {'[mass]': 1, '[length]': 1, '[time]': -2, '[substance]': -1}: 'FORCE',
        {'[mass]': 1, '[length]': -1, '[time]': -2}: 'PRESSURE',
        {'[time]': -2, '[mass]': 1, '[length]': 2}: 'ENERGY_TRANSFER'
    }

    try:
        quantity_type = quantity_types[dict(quantity.dimensionality)]
    except KeyError as error:
        raise KeyError("The quantity you're attempting to convert is not of a type "
                       "recognised by MDMC. Recognised quantity types are: "
                       f"{quantity_types.values()}") from error
    try:
        target_system = systems[target]
    except KeyError as error:
        raise KeyError("The system you are attempting to convert to is not recognised by MDMC. "
                       f"Recognised systems are: {systems.keys()}") from error

    try:
        quantity.to(target_system[quantity_type])
    except KeyError as error:
        raise KeyError(f"The quantity type {quantity_type} is not recognised by "
                       f"target system {target}") from error
    except pint.DimensionalityError:
        if quantity_type == 'MASS':  # amu <-> g/mol conversion is not currently supported by pint
            if dict(quantity.dimensionality) == {'[mass]': 1}:
                quantity = quantity.to(
                    UREG.amu).magnitude * (UREG.g / UREG.mol)
            if dict(quantity.dimensionality) == {'[mass]': 1, '[substance]': -1}:
                quantity = quantity.to(
                    (UREG.g / UREG.mol)).magnitude * UREG.amu
            quantity.to(target_system[quantity_type])

    return quantity


def scrub_unit(obj: Any) -> Any:
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
