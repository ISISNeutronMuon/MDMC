# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Read TINKER ``.prm`` files and convert them to MDMC force field python files.
"""

import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import periodictable

from MDMC.common import units
from MDMC.common.decorators import wrap_docstring
from MDMC.common.df_operations import filter_dataframe
from MDMC.MD import force_fields


def read_prm(fname: str, ncols: int = 14) -> pd.DataFrame:
    """
    Read a TINKER ``.prm`` (force field) file.

    Ignores all errors raised by incorrectly formatted data, so will be
    susceptible to incorrectly formatted ``.prm`` files.

    Parameters
    ----------
    fname : str
        The name of the ``.prm`` file.
    ncols : int, optional
        The largest number of columns that should be read. The default is 14,
        which is the number of columns required for the torsion (proper
        dihedral) in ``oplsaa.prm``.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns (named c0, c1, ...) up to (but not including)
        ncols. Each row contains a row read correctly (i.e. no Error thrown)
        from the ``.prm`` file.
    """

    return pd.read_csv(
        fname,
        delim_whitespace=True,
        error_bad_lines=False,
        header=None,
        names=dummy_headers(0, ncols),
    )


def parse_prm(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """
    Parse a ``.prm`` file into DataFrames.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A dataframe of raw data read from a ``.prm`` file.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        A tuple of dataframes, with each dataframe possessing only the entries
        starting with the corresponding definition.
    """

    ncols = len(dataframe.columns)
    dtypes = np.array(
        [
            [
                "atom",
                ["c3"] + dummy_headers(8, ncols),
                ["atom_type", "atom_group", "name", "atomic_num", "mass", "nbonds"],
            ],
            ["vdw", dummy_headers(4, ncols), ["atom_type", "sigma", "epsilon"]],
            [
                "bond",
                dummy_headers(5, ncols),
                ["atom_group1", "atom_group2", "potential_strength", "equilibrium_state"],
            ],
            [
                "angle",
                dummy_headers(6, ncols),
                [
                    "atom_group1",
                    "atom_group2",
                    "atom_group3",
                    "potential_strength",
                    "equilibrium_state",
                ],
            ],
            [
                "imptors",
                dummy_headers(8, ncols),
                ["atom_group2", "atom_group3", "atom_group1", "atom_group4", "K1", "d1", "n1"],
            ],
            [
                "torsion",
                [],
                [
                    "atom_group1",
                    "atom_group2",
                    "atom_group3",
                    "atom_group4",
                    "K1",
                    "d1",
                    "n1",
                    "K2",
                    "d2",
                    "n2",
                    "K3",
                    "d3",
                    "n3",
                ],
            ],
            ["charge", dummy_headers(3, ncols), ["atom_type", "charge"]],
        ],
    )

    atoms, disps, bonds, angles, impropers, propers, charges = tuple(
        parse_dataframe(filter_dataframe([dtype[0]], dataframe, ["c0"]), *dtype[1:])
        for dtype in dtypes
    )

    # Combine charges with atoms
    atoms.insert(3, "charge", charges.charge.to_numpy())

    # Change atomic number to element
    atoms["atomic_num"] = atoms["atomic_num"].apply(lambda x: periodictable.elements[int(x)])
    atoms.rename(columns={"atomic_num": "element"}, inplace=True)

    # Rearrange order of proper and improper dihedrals
    # For proper dihedrals, just the parameters need to be rearranged, as the
    # convention for the atom order is the same as MDMC (end1 central1 central2
    # end2).
    propers = propers[
        [
            "atom_group1",
            "atom_group2",
            "atom_group3",
            "atom_group4",
            "K1",
            "n1",
            "d1",
            "K2",
            "n2",
            "d2",
            "K3",
            "n3",
            "d3",
        ]
    ]
    # For improper dihedrals, the atom_groups are rearranged so that the central
    # atom is first (in keeping with LAMMPS anf GROMACS). This occurs because
    # the atom order in TINKER is end1, end2, central, end3 for impropers.
    impropers = impropers[
        ["atom_group1", "atom_group2", "atom_group3", "atom_group4", "K1", "n1", "d1"]
    ]

    # Convert units
    disps.epsilon = convert_units(disps.epsilon)
    bonds.potential_strength = convert_units(bonds.potential_strength)
    angles.potential_strength = convert_units(angles.potential_strength)
    impropers.K1 = convert_units(impropers.K1)
    for k_parameter in ["K1", "K2", "K3"]:
        propers[k_parameter] = convert_units(propers[k_parameter])

    return atoms, disps, bonds, angles, impropers, propers


def write_force_field_module(
    fname: str,
    atoms: pd.DataFrame,
    *interactions: pd.DataFrame,
    path: str = None,
    module_docstring: str = None,
    class_docstring: str = None,
    **settings,
):
    """
    Write a temporary module to allow Tinker potentials to be imported.

    Parameters
    ----------
    fname : str
        Name of the module (excluding the file extension) and the class.
    atoms : pandas.DataFrame
        Atoms to dump.
    *interactions
        An arbitrary number of `pandas.DataFrame` objects.
    path : str, optional
        Path to dump to.
    module_docstring : str, optional
        Docstring for created module.
    class_docstring : str, optional
        Docstring for created class.
    **settings
        Extra options.

    Notes
    -----
    Windows and Mac OS are case insensitive, so if a file exists in the
    specified path with the same case insensitive name as fname (e.g. if
    fname='spce' and 'SPCE' already exists), the existing file will be
    overwritten. So on case insensitive OS, fname should always be unique,
    without considering the case.
    """

    line_length = settings.get("line_length", 80)
    data_fname = settings.get("data_fname", fname)

    imports = "from MDMC.MD.force_fields.ff import FileForceField\n"

    if path is None:
        path = os.path.abspath(force_fields.__file__).replace("__init__.py", "")
    if module_docstring is None:
        module_docstring = (
            f'"""A module for defining the {fname} force field. This'
            " was generated from the corresponding TINKER"
            ' file."""'
        )
    if class_docstring is None:
        class_docstring = f'"""\n{fname} force field, with defined atoms and interactions\n'
    data_fname = path + "data/" + data_fname + ".dat"
    with open(path + fname + ".py", "w", encoding="UTF-8") as module:
        module.write(wrap_docstring(module_docstring, line_length) + "\n" * 2)
        module.write(imports + "\n" * 2)
        module.write(f"class {fname}(FileForceField):\n\n")
        module.write(
            textwrap.indent(wrap_docstring(class_docstring, line_length), " " * 4) + "\n\n",
        )
        module.write(textwrap.indent(f'file_name = "{data_fname}"\n', " " * 4))

    # Write a dat file which contains all atoms and interactions
    # Use pandas to output as CSV


def write_data(
    fname: str,
    atoms: pd.DataFrame,
    path: str = None,
    **settings,
) -> None:
    """
    Write forcefield data to a ``.dat`` file.

    Parameters
    ----------
    fname : str
        File name to write to.
    atoms : pd.DataFrame
        Force field to write.
    path : str, optional
        Path to read data from.
    **settings
        Extra options.

    Other Parameters
    ----------------
    orig_file : str
        Original file forcefield read from.

    Notes
    -----
    For each of `disp`, `bond`, `angle`, `proper` and `improper`, there
    exists a pair of optional arguments:
    ``<type>s`` and ``<type>_function``
    Which define the presence of parameters and the function to compute them.
    """
    inter_types = ["disps", "bonds", "angles", "propers", "impropers"]
    disps, bonds, angles, propers, impropers = (
        settings.get(inter_type, []) for inter_type in inter_types
    )

    inter_functions = [settings.get(inter_type[:-1] + "_function") for inter_type in inter_types]

    orig_file = settings.get("orig_file")

    if path is None:
        path = os.path.abspath(force_fields.__file__).replace("__init__.py", "data/")
    full_fname = path + fname + ".dat"
    # First line describes number of each data type
    description = (
        f"Atoms={len(atoms)} Dispersions={len(disps)} Bonds={len(bonds)} "
        f"BondAngles={len(angles)} Propers={len(propers)} Impropers={len(impropers)}\n"
    )
    # Second line describes types of interactions
    inter_functions = "Dispersion={} Bond={} BondAngle={} Proper={} Improper={}\n".format(
        *inter_functions,
    )
    # Metadata about the file
    original_file_str = f" It was generated from {orig_file}\n" if orig_file else "\n"
    date = datetime.today().strftime("%Y-%m-%d")
    metadata = wrap_docstring(
        f"\nThis file contains the {fname} force field. It was"
        f" created on {date}." + original_file_str,
        80,
    )
    with open(full_fname, "w", encoding="UTF-8") as out_datafile:
        out_datafile.write(description)
        out_datafile.write(inter_functions)
        out_datafile.write(metadata)

    for name, data in [
        ("Atoms", atoms),
        ("Dispersions", disps),
        ("Bonds", bonds),
        ("Bond Angles", angles),
        ("Proper Dihedrals", propers),
        ("Improper Dihedrals", impropers),
    ]:
        with open(full_fname, "a", encoding="UTF-8") as out_datafile:
            out_datafile.write("\n" + name + "\n")
            data.to_csv(out_datafile, sep="\t", index=False)


def dummy_headers(start: int, end: int) -> list[str]:
    """
    Generate dummy headers c0, c1, c2, ... until end (exclusive).

    Parameters
    ----------
    start : int
        The start of the header (inclusive).
    end : int
        The end of the header (exclusive).

    Returns
    -------
    list of str
        A `str` (``c{i}``) for each i between start and end (exclusive).
    """

    return [f"c{i}" for i in range(start, end)]


def parse_dataframe(dataframe: pd.DataFrame, drop: list, names: list) -> pd.DataFrame:
    """
    Parse a single dataframe.

    Parses a dataframe containing single datatype (e.g. atom or angles) by
    dropping any unnecessary columns and setting correct header names.

    If the dataframe has a header 'c0' this is always dropped

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe to be parsed.
    drop : list
        A list of names of headers which will be dropped.
    names : list
        A list of the header names which will be set for the remaining columns..

    Returns
    -------
    pandas.DataFrame
        A parsed ``DataFrame``.
    """

    try:
        # Assume that dataframe will have a header 'c0' which is always dropped
        dataframe = dataframe.drop(columns=["c0"] + drop)
    except KeyError:
        dataframe = dataframe.drop(columns=drop)
    dataframe.columns = names
    return dataframe


def convert_units(series: pd.Series) -> str:
    """
    Convert from TINKER units (kcal) to kJ.

    Parameters
    ----------
    series : pandas.Series
        The ``Series`` for which the parameters will be converted.

    Returns
    -------
    str
        Value in kJ.
    """

    return (series.astype(float) * units.kcal / units.mol).round(decimals=5).astype(str)
