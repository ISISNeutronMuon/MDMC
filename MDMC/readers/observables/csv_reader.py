"""Reader for MDANSE CSV files"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from MDMC.common import units
from MDMC.readers.observables.obs_reader import ObservableReader

logger = logging.getLogger(__name__)

axes_target_units = {
    "r": units.SYSTEM["LENGTH"],
    "omega": units.SYSTEM["ENERGY_TRANSFER"],
    "romega": units.SYSTEM["ENERGY_TRANSFER"],
    "Q": units.SYSTEM["LENGTH"] ** -1,
}


def get_mdanse_header(file_name: str) -> tuple[dict[int, str], dict[int, str]]:
    col_labels = {}
    col_units = {}
    with open(file_name) as source:
        for line in source:
            if "#" not in line:
                break
            toks = line.split()
            if all("col" in tok for tok in toks[1:]):
                col_labels = {index: label.split(":")[1] for index, label in enumerate(toks[1:])}
                col_units = {index: label.split(":")[2] for index, label in enumerate(toks[1:])}
    return col_labels, col_units


class csv_reader(ObservableReader):
    """
    Reads data from CSV files.
    """

    def __init__(
        self,
        file_name,
        axis_columns: Sequence[int] | None = None,
        data_column: int | None = None,
        error_column: int | None = None,
    ):
        super().__init__(file_name)
        labels, unit_dict = get_mdanse_header(file_name)
        self._independent_variables = {}
        self._dependent_variables = {}
        self._errors = {}
        self.axis_indices = []
        self.error_index = None
        self.units = unit_dict
        self.labels = labels
        for index, label in labels.items():
            if label == "data":
                self.data_index = index
            elif label == "error":
                self.error_index = index
            else:
                self.axis_indices.append(index)
        self.axis_indices = axis_columns if axis_columns else self.axis_indices
        self.data_index = data_column if data_column else self.data_index
        self.error_index = error_column if error_column else self.error_index

    def parse(self, **settings: Any) -> None:
        """Read data from the input file."""
        raw_data = []
        with open(self.file_name) as source:
            for line in source:
                if "#" in line:
                    continue
                raw_data.append([float(x) for x in line.split(",")])
        data_array = np.array(raw_data)
        for ax_ind in self.axis_indices:
            axis_array = data_array[:, ax_ind]
            if self.labels[ax_ind] in axes_target_units:
                axis_array *= units.Unit(self.units[ax_ind]).conversion_factor
            self._independent_variables[self.labels.get(ax_ind, ax_ind)] = axis_array
        self._dependent_variables[self.labels.get(self.data_index, self.data_index)] = data_array[
            :,
            self.data_index,
        ]
        if self.error_index is not None:
            self._errors[self.labels.get(self.data_index, self.data_index)] = data_array[
                :,
                self.error_index,
            ]
        else:
            self._errors[self.labels.get(self.data_index, self.data_index)] = np.sqrt(
                np.abs(
                    data_array[
                        :,
                        self.data_index,
                    ],
                ),
            )

    @property
    def independent_variables(self) -> dict:
        """
        Get the independent variables, Q (in ``Ang^-1``) and E (``meV``)

        Returns
        -------
        dict
            The independent variables Q and E
        """

        return self._independent_variables

    @property
    def dependent_variables(self) -> dict:
        """
        Get the dependent variables, SQw (in ``arb``)

        Returns
        -------
        dict
            The dependent variables, SQw (in ``arb``)
        """

        return self._dependent_variables

    @property
    def errors(self) -> dict:
        """
        Get the errors on the dependent variables

        Returns
        -------
        dict
            The error on SQw (in ``arb``)
        """

        return self._errors
