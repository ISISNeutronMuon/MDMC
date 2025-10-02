"""
Reader and writer for parametrised files.
"""

import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence, Tuple, Union

from MDMC.common.decorators import repr_decorator
from MDMC.MD.parameters import Parameter, Parameters

# Regexps to recognise numbers
FNUMBER_RE = r"(?:[+-]?(?:\d*\.?\d+|\d+\.?\d*))"
EXPNUMBER_RE = rf"(?:{FNUMBER_RE}(?:[Ee][+-]?\d{{1,3}})?)"
INTNUMBER_RE = r"(?:\d+)"
NUMBER_RE = rf"(?:{EXPNUMBER_RE}|{FNUMBER_RE}|{INTNUMBER_RE})"
WORD_RE = r"[a-zA-Z0-9_]+"

PARAM_RE = re.compile(rf"""
\{{\s*
(?P<param>{WORD_RE})    # Name of letters and underscores
\s*:\s*                 # Colon
(?P<value>{NUMBER_RE})  # A floating point or integer value
\s*\}}
""", re.VERBOSE)

BRACK_RE = re.compile(rf"\{{({WORD_RE})\}}")  # RE to capture non-matched parameters

OUT_RE = r"{\g<param>}"

PathLike = Union[str, Path]
PathsDict = dict[str, Union[str, Path]]


@repr_decorator('param_dict')
class ParamFileParser():
    """
    Class to read and parse parametrised files.

    Reads (from ``file_names``) and builds a `Parameters` object of the
    modifiable parameters.

    It can then dump a copy of the file with parameters replaced by their
    current value as stored in `_param_dict`

    Attributes
    ----------
    file_name : dict[str, Path]
        File names read into object.
    required_parameters : list[str]
        Parameters which must be present in the files for them to be valid.
    """

    def __init__(self,
                 file_names: PathsDict,
                 required_parameters: Sequence[str] = ()):
        """
        Class to read and parse parametrised files.

        Parameters
        ----------
        file_names : PathsDict
            Files to read data from.
        required_parameters : Sequence[str], optional
            Set required parameters which must be present in files
            for param file to be valid.
        """
        self.file_name = file_names
        self._files = None
        self.required_parameters = required_parameters
        self._param_dict = Parameters()

    def __call__(self, *keys, **settings) -> "ParamFileParser":
        """
        Dump selection of files and return object.

        For use in ``with`` blocks to handle closure.

        Parameters
        ----------
        *keys : str
            Keys to dump.
        **settings
            Extra options.

        Returns
        -------
        ParamFileParser
            Object holding temporary files.
        """
        # pylint: disable=consider-using-with
        to_dump = {key: self.file_name[key] for key in keys}
        name_parts = ((f"{pth.stem}_", pth.suffix) for pth in to_dump.values())

        self._files = tuple(NamedTemporaryFile(mode="w+", encoding='utf-8',
                                               prefix=pref, suffix=suff)
                            for pref, suff in name_parts)

        names = {key: file.name for key, file in zip(to_dump, self._files)}
        self.dump(names, **settings)
        for file in self._files:
            file.seek(0)
        return self

    def __enter__(self) -> Tuple[PathLike, ...]:
        """
        Dump selected keys to file and return paths to said files.

        Returns
        -------
        Tuple[PathLike, ...]
            Tuple containing dumped file paths.
        """
        return tuple(file.name for file in self._files)

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close and remove files.

        Parameters
        ----------
        exception_type : Exception
            Reason for exit.
        exception_value : str
            Reason for exit.
        traceback : Sequence[str]
            Trace on exit.
        """
        for file in self._files:
            file.close()
        self._files = None

    @property
    def file_name(self) -> dict[str, PathLike]:
        """
        Contained file names.

        Returns
        -------
        dict[str, PathLike]
            Dictionary mapping internal key to source file.
        """
        return self._file_name

    @file_name.setter
    def file_name(self, path: PathsDict):
        path = dict(path)
        self._file_name = {key: Path(val) for key, val in path.items()}

    @property
    def param_dict(self) -> dict[str, Any]:
        """
        Dictionary of found parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping param keys to current values.
        """
        return {parm.type: parm.value for parm in self.as_parameters.values()}

    @param_dict.setter
    def param_dict(self, in_dict: dict) -> None:
        self._param_dict = Parameters([
            Parameter(
                name=key,
                value=val,
                fixed=key in self.required_parameters or key.startswith("_"),
            )
            for key, val in in_dict.items()
        ])

    def update_param_dict(self, in_dict: dict) -> None:
        """
        Set internal parameters from dictionary as per :any:`dict.update`.

        Parameters
        ----------
        in_dict : dict
            Dictionary to take parameters from.
        """
        for key, val in in_dict.items():
            self._param_dict[key].fixed = False
            self._param_dict[key].value = val
            self._param_dict[key].fixed = True

    @property
    def as_dict(self) -> dict[str, Any]:
        """
        Return dictionary of found parameters s a `dict` object.

        Returns
        -------
        dict[str, Any]
            Found parameters as `dict` object.
        """
        return self.param_dict

    @property
    def as_parameters(self) -> Parameters:
        """
        Return dictionary of found parameters as a `Parameters` object.

        Returns
        -------
        Parameters
            Found parameters as `Parameters` object.
        """
        return self._param_dict

    def parse(self, **settings: dict) -> None:
        """
        Parse the data file into a `Parameters` object.

        Parses the file data so that it is in a format expected by the class
        calling the data reader.

        For readers which are not specific to one data type, the calling class
        must be determined so that the file data can be parsed into
        the appropriate data type.

        Parameters
        ----------
        **settings : dict
            Dictionary of settings from user.

        Raises
        ------
        KeyError
            If file doesn't contain all required parameters.
        ValueError
            If doubly defined or undefined parameter.
        """

        param_dict = {}

        for filename in self.file_name.values():
            with open(filename, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    for match in PARAM_RE.finditer(line):
                        param, value = match['param'], float(match['value'])

                        if param in param_dict and value != param_dict[param]:
                            print(f"Warning {file.name}:line {i}:"
                                  f"parameter {param} already defined.\n"
                                  f"Overriding value ({param_dict[param]}) with {value}")

                        param_dict[param] = value

                    # Catch invalid groups
                    for match in BRACK_RE.finditer(line):
                        if match[1] not in param_dict and not match[1].startswith("_"):
                            raise ValueError(f"{file.name}:line {i}:"
                                             f"Unrecognised parameter {match[0]}: "
                                             "no/invalid initialiser")

                        # Internal param to be set internally later
                        param_dict[match[1]] = None

        if (
            any(param not in param_dict for param in self.required_parameters) and
            not settings.get("testing", False)
        ):
            raise IOError("One or more required parameters not present in files. \n"
                          f"Expected: {self.required_parameters}.\n"
                          f"Missing: {tuple(set(self.required_parameters) - param_dict.keys())}.\n"

                          f"Found: {tuple(param_dict.keys())}.")

        self.param_dict = param_dict

    def dump(self, out_files: PathsDict, **settings: dict) -> None:
        """
        Write the parametrised file with current parameters.

        Parameters
        ----------
        out_files : PathsDict
            Files to write.
        **settings : dict
            Extra user options.

        Raises
        ------
        KeyError
            Invalid source file.
        """
        for key, out_filename in out_files.items():
            if key not in self.file_name:
                raise KeyError(f"No file in known files called {key}")

            in_filename = self.file_name[key]

            with (open(in_filename, 'r', encoding='utf-8') as in_file,
                  open(out_filename, 'w', encoding='utf-8') as out_file):
                for line in in_file:

                    line = re.sub(PARAM_RE, OUT_RE, line)
                    print(line.format(**self.param_dict), file=out_file, end="")
