"""
Reader and writer for parametrised files
"""

import re
from pathlib import Path

from MDMC.MD.parameters import Parameters, Parameter
from MDMC.readers.reader import Reader

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

BRACK_RE = re.compile(r"\{([^: \t]*)\s*:?.*\}")  # RE to capture non-matched parameters

OUT_RE = r"{\g<param>}"


class ParamFileParser(Reader):
    """ParamFileParser reads parametrised files (from `file_names`) and builds a
    dictionary of the modifiable parameters.

    It can then dump a copy of the file with parameters replaced by their
    current value as stored in `_param_dict`
    """

    def __init__(self, file_names: str | list[str]):
        Reader.__init__(self, file_names)
        self._param_dict = Parameters()

    def __enter__(self) -> None:
        self.file = map(lambda fn: open(fn, 'r', encoding='utf-8'), self.file_name)  # noqa: SIM115
        return self.file

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        for file in self.file:
            file.close()
        self.file = None

    @property
    def file_name(self):
        """ Contained file names """
        return self._file_name

    @file_name.setter
    def file_name(self, path: str):
        if isinstance(path, str):
            path = (path,)
        self._file_name = tuple(Path(file) for file in path)

    @property
    def param_dict(self):
        """ Dictionary of found parameters """
        return self._param_dict

    def parse(self, **settings: dict) -> None:
        """
        Parses the file data so that it is in a format expected by the class
        calling the data reader

        For readers which are not specific to one data type, the calling class
        must be determined so that the file data can be parsed into
        the appropriate data type.

        Parameters
        ----------
        **settings: dict
            dictionary of settings for reader
        """

        param_dict = {}

        with self as files:
            for file in files:
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
                        if match[1] not in param_dict:
                            raise ValueError(f"{file.name}:line {i}:"
                                             f"Unrecognised parameter {match[0]}:"
                                             "no/invalid initialiser")

        self._param_dict = Parameters([Parameter(name=key, value=val)
                                       for key, val in param_dict.items()])

    def dump(self, out_files: str | list[str], **settings: dict) -> None:
        """
        Write the parametrised file out with the parameters
        replaced with the appropriate values
        """
        if isinstance(out_files, str):
            out_files = (out_files,)

        if len(out_files) != len(self.file_name):
            raise IndexError(f"Number of out files {len(out_files)} "
                             f"does not match number of in files {len(self.file_name)}")

        for in_filename, out_filename in zip(self.file_name, out_files):
            with (open(in_filename, 'r', encoding='utf-8') as in_file,
                  open(out_filename, 'w', encoding='utf-8') as out_file):
                for line in in_file:
                    line = re.sub(PARAM_RE, OUT_RE, line)

                    print(line.format(**self.param_dict), file=out_file, end="")
