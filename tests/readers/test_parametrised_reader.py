"""Tests for parametrised reading and dumping

Creates a number of temporary StringIO file objects
"""
import tempfile
import pytest

from MDMC.readers.simulations.param_file import ParamFileParser
from MDMC.MD.parameters import Parameters


class TestClassRead:
    """
    Test that ParamFileParser correctly pulls values from filess
    """

    def test_parse_simple_file(self):
        """
        Ensure parser can read the file
        """
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            tmp.write("""The param is {param: 3.5}""")
            tmp.seek(0)
            parser = ParamFileParser({'main': tmp.name})
            parser.parse(testing=True)
            self.compare_params(parser.param_dict, {'param': 3.5})

    def test_parse_multiple_keys(self):
        """
        Ensure parser can read multiple keys
        """
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            tmp.write("""The param is {param: 3.5} {paramb: 10}""")
            tmp.seek(0)
            parser = ParamFileParser({'main': tmp.name})
            parser.parse(testing=True)
            self.compare_params(parser.param_dict, {'param': 3.5, 'paramb': 10.0})

    def test_parse_multiple_keys_lines(self):
        """
        Ensure parser can read multiple keys on multiple lines
        """
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            tmp.write("""The param is {param: 3.5}
                         The second param is {paramb: 10}""")
            tmp.seek(0)
            parser = ParamFileParser({'main': tmp.name})
            parser.parse(testing=True)
            self.compare_params(parser.param_dict, {'param': 3.5, 'paramb': 10.0})

    def test_parse_same_key(self):
        """
        Ensure parser reads key once does not fail second read
        """
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            tmp.write("""The param is {param: 3.5} {param}""")
            tmp.seek(0)
            parser = ParamFileParser({'main': tmp.name})
            parser.parse(testing=True)
            self.compare_params(parser.param_dict, {'param': 3.5})

    def test_parse_override_key(self):
        """
        Ensure parser overrides key with warning
        """
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            tmp.write("""The param is {param: 3.5} {param: 10}""")
            tmp.seek(0)
            parser = ParamFileParser({'main': tmp.name})
            parser.parse(testing=True)
            self.compare_params(parser.param_dict, {'param': 10.0})

    def test_parse_multiple_files(self):
        """
        Ensure parse reads multiple files"
        """
        with (tempfile.NamedTemporaryFile(mode="w+") as tmp1,
              tempfile.NamedTemporaryFile(mode="w+") as tmp2):
            tmp1.write("""The param is {param: 3.5}""")
            tmp1.seek(0)
            tmp2.write("""The param is {paramb: 10}""")
            tmp2.seek(0)

            parser = ParamFileParser({'a': tmp1.name, 'b': tmp2.name})
            parser.parse(testing=True)
            self.compare_params(parser.param_dict, {'param': 3.5, 'paramb': 10.0})

    @staticmethod
    def compare_params(parameters_list: Parameters, other: dict):
        """
        Convert Parameters array back into dictionary for simple comparison
        """
        assert parameters_list == other


class TestClassFailRead:
    def test_parse_invalid_file(self):
        """
        Ensure parser fails if param not defined
        """
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            tmp.write("""The param is {param}""")
            tmp.seek(0)
            parser = ParamFileParser({'main': tmp.name})
            with pytest.raises(ValueError):
                parser.parse(testing=True)

    def test_parse_invalid_file_empty_value(self):
        """
        Ensure parser fails if value not defined
        """
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            tmp.write("""The param is {param:}""")
            tmp.seek(0)
            parser = ParamFileParser({'main': tmp.name})
            with pytest.raises(ValueError):
                parser.parse(testing=True)

    def test_parse_invalid_file_empty(self):
        """
        Ensure parser fails if param not defined
        """
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            tmp.write("""The param is {}""")
            tmp.seek(0)
            parser = ParamFileParser({'main': tmp.name})
            with pytest.raises(ValueError):
                parser.parse(testing=True)

class TestClassWrite:
    pass

class TestClassFailWrite:
    pass
