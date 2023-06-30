"""Test for reading in data with unit conversion
MDANSESQw reader is the only data reader in MDMC
which can accept input files formatted differently
and expressed in different units. The tests included
here ensure that the resulting data are the same
independent of the format of the input."""

import tempfile
from io import StringIO

import pytest
import numpy as np

from MDMC.readers.observables.obs_reader_factory import ObservableReaderFactory

mdanse_data_ascii_output = """# variable name: s(q,f)_total
# 	type: surface
# 	axis: q|omega
# 	units: nm2/ps

# 1st column: q (1/nm)
# 1st row: omega (rad/ps)

0.000000000000000000e+00 -1.709779858788470719e+01 -1.282334894091352950e+01 -8.548899293942353594e+00 -4.274449646971176797e+00 0.000000000000000000e+00 4.274449646971176797e+00 8.548899293942353594e+00 1.282334894091352950e+01 1.709779858788470719e+01
1.219999999999999929e+01 3.950986190596459703e-04 1.392843467136946367e-04 5.612785519136486692e-05 1.716099353001354123e-05 8.161975326149337705e-04 1.716099353001355140e-05 5.612785519136484659e-05 1.392843467136946096e-04 3.950986190596460245e-04
2.419999999999999574e+01 1.267940220582672957e-03 2.017535836327090035e-03 1.967565193000387052e-03 2.662871947470733195e-03 7.601364029083551577e-01 2.662871947470733195e-03 1.967565193000387052e-03 2.017535836327089167e-03 1.267940220582672089e-03
3.619999999999999574e+01 1.746544434807437065e-03 4.828235294063881050e-03 7.125406412215911506e-03 9.161006681950002728e-03 1.552414642625137731e+00 9.161006681950000993e-03 7.125406412215912373e-03 4.828235294063879315e-03 1.746544434807435981e-03
"""

mdanse_data_plotter_output = """# Data         : s(q,f)_total
# First row    : q (1/ang)
# First column : omega (meV)
  0.0000e+00    1.2200e+00    2.4200e+00    3.6200e+00
 -1.1254e+01    3.9510e-04    1.2679e-03    1.7465e-03
 -8.4405e+00    1.3928e-04    2.0175e-03    4.8282e-03
 -5.6270e+00    5.6128e-05    1.9676e-03    7.1254e-03
 -2.8135e+00    1.7161e-05    2.6629e-03    9.1610e-03
  0.0000e+00    8.1620e-04    7.6014e-01    1.5524e+00
  2.8135e+00    1.7161e-05    2.6629e-03    9.1610e-03
  5.6270e+00    5.6128e-05    1.9676e-03    7.1254e-03
  8.4405e+00    1.3928e-04    2.0175e-03    4.8282e-03
  1.1254e+01    3.9510e-04    1.2679e-03    1.7465e-03
"""

@pytest.fixture(scope = 'module')
def mdanse_textfile():
    target = tempfile.NamedTemporaryFile('w')
    target.write(mdanse_data_ascii_output)
    target.flush()
    yield target
    target.close()

@pytest.fixture(scope = 'module')
def mdanse_plotterfile():
    target = tempfile.NamedTemporaryFile('w')
    target.write(mdanse_data_plotter_output)
    target.flush()
    yield target
    target.close()

def read_and_parse(file_obj: tempfile._TemporaryFileWrapper):
    reader = ObservableReaderFactory.create_reader('MDANSESQw', file_obj.name)
    with reader:
        reader.parse()
    return reader

@pytest.fixture()
def parsed_textfile(mdanse_textfile):
    parsed_reader = read_and_parse(mdanse_textfile)
    return parsed_reader

@pytest.fixture()
def parsed_plotterfile(mdanse_plotterfile):
    parsed_reader = read_and_parse(mdanse_plotterfile)
    return parsed_reader

def test_axes(parsed_textfile, parsed_plotterfile):
    q_axis_1 = parsed_textfile.independent_variables['Q']
    q_axis_2 = parsed_plotterfile.independent_variables['Q']
    e_axis_1 = parsed_textfile.independent_variables['E']
    e_axis_2 = parsed_plotterfile.independent_variables['E']
    assert np.allclose(q_axis_1, q_axis_2, rtol=1e-4, atol = 1e-3)
    assert np.allclose(e_axis_1, e_axis_2, rtol=1e-4, atol = 1e-3)
     
def test_data(parsed_textfile, parsed_plotterfile):
    data_1 = parsed_textfile.SQw
    data_2 = parsed_plotterfile.SQw
    assert len(data_1.shape) == len(data_2.shape)
    assert np.all(np.array(data_1.shape) == np.array(data_2.shape))
    assert np.allclose(data_1, data_2, rtol=1e-4, atol=1e-3)
    


