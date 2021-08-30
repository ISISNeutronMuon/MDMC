"""System tests for the pair distribution function

Compares the calculated partial PDFs against PDFs calculated using nMOLDYN. The
total PDF is a simple sum of these partials, but nMOLDYn uses a different
scaling, so this is not tested here."""

from netCDF4 import Dataset
import numpy as np
import pytest

from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from tests.test_data import data
from tests.system_tests.observables.data_manager import trajectory


ATOL = 1e-10
RTOL = 5e-4

@pytest.fixture(scope="module")
def PDF_file():

    """
    Returns
    -------
    Dataset
        Reference PDF calculations from nMOLDYN using the same trajectory as is
        used here
    """

    return Dataset(data.OBS_DATA['PDF'], 'r')

@pytest.fixture(scope="module")
def PDF(trajectory, PDF_file):

    """
    Returns
    -------
    PDF
        The calculated PDF which is compared with the reference. The independent
        variable (r) values are taken from the reference.
    """

    # Scale units as nMOLDYN uses nm, rather than Ang
    r = np.array(PDF_file.variables['r'][:]) * 10.
    pdf = ObservableFactory.create_observable('PDF')
    pdf.calculate_from_MD(trajectory, n_frames=5, r=r,
                          dimensions=[39.4221067]*3)
    return pdf


@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDFs(PDF, PDF_file, partial_str):

    """
    Tests that each partial pair is within a tolerance of the nMOLDYN values
    """

    ref_str = 'pdf-{0}-{1}'.format(partial_str[0], partial_str[1])
    ref_partial = np.array(PDF_file.variables[ref_str][:])
    partial = PDF.partial_pdfs[partial_str]
    assert len(ref_partial) == len(partial)
    assert np.all(np.isclose(ref_partial, partial, atol=ATOL, rtol=RTOL))
