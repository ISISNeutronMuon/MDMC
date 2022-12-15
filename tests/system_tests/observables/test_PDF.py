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
from scipy.signal import find_peaks

pytestmark = [pytest.mark.lammps]

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

    return Dataset(data.OBS_DATA['netcdf_PDF'], 'r')


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
    pdf.calculate_from_MD(trajectory, n_frames=5, r=r, dimensions=[39.4221067]*3)
    return pdf


@pytest.fixture(scope="module")
def averaged_PDF(trajectory, PDF_file):

    """
    Returns
    -------
    PDF
        The calculated PDF which was calculated using an average of the individual PDFs
        of the frames used from the trajectory. The independent variable (r) values
        are taken from the reference.
    """

    # Scale units as nMOLDYN uses nm, rather than Ang
    r = np.array(PDF_file.variables['r'][:]) * 10.
    pdf = ObservableFactory.create_observable('PDF')
    pdf.calculate_from_MD(trajectory, n_frames=5, r=r, use_average=True, dimensions=[39.4221067]*3)
    return pdf


@pytest.fixture(scope="module")
def expected_peak_r_values():
    """
    Returns
    -------
    dict
        A dictionary containing partial pairs as the key, and the corresponding r value of
        its peak as the corresponding value
    """
    return {('H', 'H'): 1.65, ('H', 'O'): 1.05, ('O', 'O'): 2.65}


@pytest.fixture(scope="module")
def expected_limiting_behaviour_value(PDF):
    """
    Returns the value that the PDF reaches when the value of :math:`r` is less than the
    shortest distance that any two atoms may approach each other (:math:`r_0`).
    This is defined by the following equation:
        .. math::

            G(r<r_0)=-(\sum_{i=1}^n c_i\overline{b}_i)^2

    Parameters
    ----------
    PDF
        The PDF object

    """
    limiting_value = 0.
    total_number_of_particles = np.sum(list(PDF.numbers.values()))
    for elem in PDF.elements:
        limiting_value += (PDF.numbers[elem] / total_number_of_particles) * PDF.weights[elem]

    return -(limiting_value)**2

# Non-averaged PDF tests
@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDFs(PDF, PDF_file, partial_str):
    """Tests that each partial pair is within a tolerance of the nMOLDYN values"""

    ref_str = 'pdf-{0}-{1}'.format(partial_str[0], partial_str[1])
    ref_partial = np.array(PDF_file.variables[ref_str][:])
    partial = PDF.partial_pdfs[partial_str]
    assert len(ref_partial) == len(partial)
    assert np.allclose(ref_partial, partial, atol=ATOL, rtol=RTOL)

# The below tests are property-based, as there is (currently)
# no calculated 3rd party validated data to test against.
def test_total_PDF_peaks(PDF, expected_peak_r_values):
    """Tests that the total PDF contains peaks in the correct known places"""
    peak_r_values = list(expected_peak_r_values.values())
    # Get the absolute values as the O-H peak ends up being a massively negative value in total PDF
    abs_pdf = np.abs(PDF.PDF)
    # Check that peaks exist in the right places
    peak_indexes, properties = find_peaks(abs_pdf, height=2)
    peak_actual_r_values = [PDF.r[i] for i in peak_indexes]
    assert np.isin(peak_r_values, peak_actual_r_values)


def test_total_PDF_starts_correctly(PDF, expected_limiting_behaviour_value):
    """
    Tests that the total PDF begins with the correct values.
    """
    beginning_values = PDF.PDF[:8]
    assert np.allclose(beginning_values, expected_limiting_behaviour_value, atol=ATOL, rtol=RTOL)


def test_total_PDF_converges_correctly(PDF):
    """Tests that the total PDF converges on 0"""
    end_values = PDF.PDF[-1:-7]
    assert np.allclose(end_values, 0., atol=ATOL, rtol=RTOL)


# Averaged PDF Tests
@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDF_starts_correctly(averaged_PDF, partial_str):
    """Tests that the beginning values of the partial PDFs begin with 0"""
    beginning_values = averaged_PDF.partial_pdfs[partial_str][:8]
    assert np.allclose(beginning_values, 0., atol=ATOL, rtol=RTOL)


def test_total_PDFs_start_correctly(averaged_PDF, expected_limiting_behaviour_value):
    """Tests that the beginning values of the total PDF begin with thr correct value"""
    beginning_values = averaged_PDF.PDF[:8]
    assert np.allclose(beginning_values, expected_limiting_behaviour_value, atol=ATOL, rtol=RTOL)


def test_total_PDFs_converge_correctly(averaged_PDF):
    """Tests that the values of the total PDF converge on 0"""
    end_values = averaged_PDF.PDF[-1:-7]
    assert np.allclose(end_values, 0., atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDFs_converge_correctly(averaged_PDF, partial_str, expected_limiting_behaviour_value):
    """Tests that the values of the partial PDFs converge on the right value"""
    end_values = averaged_PDF.partial_pdfs[partial_str][-1:-7]
    assert np.allclose(end_values, expected_limiting_behaviour_value, atol=ATOL, rtol=RTOL)


def test_total_avg_PDF_peaks(averaged_PDF, expected_peak_r_values):
    """Tests that the averaged total PDF contains peaks in the correct known places"""
    peak_r_values = list(expected_peak_r_values.values())
    # Get the absolute values as the O-H peak ends up being a massively negative value in total PDF
    abs_pdf = np.abs(averaged_PDF.PDF)
    # Check that peaks exist in the right places
    peak_indexes, properties = find_peaks(abs_pdf, height=2)
    peak_actual_r_values = [averaged_PDF.r[i] for i in peak_indexes]
    assert np.isin(peak_r_values, peak_actual_r_values)


@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDF_peaks(averaged_PDF, partial_str, expected_peak_r_values):
    """Tests that the values of the partial PDF peaks are correct"""
    peak_r_values = expected_peak_r_values[partial_str]
    abs_pdf = np.abs(averaged_PDF.partial_pdfs[partial_str])
    peak_indexes, properties = find_peaks(abs_pdf, height=2)
    peak_actual_r_values = [averaged_PDF.r[i] for i in peak_indexes]
    assert np.isin(peak_r_values, peak_actual_r_values)
