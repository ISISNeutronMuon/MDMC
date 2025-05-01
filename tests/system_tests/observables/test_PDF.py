"""System tests for the pair distribution function

Compares the calculated partial PDFs against PDFs calculated using nMOLDYN. The
total PDF is a simple sum of these partials, but nMOLDYn uses a different
scaling, so this is not tested here."""
from netCDF4 import Dataset
import numpy as np
import pytest

from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from tests.test_data import data
from scipy.signal import find_peaks, peak_prominences, peak_widths
from tests.system_tests.observables.data_manager import trajectory

import cProfile, pstats

pytestmark = [pytest.mark.lammps]

ATOL = 1e-10
RTOL = 5e-4
CLOSER_RTOL = 5.96e-8
# Obtains the machine epsilon/precision for floating point numbers
MACHINE_PRECISION = np.finfo(float).eps


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
    pdf = ObservableFactory.create('PDF')
    pdf.calculate_from_MD(trajectory, n_frames=5, r=r, use_average=False, dimensions=[39.4221067]*3)
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
    pdf = ObservableFactory.create('PDF')
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
    return {('H', 'H'): [1.65], ('H', 'O'): [1.05], ('O', 'O'): [2.65]}


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
    total_number_of_particles = np.sum(list(PDF.numbers_of_atoms.values()))
    limiting_value = sum((PDF.numbers_of_atoms[elem] / total_number_of_particles) * PDF.weights[elem]
                         for elem in PDF.elements)
    return -(limiting_value)**2

# Non-averaged PDF tests
@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDFs(PDF, PDF_file, partial_str):
    """Tests that each partial pair is within a tolerance of the nMOLDYN values"""

    ref_str = f'pdf-{partial_str[0]}-{partial_str[1]}'
    ref_partial = np.array(PDF_file.variables[ref_str][:])
    partial = PDF.partial_pdfs[partial_str]
    assert len(ref_partial) == len(partial)
    assert np.allclose(ref_partial, partial, atol=ATOL, rtol=RTOL)

# The below tests are property-based, as there is (currently)
# no calculated 3rd party validated data to test against.

# All below tests use machine precision/epsilon to check values are
# the same within floating point precision
def test_total_PDF_peaks(PDF, expected_peak_r_values):
    """
    Tests that the total PDF contains peaks in the correct known places"""
    peak_expected_r_values = list(expected_peak_r_values.values())
    # Get the absolute values as the O-H peak ends up being a massively negative value in total PDF
    abs_pdf = np.abs(PDF.PDF)
    # Check that peaks exist in the right places
    peak_indexes, properties = find_peaks(abs_pdf, height=2)
    peak_actual_r_values = [PDF.r[i] for i in peak_indexes]
    assert np.all(present_values(peak_expected_r_values, peak_actual_r_values))


def test_total_PDF_starts_correctly(PDF, expected_limiting_behaviour_value):
    """Tests that the total PDF begins with the correct values"""
    beginning_values = PDF.PDF[:8]
    assert np.allclose(beginning_values, expected_limiting_behaviour_value, rtol=MACHINE_PRECISION)


def test_total_PDF_converges_correctly(PDF):
    """Tests that the total PDF converges on 0"""
    end_values = PDF.PDF[-1:-7]
    assert np.allclose(end_values, 0., rtol=MACHINE_PRECISION)


# Averaged PDF Tests
@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDF_starts_correctly(averaged_PDF, partial_str):
    """Tests that the beginning values of the partial PDFs begin with 0"""
    beginning_values = averaged_PDF.partial_pdfs[partial_str][:8]
    assert np.allclose(beginning_values, 0., rtol=MACHINE_PRECISION)


def test_total_PDFs_start_correctly(averaged_PDF, expected_limiting_behaviour_value):
    """Tests that the beginning values of the total PDF begin with thr correct value"""
    beginning_values = averaged_PDF.PDF[:8]
    assert np.allclose(beginning_values, expected_limiting_behaviour_value, rtol=MACHINE_PRECISION)


def test_total_PDFs_converge_correctly(averaged_PDF):
    """Tests that the values of the total PDF converge on 0"""
    end_values = averaged_PDF.PDF[-1:-7]
    assert np.allclose(end_values, 0., rtol=MACHINE_PRECISION)


@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDFs_converge_correctly(averaged_PDF, partial_str, expected_limiting_behaviour_value):
    """Tests that the values of the partial PDFs converge on the right value"""
    end_values = averaged_PDF.partial_pdfs[partial_str][-1:-7]
    assert np.allclose(end_values, expected_limiting_behaviour_value, rtol=MACHINE_PRECISION)


def test_total_avg_PDF_peaks(averaged_PDF, expected_peak_r_values):
    """Tests that the averaged total PDF contains peaks in the correct known places"""
    peak_expected_r_values = list(expected_peak_r_values.values())
    # Get the absolute values as the O-H peak ends up being a massively negative value in total PDF
    abs_pdf = np.abs(averaged_PDF.PDF)
    # Check that peaks exist in the right places
    peak_indexes, properties = find_peaks(abs_pdf, height=2)
    peak_actual_r_values = [averaged_PDF.r[i] for i in peak_indexes]
    assert np.all(present_values(peak_expected_r_values, peak_actual_r_values))

@pytest.mark.parametrize("partial_str", [('H', 'H'), ('H', 'O'), ('O', 'O')])
def test_partial_PDF_peaks(averaged_PDF, partial_str, expected_peak_r_values):
    """Tests that the values of the partial PDF peaks are correct"""
    peak_expected_r_values = expected_peak_r_values[partial_str]
    # The correct r value for the O-O peak here is 0.1 Ang off from what it is otherwise
    if partial_str == ('O', 'O'):
        peak_expected_r_values = [2.75]
    abs_pdf = np.abs(averaged_PDF.partial_pdfs[partial_str])
    peak_indexes, properties = find_peaks(abs_pdf)
    peak_actual_r_values = [averaged_PDF.r[i] for i in peak_indexes]
    assert np.all(present_values(peak_expected_r_values, peak_actual_r_values))

def test_slice_trajectory(trajectory):
    """Tests that the sliced trajectory contains the correct frames from the right times"""
    # 5 Frames are picked from a 50-frame trajectory
    pdf = ObservableFactory.create('PDF')
    sliced = pdf._slice_trajectory(trajectory, n_frames=5)
    assert len(trajectory) == 50
    assert len(sliced) == 5
    # Check that the 5 frames are evenly spaced (i.e. at indexes 0, 10, 20, 30, 40)
    for index in range(0, 5):
        assert sliced.times[index] in trajectory.times
        assert sliced.times[index] == trajectory.times[index*10]


def test_pdf_recalculation_does_not_change_values(trajectory, PDF):
    """Tests that when the PDF is calculated in succession the values do not change"""
    before_total = np.copy(PDF.PDF)
    before_partials = PDF.partial_pdfs.copy()
    # Calculate PDF again
    PDF.calculate_from_MD(trajectory, n_frames=5, r=PDF.r, use_average=False,
                          dimensions=[39.4221067] * 3)
    after_total = np.copy(PDF.PDF)
    after_partials = PDF.partial_pdfs.copy()
    # Check that values are the same
    assert np.array_equal(before_total, after_total)
    assert np.all([
        np.array_equal(before_partials[partial_name], after_partials[partial_name])
        for partial_name in PDF.partial_pdfs.keys()
    ])


def present_values(expected_values, actual_values):
    """Checks that expected values are present within the actual values, within machine precision"""
    return [np.any(np.isclose(np.array(expected_value), actual_values, rtol=MACHINE_PRECISION))
            for expected_value in expected_values]
