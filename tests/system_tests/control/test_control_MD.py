"""System tests for the Control object with a real MD engine plugged in."""
import logging
import re

import numpy as np
import pandas as pd
import pytest

from MDMC.control import Control
from MDMC.MD import Atom, Dispersion, LennardJones, Simulation, Universe
from tests.control.test_control import exp_datasets, simulation


pytestmark = [pytest.mark.lammps]

@pytest.fixture()
def argon_control(exp_datasets) -> callable:
    """
    A Control object for a simulation of argon atoms.

    Returns
    -------
    control : object
        control object from setting up a universe identical to that of the Argon tutorial
    fit_parameters : dict
        dictionary of force field parameters

    """
    def _argon_control(file_name, constraints: list = [[1.0,5.0],[0.5, 5.0]],
                                          values: list = [3.36,1.02]):
        density = 0.0176
        universe = Universe(dimensions=23.0668)
        Ar = Atom('Ar', charge=0., mass=36.0)

        n_ar_atoms = int(density * np.prod(universe.dimensions))
        print(f'Number of argon atoms = {n_ar_atoms}')
        universe.fill(Ar, num_struc_units=(n_ar_atoms))

        Ar_dispersion = Dispersion(universe,
                                (Ar.atom_type, Ar.atom_type),
                                cutoff=8.,
                                function=LennardJones(epsilon=values[1], sigma=values[0]))

        simulation = Simulation(universe,
                                engine="openmm",
                                time_step=10.18893,
                                temperature=120.,
                                traj_step=15)

        dataset = exp_datasets(file_name=file_name)
        fit_parameters = universe.parameters

        fit_parameters['sigma'].constraints = constraints[0]
        fit_parameters['epsilon'].constraints = constraints[1]

        control = Control(simulation=simulation,
                    exp_datasets=dataset,
                    fit_parameters=fit_parameters,
                    minimizer_type="GPO",
                    reset_config=True,
                    MD_steps=4000,
                    equilibration_steps=4000,
                    data_printer='ipython')
        return control, fit_parameters
    return _argon_control


def test_control_q_value_trimming(argon_control):
    """
    Tests that the q_value trimming is done correctly. This uses modified experimental Argon data
    with reduced Q_values. There is a specific method that does the trimming but this test
    doesn't test that method in isolation; it tests that on a general refinement step, that trimming
    functions as expected.
    """
    ctrl, _ = argon_control(file_name='Argon_test_data.xml')
    ctrl.equilibrate(n_steps=100)

    recreated_q_values_pos = [6,9]
    manually_trimmed_obs_arrays = [ctrl.observable_pairs[0].exp_obs.dependent_variables['SQw'][0][pos]
                               for pos in recreated_q_values_pos]
    manually_trimmed_errors_arrays = [ctrl.observable_pairs[0].exp_obs.errors['SQw'][0][pos]
                               for pos in recreated_q_values_pos]

    ctrl.refine(n_steps=1)
    auto_trimmed_obs_arrays = ctrl.observable_pairs[0].exp_obs.dependent_variables['SQw'][0]
    auto_trimmed_errors_arrays = ctrl.observable_pairs[0].exp_obs.errors['SQw'][0]

    assert np.array_equal(manually_trimmed_obs_arrays, auto_trimmed_obs_arrays)
    assert np.array_equal(manually_trimmed_errors_arrays, auto_trimmed_errors_arrays)

def test_control_q_value_trimming_warning(argon_control, caplog):
    """
    Tests that the correct warning is given when some experimental Q_values cant be recreated.
    This uses modified experimental Argon data with reduced Q_values.
    """
    ctrl, _ = argon_control(file_name='Argon_test_data.xml')
    ctrl.equilibrate(n_steps=100)

    caplog.set_level(logging.WARNING)
    ctrl.refine(n_steps=1)

    log_message = ("The specified universe dimensions were not able to recreate the lowest q" \
                " values of the experimental data and so this data has been" \
                " trimmed accordingly.")

    assert log_message in caplog.text

@pytest.mark.parametrize('eps, sig',[(1.02, 3.36),
                                (2.0, 3.0),
                                (3.0,4.0),
                                (4.0,5.0),])
def test_control_bad_params(argon_control, simulation, eps, sig):
    """
    Tests that given a set of bad parameters (which crash the refinement), the equilibration
    and production runs handle this.
    """
    constraints = [[sig-0.5, sig+0.5], [eps-0.5, eps+0.5]]
    values = [sig,eps]
    ctrl, fit_parameters = argon_control(file_name='Well_s_q_omega_Ar_data.xml',
                                    constraints=constraints, values=values)
    fit_parameters['epsilon'].value = eps
    fit_parameters['sigma'].value = sig

    ctrl.equilibrate(n_steps=100)
    ctrl.refine(4)

@pytest.mark.parametrize('eps_constr, sig_constr',[([0.02,2.02], [2.36, 4.36]),
                                                             ([0.5,20.0], [1.0,20.0])])

def test_control_bad_constraints(argon_control, eps_constr, sig_constr):
    """
    Tests that given different sets of constraints on the parameter values (which can possibly crash
    the refinement), the equilibration and production runs handle this.
    """
    constraints = [sig_constr, eps_constr]
    ctrl, fit_parameters = argon_control(file_name='Well_s_q_omega_Ar_data.xml',
                                                            constraints=constraints)
    fit_parameters['epsilon'].value = 1.02
    fit_parameters['sigma'].value = 3.36

    ctrl.equilibrate(n_steps=100)
    ctrl.refine(4)

def test_control_trial_reduce_time_step(argon_control):
    """
    Tests that the method for varying time_step (upon a failed equilibration), changes the
    time_step and traj_step accurately.
    """

    ctrl, _ = argon_control(file_name='Well_s_q_omega_Ar_data.xml')
    original_time_step = ctrl.simulation.time_step
    reduction_factor = 0.6
    ctrl.trial_reduce_time_step(reduction_factor=reduction_factor)
    assert ctrl.simulation.time_step == original_time_step * reduction_factor
