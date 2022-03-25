"""The Gaussian-Process-Regression minimizer class"""
import numpy as np
import scipy.stats as st
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
import itertools
from MDMC.refinement.minimizers.minimizer_abs import Minimizer


class GPR(Minimizer):

    """
    ``Minimizer`` employing Gaussian Process Regression 

    Parameters
    ----------
    kernel : string
        Type of kernel to be used, defaults to 'RBF' 
    alpha : float
        Hyperparameter for the fitting, also can represent additional Gaussian noise in measrement points 
    length_scale : float
        Lengthscale parameter for the kernel
    

    Attributes
    ----------
    history_columns: list[str]
        list of the column titles for the minimizer history
    """


    def __init__(self, parameters, distribution, max_parameter_change, n_points, **settings):
        super().__init__(parameters, distribution, max_parameter_change)
        self.kernel = settings.get('kernel', 'RBF')
        self.alpha = settings.get('alpha', 0.0001)
        self.length_scale = settings.get('length_scale', 0.1)
        self.parameter_point_array = self.create_parameter_point_array(parameters, n_points)

    def create_parameter_point_array(self, parameters, points=2):
        """
        Takes the constraints of the parameters to be minimised, makes an array of length points
        then performs the Cartesian product, to give every set of coordinates. If no constraints
        are present, then some arbitrary ones are set based on the parameter value.
        
        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined.
        points : int
            Number of points across the constraint range to take, defaults to 2.

        Returns
        -------
            array 
                all coordinates to be simulated.
        """
        bounds_array=[]
        for parameter in parameters:

            if parameter.constraints:
                min_bound = parameter.constraints[0]
                max_bound = parameter.constraints[1]
            else:
                min_bound = parameter.value*0.1-0.1
                max_bound = parameter.value*5+0.1  # This creates some finite bounds and also deals with zeros

            bounds = np.linspace(min_bound, max_bound, points)
            bounds_array.append(bounds)

        point_array =  list(map(list, itertools.product(*bounds_array))) # * is necessary for unpacking the arrays 
        print(point_array)
        return point_array


    def has_converged(self, conv_tol: float = 1e-5, min_steps: int =1) -> bool:
        """
        Checks if the refinement process has converged on a stable solution.
        Specifically, it checks if the certainty of the parameters being refined have all
        become less than the relative conversion tolerance (`conv_tol`). 
        It also allows specifying a minimum
        number of refinement steps (`min_steps`) that must have been simulated
        before checking for convergence.

        Parameters
        ----------
        conv_tol : float, optional
            The relative tolerance of the convergence check. Defaults to `1e-5`
        min_steps : int, optional
            The number of refinement steps after which
            convergence is checked. If the number of accepted state changes is less than this,
            then the refinement is deemed as not converged.
            Defaults to `1`.

        Returns
        -------
        bool
            Whether or not the minimizer has converged.
        """
        min_steps = np.max([min_steps,len(self.parameter_point_array)])
        if len(self.history) >= min_steps:
            #self.GPR_fit()
            converged = True
        else:
            converged = False

        return converged

    def change_state(self):
        change_state = self.comm.bcast(change_state, root=0)
        return change_state

    @property
    def history_columns(self):

        return ['FoM', 'Change state'] + [p.name for p in self.parameters]

    # pylint: disable=arguments-differ
    # we allow implementations of the abstract method to have different arguments

    def change_parameters(self, parameters):
        """
        Selects a new value for each parameter from the array of parameter values to interrogate

        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined
        """

        # Change parameters by same amount on all processes
        for i, parameter in enumerate(parameters):
            point_to_calculate = len(self._history)
            parameter.value = self.parameter_point_array[point_to_calculate][i]

    def step(self, FoM):
        """
        Increments the minimization by a step
        """

        self.FoM = FoM
        values = np.array([p.value for p in self.parameters])
        history = [self.FoM]
        history.append('Accepted')
        self.FoM_old = self.FoM
        self.state_changed = True

        history.extend(values)
        self._history.append(history)
        self.change_parameters(self.parameters)

    def GPR_fit(self):
        n_dims = len(self.parameters)
        kernel = kernels.RBF(length_scale = np.ones(np.ones(n_dims))*self.length_scale)
        gpr = GPR(kernel, n_restarts_optimizer=10)

        gpr.fit(self.parameter_point_array, self.MD_sim_output)
        y_mean, y_cov = gpr.predict(self.parameter_point_array, return_cov=True)

        posteriors = st.multivariate_normal.rvs(mean=y_mean, cov=y_cov)
        return posteriors

    