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


    def __init__(self, parameters, distribution, max_parameter_change, **settings):
        super().__init__(parameters, distribution, max_parameter_change)
        self.kernel = settings.get('kernel', 'RBF')
        self.alpha = settings.get('alpha', 0.0001)
        self.length_scale = settings.get('length_scale', 0.1)
        self.parameter_point_array = self.create_parameter_point_array(parameters)

    def create_parameter_point_array(self, parameters, points=10):
        
        bounds_array=[]
        for parameter in parameters:
            min_bound = parameter.value*0.1
            max_bound = parameter.value*5
            bounds = np.linspace(min_bound, max_bound, points)
            bounds_array.append(bounds)
        point_array =  list(map(list, itertools.product(*bounds_array))) # * is necessary for unpacking the arrays 
        return point_array


    def has_converged(self, **settings) -> bool:
        min_steps = self.parameter_point_array.size
        if len(self.history) >= min_steps:
            converged = True
        else:
            converged = False

        return converged

    def change_state(self):
        return True

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
            parameter.value = self.parameter_point_array[i][point_to_calculate]

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

    