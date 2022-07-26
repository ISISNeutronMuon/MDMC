from abc import ABC
from MDMC.refinement.minimizers.minimizer_abs import Minimizer


class GaussianMinimizer(ABC, Minimizer):
    """
    An abstract class containing behaviour common to all
    gaussian minimizers
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def extract_result(self) -> list:
        """
        Extracts the measured & predicted FoM and point(s)

        Returns
        -------
        list
            A list of output values in the following order:
            1. Coordinates of minimum measured point
            2. Minimum value of FoM at measured point
            3. Coordinates of minimum predicted point
            4. Minimum value of FoM at predicted point
        """
        fit, min_FoM_measured, min_parameters_measured = self.GPR_fit()
        points, FoMs = self.GPR_predict(fit)
        min_parameters_predicted, min_FoM_predicted = self.global_minimum_position(FoMs, points)
        self.set_parameter_values(self.parameter_names, min_parameters_predicted)

        list_of_outputs = [
            min_parameters_measured,
            min_FoM_measured,
            min_parameters_predicted,
            min_FoM_predicted
        ]
        return list_of_outputs

    def format_result_string(self, minimizer_output: list) -> str:
        """
        Parameters
        ----------
        minimizer_output
            A list of values in the following order:
            1. Coordinate of the lowest measured point (ideally tuple or string)
            2. FoM value of lowest measured point
            3. Coordinate of the lowest predicted point (ideally tuple or string)
            4. FoM value of lowest predicted point

        Returns
        -------
        An output string, formatted with the appropriate information about measured
        and predicted points
        """

        if self.has_converged():
            converged_message = '\nThe refinement has converged.'
        else:
            converged_message = "\nThe refinement has not converged."

        output_string = (f'{converged_message} \n \n'
                         f'Minimum measured point is: \n'
                         f'{minimizer_output[0]} with an '
                         f'FoM of {minimizer_output[1]}. \n \n'
                         f'Minimum point predicted is: \n'
                         f'{minimizer_output[2]} for an'
                         f'FoM of {minimizer_output[3]}.\n \n ')

        return output_string