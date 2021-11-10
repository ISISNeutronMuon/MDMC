from abc import ABC, abstractmethod

import numpy as np
from scipy import signal


class Resolution(ABC):
	"""
	An abstract base class for resolution functions.
	"""

	def apply(self, array, x, frequency_space=False):
		"""
		Applies resolution to an array.

		Parameters
		----------
		array: the array to which resolution is applied.
		x: the variable to which resolution is applied.
		frequency_space: a bool which states whether
			the resolution is convolved (if True) or
			multiplied (if False) with the array.

		Returns
		-------
		The array with the resolution function applied to it.
		"""

		N_Q, N_x = np.shape(array)
		window = self._calculate_resolution_window(x[:N_x], frequency_space)

		if frequency_space:
			return signal.convolve(array, np.broadcast_to(window, (N_Q, N_x)), mode="same")
		else:
			return np.broadcast_to(window, (N_Q, N_x)) * array

	@abstractmethod
	def _calculate_resolution_window(self, x, frequency_space=False):
		"""
		Calculate the resolution window to be applied.
		"""
		raise NotImplementedError
