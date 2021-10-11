from abc import ABC, abstractmethod


class Resolution(ABC):
	"""
	An abstract base class for resolution functions.
	"""

	@abstractmethod
	def apply(self, array, x):
		"""
		Applies resolution to an array.

		Parameters
		----------
		array: the array to which resolution is applied (usually SQw or FQt)
		x: the variable along which the resolution window is calculated, usually t for FQt and w for SQw

		Returns
		-------
		The array with the resolution function applied to it.
		"""
	
		raise NotImplementedError

