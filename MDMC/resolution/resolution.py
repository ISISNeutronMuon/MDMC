from abc import ABC, abstractmethod


class Resolution(ABC):
	"""
	An abstract base class for resolution functions.
	"""

	@abstractmethod
	def apply(self, FQt, t, Q):
		"""
		Applies resolution to an array.

		Parameters
		----------
		FQt: the FQt array to which resolution is applied.
		t: the variable to which resolution is applied.

		Returns
		-------
		The array with the resolution function applied to it.
		"""
	
		raise NotImplementedError

