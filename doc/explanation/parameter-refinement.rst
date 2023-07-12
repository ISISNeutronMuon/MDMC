.. _parameter-refinement-label:

Parameter Refinement
====================

Running a refinement requires:

 - A setup :class:`~MDMC.MD.simulation.Simulation`.
 - One or more experimental datasets in a format that can be read by MDMC.
 - One or more :class:`~MDMC.MD.parameters.Parameters` to refine.

These objects are then passed to a :class:`~MDMC.control.control.Control` object
along with other parameters such as the minimizer type.  This
:class:`~MDMC.control.control.Control` object is then
used to :py:meth:`~MDMC.control.control.Control.refine` the
:class:`~MDMC.MD.parameters.Parameters`.

.. rubric:: Related Tutorials

| `Selecting fitting Parameters <../tutorials/selecting-fitting-parameters.ipynb>`_
| `Running a Refinement <../tutorials/running-a-refinement.ipynb>`_
