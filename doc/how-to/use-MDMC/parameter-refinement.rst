.. _parameter-refinement-label:

Parameter Refinement
====================

MDMC runs refinements to fit simulation parameters to experimental data.

Running a refinement requires:

 - A setup :class:`~MDMC.MD.simulation.Simulation`.
 - One or more experimental datasets in a format that can be read by MDMC.
 - One or more :class:`~MDMC.MD.parameters.Parameters` to refine.

These objects are then passed to a :class:`~MDMC.control.control.Control` object
along with other parameters such as the minimizer type.  This
:class:`~MDMC.control.control.Control` object is then
used to :py:meth:`~MDMC.control.control.Control.refine` the
:class:`~MDMC.MD.parameters.Parameters`.

The MDMC interface for these objects is explained in more detail in the following notebooks:

.. toctree::
   :maxdepth: 2

   notebooks/selecting-fitting-parameters.ipynb
   ../../tutorials/running-a-refinement.ipynb