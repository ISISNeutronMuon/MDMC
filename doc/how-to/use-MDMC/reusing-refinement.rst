.. _reusing-refinement-label:
.. _H5MD Website: https://h5md.nongnu.org/h5md.html
.. _MDANSE: https://mdanse.readthedocs.io/en/latest/index.html

Reusing Refinements
===================

MDMC allows for trajectories to be created and stored at the refinement stage to the simulation.
The trajectories are following a standardised H5MD file format found at the `H5MD Website`_.

MDMC has the functionality for the H5MD trajectory files to be read back in to MDMC as a compact trajectory or
used with software that supports the standardised H5MD file format.

This can be useful as it allows for the trajectories to be reused in MDMC and other software without taking time to
re-run the simulations.

Creating a trajectory File
--------------------------
In MDMC a H5MD file can be created at the refinement stage of the simulation. At this stage it can be chosen to
ether create a H5MD file from the trajectory with the best figure of merit or from every trajectory generated.

To get an H5MD trajectory file from the refinement stage the ``Dump.every``, to create a H5MD file from every trajectory,
or ``Dump.Best``, to create a H5MD file from the trajectory with the best figure of merit, must be passed to the ``Control`` object.

This will result in files name "<timestamp>trajectory.h5" being created within the MDMC files.

Optional
^^^^^^^^
Optionally, additional parameters can be used to change how or where the File is created:

* ``h5md_file_name`` can be set to a preferred name of the H5MD trajectory files,
* ``h5md_file_loc`` can be set to change where the file is stored,
* ``h5md_timestamp`` can be set to True or False, adding or removing the time stamp at the end of the file name respectively.

.. note::

    MDMC will not add the ``.h5`` suffix to the names of the file. This will not break the file but it is suggested
    to be added to the end of the file name to make the file easier to find.

Examples
--------
.. code-block::

        control = Control(simulation=simulation,
                    exp_datasets=exp_datasets,
                    fit_parameters=fit_parameters,
                    MD_steps=570,
                    h5md_dump=Dump.BEST)

        control.refine(n_steps=10,
                    h5md_file_name='best_trajectory',
                    h5md_file_loc='files/files/filed',
                    h5md_timestamp=False)

This code snippet shows an example of the parameters that may be used to get a file containing the best H5MD trajectory.
As can be seen in the example code a :class:`~MDMC.MD.simulation.Simulation` needs to be created as explained :ref:`simulation-label`.

.. warning::

    If printing all trajectories ``h5md_timestamp`` should be set to true, if not the file will be continually overwritten
    and the file will only contain the last trajectory.

External Use
------------

The H5MD files can then be used within External programs that have comparability with the standardised H5MD file format.

Examples of this includes:

* `MDANSE`_: Simulation software that can be used for trajectory visualization and computation of properties.

How to be used with MDANSE
^^^^^^^^^^^^^^^^^^^^^^^^^^

Useful Links
------------

* :class:`MDMC.writers.H5MD_build`: API documentation intended for developers and users with advanced understanding of software development.
