.. _reusing-refinement-label:
.. _H5MD Website: https://h5md.nongnu.org/h5md.html
.. _MDANSE: https://mdanse.readthedocs.io/en/latest/index.html
.. _MDANSE install docs: https://mdanse.readthedocs.io/en/latest/pages/H_start.html
.. _MDANSE Turorials: https://mdanse.readthedocs.io/en/latest/pages/T_sim.html

Reusing Refinements
===================

MDMC allows for trajectories to be created and stored at the refinement stage of the simulation.
The trajectories are saved in a standardised H5MD file format found at the `H5MD Website`_.

H5MD trajectory files can be read back into MDMC as a ``CompactTrajectory`` or
used with software that supports the standardised H5MD file format.

This can be useful as it allows for the trajectories to be reused in MDMC and other software without taking time to
re-run the simulations.

Creating a trajectory File
--------------------------
In MDMC, an H5MD file can be created at the refinement stage of the simulation. It is possible to
either create a H5MD file only from the trajectory with the best figure of merit or from every trajectory generated.

To get an H5MD trajectory file from the refinement stage pass ``Dump.EVERY`` to the control object to create a H5MD file from every trajectory,
or ``Dump.BEST``, to create a H5MD file from the trajectory with the best figure of merit.

This will result in files name "<timestamp>trajectory.h5" being created within the MDMC files.

Optional
^^^^^^^^
Optionally, additional parameters can be used to change how or where the File is created:

* ``h5md_filename`` can be set to a preferred name of the H5MD trajectory files.
* ``h5md_file_loc`` can be set to change where the file is stored,
* ``h5md_timestamp`` can be set to True or False, adding or removing the time stamp at the end of the file name respectively.

.. note::

    MDMC will not add the ``.h5`` extension to the name of the file and will use exactly the name you define. 
    Not specifying this will not break the file but it is recommended, as some operating systems rely on the extension
    to work out the appropriate reader, as well as simply making the file easier to find.

Examples
--------

This example shows how to set up a ``Control`` object to dump the best trajectory in a subfolder of the folder in which you are running
MDMC, called ``trajectories``; i.e. if you are running a refinement in a folder ``my_refinement_folder``, this would dump the trajectory
to the file ``my_refinement_folder/trajectories/best_trajectory.h5``.

.. code-block:: Python

  control = Control(simulation=simulation,
              exp_datasets=exp_datasets,
              fit_parameters=fit_parameters,
              MD_steps=570,
              h5md_dump="best",
              h5md_file_loc="./trajectories"
              h5md_filename="best_trajectory.h5",
              h5md_timestamp=False)

We can set h5md_dump to dump every trajectory created by the analysis with a timestamp. This example ``Control`` object would dump trajectories
to the same ``trajectories`` subfolder as previously, with the name ``traj_DDMMYYYY-HH.MM.SS``, i.e. trajectory dumped
at 12:23:15pm on the 18th of October 2025, the name would be ``traj_18102025-12.23.15``.

.. code-block:: Python

  control = Control(simulation=simulation,
              exp_datasets=exp_datasets,
              fit_parameters=fit_parameters,
              MD_steps=570,
              h5md_dump="best",
              h5md_file_loc="./trajectories"
              h5md_filename="traj",
              h5md_timestamp=True)


.. warning::

    If printing all trajectories ``h5md_timestamp`` should be set to true, if not the file will be continually overwritten
    and the file will only contain the last trajectory.

External Use
------------

The H5MD files can then be used within external programs that have support the standard H5MD file format.

Examples of this include:

* `MDANSE`_: Simulation software that can be used for trajectory visualization and analysis.

How to be used with MDANSE
^^^^^^^^^^^^^^^^^^^^^^^^^^
``MDANSE`` installation instructions can be found at `MDANSE install docs`_.

Once ``MDANSE`` is installed, you can use the User Interface to visualise the trajectory:
1. Using the tabs along the top of the interface, navigate to the `Trajectory` tab.
2. Under that tab, select the button "Load .MDT Trajectory".
3. Your file explorer should appear, and select your `.h5` trajectory file.
4. Your file name should appear below the load trajectory button. Double-click it, and a visualisation of the trajectory should appear in the model box.
5. You can now use the bar or buttons below the visualisation to see the visualisation through time.

``MDANSE`` can also be used for trajectory analysis:
1. Load a trajectory as seen in above instructions.
2. Navigete to the `Actions` tab and select Analysis.
3. Chose an analysis type and configer the paramiters.
4. Run analysis with the run button.

``MDANSE`` has futher fcuntionality and for more infomation use the MDANSE documentation `MDANSE`_ or the MDANSE Tutorials `MDANSE Turorials`_.

Useful Links
------------

* :class:`MDMC.writers.H5MD_build`: API documentation intended for developers.
* `MDANSE`: MDANSE documentation.
