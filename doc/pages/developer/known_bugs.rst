.. _dev_doc_known_bugs-label:

Known bugs and unimplemented features
=======================
This page contains known bugs in MDMC, as well as features that are either unimplementable, or surplus to user requirements.

Known bugs
----------


Unimplementable features
------------------------
- Allowing resolution to be applied to a S(Q, w) array via convolution, and allowing S(Q, w) arrays to be inverse Fourier transformed into F(Q, t) arrays. Due to the way these arrays are implemented, this cannot be done accurately. See the `Github repo pull request #764 <https://github.com/MDMCproject/MDMCv0.2_pilot/pull/764>` for more details.
- Redirecting PyLammps messages to the log rather than to console output; this is difficult to do due to how LAMMPS handles stdout. See `issue #471 <https://github.com/MDMCproject/MDMCv0.2_pilot/issues/471>`.

Unimplemented features
----------------------
- Allowing users to directly use Readers to read in observables and configurations, rather than reading them as part of Control objects. This is possible, but complicates the Reader classes so will not be implemented unless there is user demand.
