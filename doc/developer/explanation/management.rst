.. _dev_doc_management-label:

Github
======
This page details how Github issues are managed for MDMC.

If writing an issue, please ensure that the issue is actionable; i.e. it explicitly
provides instructions towards something that must be done. This means it should include:

* A description of the problem that has led to the issue or suggestion. If the issue
  is a bug, state what you expect to happen versus what actually happens.
* If the issue is a bug, a way in which a developer can reproduce the bug, such
  as example code (ideally a `minimal working example <https://stackoverflow.com/help/minimal-reproducible-example>`_)
* The solution that you'd like, and any additional details.

Priorities
----------
All issues are assigned a priority label.  Theses are inspired by
`Quantified Task Management <https://standards.mousepawmedia.com/qtm.html>`_.

**P5:** (*IMMEDIATE*) For emergencies. Drop everything and fix this issue.

**P4:** (*IMPORTANT*) Top priority in the development cycle, except for emergencies.

**P3:** (*NORMAL*) Should be completed in current development cycle.

**P2:** (*SOONISH*) Might be completed in current development cycle but not essential.

**P1:** (*EVENTUAL*) For future development cycles.

**P0:** (*OPTIONAL*) May not ever be completed.

**PT:** (*TRIAGE*) Yet to be prioritised.

By default, all newly created issues will have priority **PT**.  Issue contributors
are instructed not to assign priorities, as this should be done by active
developers.

Other labels
------------

Icebox
^^^^^^
Issues which we do not intend to complete in the current development cycle will be
triaged as 'iceboxed'. This means that they are known issues, but they are currently
either unimplementable (due to a hitch in how MDMC works) or a large undertaking of
development hours which are better spent on the focus of the current version.

Needs breaking down
^^^^^^^^^^^^^^^^^^^
An issue marked 'needs breaking down' is one which is too big to be actionable, such
as "Rewrite all algorithms to use type hints". This should be broken down into chunks
such as "Rewrite observable calculation to use type hints" and "Rewrite the Control class
to use type hints", else taking on the issue completely is a huge undertaking for
any developer.

Needs more info
^^^^^^^^^^^^^^^
An issue marked 'needs more info' is not actionable due to lacking information.
The way it is currently written is confusing as to how a new feature should be implemented,
or what the problem actually *is*. The issue should be edited to provide this information.
