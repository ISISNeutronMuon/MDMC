.. _dev_doc_management-label:

MDMC Software Management
========================
.. note::

  This section is only relevant to active developers of MDMC, rather than
  general contributors.

MDMC uses Git/GitHub for version control and ZenHub to implement aspects of the
Agile methodology.

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

Story points
------------
All issues are assigned a number of story points which attempt to quantify the
expected development resources required for the issue. Story points are a
relative measure and do no correspond to a specific amount of developer hours,
as this will vary dependent on the currently active developers; however an
attempt is made to equate 1 story point to ~0.5 developer days.

As with priorities, story points should be specified by active developers.

Pipelines
---------

New Issues
^^^^^^^^^^
All newly created issues should go here. An active developer should
then determine:

- the priority of the issue;
- the number of story points of the issue;
- and if it needs to be included in the current development cycle or not (move
  to **Backlog** if yes, move to **Icebox** if no).

Icebox
^^^^^^
Issues which will not be worked on in the current development cycle. No issue
with a priority higher than **P2** should be put in the **Icebox**. Before a new
development cycle, **Icebox** issues should be checked to determine if they
should be moved to **Backlog**.

Backlog
^^^^^^^
Issues worked on during the current development cycle.  No issue with a priority
lower than **P2** should be in the **Backlog**. Issues should be ordered based
on priority i.e. all **P5** at the top, all **P2** at the bottom. During sprint
planning, the **Backlog** should be surveyed to determine which issues will be
included in the next sprint.

In Progress
^^^^^^^^^^^
Issues that are being worked on should be moved to the **In Progress** pipeline;
this helps protect against duplication of effort and reduces the likelihood two
developers will be working on the same part of the codebase.  Issues that are
**In Progress** may or may not have an associated pull request (PR).

Review/QA
^^^^^^^^^
Issues should be moved to **Review/QA** once they have a PR which has a reviewer
allocated.

Closed
^^^^^^
Issues which are closed will automatically move to this pipeline.


Sprints
-------
Each sprint is represented by a Milestone; this allows ZenHub to provide
burndown reports and velocity tracking. During sprint planning, any issues to be
added to the sprint should have the relevant Milestone set. The issues within a
specific sprint can then be viewed by filtering the ZenHub board by Milestone.
