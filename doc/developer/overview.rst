.. _dev_doc_overview-label:

Overview
========

MDMC is open source software. Code contributions to MDMC are welcomed; please
visit the `MDMC GitHub <https://github.com/MDMCproject/MDMCv0.2_pilot>`_ in
order to contribute.

The process by which to contribute code is:

1. Create a fork from the master branch of the MDMC repository
2. Create a branch on your forked repository.
3. Commit changes to the branch
4. Create a pull request to the MDMC master.
5. Code is reviewed and any required changes are requested.
6. After changes are made (and any further reviews occur) branch is merged to
   master by one of the core MDMC developers.

If you require further details about any of these steps, please contact MDMC
support: support@mdmcproject.org

Please read the following developer documentation sections before contributing
to MDMC.

- `Coding Standards <explanation/coding_standards.rst>`_
- `Units <how-to/units.rst>`_

Please note that, at a bare minimum, any pull request must do the following:

- Pass the continuous integration testing that MDMC uses. This includes both
  unit testing via `pytest <https://docs.pytest.org/>`_, and style testing via
  `pylint <https://docs.pylint.org/>`_.
- Ensure that all code added by the pull request is equipped with unit tests.
- Ensure that all code added is documented by docstrings, comments and pages in the
  documentation. Our documentation follows the :ref:`diataxis-label` framework, and
  new documentation should be written with these principles in mind.

OnBoarding Instructions
-----------------------

When onboarding as a developer first read the Overview above.

Next, read the pages in the :ref:`Explanation <dev_doc_explanation-label>` section of the 
Developer Documentation.

Instructions to an get start with MDMC in VSCode:
 1. :ref:`Set Up Docker <dev_doc_setupdocker-label>`
 2. :ref:`Clone Source Code <dev_sourcecode-label>`. Work on a branch - never Master.
 3. :ref:`Setup VSCode inside a Container <dev_doc_setupvscode-label>`
 4. :ref:`Build Documentation <dev_doc_builddocs-label>`
 5. :ref:`Run Tutorials <dev_doc_runtutorials-label>`. Find the correct tutorial order in Tutorials section of User Documentation.
