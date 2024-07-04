.. _dev_doc_coding_standards-label:

Coding Standards
================
MDMC code follows standards described in
`PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_, and pylint is
used to enforce this; pull requests will have their continuous integration
testing fail if the code raises any pylint messages.

Jupyter notebooks
-----------------
MDMC's tutorials and how-to guides consist heavily of notebooks. These notebooks
are tested in new pull-requests, but only to ensure they work; as long as they
can run without errors, the output is not under version control. It is the responsibility
of developers to ensure outputs are what we expect them to be.