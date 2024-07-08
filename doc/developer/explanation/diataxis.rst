.. _diataxis-label :

Diátaxis
========

The MDMC documentation uses the `Diátaxis framework <https://diataxis.fr/>`_, which helps documentation
stay focused around user needs. This improves the quality and variety of documentation by
separating it into four 'user contexts', explained below.

Why?
----
Diátaxis creates a 'compass' for pitching documentation. This makes
it easier to write higher-quality, focused documentation on two levels.

-  Page-by-page focus: each page of documentation has a defined goal for
   what user need it aims to satisfy. The developer keeps this goal in
   mind when writing the page, and writes with purpose as a result.

-  Documentation-wide focus: documentation aims to cover all user needs
   with the software, rather than aiming for the usual 'code coverage'
   of 'this function is mentioned in the docs, so it has been documented'

-  Tutorials and explanations mean workshops and technical reports plan
   themselves!

-  On-boarding new developers is easier, as documentation has been
   written with understanding in mind.


The four contexts of Diátaxis
-----------------------------

Tutorials
^^^^^^^^^

-  A tutorial is 'learning-oriented'. It takes a beginner to basic competence, enough that they can
   'take away' your software and start using it for their own purposes.

-  It is for to learning and developing familiarity with your software,
   not 'getting things done'.

-  A tutorial is usually what you're giving if you run a workshop;
   you're assuming no existing familiarity, and you're guiding users
   down a single line to competence.

How-to Guides
^^^^^^^^^^^^^

-  A how-to guide is 'task-oriented'. It *assumes familiarity* and shows how to get a
   particular result. They're for a user who wants to know how to
   complete a specific task.

-  They address a particular question. The purpose of the guide should
   fill in the blank: “How to _______”

-  If a tutorial is teaching someone how to cook, a how-to guide is
   giving someone a recipe.

Reference
^^^^^^^^^

-  Reference is an encyclopedic description of the software and its
   interface. This is for users who want information and technical
   reference material.

-  In MDMC, the reference material is generated automatically by
   sphinx-autodoc from source code docstrings. These docstrings should
   comply to the `Numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.

Explanation
^^^^^^^^^^^

-  Explanation is 'understanding-oriented'. It gives a 'high-level'
   discussion of the algorithms used and the ideas underlying them.
   It is for users (and especially future developers)
   who want to understand the theory and design ideas of the software.

-  It should make sense to someone who doesn't have the software
   directly to hand; explanation is *not* instruction.

-  This would be the 'background' and 'discussion' sections of a paper
   on the software; in fact, both of these would be suitable alternative
   names for this section.

How do I use this?
------------------
When writing MDMC documentation, you will have to put your new documentation in
one of the four categories, because the documentation source is structured that way.
Once you have done so, the Diátaxis framework provides guidelines to answer
the following questions:

- Who is this aimed at? (e.g. new users, 'power users', developers)
- What do I want them to learn? (e.g. familiarity with this framework, how to use this class to do X)
- What do I assume they know? (familiarity with MD, optimisation, ``matplotlib``, Docker)
