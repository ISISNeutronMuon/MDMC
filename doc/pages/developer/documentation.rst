.. _dev_doc_documentation-label:

Documentation Standards
=======================
MDMC follows the `NumPy documentation style <https://numpydoc.readthedocs.io/en/latest/format.html>`_,
which is consistent with PEP 257. The exceptions to this are:

- The Attributes section should be the last section in a class docstring. This
  is because Sphinx formats Attributes in the same manner to methods (without a
  heading), so it leads on to this.
- Properties should not be documented in the Attributes section, as then they
  are duplicated in Sphinx. Instead they should only be documented in the getter
  method. As long as properties precede methods, the properties will follow the
  attributes in the Sphinx documentation.
- Parameters that can only assume one of a fixed set of values, in NumPy these
  values are listed instead of the type. This is not the case in MDMC as at this
  early development stage these values are still in flux. Therefore instead just
  state the type:

.. code-block:: python

  """
  Parameters
  ----------
  MDMC : str
      This is how MDMC does it.
  NumPy : {'C', 'F', 'A'}
      This is how NumPy does it.
  """

Sphinx
------
- To build the documentation using Sphinx, `pandoc <https://pandoc.org/>`_ must
  be installed. **This is not the same as the pandoc which is available from**
  `pypi <https://pypi.org>`_ **(e.g. which is installed using**
  :code:`pip install pandoc` **).**
- All images should be included in doc/_static/images
- All .ipynb tutorials should be included in doc/tutorials. When modifying these
  tutorials, :code:`jupyter notebook` must be run from the doc directory, not
  the tutorials directory. This is because the Jupyter server then has access to
  the _static/images folder, which is required for some tutorials.
