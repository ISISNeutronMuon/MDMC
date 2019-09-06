"""Tests for decorators and associated functions"""

import pytest

from MDMC.common import decorators


@pytest.fixture
def docstring():

    """
    Returns an example docstring
    """

    doc = (r"""
           This is a docstring with several sections

           Here is the extended description. It wraps over two lines due to its
           extended length.

           Parameters
           ----------
           a : type
               This is parameter 'a'.
           b : type
               This is parameter 'b'. It has a longer description to test for
               line wrapping and indenting of Parameters.
           *args
                These are the *args.
           **kwargs
                c : type
                    **kwargs are indented further

           Returns
           -------
           type
               This is the return type

           Examples
           --------
           Here is an example::

               >>> print('Example')
               Example

           Notes
           -----
           Here is an equation:

           ..math::

               {\Phi _{12}(r)=A\exp \left(-Br\right)-{\frac {C}{r^{6}}}}
           """)

    return doc


@pytest.mark.parametrize('length', [100, 80, 60, 40])
def test_wrap_docstring_wrapping(docstring, length):

    """
    This tests wrapping a docstring to a specific line length
    """

    # Determine expected line lengths

    wrap = decorators._wrap_docstring(docstring, length)
    for line in wrap.split('\n'):
        assert len(line) <= length

