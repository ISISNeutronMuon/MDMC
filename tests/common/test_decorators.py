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

@pytest.fixture
def wrap(docstring):

    """
    docstring wrapped with a line length of 40
    """

    return decorators._wrap_docstring(docstring, 40)


@pytest.mark.parametrize('length, n_expected_lines', [(100, 39),
                                                      (80, 39),
                                                      (60, 41),
                                                      (40, 48)])
def test_wrap_docstring_wrapping(docstring, length, n_expected_lines):

    """
    This tests wrapping a docstring to a specific line length

    Tests that the the wrapping results in the expected number of lines, and
    that none of the wrapped lines have a length greater than the wrapping line
    length.
    """

    wrap = decorators._wrap_docstring(docstring, length)
    for line in wrap.split('\n'):
        assert len(line) <= length
    assert len(wrap.split('\n')) == n_expected_lines


@pytest.mark.parametrize('length', [16, 10])
def test_wrap_docstring_invalid_length(docstring, length):

    """
    Tests that a wrapping length greater than the number of characters in one or
    more indents raises a ValueError
    """

    with pytest.raises(ValueError):
        decorators._wrap_docstring(docstring, length)



def test_wrap_docstring_blank_lines():

    pass


def test_wrap_docstring_indentation():

    pass
