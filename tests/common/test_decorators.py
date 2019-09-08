"""Tests for decorators and associated functions"""

from textwrap import dedent

import pytest

from MDMC.common import decorators


@pytest.fixture
def docstring():

    """
    Returns
    -------
    str
        An example docstring
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
def modified_docstring():

    """
    Returns
    -------
    dict
        A dict with 'after' (str with the docstring after it has been modified),
        and 'replacements' (dict containing keys with str to be replaced, and
        values with the replacement str).
    """

    mod = {}
    mod['replacements'] = {'int':'float',
                           'An ':'A ',
                           'Arguments':'Parameters',
                           'longer':'much much much much much longer'}
    mod['after'] = (
        """
        This is a docstring with parts to be replaced

        Parameters
        ----------
        a : float
            A float
        b : float
            Another float

        Returns
        -------
        str
            Replace this description with a much much much much much longer
            description
        """)

    return mod


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


def test_wrap_docstring_blank_lines(docstring, wrap):
    """
    Tests that wrapping preserves blank lines (with different indentation)
    """

    assert wrap.split('\n').count('') == docstring.split('\n').count('')


def test_wrap_docstring_indentation(wrap):

    """
    Tests that indentation is maintained
    """

    # The expected indent sizes when wrapping the docstring fixture with a line
    # length of 40
    EXPECTED_INDENTS = [0, 12, 12, 0, 12, 12, 12, 12, 0, 12, 12, 12, 16, 12, 16,
                        16, 16, 16, 16, 12, 16, 12, 16, 20, 20, 0, 12, 12, 12,
                        16, 0, 12, 12, 12, 0, 16, 16, 0, 12, 12, 12, 0, 12, 0,
                        16, 16, 16]

    for index, indent in enumerate(EXPECTED_INDENTS):
        line = wrap.split('\n')[index]
        assert indent == len(line) - len(line.strip())


def test_wrap_docstring_math():

    """
    Tests that text wrapping doesn't prepend text in front of a math command,
    which would stop the command functioning
    """

    math_doc = (r"""
                 Docstring with inline math
                 .. math:: x^2
                 """)

    math_wrap = decorators._wrap_docstring(math_doc, 40).split('\n')
    assert len(math_wrap) == 5
    assert dedent(math_wrap[-2]) == r'.. math:: x^2'

