"""
This module is the interface to the Atomic Simulation Environment (ASE,
https://wiki.fysik.dtu.dk/ase/) ``GUI``, which enables a molecular viewer to be
launched. This viewer allows the visualization of atomic positions and bonds.

"""

from functools import partial
from io import StringIO
from typing import TYPE_CHECKING, Union
import warnings
from tkinter import TclError
import numpy as np

from ase.gui.gui import GUI
from ase.gui.i18n import _
from ase.gui.images import Images
from ase.gui.ui import MenuItem
from ase.gui.view import get_cell_coordinates
from IPython.display import HTML

from MDMC.MD.ase.conversions import get_ase_atoms

if TYPE_CHECKING:
    from MDMC.MD.structures import Atom
    from MDMC.MD.ase import ASEAtoms


def view(atoms: 'list[Atom]',
         viewer: str = 'X3DOM',
         cell: np.ndarray = None,
         **settings: dict) -> Union[HTML, None]:
    """
    Launches the ASE ``GUI`` for a collection of atoms

    Parameters
    ----------
    atoms : list
        A `list` of ``Atom`` (``MDMC.MD.structures.Atom``) to view
    viewer : str, optional
        The viewer used to display the visualization. This can be 'X3DOM', which
        allows for inline visualization in Jupyter notebooks, or 'ASE', which
        displays in an external window. The default is 'X3DOM'.
    cell : numpy.ndarray, optional
        An ``array`` of `float` specifying the dimensions of the cell to view.
        The default is `None`.
    **settings
        ``max_atoms`` (`int`)
            Sets the maximum number of atoms that will be viewed
    """

    atoms = get_ase_atoms(atoms, cell=cell)

    if viewer == 'X3DOM':
        return view_x3dom(atoms, **settings)
    if viewer == 'ASE':
        return view_ase(atoms, **settings)
    raise ValueError('Unrecognised viewer. Specify either "X3DOM" or "ASE"')


def view_ase(atoms: 'ASEAtoms', **settings: dict) -> None:
    """
    View atom using the ASE viewer

    Parameters
    ----------
    ase_atoms : ASEAtoms
        The ``ASEAtoms`` object to be visualized using the ASE viewer
    **settings
        ``max_atoms`` (`int`)
            Sets the maximum number of atoms that will be viewed
    """
    atoms = limit_atoms(atoms, settings.get('max_atoms', 8000))

    atom_images = Images()
    atom_images.initialize([atoms])

    try:
        viewer = Viewer(atom_images)
    except TclError as err:
        raise TclError(str(err) + '. This may be because MDMC is being run in'
                                  ' Docker and X11 forwarding has not been'
                                  ' enabled.') from err
    viewer.run()


def view_x3dom(atoms: 'ASEAtoms', **settings: dict) -> HTML:
    """
    View atoms using the X3D viewer, which enables inline visualization within
    a IPython/Jupyter notebook

    Parameters
    ----------
    ase_atoms : ASEAtoms
        The ``ASEAtoms`` object to be visualized using the X3D viewer
    **settings
        ``max_atoms`` (`int`)
            Sets the maximum number of atoms that will be viewed
    """

    atoms = limit_atoms(atoms, settings.get('max_atoms', 2000))

    output = StringIO()
    atoms.write(output, format='X3DOM')
    data = output.getvalue()
    output.close()
    return HTML(data)


def limit_atoms(atoms: 'ASEAtoms', max_atoms: int) -> 'ASEAtoms':
    """
    Limits the number of atoms that are passed to a visualizer

    Parameters
    ----------
    atoms : ASEAtoms
        The ``ASEAtoms`` object to be visualized
    max_atoms : int
        The maximum number of atoms which can be passed to the visualizer

    Warns
    -----
    warnings.warn
        If the number of atoms is greater than `max_atoms`, the user is warned
        that `atoms` will be capped at this size
    """

    if len(atoms) < max_atoms:
        return atoms
    warnings.warn(f'The number of atoms visualized has been capped to {max_atoms}. To'
                  ' increase this, pass a larger `max_atoms`')
    return atoms[:max_atoms]


def get_bonds(atoms: 'ASEAtoms') -> np.ndarray:
    """
    Adds ``(0, 0, 0,)`` to each bonded atom pair defined within an ``ASEAtoms``
    object

    Parameters
    ----------
    atoms : ASEAtoms
        The ``ASEAtoms`` object for which the ``bonds`` are required for
        plotting
    """

    bonds = [pair + (0, 0, 0) for pair in atoms.bonds]
    return np.array(bonds)


class Viewer(GUI):

    """
    Subclasses the ASE ``GUI`` to provide a molecular viewer for MDMC.

    It modifies how ``bonds`` are plotted by using an alternative ``get_bonds``
    function in the ``set_atoms`` method.

    It removes ``GUI`` menu options that are not applicable in MDMC.
    """

    def __init__(self, images: Images  = None, rotations: str = '', expr: str = None):

        # Override in order to set show bonds
        super().__init__(images=images, rotations=rotations, show_bonds=True,
                         expr=expr)
        self.X = None
        self.X_pos = None
        self.X_cell = None
        self.X_bonds = None
        self.B = None

    def set_atoms(self, atoms: 'ASEAtoms'):
        """
        Almost an exact copy from ASE

        This method is defined purely so that an alternative to the
        ``get_bonds`` function is used. Now the ``bonds`` are set during
        ``__init__``.

        Parameters
        ----------
        atoms : ASEAtoms
            The ``atoms`` which will be set
        """

        natoms = len(atoms)

        if self.showing_cell():
            B1, B2 = get_cell_coordinates(atoms.cell,
                                          self.config['shift_cell'])
        else:
            B1 = B2 = np.zeros((0, 3))

        if self.showing_bonds():
            atomscopy = atoms.copy()
            atomscopy.cell *= self.images.repeat[:, np.newaxis]
            bonds = get_bonds(atoms)
        else:
            bonds = np.empty((0, 5), int)

        # X is all atomic coordinates, and starting points of vectors
        # like bonds and cell segments.
        # The reason to have them all in one big list is that we like to
        # eventually rotate/sort it by Z-order when rendering.

        # Also B are the end points of line segments.

        self.X = np.empty((natoms + len(B1) + len(bonds), 3))
        self.X_pos = self.X[:natoms]
        self.X_pos[:] = atoms.positions
        self.X_cell = self.X[natoms:natoms + len(B1)]
        self.X_bonds = self.X[natoms + len(B1):]

        cell = atoms.cell
        ncellparts = len(B1)
        nbonds = len(bonds)

        self.X_cell[:] = np.dot(B1, cell)
        self.B = np.empty((ncellparts + nbonds, 3))
        self.B[:ncellparts] = np.dot(B2, cell)

        if nbonds > 0:
            P = atoms.positions
            Af = self.images.repeat[:, np.newaxis] * cell
            a = P[bonds[:, 0]]
            b = P[bonds[:, 1]] + np.dot(bonds[:, 2:], Af) - a
            d = (b**2).sum(1)**0.5
            r = 0.65 * self.get_covalent_radii()
            x0 = (r[bonds[:, 0]] / d).reshape((-1, 1))
            x1 = (r[bonds[:, 1]] / d).reshape((-1, 1))
            self.X_bonds[:] = a + b * x0
            b *= 1.0 - x0 - x1
            b[bonds[:, 2:].any(1)] *= 0.5
            self.B[ncellparts:] = self.X_bonds + b

    def update_labels(self):
        index = self.window['show-labels']
        if index == 0:
            self.labels = None
        elif index == 1:
            self.labels = self.atoms.IDs
        elif index == 3:
            Q = self.atoms.get_initial_charges()
            self.labels = ['{0:.4g}'.format(q) for q in Q]
        else:
            self.labels = self.atoms.get_chemical_symbols()

    def get_menu_data(self):
        """
        Subset of default ASE ``GUI`` menu options which are applicable to MDMC
        """

        M = MenuItem
        return [
            (_('_File'),
             [M(_('_Open'), self.open, 'Ctrl+O'),
              M(_('_New'), self.new, 'Ctrl+N'),
              M(_('_Save'), self.save, 'Ctrl+S'),
              M('---'),
              M(_('_Quit'), self.exit, 'Ctrl+Q')]),

            (_('_Edit'),
             [M(_('Select _all'), self.select_all),
              M(_('_Invert selection'), self.invert_selection),
              M('---'),
              M(_('Hide selected atoms'), self.hide_selected),
              M(_('Show selected atoms'), self.show_selected),
              M('---'),
              M(_('_First image'), self.step, 'Home'),
              M(_('_Previous image'), self.step, 'Page-Up'),
              M(_('_Next image'), self.step, 'Page-Down'),
              M(_('_Last image'), self.step, 'End'),
              M(_('Append image copy'), self.copy_image)]),

            (_('_View'),
             [M(_('Show _unit cell'), self.toggle_show_unit_cell, 'Ctrl+U',
                value=self.config['show_unit_cell']),
              M(_('Show _axes'), self.toggle_show_axes,
                value=self.config['show_axes']),
              M(_('Show _velocities'), self.toggle_show_velocities, 'Ctrl+G',
                value=False),
              M(_('Show _forces'), self.toggle_show_forces, 'Ctrl+F',
                value=False),
              M(_('Show _bonds'), self.toggle_show_bonds, 'Ctrl+B',
                value=self.config['show_bonds']),
              M(_('Show _Labels'), self.show_labels,
                choices=[_('_None'),
                         _('Atom _ID'),
                         _('_Element Symbol'),
                         _('_Charges'),
                         ]),
              M('---'),
              M(_('Quick Info ...'), self.quick_info_window, 'Ctrl+I'),
              M(_('Repeat ...'), self.repeat_window, 'R'),
              M(_('Rotate ...'), self.rotate_window),
              M(_('Colors ...'), self.colors_window, 'C'),
              # TRANSLATORS: verb
              M(_('Focus'), self.focus, 'F'),
              M(_('Zoom in'), self.zoom, '+'),
              M(_('Zoom out'), self.zoom, '-'),
              M(_('Change View'),
                submenu=[
                    M(_('Reset View'), self.reset_view, '='),
                    M(_('xy-plane'), self.set_view, 'Z'),
                    M(_('yz-plane'), self.set_view, 'X'),
                    M(_('zx-plane'), self.set_view, 'Y'),
                    M(_('yx-plane'), self.set_view, 'Alt+Z'),
                    M(_('zy-plane'), self.set_view, 'Alt+X'),
                    M(_('xz-plane'), self.set_view, 'Alt+Y'),
                    M(_('a2,a3-plane'), self.set_view, '1'),
                    M(_('a3,a1-plane'), self.set_view, '2'),
                    M(_('a1,a2-plane'), self.set_view, '3'),
                    M(_('a3,a2-plane'), self.set_view, 'Alt+1'),
                    M(_('a1,a3-plane'), self.set_view, 'Alt+2'),
                    M(_('a2,a1-plane'), self.set_view, 'Alt+3')]),
              M(_('Settings ...'), self.settings),
              M('---'),
              M(_('VMD'), partial(self.external_viewer, 'vmd')),
              M(_('RasMol'), partial(self.external_viewer, 'rasmol')),
              M(_('xmakemol'), partial(self.external_viewer, 'xmakemol')),
              M(_('avogadro'), partial(self.external_viewer, 'avogadro'))])]
