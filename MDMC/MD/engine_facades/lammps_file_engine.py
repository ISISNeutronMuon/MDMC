from MDMC.MD.engine_facades.file_facade import FileEngine, FileUniverse, ControlFilenames, Stages
from MDMC.MD.engine_facades.lammps_engine import LAMMPSEngine
from MDMC.MD.simulation import Universe

from lammps import PyLammps


class LAMMPSFileUniverse(FileUniverse):
    def __init__(self, init_filename, field_filename, lmp):
        self.lmp = lmp

        self.lmp.file(init_filename)
        with self as field_file:
            self.lmp.file(field_file)

    def __enter__(self, typ):
        tmp_files = FileUniverse.__enter__(self)

        self.ensemble.remove_ensemble_fixes()
        self.lmp_universe.apply_constraints()
        self.ensemble.apply_ensemble_fixes()

        if typ == Stages.MIN:
            pass
        elif typ == Stages.EQUIL:
            pass
        elif typ == Stages.RUN:
            pass

    def __exit__(self):
        self.lmp.clear()
        pass

    @property
    def dimensions(self):
        return self.lmp.region


class LAMMPSFileEngine(FileEngine, LAMMPSEngine):
    def __init__(self, **settings):
        LAMMPSEngine.__init__(self)

    def setup_universe(self, universe: Universe, **settings: dict) -> None:
        """
        Creates the simulation box, the atomic configuration, and the topology
        in LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` which will be ignored.
        **settings
            ``atom_style`` (`str`)
                A LAMMPS ``atom_style`` `str`. The default setting of ``real``
                will generally be appropriate.
        """

        control_fnames = ControlFilenames(
            min_control=settings.get('min_control', ()),
            equil_control=settings.get('equil_control', ()),
            run_control=settings.get('run_control', ())
        )

        fnames = (settings['data_filename'],
                  settings['field_filename'])

        FileEngine.__init__(self, fnames, control_fnames)

        print(self.lmp.region)

        self.lmp_universe: LAMMPSFileUniverse = LAMMPSFileUniverse(*self.parametrised_file_names,
                                                                   self.lmp)

        self.lmp.clear()
        with self.lmp_universe as inputs:
            self.lmp.file(inputs[1])
            self.lmp.read_data(inputs[0])

    def setup_simulation(self, **settings: dict) -> None:
        """
        Sets simulation parameters in LAMMPS, such as the thermodynamic
        variables, thermostat/barostat parameters and trajectory settings

        Parameters
        ----------
        **settings
            Passed to ``LAMMPSSimulation``
        """
        self.lmp_simulation = self.universe

    def run(self, n_steps: int, equilibration=False, output_log: str = None,
            work_dir: str = None, **settings: dict):

        self.control_parsers.run_control.param_dict['n_steps'] = n_steps

        self.lmp.clear()
        with self.universe as inputs, self.control_parsers.run_control as cont:
            self.lmp.file(inputs[1])
            self.lmp.read_data(inputs[0])

            for file in cont:
                self.lmp.file(file)

        self.lmp.run(n_steps)

        # # Reset LAMMPS fixes
        # self.ensemble.remove_ensemble_fixes()
        # self.lmp_universe.apply_constraints()
        # self.ensemble.apply_ensemble_fixes()

    def minimize(self, n_steps: int, output_log: str = None,
                 work_dir: str = None, **settings: dict) -> None:
        self.control_parsers.min_control.param_dict['n_steps'] = n_steps

        self.lmp.clear()

        with self as inputs, self.control_parsers.min_control as cont:
            self.lmp.file(inputs[1])
            self.lmp.read_data(inputs[0])

            for file in cont:
                self.lmp.file(file)

        # # Reset LAMMPS fixes
        # self.ensemble.remove_ensemble_fixes()
        # self.lmp_universe.apply_constraints()
        # self.ensemble.apply_ensemble_fixes()
