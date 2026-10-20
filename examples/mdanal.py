import MDAnalysis as mda
import h5py
import MDAnalysis.analysis.rdf as rdf

try:
    import cPickle as pickle
except ImportError:
    import pickle
import zlib

from MDMC.readers import H5MD_reader
from MDMC.exporters.trajectories import H5MD_build
from MDMC.MD.interactions import Dispersion
from tests.test_data import data
from ase import Atoms, io

fn = "Test_traj.h5"
topology_file_name = "topology.lammps"

# Unzip and unpickle the trajectory
with open(data.OBJECT_DATA['compact_trajectory'], 'rb') as compressed_trajectory:
    pickled_trajectory = zlib.decompress(compressed_trajectory.read())
H5MD_build.write_H5MD(pickle.loads(pickled_trajectory, encoding='latin-1'),
                      filename=fn,
                      timestamp=False)

with h5py.File(fn, 'r') as file:
    for x in H5MD_reader.read_dataset(file, 'position'):
        atoms = Atoms(H5MD_reader.read_dataset(file, 'atom_symbols'),
                      x, cell=H5MD_reader.read_dataset(file, '/box/edges'))

    io.write(topology_file_name, atoms, 'dlp4')

u = mda.Universe(topology_file_name, fn, format="h5md", topology_format='config',
                 convert_units=False)

print(u.atoms)
oxi = u.select_atoms('name O')
hydr = u.select_atoms('name H')

# Only using a few atoms for speed
grp1 = mda.AtomGroup([oxi[0]])
grp2 = mda.AtomGroup([hydr[0], hydr[1],hydr[3]])
asg = [[grp1, grp2], [grp1, grp2]]

rdf1 = rdf.InterRDF_s(u, asg)
rdf1.run(0, 49, 1)
plt.plot(rdf1.results.bins, rdf1.results.rdf[0][0, 0])
plt.show()
