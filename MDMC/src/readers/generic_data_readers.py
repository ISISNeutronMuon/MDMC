"""Readers for generic data of different file formats

The file formats of these readers are not associated with a specific type of
data

AUTHOR :    Thomas Farmer        START DATE :    2018-6-5 20:58:20"""

# TODO: Consider if an abstract static data class is required

import h5py

from MDMC.src.readers.readers import Reader
from MDMC.src.utilities.introspection import get_calling_class
from MDMC.src.trajectory_analysis.observables import RDF, SQW

class GenericReader(object):

    def __new__(cls, *args, **kwargs):
        if cls is GenericReader:
            raise TypeError("GenericReader may not be instantiated")
        return object.__new__(cls, *args, **kwargs)

    def get_data_type(self):
        self.type = get_calling_class(levels_up=2)


class HDFReader(Reader, GenericReader):

    def open(self, file_name):
        self.file = h5py.File(file_name, 'r')

    # TODO: Probably want to strip this out into GenericReader - might need to check MRO to ensure this implements parse
    def parse(self):
        self.get_data_type()
        if self.type == RDF.RadialDistributionFunction:
            self.parse_RDF()
        elif self.type == SQW.DynamicStructureFactor:
            raise NotImplementedError

    def parse_RDF(self):
        raise NotImplementedError

    def parse_SQW(self):
        raise NotImplementedError
