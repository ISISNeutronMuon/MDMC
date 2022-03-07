"""Reader for radial distribution functions"""

import numpy as np

from MDMC.common import units
from MDMC.common.decorators import unit_decorator
from MDMC.readers.observables.obs_reader import ObservableReader


class LAMPRDF(ObservableReader):


    """
    A class for reading RDF files from LAMP
    LAMP's ascii output uses one file
    The file's structure is the following:   
    Row-Number  Distance  rdf1  rdf2  ...  rdfN 
   
    Attributes
    ----------
    file_indep : file
        File containing the output from the calculation
    """



    def parse(self,**settings):

        """
        Parse into RDF format
        Dist is the energy transfer (in nm)
        RDF is wavevector transfer (in barn)
        """

        self.r = self.parse_indep_var(self.Distt)
        self.PDF = self.parse_dep_var(self.rdff)


    @property
    def r(self):

        """
        Get or set the distance between pairs, r, in ``Å``

        Returns
        -------
        numpy.ndarray
            Distance between pairs, r, in ``Å``
        """

        return self._r

    @Dist.setter
    @unit_decorator(unit=units.LENGTH)
    def r(self, value):

        self._r = value

    @property
    def PDF(self):

        """
        Get or set the RDF between pairs, in ``barn``

        Returns
        -------
        numpy.ndarray
            RDF between pairs, RDF, in ``barn``
        """

        return self._PDF

    @rdf.setter
    @unit_decorator(unit=units.Unit('barn'))
    def PDF(self, value):
        """ 
        Get or set the value of the radial distribution function (in ``barn``)
        """

        self._PDF = value


    @property
    def independent_variables(self):

        """
        Get the independent variable, r (in ``Å``)

        Returns
        -------
        dict
            The independent variables r
        """

        return {"r":self.r}

    @property
    def dependent_variables(self):

        """
        Get the dependent variables, rdf (in ``barn``)

        Returns
        -------
        dict
            The dependent variables, rdf (in ``barn``)
        """

        return {"PDF": [self.PDF]}

    def parse_indep_var(self, ffile):

        """
        Parses the independent variables
        Splits the file so that the data can be extracted into a ``array`` 
        Since the the output file contains both the independent and dependent variables,        
        the parsing procedure for the same file is separated into two steps.     
        The file is basically the same for both dependent and independent variables,         
        but the procedure is separated into 2 steps for the clarity.

        Parameters
        ----------
        ffile : file
            Open file containing independent data
            Generally, the file is the same fo rboth dependent and independent parameters

        Returns
        -------
        array
            r -  distance
        """
        ffile1=open(ffile, 'r')  
        k=0
        for i, line in enumerate(ffile1):
           if i==3:
             line=line.strip()
             column=line.split()
             time_step=int(column[0])
             num_rows=int(column[1]) 
             # Initialization of arrays
             c_number=np.zeros(num_rows)
             distance=np.zeros(num_rows)    
           elif i>3:
             line=line.strip()
             column=line.split()
             # c_number or the column[0] represents the counter for the number of rows,
             # this counter is not interesting for the data analysis.
             c_number[k]=int(column[0])
             # distance[k] or column[1] is the distance for which the rdfs were calculated.
             distance[k]=float(column[1])
           k=i-3
        self.dist=distance
        ffile1.close()
        return self.dist

    def parse_dep_var(self, numrdf, ffile):

        """
        Parses the dependent variables (radial distribution functions)
        Since the the output file contains both the independent and dependent variables,        
        the parsing procedure for the same file is separated into two steps.

        Parameters
        ----------
        file : file
            Open file containing independent data

        Returns
        -------
        numpy.ndarray
            A 2d array with dimensions of the several dependent variables (several RDFs)
        """

        # Give the number of radial distribution functions
        num_rdf=numrdf
        ffile2=open(ffile,'r')
        for i, line in enumerate(ffile2):
        # The 3rd line is interesting since it is containing the important information for the initialization
        # of the arrays, like distances and rdfs. Particularly, the second parameter in the second column is of the interest.
        # The first parameter is not interesting.
           if i==3:
             line=line.strip()
             column=line.split()
             time_step=int(column[0])
             num_rows=int(column[1]) 
             NR=num_rows
        # Closing the file after obtaining the right number for the initialization of arrays.
        ffile2.close()
        ar=(NR, num_rdf) 
        # Initialization of the arrays by allocating the space and setting all numbers to zero.                         
        rdf_ar=np.zeros(ar)
        c_number=np.zeros(num_rows)
        distance=np.zeros(num_rows) 

        ffile2=open(ffile, 'r')
        k=0
        for i, line in enumerate(ffile2):
           # First 2 lines are comments on what rdfs were computed
           # The 3rd line contains the time-step which is not important for the current calculation
           # and the number of rows for rdf which is very important for the definition of the size of the array.
           # The parameter is called as NR=num_rows
           if i>3:
             line=line.strip()
             column=line.split()
           # These numbers won't be used for the reader here. They kept for the clarity of the column count,
           # since the file is the same.
             c_number=int(column[0])
             distance[k]=float(column[1])
             l=0
         # Since the actual rdfs are starting from the 3rd column, labeled as number 2 in python,
         # and the counter starts from zero, we take that column as a column for rdf, which will be counted as l+2.
             for l in range(num_rdf):
               rdf_ar[k][l]=float(column[l+2])
           k=i-3 
        # RDF then will be an array with the size (number of rows x number of rdfs) 
        self.rdf=rdf_ar
        ffile2.close()
        return self.rdf
