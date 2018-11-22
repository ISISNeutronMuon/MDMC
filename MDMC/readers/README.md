# readers
This package contains all of the data readers, including readers for experimental observables data and for molecular configurations.  It also contains a module which defines a factory pattern for instantiating a reader class.  

Additional readers can be added by creating a module with the reader name within this package.  This module must contain a class with the same name which implements the methods defined in the Reader class.
