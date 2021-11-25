Resolution
==========

This directory contains files for the Resolution class, used in control.py and sqw.py for applying instrument resolution to data.

Contents:
---------
- `resolution.py`: contains the `Resolution` abstract base class, defining has the abstract method `apply()`. `apply()` applies instrument resolution to an array. It has the subclasses:
  - `gaussian.py`: `GaussianResolution`, a numeric resolution function with Gaussian shape. It is initialised with the parameter `e_res`, which is the FWHM of the function.
  - `lorentzian.py`: `LorentzianResolution`, a numeric resolution function with Lorentzian shape. It is initialised with the parameter `e_res`, which is the FWHM of the function.
  - `from_file.py`: `FileResolution`, methods for defining resolution using a vanadium run file. Takes the parameters `file_name`, the path of the resolution file; `file_type`, the type of observable; `file_reader`, the type of reader; and `dt`, the time separation in of frames for the Observable's simuatlion.
  - `null.py`: `NullResolution`, the null object pattern for `Resolution`.
- `resolution_factory.py`: contains `ResolutionFactory`, a factory pattern for creating `Resolution` objects. The factory pattern also contains handling for 'lazy' resolution passing, e.g. passing resolution as a string or float.