Benchmarks
===========================

The benchmark suite
-------------------
MDMC includes a suite of benchmarks to measure the performance of the minimizers availaible in the program (MMC, GPO and GPR).
The benchmarks measure:

- Time to run
- Peak memory use
- Figure of merit returned

No MD is run during the benchmarks, instead a mocked figure of merit (FoM) 
is calculated using a `Schwefel function <https://www.sfu.ca/~ssurjano/schwef.html>`_.
This serves as a complex enough function challenge the minimizers, whilst having a known
global minimum and being simple to calculate. It also means that we can measure the performance
of MDMC directly rather than runtime being dominated by MD calculations.

Running the benchmark suite
---------------------------

Benchmarking is run using the `airspeed velocity <https://asv.readthedocs.io/en/latest/>`_ tool (asv). 
asv is primarily designed to track the performance of a repository over time, so works by benchmarking individual commits.
Benchmarks are configured to run in Github actions on every pull request, with longer benchmarks
run weekly.
Benchmarks are run with varying numbers of parameters to fit and refinement steps.
The starting values of the parameters are randomised at the start of each refinement.

The benchmark suite can also be run locally using the command:

.. code-block:: bash

    asv run -b '.*(?<!_long)\(' commit

Where ``commit`` is the hash of the commit you want to benchmark. This will be run in a python 3.10
virtual environment, and excludes any benchmarks with ``_long`` in their name.

This will output results in the terminal, and save them as json in the ``benchmarks/results`` directory.

**NOTE:** if you do not include a commit hash, asv will default to running the benchmarks on every commit in the repo,
which will take a very long time!

Benchmarks can also be run on the current version of the repo (whether it is committed or not) using your local python installation using:

.. code-block:: bash

    asv run -b '.*(?<!_long)\(' --python=same

However this will only output the results to the terminal, they are not saved as JSON.

You can select a subset of benchmarks to run using the ``-b`` option and
providing a python regular expression as above.
The function names matched against this regex will have parameters appended, for example ``time_GPR(5, 10)``.
Be aware of this when writing your regular expression.

Running:

.. code-block:: bash

    python3 benchmarks/scripts/results_parser.py commit

Will generate a ``benchmark_results.md`` file with more human-friendly formatting for comparing results.
``commit`` must be the 8-character short hash for the commit you wish to benchmark.
If you have run the benchmarks using a version of python that is not ``3.10``, you will also have to
specify your version with the ``--python_version`` argument.

To compare the results of different commits use:

.. code-block:: bash

    asv compare --only-changed --split commit_1 commit_2

Adding new benchmarks
---------------------

Benchmark functions are defined in ``benchmarks/benchmarks.py``.
asv runs all functions in the top level of the benchmark directory with one of its accepted prefixes, listed 
`here <https://asv.readthedocs.io/en/latest/writing_benchmarks.html#benchmark-types>`_.

Different suites of benchmarks can be separated into classes.
Each class includes benchmarks as methods, and can also include a ``setup`` method which is run before each benchmark.
You can also define a list of parameters in the class to run benchmarks with.
Parameters are passed as positional arguments to each benchmark function, as well as ``setup``.

For more information on writing benchmarks, see the 
`asv documentation <https://asv.readthedocs.io/en/latest/writing_benchmarks.html>`_.

asv configuration
-----------------

Configuration for asv runs, such as installation commands for the virtual environment and results directories is stored in ``asv.conf.json``.
See the `docs <https://asv.readthedocs.io/en/latest/asv.conf.json.html>`_ for more information.