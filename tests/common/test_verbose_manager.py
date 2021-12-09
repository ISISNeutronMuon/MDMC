import pytest
from pytest_cases import parametrize
from time import sleep

from MDMC.common.verbose_manager import VerboseManager


@pytest.fixture
def process():
    """A mock process for testing."""

    def _process(verbose):
        vman = VerboseManager.instance()
        vman.start(2, verbose=verbose)
        sleep(0.01)
        vman.step("Process step 1")
        sleep(0.01)
        vman.step("Process step 2")
        sleep(0.01)
        vman.finish("Process")

    return _process


@pytest.fixture
def bigger_process(process):
    """A mock process that contains process()"""

    def _bigger_process(verbose):
        vman = VerboseManager.instance()
        vman.start(3, verbose=verbose)
        sleep(0.01)
        vman.step("Bigger process step 1")
        sleep(0.01)
        vman.header("Subprocess")
        process(verbose=verbose)
        vman.finish("Bigger process")

    return _bigger_process


def test_progress():
    """Tests that progress increases and decreases as intended with different methods"""
    vman = VerboseManager.instance()
    vman.start(1, verbose=1)
    assert vman.progress == 0
    vman.header("Header")
    assert vman.progress == 0
    vman.step("Step")
    assert vman.progress == 1
    vman.finish("start, step and header test")
    vman.start(0, verbose=1)
    assert vman.progress == 0
    vman.finish("restart test")


# stdout for the fixtures at each verbosity level
process_stdout = {1: "Process complete in 0.03 seconds.",
                  2: ("Process complete in 0.03 seconds.\n"
                      "Timings per step:\n"
                      "Initialising: 0.01\n"
                      "Process step 1: 0.01\n"
                      "Process step 2: 0.01\n"),
                  3: ("\n\r[                    ] 0%  Initialising "
                      "\r[==========          ] 50%  Process step 1; previous step took 0.01 seconds. "
                      "\r[====================] 100%  Process step 2; previous step took 0.01 seconds. \n"
                      "Process complete in 0.03 seconds.\n"
                      "Timings per step:\n"
                      "Initialising: 0.01\n"
                      "Process step 1: 0.01\n"
                      "Process step 2: 0.01\n")}

bigger_process_stdout = {1: 'Bigger process complete in 0.05 seconds.\n',
                         2: 'Bigger process complete in 0.05 seconds.\n'
                            'Timings per step:\n'
                            'Initialising: 0.01\n'
                            'Bigger process step 1: 0.02\n'
                            'Subprocess: \n'
                            '|Process step 1: 0.01\n'
                            '|Process step 2: 0.01\n',
                         3: '\n\r[                    ] 0%  Initialising '
                            '\r[======              ] 33%  Bigger process step 1; previous step took 0.01 seconds. '
                            '\r[=============       ] 66%  Process step 1; previous step took 0.02 seconds.        '
                            '\r[====================] 100%  Process step 2; previous step took 0.01 seconds. \n'
                            'Bigger process complete in 0.05 seconds.\n'
                            'Timings per step:\n'
                            'Initialising: 0.01\n'
                            'Bigger process step 1: 0.02\n'
                            'Subprocess: \n'
                            '|Process step 1: 0.01\n'
                            '|Process step 2: 0.01\n'}


@parametrize("verbose", [1, 2, 3])
@parametrize("func, exp_stdout", [(process, process_stdout),
                                  (bigger_process, bigger_process_stdout)])
def test_stdout(func, exp_stdout, capsys, verbose):
    func(verbose=verbose)
    stdout = capsys.readouterr().out
    assert exp_stdout[verbose] in stdout
