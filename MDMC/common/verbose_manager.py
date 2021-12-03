import sys
import warnings
from time import time

import numpy as np


class VerboseManager:
    """
    VerboseManager is a Singleton pattern class which manages verbose printing of a process.
    """
    _instance = None

    def __init__(self):
        raise RuntimeError("VerboseManager should not be instantiated directly. Use VerboseManager.instance().")

    @classmethod
    def instance(cls):
        """
        Instantiates a VerboseManager if one does not exist, and creates one if it does exist.
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.times = False
            cls._instance.bar = False
            cls._instance.step_times = False
            cls._instance._in_progress = False
            cls._instance.subprocesses = 0
            cls._instance.prev_message_len = 0
            cls._instance.timings_list = []
        return cls._instance

    def start(self, maximum: int, verbose: int = 0):
        """
        Initialises progress management, including times and a progress bar.
        If a progress bar is already running, this is ignored.

        Parameters
        ----------
        maximum: int
            The amount of steps involved in the process.
        verbose: int
            The level of verbosity:
            Verbose level 0 gives no information.
            Verbose level 1 gives final time for a whole process.
            Verbose level 2 gives final time and also a progress bar.
            Verbose level 3 gives final time, a progress bar, and time per step.
        """
        if verbose > 0:
            self.times = True
        if verbose > 1:
            self.bar = True
        if verbose > 2:
            self.step_times = True

        if not self._in_progress:
            self._in_progress = True
            self.progress = 0
            self.maximum = maximum

            if self.times:
                self.start_time = time()
            if self.bar:
                sys.stdout.write('\n')
                self._print_progress(0, self.maximum, "Initialising")
            if self.step_times:
                self.prev_message = "Initialising"
        else:
            # else, a subprocess called this method; account for it
            self.subprocesses += 1

    def step(self, message: str):
        """
        Increases progress by one 'step' towards maximum, updating progress bar if necessary.

        Parameters
        ----------
        message: str
            a message which says what the current step is doing.
        """
        self.progress += 1

        if self.step_times:
            if self.progress == 1:
                message_time = time() - self.start_time
            else:
                message_time = time() - self.step_time
            # append previous step to timings list
            self.timings_list.append((self.prev_message, np.round_(message_time, 2)))
            self.prev_message = message
            message += f"; previous step took {np.round_(message_time, 2)} seconds."
            self.step_time = time()

        if self.bar:
            self._print_progress(self.progress, self.maximum, message)
            if self.progress == self.maximum:
                self._print_progress(self.progress, self.maximum, "Complete")

            # keep track of previous string; if we do not add blankspace to the end of the message,
            # it will show older messages under new ones
            self.prev_message_len = len(message)

    def finish(self, process_name: str):
        """
        Prints final times for the process.
        Parameters
        ----------
        process_name: str
            a message which gives the name of the process being managed.
        """
        if self._in_progress and self.subprocesses == 0:
            # give warning to developer if self.maximum is set incorrectly
            if self.progress != self.maximum:
                warnings.warn(f"maximum steps for process \"{process_name}\" is set incorrectly:"
                              f" it is equal to {self.maximum}, but the process took {self.progress} steps.",
                              stacklevel=2)

            if self.times:
                timings = time()
                # the newline spaces are nice if the bar is there, but too spacious without it.
                if self.bar:
                    sys.stdout.write('\n')
                print(f"{process_name} complete in {np.round_(timings - self.start_time, 2)} seconds.")
                if self.step_times:
                    # append final step to timings list and then print timings per step
                    self.timings_list.append((self.prev_message, np.round_(time() - self.step_time, 2)))
                    print("Timings per step:")
                    for step_timing in self.timings_list:
                        print(f"{step_timing[0]}: {step_timing[1]}")

                    # save timings list for return after we reset it
                    timings = self.timings_list

            # reset parameters to how they were when VerboseManager was initialised
            self.times = False
            self.bar = False
            self.step_times = False
            self._in_progress = False
            self.subprocesses = 0
            self.prev_message_len = 0
            self.timings_list = []

            try:
                return timings
            except NameError:
                return None

        elif self._in_progress:
            # else, a subprocess called this method; account for it
            self.subprocesses -= 1
        else:
            warnings.warn("VerboseManager.finish() was called, but no management process was running.")


    def _print_progress(self, i, maximum, message):
        """
        Prints out a progress bar, which looks like (e.g.)
        [================    ] 80%  message
        """
        bar_size = 20
        progress = i / maximum

        # calculate how much trailing whitespace is needed
        # to avoid previous message being visible under new one
        eraser_diff = self.prev_message_len - len(message)
        if eraser_diff <= 0:
            eraser_diff = 0
        eraser = f'{" " * eraser_diff}'

        # we use sys.stdout.write() instead of print() because print() creates a new line at the end;
        # we don't want this, we want to stay on the same line so we can use \r to overwrite the bar.
        # \r is 'carriage return' - it returns to the start of line so it can be overwritten.
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'=' * int(bar_size * progress):{bar_size}s}] {int(100 * progress)}%  {message} {eraser}")
