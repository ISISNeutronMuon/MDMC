import sys
import warnings
from time import time, asctime, localtime
from typing import Union


class VerboseManager:
    """VerboseManager is a Singleton pattern class which manages verbose printing of a process."""
    _instance = None
    # max_print_len decides how many lines of output are printed to stdout
    # before they are printed to file instead; this can be changed by user
    max_output_len = 20

    def __init__(self):
        raise RuntimeError("VerboseManager should not be instantiated directly. Use VerboseManager.instance().")

    @classmethod
    def instance(cls):
        """Instantiates a VerboseManager if one does not exist, and returns the existing one if it does exist."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        """
        Sets variables to how they should be set when VerboseManager is instantiated
        """
        self.times = False
        self.bar = False
        self.step_times = False
        self._in_progress = False
        self.subprocesses = 0
        self.prev_message_len = 0
        self.timings_list = []
        self.subprocess_start_times = []

    def start(self, n_steps: int, verbose: int = 0):
        """
        Initialises progress management, including times and a progress bar.
        If a progress bar is already running, this is ignored.

        Parameters
        ----------
        n_steps: int
            The amount of steps involved in the process.
        verbose: int
            The level of verbosity:
            Verbose level 0 gives no information.
            Verbose level 1 gives final time for a whole process.
            Verbose level 2 gives final time and also a progress bar.
            Verbose level 3 gives final time, a progress bar, and time per step.
        """
        # activate features based on verbosity level
        if verbose >= 1:
            self.times = True
        if verbose >= 2:
            self.step_times = True
        if verbose >= 3:
            self.bar = True

        if not self._in_progress:
            # if not in progress, initialise process
            self._in_progress = True
            self.progress = 0
            self.maximum = n_steps

            if self.times:
                self.start_time = time()
            if self.bar:
                sys.stdout.write('\n')
                self._print_progress(0, self.maximum, "Initialising")
            if self.step_times:
                self.prev_message = "Initialising"
                self.step_time = self.start_time
        else:
            # else, a subprocess called this method; account for it
            self.subprocesses += 1
            self.subprocess_start_times.append(time())

    def step(self, message: str):
        """
        Increases progress by one 'step' towards maximum, updating progress bar if necessary.

        Parameters
        ----------
        message: str
            a message which says what the current step is doing.
        """
        self.progress += 1

        # if verbosity is >=2, give step time with bar message and record it for final result
        if self.step_times:
            message_time = time() - self.step_time
            # append previous step to timings list
            self.timings_list.append((self.prev_message, round(message_time, 2)))
            # add vertical bars to denote nesting level in final list
            # and add previous timing to progress bar message
            self.prev_message = f"{'|' * self.subprocesses}{message}"
            message += f"; previous step took {round(message_time, 2)} seconds."
            self.step_time = time()

        # if verbosity is 3, print progress bar
        if self.bar:
            self._print_progress(self.progress, self.maximum, message)

            # keep track of previous string; if we do not add blankspace to the end of the message,
            # it will show older messages under new ones.
            # this isn't len(self.prev_message) as self.prev_message doesn't include prev step timings
            self.prev_message_len = len(message)

    def finish(self, process_name: str) -> Union[None, float, list]:
        """
        Prints final times for the process.

        Parameters
        ----------
        process_name: str
            a message which gives the name of the process being managed.

        Returns
        -------
        float
            the runtime in seconds for the process or subprocess that just finished.
        """
        # fallback to timings as None if there's no timings to return
        timings = None

        # if all processes are finished, finalise verbose printing
        if self._in_progress and self.subprocesses == 0:
            # give warning to developer if self.maximum is set incorrectly
            if self.progress != self.maximum:
                warnings.warn(f"maximum steps for process \"{process_name}\" is set incorrectly:"
                              f" it is equal to {self.maximum}, but the process took {self.progress} steps.",
                              stacklevel=2)

            # print final results
            if self.times:
                timings = time()
                # the newline spaces are nice if the bar is there, but too spacious without it.
                if self.bar:
                    sys.stdout.write('\n')
                print(f"{process_name} complete in {round(timings - self.start_time, 2)} seconds.")
                if self.step_times:
                    # append final step to timings list and then print timings per step
                    self.timings_list.append((self.prev_message, round(time() - self.step_time, 2)))
                    self._print_step_timings(process_name)

                    # save timings list for return after we reset it
                    timings = self.timings_list

            # reset parameters to how they were when VerboseManager was initialised
            self._init()

        elif self._in_progress:
            # else, a subprocess called this method; account for it
            # but also return whole subprocess timing
            self.subprocesses -= 1
            timings = time() - self.subprocess_start_times[-1]
            self.subprocess_start_times.pop()

        else:
            warnings.warn("VerboseManager.finish() was called, but no management process was running.")

        return timings

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

    @classmethod
    def _print_step_timings(cls, process_name):
        """
        Prints final step timings, either to stdout (if below max_print_len)
        or to file (if above max_print_len)

        This is a class method as we need to access cls.max_print_len.
        """

        if len(cls._instance.timings_list) > cls.max_output_len:
            print("Timings list is too long, so has been printed to the file timings.log."
                  " To change how many steps are recorded until we print to file instead"
                  " of output, set max output length with VerboseManager.max_output_len = [your desired length]")
            with open("timings.log", "a") as logfile:
                # prints title line in case multiple timings are printed to same log
                logfile.write(f"\nTimings for {process_name.lower()} on {asctime(localtime(time()))}\n")
                for step_timing in cls._instance.timings_list:
                    logfile.write(f"{step_timing[0]}: {step_timing[1]}\n")
        else:
            print("Timings per step:")
            for step_timing in cls._instance.timings_list:
                print(f"{step_timing[0]}: {step_timing[1]}")

