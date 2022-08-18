"""
This file contains a class designed to keep track of execution time,
recording it on a per-function level.
"""

from time import time
from collections import defaultdict

class TimeKeeper():
    """
    A class designed for storing funtion timing information in
    class variable. The idea is to access the class variables
    throught a class instance, typically in a function decorator
    called time_function_execution.
    """
    number_of_calls = defaultdict(int)
    execution_time = defaultdict(float)
    started = time()
    def __init__(self):
        pass

    def function_called(self, fname: str):
        """
        Increments the counter of the function calls for the
        specified function by 1.

        Args:
            fname (str) : a descriptive name of the function that
            has been called.
        """
        self.number_of_calls[fname] +=1

    def time_passed(self, fname: str, exp_time: float):
        """
        Adds the execution time of the function to the accumulated
        execution time from all the times that function has been called.

        Args:
            fname (str) : a descriptive name of the function that
            has been called.
            exp_time (float) : the wall time that has expired during the
            function execution.
        """
        self.execution_time[fname] += exp_time

    def summarise_results(self) -> list:
        """
        Convert all the recorded function timing data into a list
        and return it.

        Returns:
            list[str, int, float] : list of function names,
            number of function calls and accumulated execution time
        """
        results = []
        for kk, nc in self.number_of_calls.items():
            strk = str(kk)
            if strk in self.execution_time.keys():
                results.append([strk,
                               nc,
                               self.execution_time[kk]])
        return results

    def total_time(self) -> float:
        """
        Returns the total wall time since we started timing.

        Returns:
            (float) : Number of seconds since we started timing
            any of the functions.
        """
        return time() - self.started
