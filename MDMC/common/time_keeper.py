"""
This file contains a class designed to keep track of execution time,
recording it on a per-function level.
"""

from time import time

class TimeKeeper():
    number_of_calls = {}
    execution_time = {}
    started = time()
    def __init__(self):
        pass
    def function_called(self, fname: str):
        if fname in self.number_of_calls.keys():
            self.number_of_calls[fname] +=1
        else:
            self.number_of_calls[fname] = 1
    def time_passed(self, fname: str, exp_time: float):
        if fname in self.execution_time.keys():
            self.execution_time[fname] += exp_time
        else:
            self.execution_time[fname] = exp_time
    def summarise_results(self) -> list:
        results = []
        for kk in self.number_of_calls.keys():
            strk = str(kk)
            if strk in self.execution_time.keys():
                results.append([strk,
                               self.number_of_calls[kk],
                               self.execution_time[kk]])
        return results
    def total_time(self) -> float:
        return time() - self.started


