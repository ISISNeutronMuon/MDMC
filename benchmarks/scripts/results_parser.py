import json
import pandas as pd
from itertools import product
from pathlib import Path
import glob
import os

results_dir = Path(__file__).parent.parent.absolute() / "results/"

#Assume we only have one machine
machine_dir = [x.name for x in os.scandir(results_dir) if x.is_dir()][0]
results_dir = results_dir / machine_dir

filenames = glob.glob("*.json", root_dir=results_dir)
filenames.remove("machine.json")

#Assumes we only have one results file
filename = results_dir / filenames[0]

with open(filename) as json_file:
    data = json.load(json_file)

results = {k.split(".")[-1]:v[0] for k, v in data["results"].items()}

#Take params from first benchmark's result
#Tuple of (number of parameters, number of refinement steps)
results_vals = list(data["results"].values())
params = list(product(*results_vals[0][1]))

tuple_results = {}

for k in results.keys():
    result_type, result_name = k.split("_")
    if result_type == "time":
        tuple_results[("Time (s)", result_name)] = results[k]

        time_per_step = [r / int(p[1]) if r is not None else r for r, p in zip(results[k], params)]
        tuple_results[("Time per step (s)", result_name)] = time_per_step

    elif result_type == "peakmem":
        mem_vals = [r / 1e+9 if r is not None else r for r in results[k] ]
        tuple_results[("Peak Memory (GB)", result_name)] = mem_vals

    elif result_type == "track":
        tuple_results[("FoM", result_name)] = results[k]

#Get columns in order
time_results = [t for t in tuple_results if t[0] == "Time (s)"]
time_per_step_results = [t for t in tuple_results if t[0] == "Time per step (s)"]
fom_results = [t for t in tuple_results if t[0] == "FoM"]
memory_results = [t for t in tuple_results if t[0] == "Peak Memory (GB)"]

col_order = time_results + time_per_step_results + fom_results + memory_results

cols = pd.MultiIndex.from_tuples(col_order)
rows = pd.MultiIndex.from_tuples(params)

df = pd.DataFrame(tuple_results,  columns=cols, index=rows)

df.index.names = ["Parameters", "Steps"]

print(df[time_results])
print("-"*20)
print(df[time_per_step_results])
print("-"*20)
print(df[fom_results])
print("-"*20)
print(df[memory_results])