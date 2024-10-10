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
results_vals = list(data["results"].values())
params = product(*results_vals[0][1])

params = [f"{p[0]} parameters, {p[1]} steps" for p in params]

tuple_results = {}

for k in results.keys():
    result_type, result_name = k.split("_")
    if result_type == "time":
        tuple_results[("time", result_name)] = results[k]
    elif result_type == "track":
        tuple_results[("FoM", result_name)] = results[k]

cols = pd.MultiIndex.from_tuples(tuple_results.keys())

df = pd.DataFrame(tuple_results,  columns=cols, index=params)

print(df)
