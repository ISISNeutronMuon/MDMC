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
        tuple_results[(result_name, "Time")] = results[k]
        
        time_per_step = [r / int(p[1]) for r, p in zip(results[k], params)]
        tuple_results[(result_name, "Time per step")] = time_per_step
    elif result_type == "track":
        tuple_results[(result_name, "FoM")] = results[k]


cols = pd.MultiIndex.from_tuples(tuple_results.keys())
rows = pd.MultiIndex.from_tuples(params)

df = pd.DataFrame(tuple_results,  columns=cols, index=rows)

df.insert(3, ("Peak Memory (GB)", "refineGPR"), df.pop(("Peak Memory (GB)", "refineGPR")))
df.insert(6, ("FoM", "refineGPR"), df.pop(("FoM", "refineGPR")))

df.index.names = ["Parameters", "Steps"]

cols = df.columns.to_list()

#Quick and dirty way to group columns by benchmark
df.insert(2, ("refineGPO", "FoM"), df.pop(("refineGPO", "FoM")))

print(df)