import json
import pandas as pd
from itertools import product
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    prog='MDMC ASV results parser',
    description='Parse the results of an MDMC benchmarking run to make it more human readable'
)

parser.add_argument(
    "benchmark_commit",
    help="commit get results for (8 character hash)"
)

parser.add_argument(
    "--python_version", 
    default="3.10", 
    help="python version benchmarks were run with. Defaults to 3.10"
)

args = parser.parse_args()

filename = f"{args.benchmark_commit}-virtualenv-py{args.python_version}.json"

results_dir = Path(__file__).parent.parent.absolute() / "results/"

#Assume we only have one machine
machine_dir = [x.name for x in results_dir.iterdir() if x.is_dir()][0]
results_dir = results_dir / machine_dir

filename = results_dir / filename

with filename.open("r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    
results = {k.rsplit(".", 1)[1]: v[0] for k, v in data["results"].items()}

#Take params from first benchmark's result
#Tuple of (number of parameters, number of refinement steps)
results_vals = list(data["results"].values())
params = list(product(*results_vals[0][1]))

results_dict = {
    "Time (s)": {},
    "Peak Memory (GB)": {},
    "FoM": {},
}

for k in results:
    result_type, result_name = k.split("_")
    if result_type == "time":
        results_dict["Time (s)"][result_name] = results[k]
    elif result_type == "track":
        results_dict["FoM"][result_name] = results[k]
    elif result_type == "peakmem":
        mem_vals = [r / 1e+9 if r is not None else r for r in results[k] ]
        results_dict["Peak Memory (GB)"][result_name] = mem_vals

cols = pd.MultiIndex.from_tuples(results_dict.keys())
rows = pd.MultiIndex.from_tuples(params)

df = pd.DataFrame(results_dict,  columns=cols, index=rows)

df.insert(3, ("Peak Memory (GB)", "refineGPR"), df.pop(("Peak Memory (GB)", "refineGPR")))
df.insert(6, ("FoM", "refineGPR"), df.pop(("FoM", "refineGPR")))

df.index.names = ["Parameters", "Steps"]

cols = df.columns.to_list()

#Quick and dirty way to group columns by benchmark
df.insert(2, ("refineGPO", "FoM"), df.pop(("refineGPO", "FoM")))

print(df)
