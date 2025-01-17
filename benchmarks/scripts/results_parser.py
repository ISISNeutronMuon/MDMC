import json
import pandas as pd
from itertools import product
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    prog='MDMC ASV results parser',
    description='Parse the results of an MDMC benchmarking run to make it more human readable'
)

parser.add_argument("benchmark_commit")
args = parser.parse_args()

filename = f"{args.benchmark_commit}-virtualenv-py3.10.json"

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

tuple_results = {}

for k in results:
    result_type, result_name = k.split("_", 1)
    match result_type:
        case "time":
            tuple_results[("Time (s)", result_name)] = results[k]

            time_per_step = [r / int(p[1]) if r is not None else r for r, p in zip(results[k], params)]
            tuple_results[("Time per step (s)", result_name)] = time_per_step

        case "peakmem":
            mem_vals = [r / 1e+9 if r is not None else r for r in results[k] ]
            tuple_results[("Peak Memory (GB)", result_name)] = mem_vals

        case "track":
            tuple_results[("FoM", result_name)] = results[k]

#Get columns in order
time_results = [t for t in tuple_results if t[0] == "Time (s)"]
time_per_step_results = [t for t in tuple_results if t[0] == "Time per step (s)"]
fom_results = [t for t in tuple_results if t[0] == "FoM"]
memory_results = [t for t in tuple_results if t[0] == "Peak Memory (GB)"]

col_order = time_results + time_per_step_results + fom_results + memory_results
col_groups = [time_results, time_per_step_results, fom_results, memory_results]

cols = pd.MultiIndex.from_tuples(col_order)
rows = pd.MultiIndex.from_tuples(params)

df = pd.DataFrame(tuple_results,  columns=cols, index=rows)

df.index.names = ["Parameters", "Steps"]

md_filename = f"benchmark_results_{args.benchmark_commit}.md"

with open(md_filename, "w") as f:
    pass

for c in col_groups:
    df[c].to_markdown(md_filename, mode="a")
    with open(md_filename, "a") as f:
        f.write("\n\n\n")