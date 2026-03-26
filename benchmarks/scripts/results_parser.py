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

results_dict = {
    "Time (s)": {},
    "Peak Memory (GB)": {},
    "FoM": {},
}

for k in results:
    result_type, result_name = k.split("_", 1)
    match result_type:
        case "time":
            results_dict["Time (s)"][result_name] = results[k]

        case "peakmem":
            mem_vals = [r / 1e+9 if r is not None else r for r in results[k] ]
            results_dict["Peak Memory (GB)"][result_name] = mem_vals

        case "track":
            results_dict["FoM"][result_name] = results[k]

md_filename = "benchmark_results.md"

with open(md_filename, "a") as f:
    f.write("<details> <summary> Full benchmark results </summary> \n\n")


#Construct dataframe for each result type and save as markdown
rows = pd.MultiIndex.from_tuples(params, names=["Parameters", "Steps"])

for k, v in results_dict.items():
    df = pd.DataFrame(v, index=rows)

    with open(md_filename, "a") as f:
        f.write(f"{k}\n\n")

    df.reset_index().to_markdown(md_filename, index=False, mode="a", tablefmt='github')

    with open(md_filename, "a") as f:
        f.write("\n\n\n")
        
with open(md_filename, "a") as f:
    f.write("</details>")