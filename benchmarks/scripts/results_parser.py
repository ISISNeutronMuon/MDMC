import json
import pandas as pd
import numpy as np
from itertools import product
from pprint import pprint

filename = "results/a23cc70a78d3/e3e2bb98-virtualenv-py3.12.json"

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
