"""
Tests the Jupyter notebooks in ./doc/tutorials/
This script is intended to be run from the main MDMCv0.2_pilot directory
"""

import papermill as pm
import os

notebooks = []
for _, _, files in os.walk("./doc/tutorials/"):
    for file in files:
        if file.endswith(".ipynb"):
            notebooks.append(f"./doc/tutorials/{file}")

failures = 0
for notebook in notebooks:
    try:
        pm.execute_notebook(notebook, "_.ipynb")
    except pm.exceptions.PapermillExecutionError as error:
        print(f"Notebook {notebook} failed with error: {error}")
        failures += 1
        
if failures > 0:
    os._exit(1)
os._exit(0)
