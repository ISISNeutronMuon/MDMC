c = get_config()
c.ExecutePreprocessor.enabled = True
c.NbConvertApp.export_format = "notebook"
c.NbConvertApp.notebooks = [
    "applying-a-forcefield.ipynb",
    "Argon-a-to-z.ipynb",
    "building-a-universe.ipynb",
    "creating-an-observable.ipynb",
    "running-a-refinement.ipynb",
    "running-a-simulation.ipynb",
    "selecting-fitting-parameters.ipynb",
    "solvating-a-universe.ipynb",
    "units.ipynb"
]