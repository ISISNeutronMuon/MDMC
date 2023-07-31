c = get_config()
c.NbConvertApp.export_format = "notebook"
c.NbConvertApp.notebooks = [
    "doc/tutorials/Argon-a-to-z.ipynb",
    "doc/how-to/use-MDMC/notebooks/applying-a-forcefield.ipynb",
    "doc/how-to/use-MDMC/notebooks/creating-atomic-configurations.ipynb",
    "doc/how-to/use-MDMC/notebooks/defining-molecule-interactions.ipynb",
    "doc/how-to/use-MDMC/notebooks/creating-an-observable.ipynb",
    "doc/how-to/use-MDMC/notebooks/running-a-simulation.ipynb",
    "doc/how-to/use-MDMC/notebooks/selecting-fitting-parameters.ipynb",
    "doc/how-to/use-MDMC/notebooks/solvating-a-universe.ipynb",
    "doc/how-to/use-MDMC/notebooks/units.ipynb",
    "doc/tutorials/filling-with-packmol.ipynb"
]