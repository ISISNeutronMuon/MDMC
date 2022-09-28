c = get_config()
c.NbConvertApp.export_format = "notebook"
c.NbConvertApp.notebooks = [
    "MDMC/doc/tutorials/applying-a-forcefield.ipynb",
    "MDMC/doc/tutorials/Argon-a-to-z.ipynb",
    "MDMC/doc/tutorials/building-a-universe.ipynb",
    "MDMC/doc/tutorials/creating-an-observable.ipynb",
    "MDMC/doc/tutorials/running-a-refinement.ipynb",
    "MDMC/doc/tutorials/running-a-simulation.ipynb",
    "MDMC/doc/tutorials/selecting-fitting-parameters.ipynb",
    "MDMC/doc/tutorials/solvating-a-universe.ipynb",
    "MDMC/doc/tutorials/units.ipynb"
]
c.ExecutePreprocessor.enabled = True
