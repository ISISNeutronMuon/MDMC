"""An """
from MDMC.exporters.configurations.conf_exporter import ConfigurationExporter
from MDMC.MD import Structure

class ProteinDataBankExporter(ConfigurationExporter):

    def __init__(self, file_name: str):
        super().__init__(file_name)

    def write(self, structure: Structure, **settings: dict) -> None:
        pass