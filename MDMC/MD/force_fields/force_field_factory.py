"""Factory class for generating force fields"""

from pathlib import Path

from MDMC.common.factory import ModuleFactory
from MDMC.MD.force_fields.ff import ForceField


class ForceFieldFactory(ModuleFactory[ForceField]):
    """
    Provides a factory for creating a ``ForceField``.

    Any force field within the force fields folder can be created with
    a string of the class name, as long as it is a subclass of
    ``ForceField``.
    """
    registry: dict[str, ForceField] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "force_field_factory.py")

ForceFieldFactory.scan()
