from dataclasses import dataclass
import re

from yaml2nuctrans.config.ogs_sim_config import SimulationConfig
from yaml2nuctrans.primary_parameters.geometry import Units1D


@dataclass
class MediumIds:
    """Map OGS medium IDs to geological layer names.

    Attributes:
        by_id: Mapping from an OGS medium ID to a unit name.
            For example, ``{"0": "Muschelkalk", "1": "Keuper"}``.
    """

    by_id: dict[str, str]

    @classmethod
    def create_ids(cls, sim_config, units):
        """Create medium IDs appropriate for the mesh dimension.

        For a 1D mesh, each key ending in ``_bottom`` defines the
        bottom interface of one geological layer. IDs are assigned from
        bottom to top: the deepest layer receives ID ``"0"``.

        Args:
            sim_config (SimulationConfig): Parsed simulation configuration.
            units (Units1D): Geological unit interface elevations.

        Returns:
            MediumIds: Mapping from OGS medium ID to layer name.

        Raises:
            NotImplementedError: If the mesh dimension is not 1.
        """
        if sim_config.mesh_dimension != 1:
            raise NotImplementedError(
                "Medium-ID creation is currently implemented "
                "only for 1D meshes."
            )

        return cls._from_units_1d(units)

    @classmethod
    def _from_units_1d(cls, units):
        """Create bottom-to-top medium IDs from 1D unit interfaces.

        Args:
            units (Units1D): Geological unit interface elevations.

        Returns:
            MediumIds: Mapping from OGS medium ID to layer name.
        """
        rock_layers = [
            re.sub(r"_bottom$", "", unit_name)
            for unit_name in units.interfaces
            if unit_name.endswith("_bottom")
        ]

        number_of_layers = len(rock_layers)

        by_id = {
            str(number_of_layers - 1 - index): layer_name
            for index, layer_name in enumerate(rock_layers)
        }

        return cls(by_id=by_id)

    @property
    def by_layer(self) -> dict[str, str]:
        """Return the inverse mapping: layer name to OGS medium ID."""
        return {
            layer_name: medium_id
            for medium_id, layer_name in self.by_id.items()
        }

    def id_for(self, layer_name: str) -> str:
        """Return the OGS medium ID for one geological layer.

        Args:
            layer_name: Unit name without the ``_bottom`` suffix.

        Returns:
            str: OGS medium ID.

        Raises:
            KeyError: If the layer is not in this medium-ID mapping.
        """
        return self.by_layer[layer_name]