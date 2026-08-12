# mesh_builder.py
from dataclasses import dataclass
from pathlib import Path

from yaml2nuctrans.meshing.mesh_1d import CreateMesh1D



@dataclass
class MeshFiles1D:
    """Store generated OGS mesh paths and mesh names.

    Attributes:
        file_path: Directory containing all generated mesh files.
        dimension: Spatial mesh dimension.
        domain: Full-domain mesh name.
        top: Top-boundary point mesh name.
        bottom: Bottom-boundary point mesh name.
        reactive_domain: Mesh for the source-containing geological unit.
    """

    file_path: Path
    dimension: int
    domain: str
    top: str
    bottom: str
    reactive_domain: str

class MeshBuilder:
    """Create OGS meshes from units, source geometry, and configuration.
    """
    def __init__(self):
        pass

    @staticmethod
    def build(sim_config, units, output_directory):
        """create OGS meshes based on the simulation setup and units.

        Args:
            sim_config (SimulationConfig): The simulation configuration.
            units (Units1D): The units for the simulation.
            output_directory (str): The directory to output the generated mesh files.

        Returns:
            MeshFiles1D: Paths and standard names of the required meshes.
        Raises:
            NotImplementedError: If 2D or 3D mesh generation is selected.
        """
        if sim_config.mesh_dimension == 1:
            return MeshBuilder.build_1d(units, sim_config, output_directory)

        if sim_config.mesh_dimension == 2:
            return MeshBuilder.build_2d()

        return MeshBuilder.build_3d()
    
    @staticmethod
    def build_1d(units, sim_config, output_directory):
        """create 1D mesh

        Args:
            units (Units1D): The units for the simulation.
            sim_config (SimulationConfig): The simulation configuration.
            output_directory (str): The directory to output the generated mesh files.

        Returns:
            MeshFiles1D: Paths and standard names of the required meshes.
        """
        

        if sim_config.create_meshes:
            CreateMesh1D.generate_1d_meshes(units=units, source=sim_config.source, output_msh_path=output_directory)
            output_directory = "" # relative to the project file
        return MeshFiles1D(
            file_path=output_directory, 
            dimension=sim_config.meshes.dimension,
            domain=sim_config.meshes.domain,
            top=sim_config.meshes.top,
            bottom=sim_config.meshes.bottom,
            reactive_domain=sim_config.meshes.reactive_domain
        )

    @staticmethod
    def build_2d():
        raise NotImplementedError(
            "2D mesh creation not implemented yet"
        )
    @staticmethod
    def build_3d():

        raise NotImplementedError(
            "3D mesh creation not implemented yet"
        )