import logging
from pathlib import Path
from typing import Optional, Tuple

import xarray as xr

from .assets import SceneAssets
from .config import SceneGenConfig
from .exceptions import DataNotFoundError, ProcessingError
from ..assets.dem import DEMProcessor
from ..assets.landcover import LandCoverProcessor
from ..assets.mesh import MeshGenerator
from ..assets.texture import TextureGenerator
from ..assets.wrb_soil import WRBSoilProcessor
from ..scene import create_s2gos_scene
from ..utils import create_aoi_polygon


class SceneGenerationPipeline:
    """Main pipeline orchestrator for generating 3D scenes from earth observation data."""

    def __init__(self, config: SceneGenConfig):
        """Initialize the pipeline with a configuration object."""
        self.config = config
        self.assets = SceneAssets()

        dem_index_path = config.data_sources.dem_index_path
        dem_root_dir = config.data_sources.dem_root_dir
        landcover_index_path = config.data_sources.landcover_index_path
        landcover_root_dir = config.data_sources.landcover_root_dir
        scene_name = config.scene_name

        self.dem_processor = DEMProcessor(
            index_path=dem_index_path, dem_root_dir=dem_root_dir
        )

        self.landcover_processor = LandCoverProcessor(
            index_path=landcover_index_path, landcover_root_dir=landcover_root_dir
        )

        self.mesh_generator = MeshGenerator()
        # Initialize TextureGenerator with materials configuration for proper soil material mapping
        materials_config_path = config.data_sources.material_config_path
        self.texture_generator = TextureGenerator(materials_config_path=materials_config_path)
        
        # Initialize WRB soil processor (optional - only if WRB data is available)
        self.wrb_soil_processor = None
        wrb_data_path = "/home/gonzalezm/wrb/MostProbable.vrt"  
        try:
            if Path(wrb_data_path).exists():
                self.wrb_soil_processor = WRBSoilProcessor(wrb_data_path)
                logging.info("WRB soil processor initialized successfully")
            else:
                logging.warning(f"WRB data not found at {wrb_data_path}, soil-aware texturing disabled")
        except Exception as e:
            logging.warning(f"Failed to initialize WRB soil processor: {e}, soil-aware texturing disabled")

        self._setup_output_directories()

        logging.info(f"Pipeline initialized for scene '{scene_name}'")

    @property
    def scene_name(self) -> str:
        """Get scene name from the configuration."""
        return self.config.scene_name

    @property
    def center_lat(self) -> float:
        """Get center latitude from the configuration."""
        return self.config.location.center_lat

    @property
    def center_lon(self) -> float:
        """Get center longitude from the configuration."""
        return self.config.location.center_lon

    @property
    def aoi_size_km(self) -> float:
        """Get AOI size from the configuration."""
        return self.config.location.aoi_size_km

    @property
    def target_resolution_m(self) -> float:
        """Get target resolution from the configuration."""
        return self.config.processing.target_resolution_m

    @property
    def generate_texture_preview(self) -> bool:
        """Get texture preview setting from the configuration."""
        return self.config.processing.generate_texture_preview

    @property
    def handle_dem_nans(self) -> bool:
        """Get DEM NaN handling setting from the configuration."""
        return self.config.processing.handle_dem_nans

    @property
    def dem_fillna_value(self) -> float:
        """Get DEM fill value from the configuration."""
        return self.config.processing.dem_fillna_value

    @property
    def enable_buffer(self) -> bool:
        """Get buffer enable setting from the configuration."""
        return self.config.has_buffer

    @property
    def buffer_size_km(self) -> Optional[float]:
        """Get buffer size from the configuration."""
        return self.config.buffer.buffer_size_km if self.config.buffer else None

    @property
    def buffer_resolution_m(self) -> float:
        """Get buffer resolution from the configuration."""
        return self.config.buffer.buffer_resolution_m if self.config.buffer else 100.0

    @property
    def background_elevation(self) -> float:
        """Get background elevation from the configuration."""
        return self.config.buffer.background_elevation if self.config.buffer else 0.0

    @property
    def atmosphere_config(self):
        """Get atmosphere configuration from the configuration."""
        return self.config.atmosphere

    def _setup_output_directories(self) -> None:
        """Create the output directory structure."""
        self.output_dir = self.config.scene_output_dir
        self.data_dir = self.output_dir / "data"
        self.meshes_dir = self.output_dir / "meshes"
        self.textures_dir = self.output_dir / "textures"

        for directory in [
            self.output_dir,
            self.data_dir,
            self.meshes_dir,
            self.textures_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def generate_aoi(self) -> None:
        """Generate the Area of Interest polygon."""
        self.aoi_polygon = create_aoi_polygon(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            side_length_km=self.aoi_size_km,
        )
        logging.info(
            f"Created AOI polygon: {self.aoi_size_km}km x {self.aoi_size_km}km"
        )

    def process_dem(self) -> Path:
        """Process DEM data for the AOI."""
        logging.info("=== Processing DEM Data ===")

        dem_filename = f"dem_{self.scene_name}_{self.target_resolution_m}m.zarr"
        dem_output_path = self.data_dir / dem_filename

        self.dem_processor.generate_dem(
            aoi_polygon=self.aoi_polygon,
            output_path=dem_output_path,
            fillna_value=self.dem_fillna_value,
            target_resolution_m=self.target_resolution_m,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.aoi_size_km,
        )

        self.assets.dem_file = dem_output_path
        logging.info(f"DEM processing complete: {dem_output_path}")
        return dem_output_path

    def process_landcover(self) -> Path:
        """Process land cover data for the AOI."""
        logging.info("=== Processing Land Cover Data ===")

        landcover_filename = (
            f"landcover_{self.scene_name}_{self.target_resolution_m}m.zarr"
        )
        landcover_output_path = self.data_dir / landcover_filename

        self.landcover_processor.generate_landcover(
            aoi_polygon=self.aoi_polygon,
            output_path=landcover_output_path,
            target_resolution_m=self.target_resolution_m,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.aoi_size_km,
        )

        self.assets.landcover_file = landcover_output_path
        logging.info(f"Land cover processing complete: {landcover_output_path}")
        return landcover_output_path

    def process_wrb_soil(self) -> Optional[Path]:
        """Process WRB soil classification data for the AOI."""
        if not self.wrb_soil_processor:
            logging.info("WRB soil processor not available, skipping soil processing")
            return None
            
        logging.info("=== Processing WRB Soil Classification Data ===")

        wrb_filename = f"wrb_soil_{self.scene_name}_{self.target_resolution_m}m.zarr"
        wrb_output_path = self.data_dir / wrb_filename

        self.wrb_soil_processor.generate_wrb_soil_data(
            aoi_polygon=self.aoi_polygon,
            output_path=wrb_output_path,
            target_resolution_m=self.target_resolution_m,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.aoi_size_km,
        )

        self.assets.wrb_soil_file = wrb_output_path
        logging.info(f"WRB soil processing complete: {wrb_output_path}")
        return wrb_output_path

    def process_buffer_wrb_soil(self) -> Optional[Path]:
        """Process WRB soil classification data for buffer area."""
        if not self.enable_buffer or not self.buffer_size_km or not self.wrb_soil_processor:
            return None
            
        logging.info("=== Processing Buffer WRB Soil Classification Data ===")

        buffer_aoi = create_aoi_polygon(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            side_length_km=self.buffer_size_km,
        )

        wrb_filename = f"wrb_soil_buffer_{self.scene_name}_{self.buffer_resolution_m}m.zarr"
        wrb_output_path = self.data_dir / wrb_filename

        self.wrb_soil_processor.generate_wrb_soil_data(
            aoi_polygon=buffer_aoi,
            output_path=wrb_output_path,
            target_resolution_m=self.buffer_resolution_m,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.buffer_size_km,
        )

        self.assets.buffer_wrb_soil_file = wrb_output_path
        logging.info(f"Buffer WRB soil processing complete: {wrb_output_path}")
        return wrb_output_path

    def process_background_wrb_soil(self) -> Optional[Path]:
        """Process WRB soil classification data for background area."""
        if not self.enable_buffer or not hasattr(self.config.buffer, "background_size_km") or not self.wrb_soil_processor:
            return None
            
        logging.info("=== Processing Background WRB Soil Classification Data ===")

        background_aoi = create_aoi_polygon(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            side_length_km=self.config.buffer.background_size_km,
        )

        wrb_filename = f"wrb_soil_background_{self.scene_name}_{self.config.buffer.background_resolution_m}m.zarr"
        wrb_output_path = self.data_dir / wrb_filename

        self.wrb_soil_processor.generate_wrb_soil_data(
            aoi_polygon=background_aoi,
            output_path=wrb_output_path,
            target_resolution_m=self.config.buffer.background_resolution_m,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.config.buffer.background_size_km,
        )

        self.assets.background_wrb_soil_file = wrb_output_path
        logging.info(f"Background WRB soil processing complete: {wrb_output_path}")
        return wrb_output_path

    def generate_mesh(self, dem_file_path: Path) -> Path:
        """Generate 3D mesh from DEM data."""
        logging.info("=== Generating 3D Mesh ===")

        mesh_path = self.meshes_dir / f"{self.scene_name}_terrain.ply"
        mesh = self.mesh_generator.generate_mesh_from_dem_file(
            dem_file_path=dem_file_path,
            output_path=mesh_path,
            add_uvs=True,
            handle_nans=self.handle_dem_nans,
        )
        self.assets.mesh_file = mesh_path

        mesh_info = self.mesh_generator.get_mesh_info(mesh)
        logging.info(
            f"Generated mesh: {mesh_info['vertices']} vertices, {mesh_info['faces']} faces"
        )

        return mesh_path

    def generate_textures(self, landcover_file_path: Path, wrb_soil_file_path: Optional[Path] = None) -> Tuple[Path, Path]:
        """Generate texture maps from land cover data with optional soil-aware mapping."""
        logging.info("=== Generating Textures ===")

        # Load landcover data from Zarr format
        landcover_dataset = xr.open_zarr(landcover_file_path)
        landcover_data = landcover_dataset["landcover"]
        if isinstance(landcover_data, xr.Dataset):
            landcover_data = landcover_data[list(landcover_data.data_vars.keys())[0]]

        # Load WRB soil data if available
        wrb_soil_data = None
        if wrb_soil_file_path and Path(wrb_soil_file_path).exists():
            try:
                logging.info("Loading WRB soil data for soil-aware texturing...")
                wrb_soil_dataset = xr.open_zarr(wrb_soil_file_path)
                wrb_soil_data = wrb_soil_dataset["wrb_soil_class"]
                if isinstance(wrb_soil_data, xr.Dataset):
                    wrb_soil_data = wrb_soil_data[list(wrb_soil_data.data_vars.keys())[0]]
                logging.info("WRB soil data loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load WRB soil data: {e}, using standard landcover mapping")
                wrb_soil_data = None

        # Generate selection texture using soil-aware method
        selection_texture_path = self.textures_dir / f"{self.scene_name}_{self.target_resolution_m}m_selection.png"
        
        if wrb_soil_data is not None:
            logging.info("Using soil-aware texture generation")
            selection_texture = self.texture_generator.landcover_to_soil_aware_selection_texture(
                landcover_data=landcover_data,
                wrb_soil_data=wrb_soil_data,
                output_path=selection_texture_path,
                flip_vertical=False,
                default_material_index=7,
            )
        else:
            logging.info("Using standard landcover texture generation")
            selection_texture = self.texture_generator.landcover_to_selection_texture(
                landcover_data=landcover_data,
                output_path=selection_texture_path,
                flip_vertical=False,
                default_material_index=7,
            )

        # Generate preview texture if requested
        preview_texture_path = None
        if self.generate_texture_preview:
            preview_texture_path = self.textures_dir / f"{self.scene_name}_{self.target_resolution_m}m_preview.png"
            self.texture_generator.create_preview_texture(
                landcover_data=landcover_data,
                output_path=preview_texture_path,
                flip_vertical=True,
            )

        self.assets.selection_texture_file = selection_texture_path
        if preview_texture_path:
            self.assets.preview_texture_file = preview_texture_path

        # Analyze landcover classes
        analysis = self.texture_generator.analyze_landcover_classes(landcover_data)
        logging.info(
            f"Texture analysis: {analysis['unique_classes']} land cover classes found"
        )

        return selection_texture_path, preview_texture_path

    def process_buffer_dem(self) -> Optional[Path]:
        """Process DEM data for buffer area (if enabled)."""
        if not self.enable_buffer or not self.buffer_size_km:
            return None

        logging.info("=== Processing Buffer DEM Data ===")

        buffer_aoi = create_aoi_polygon(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            side_length_km=self.buffer_size_km,
        )

        dem_filename = f"dem_buffer_{self.scene_name}_{self.buffer_resolution_m}m.zarr"
        dem_output_path = self.data_dir / dem_filename

        self.dem_processor.generate_dem(
            aoi_polygon=buffer_aoi,
            output_path=dem_output_path,
            fillna_value=self.dem_fillna_value,
            target_resolution_m=self.buffer_resolution_m,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.buffer_size_km,
        )

        self.assets.buffer_dem_file = dem_output_path
        logging.info(f"Buffer DEM processing complete: {dem_output_path}")
        return dem_output_path

    def process_buffer_landcover(self) -> Optional[Path]:
        """Process land cover data for buffer area (if enabled)."""
        if not self.enable_buffer or not self.buffer_size_km:
            return None

        logging.info("=== Processing Buffer Land Cover Data ===")

        buffer_aoi = create_aoi_polygon(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            side_length_km=self.buffer_size_km,
        )

        landcover_filename = (
            f"landcover_buffer_{self.scene_name}_{self.buffer_resolution_m}m.zarr"
        )
        landcover_output_path = self.data_dir / landcover_filename

        self.landcover_processor.generate_landcover(
            aoi_polygon=buffer_aoi,
            output_path=landcover_output_path,
            target_resolution_m=self.buffer_resolution_m,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.buffer_size_km,
        )

        self.assets.buffer_landcover_file = landcover_output_path
        logging.info(f"Buffer land cover processing complete: {landcover_output_path}")
        return landcover_output_path

    def process_background_landcover(self) -> Optional[Path]:
        """Process land cover data for background area (mirrors buffer system)."""
        if not self.enable_buffer or not hasattr(
            self.config.buffer, "background_size_km"
        ):
            return None

        logging.info("=== Processing Background Land Cover Data ===")

        # Create background AOI polygon
        background_aoi = create_aoi_polygon(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            side_length_km=self.config.buffer.background_size_km,
        )

        landcover_filename = f"landcover_background_{self.scene_name}_{self.config.buffer.background_resolution_m}m.zarr"
        landcover_output_path = self.data_dir / landcover_filename

        self.landcover_processor.generate_landcover(
            aoi_polygon=background_aoi,
            output_path=landcover_output_path,
            target_resolution_m=self.config.buffer.background_resolution_m,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.config.buffer.background_size_km,
        )

        self.assets.background_landcover_file = landcover_output_path
        logging.info(
            f"Background land cover processing complete: {landcover_output_path}"
        )
        return landcover_output_path

    def generate_background_textures(
        self, background_landcover_file_path: Path, background_wrb_soil_file_path: Optional[Path] = None
    ) -> Optional[Tuple[Path, Path]]:
        """Generate texture maps for background area with soil-aware mapping."""
        if not background_landcover_file_path:
            return None, None

        logging.info("=== Generating Background Textures ===")

        # Load landcover data from Zarr format
        landcover_dataset = xr.open_zarr(background_landcover_file_path)
        landcover_data = landcover_dataset["landcover"]
        if isinstance(landcover_data, xr.Dataset):
            landcover_data = landcover_data[list(landcover_data.data_vars.keys())[0]]

        # Load WRB soil data if available
        wrb_soil_data = None
        if background_wrb_soil_file_path and Path(background_wrb_soil_file_path).exists():
            try:
                logging.info("Loading background WRB soil data for soil-aware texturing...")
                wrb_soil_dataset = xr.open_zarr(background_wrb_soil_file_path)
                wrb_soil_data = wrb_soil_dataset["wrb_soil_class"]
                if isinstance(wrb_soil_data, xr.Dataset):
                    wrb_soil_data = wrb_soil_data[list(wrb_soil_data.data_vars.keys())[0]]
                logging.info("Background WRB soil data loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load background WRB soil data: {e}, using standard landcover mapping")
                wrb_soil_data = None

        # Generate selection texture using soil-aware method
        selection_texture_path = self.textures_dir / f"{self.scene_name}_background_{self.config.buffer.background_resolution_m}m_selection.png"
        
        if wrb_soil_data is not None:
            logging.info("Using soil-aware background texture generation")
            selection_texture = self.texture_generator.landcover_to_soil_aware_selection_texture(
                landcover_data=landcover_data,
                wrb_soil_data=wrb_soil_data,
                output_path=selection_texture_path,
                flip_vertical=False,
                default_material_index=7,
            )
        else:
            logging.info("Using standard background landcover texture generation")
            selection_texture = self.texture_generator.landcover_to_selection_texture(
                landcover_data=landcover_data,
                output_path=selection_texture_path,
                flip_vertical=False,
                default_material_index=7,
            )

        # Generate preview texture if requested
        preview_texture_path = None
        if self.generate_texture_preview:
            preview_texture_path = self.textures_dir / f"{self.scene_name}_background_{self.config.buffer.background_resolution_m}m_preview.png"
            self.texture_generator.create_preview_texture(
                landcover_data=landcover_data,
                output_path=preview_texture_path,
                flip_vertical=True,
            )

        self.assets.background_selection_texture_file = selection_texture_path
        if preview_texture_path:
            self.assets.background_preview_texture_file = preview_texture_path

        logging.info("Background texture generation complete")
        return selection_texture_path, preview_texture_path

    def generate_buffer_mesh(self, buffer_dem_file_path: Path) -> Optional[Path]:
        """Generate 3D mesh for buffer area."""
        if not buffer_dem_file_path:
            return None

        logging.info("=== Generating Buffer 3D Mesh ===")

        mesh_path = self.meshes_dir / f"{self.scene_name}_buffer_terrain.ply"
        mesh = self.mesh_generator.generate_mesh_from_dem_file(
            dem_file_path=buffer_dem_file_path,
            output_path=mesh_path,
            add_uvs=True,
            handle_nans=self.handle_dem_nans,
        )
        self.assets.buffer_mesh_file = mesh_path

        mesh_info = self.mesh_generator.get_mesh_info(mesh)
        logging.info(
            f"Generated buffer mesh: {mesh_info['vertices']} vertices, {mesh_info['faces']} faces"
        )

        return mesh_path

    def generate_buffer_textures(
        self, buffer_landcover_file_path: Path, buffer_wrb_soil_file_path: Optional[Path] = None
    ) -> Optional[Tuple[Path, Path]]:
        """Generate texture maps for buffer area with soil-aware mapping."""
        if not buffer_landcover_file_path:
            return None, None

        logging.info("=== Generating Buffer Textures ===")

        # Load landcover data from Zarr format
        landcover_dataset = xr.open_zarr(buffer_landcover_file_path)
        landcover_data = landcover_dataset["landcover"]
        if isinstance(landcover_data, xr.Dataset):
            landcover_data = landcover_data[list(landcover_data.data_vars.keys())[0]]

        # Load WRB soil data if available
        wrb_soil_data = None
        if buffer_wrb_soil_file_path and Path(buffer_wrb_soil_file_path).exists():
            try:
                logging.info("Loading buffer WRB soil data for soil-aware texturing...")
                wrb_soil_dataset = xr.open_zarr(buffer_wrb_soil_file_path)
                wrb_soil_data = wrb_soil_dataset["wrb_soil_class"]
                if isinstance(wrb_soil_data, xr.Dataset):
                    wrb_soil_data = wrb_soil_data[list(wrb_soil_data.data_vars.keys())[0]]
                logging.info("Buffer WRB soil data loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load buffer WRB soil data: {e}, using standard landcover mapping")
                wrb_soil_data = None

        # Generate selection texture using soil-aware method
        selection_texture_path = self.textures_dir / f"{self.scene_name}_buffer_{self.buffer_resolution_m}m_selection.png"
        
        if wrb_soil_data is not None:
            logging.info("Using soil-aware buffer texture generation")
            selection_texture = self.texture_generator.landcover_to_soil_aware_selection_texture(
                landcover_data=landcover_data,
                wrb_soil_data=wrb_soil_data,
                output_path=selection_texture_path,
                flip_vertical=False,
                default_material_index=7,
            )
        else:
            logging.info("Using standard buffer landcover texture generation")
            selection_texture = self.texture_generator.landcover_to_selection_texture(
                landcover_data=landcover_data,
                output_path=selection_texture_path,
                flip_vertical=False,
                default_material_index=7,
            )

        # Generate preview texture if requested
        preview_texture_path = None
        if self.generate_texture_preview:
            preview_texture_path = self.textures_dir / f"{self.scene_name}_buffer_{self.buffer_resolution_m}m_preview.png"
            self.texture_generator.create_preview_texture(
                landcover_data=landcover_data,
                output_path=preview_texture_path,
                flip_vertical=True,
            )

        self.assets.buffer_selection_texture_file = selection_texture_path
        if preview_texture_path:
            self.assets.buffer_preview_texture_file = preview_texture_path

        logging.info("Buffer texture generation complete")
        return selection_texture_path, preview_texture_path

    def _create_scene_config(self):
        """Create complete scene configuration from generated assets."""
        buffer_mesh_path = None
        buffer_texture_path = None
        buffer_size_km = None

        if (
            self.enable_buffer
            and self.assets.buffer_mesh_file
            and self.assets.buffer_selection_texture_file
        ):
            buffer_mesh_path = str(
                self.assets.buffer_mesh_file.relative_to(self.output_dir)
            )
            buffer_texture_path = str(
                self.assets.buffer_selection_texture_file.relative_to(self.output_dir)
            )
            buffer_size_km = self.buffer_size_km

        buffer_dem_file = None
        if self.enable_buffer and self.assets.buffer_dem_file:
            buffer_dem_file = str(
                self.assets.buffer_dem_file.relative_to(self.output_dir)
            )

        dem_index_path = str(self.config.data_sources.dem_index_path)
        landcover_index_path = str(self.config.data_sources.landcover_index_path)
        material_config_path = self.config.data_sources.material_config_path

        background_selection_texture = None
        background_size_km = None
        if (
            self.enable_buffer
            and self.assets.background_selection_texture_file
            and hasattr(self.config.buffer, "background_size_km")
        ):
            background_selection_texture = str(
                self.assets.background_selection_texture_file.relative_to(
                    self.output_dir
                )
            )
            background_size_km = self.config.buffer.background_size_km

        return create_s2gos_scene(
            scene_name=self.scene_name,
            mesh_path=str(self.assets.mesh_file.relative_to(self.output_dir)),
            texture_path=str(
                self.assets.selection_texture_file.relative_to(self.output_dir)
            ),
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            aoi_size_km=self.aoi_size_km,
            resolution_m=self.target_resolution_m,
            buffer_mesh_path=buffer_mesh_path,
            buffer_texture_path=buffer_texture_path,
            buffer_size_km=buffer_size_km,
            output_dir=self.output_dir,
            buffer_dem_file=buffer_dem_file,
            background_elevation=self.background_elevation,
            background_selection_texture=background_selection_texture,
            background_size_km=background_size_km,
            dem_index_path=dem_index_path,
            landcover_index_path=landcover_index_path,
            material_config_path=material_config_path,
            atmosphere_config=self.atmosphere_config,
        )

    def run_full_pipeline(self):
        """Execute the complete scene generation pipeline."""
        logging.info(f"Starting full pipeline for scene '{self.scene_name}'")

        try:
            self.generate_aoi()
            dem_file = self.process_dem()
            landcover_file = self.process_landcover()
            wrb_soil_file = self.process_wrb_soil()  # Process WRB soil data
            self.generate_mesh(dem_file)
            texture_selection, texture_preview = self.generate_textures(landcover_file, wrb_soil_file)

            buffer_dem_file = None
            buffer_landcover_file = None
            buffer_texture_selection = None
            buffer_texture_preview = None

            background_landcover_file = None
            background_texture_selection = None
            background_texture_preview = None

            if self.enable_buffer:
                buffer_dem_file = self.process_buffer_dem()
                buffer_landcover_file = self.process_buffer_landcover()
                buffer_wrb_soil_file = self.process_buffer_wrb_soil()  # Process buffer WRB soil data
                if buffer_dem_file:
                    self.generate_buffer_mesh(buffer_dem_file)
                if buffer_landcover_file:
                    buffer_texture_selection, buffer_texture_preview = (
                        self.generate_buffer_textures(buffer_landcover_file, buffer_wrb_soil_file)
                    )

                background_landcover_file = self.process_background_landcover()
                background_wrb_soil_file = self.process_background_wrb_soil()  # Process background WRB soil data
                if background_landcover_file:
                    background_texture_selection, background_texture_preview = (
                        self.generate_background_textures(background_landcover_file, background_wrb_soil_file)
                    )

            scene_config = self._create_scene_config()
            scene_config_file = self.output_dir / f"{self.scene_name}.yml"
            scene_config_file_json = self.output_dir / f"{self.scene_name}.json"

            scene_config.save_yaml(scene_config_file)
            scene_config.save_json(scene_config_file_json)

            self.assets.config_file = scene_config_file

            logging.info("=== Pipeline Complete ===")
            logging.info(f"Scene configuration saved to: {scene_config_file}")
            logging.info(f"Scene configuration saved to: {scene_config_file_json}")


            return scene_config

        except FileNotFoundError as e:
            raise DataNotFoundError(f"Required file not found: {e}", str(e))
        except PermissionError as e:
            raise DataNotFoundError(f"Permission denied accessing file: {e}", str(e))
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise ProcessingError("Pipeline execution failed", "pipeline", e)
