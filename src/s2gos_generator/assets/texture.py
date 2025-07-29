import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from PIL import Image

DEFAULT_MATERIALS = [
    {
        "name": "Tree cover",
        "esa_class": 10,
        "color_8bit": (40, 75, 30),
        "roughness": 0.6,
    },
    {
        "name": "Shrubland",
        "esa_class": 20,
        "color_8bit": (185, 170, 130),
        "roughness": 0.7,
    },
    {
        "name": "Grassland",
        "esa_class": 30,
        "color_8bit": (140, 155, 95),
        "roughness": 0.7,
    },
    {
        "name": "Cropland",
        "esa_class": 40,
        "color_8bit": (240, 150, 255),
        "roughness": 0.6,
    },
    {
        "name": "Built-up",
        "esa_class": 50,
        "color_8bit": (150, 150, 150),
        "roughness": 0.3,
    },
    {
        "name": "Bare / sparse vegetation",
        "esa_class": 60,
        "color_8bit": (220, 140, 90),
        "roughness": 0.8,
    },
    {
        "name": "Snow and ice",
        "esa_class": 70,
        "color_8bit": (240, 240, 240),
        "roughness": 0.2,
    },
    {
        "name": "Permanent water bodies",
        "esa_class": 80,
        "color_8bit": (0, 100, 200),
        "roughness": 0.1,
    },
    {
        "name": "Herbaceous wetland",
        "esa_class": 90,
        "color_8bit": (80, 120, 90),
        "roughness": 0.4,
    },
    {
        "name": "Mangroves",
        "esa_class": 95,
        "color_8bit": (0, 207, 117),
        "roughness": 0.4,
    },
    {
        "name": "Moss and lichen",
        "esa_class": 100,
        "color_8bit": (250, 230, 160),
        "roughness": 0.8,
    },
]


class TextureGenerator:
    """
    Generates texture maps from land cover data for use in 3D rendering.
    """

    def __init__(self, materials: Optional[List[Dict]] = None, materials_config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the texture generator.

        Args:
            materials: List of material definitions. If None, uses default materials or loads from config.
            materials_config_path: Path to materials.json file. If provided, loads materials from file.
        """
        if materials_config_path is not None:
            self.materials, self.material_name_to_index = self._load_materials_from_config(materials_config_path)
        elif materials is not None:
            self.materials = materials
            self.material_name_to_index = {mat["name"]: idx for idx, mat in enumerate(self.materials)}
        else:
            self.materials = DEFAULT_MATERIALS
            self.material_name_to_index = {mat["name"]: idx for idx, mat in enumerate(self.materials)}
            
        self.class_to_index = {
            mat["esa_class"]: idx for idx, mat in enumerate(self.materials) if mat["esa_class"] is not None
        }
        logging.info(
            f"TextureGenerator initialized with {len(self.materials)} materials"
        )

    def _load_materials_from_config(self, config_path: Union[str, Path]) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Load materials from materials.json configuration file.
        
        Args:
            config_path: Path to materials.json file
            
        Returns:
            Tuple of (materials_list, material_name_to_index_mapping)
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logging.warning(f"Materials config not found at {config_path}, using default materials")
            return DEFAULT_MATERIALS, {mat["name"]: idx for idx, mat in enumerate(DEFAULT_MATERIALS)}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            materials_config = config.get("materials", {})
            landcover_mapping = config.get("landcover_mapping", {})
            
            # Build materials list from configuration 
            materials = []
            material_name_to_index = {}
            
            # Add landcover materials in the EXACT same order as scene generation
            # This must match the hardcoded order in scene/config.py:216-228
            landcover_order = [
                "tree_cover", "shrubland", "grassland", "cropland", "built_up", 
                "bare_sparse_vegetation", "snow_and_ice", "permanent_water_bodies", 
                "herbaceous_wetland", "mangroves", "moss_and_lichen"
            ]
            
            for landcover_class in landcover_order:
                if landcover_class in landcover_mapping:
                    material_name = landcover_mapping[landcover_class]
                    if material_name in materials_config:
                        # Map landcover classes to ESA classes (must match scene/config.py landcover_ids)
                        landcover_to_esa = {
                            "tree_cover": 10, "shrubland": 20, "grassland": 30, "cropland": 40,
                            "built_up": 50, "bare_sparse_vegetation": 60, "snow_and_ice": 70,
                            "permanent_water_bodies": 80, "herbaceous_wetland": 90,
                            "mangroves": 95, "moss_and_lichen": 100
                        }
                        esa_class = landcover_to_esa.get(landcover_class)
                        
                        if esa_class is not None:
                            mat_info = {
                                "name": material_name,
                                "esa_class": esa_class,
                                "color_8bit": next((m["color_8bit"] for m in DEFAULT_MATERIALS if m["esa_class"] == esa_class), (128, 128, 128)),
                                "roughness": next((m["roughness"] for m in DEFAULT_MATERIALS if m["esa_class"] == esa_class), 0.5)
                            }
                            materials.append(mat_info)
                            material_name_to_index[material_name] = len(materials) - 1
            
            # Add additional materials (like soil types) that aren't in landcover mapping
            for material_name in materials_config.keys():
                if material_name not in material_name_to_index:
                    mat_info = {
                        "name": material_name,
                        "esa_class": None,  # No ESA class for additional materials
                        "color_8bit": (200, 150, 100),  # Default soil color
                        "roughness": 0.8
                    }
                    materials.append(mat_info)
                    material_name_to_index[material_name] = len(materials) - 1
            
            logging.info(f"Loaded {len(materials)} materials from {config_path}")
            return materials, material_name_to_index
            
        except Exception as e:
            logging.error(f"Failed to load materials from {config_path}: {e}")
            logging.warning("Falling back to default materials")
            return DEFAULT_MATERIALS, {mat["name"]: idx for idx, mat in enumerate(DEFAULT_MATERIALS)}

    def landcover_to_selection_texture(
        self,
        landcover_data: xr.DataArray,
        output_path: Path,
        flip_vertical: bool = False,
        default_material_index: int = 7,
    ) -> np.ndarray:
        """
        Converts land cover classification data to a material selection texture.

        Args:
            landcover_data: xarray DataArray containing land cover class values.
            output_path: Path where the texture PNG will be saved.
            flip_vertical: If True, flips the texture vertically (for Mitsuba compatibility).
            default_material_index: Material index to use for unknown classes.

        Returns:
            The selection texture as a numpy array.
        """
        logging.info("Converting land cover data to selection texture...")
        landcover_data.load()

        class_values = landcover_data.values

        selection_texture = np.full_like(
            class_values, default_material_index, dtype=np.uint8
        )

        for esa_class, material_index in self.class_to_index.items():
            mask = class_values == esa_class
            selection_texture[mask] = material_index
            logging.debug(
                f"Mapped {np.sum(mask)} pixels from ESA class {esa_class} to material index {material_index}"
            )

        if flip_vertical:
            selection_texture = np.flipud(selection_texture)
            logging.info("Applied vertical flip for rendering engine compatibility")

        self._save_selection_texture(selection_texture, output_path)

        logging.info(f"Selection texture saved to {output_path}")
        return selection_texture

    def landcover_to_soil_aware_selection_texture(
        self,
        landcover_data: xr.DataArray,
        wrb_soil_data: Optional[xr.DataArray],
        output_path: Path,
        flip_vertical: bool = False,
        default_material_index: int = 7,
        bare_vegetation_esa_class: int = 60,
    ) -> np.ndarray:
        """
        Converts land cover data to soil-aware material selection texture.
        
        For bare/sparse vegetation areas (ESA class 60), uses WRB soil data to assign
        specific soil materials. For all other areas, uses standard landcover mapping.

        Args:
            landcover_data: xarray DataArray containing land cover class values.
            wrb_soil_data: Optional xarray DataArray containing WRB soil material indices.
            output_path: Path where the texture PNG will be saved.
            flip_vertical: If True, flips the texture vertically (for Mitsuba compatibility).
            default_material_index: Material index to use for unknown classes.
            bare_vegetation_esa_class: ESA class value for bare/sparse vegetation.

        Returns:
            The selection texture as a numpy array.
        """
        logging.info("Converting land cover data to soil-aware selection texture...")
        landcover_data.load()
        
        if wrb_soil_data is not None:
            wrb_soil_data.load()
            logging.info("Using WRB soil data for bare terrain subdivision")
        else:
            logging.warning("No WRB soil data provided, falling back to standard landcover mapping")

        class_values = landcover_data.values
        selection_texture = np.full_like(
            class_values, default_material_index, dtype=np.uint8
        )

        # First, apply standard landcover mapping for all non-bare areas
        bare_mask = class_values == bare_vegetation_esa_class
        for esa_class, material_index in self.class_to_index.items():
            if esa_class != bare_vegetation_esa_class:  # Skip bare vegetation for now
                mask = class_values == esa_class
                selection_texture[mask] = material_index
                logging.debug(
                    f"Mapped {np.sum(mask)} pixels from ESA class {esa_class} to material index {material_index}"
                )

        # Handle bare/sparse vegetation areas with soil-aware mapping
        if np.any(bare_mask):
            bare_pixel_count = np.sum(bare_mask)
            logging.info(f"Processing {bare_pixel_count} bare vegetation pixels with soil data...")
            
            if wrb_soil_data is not None:
                try:
                    # Get WRB soil material indices for bare areas
                    wrb_values = wrb_soil_data.values
                    
                    # Create soil material index mapping using proper material names
                    soil_material_names = {
                        0: "baresoil",      # fallback
                        1: "arenosols",     # arenosols  
                        2: "regosols",      # regosols
                        3: "leptosols",     # leptosols
                        4: "calcisols",     # calcisols
                        5: "solonchaks",    # solonchaks
                    }
                    
                    # Map WRB soil classes to actual material indices
                    soil_to_material_idx = {}
                    for wrb_class, material_name in soil_material_names.items():
                        if wrb_class == 0:
                            # Use baresoil from landcover mapping as fallback
                            fallback_idx = self.class_to_index.get(bare_vegetation_esa_class, default_material_index)
                            soil_to_material_idx[0] = fallback_idx
                            logging.info(f"WRB class 0 (fallback) → material index {fallback_idx}")
                        else:
                            # Look up material index by name
                            material_idx = self.material_name_to_index.get(material_name)
                            if material_idx is not None:
                                soil_to_material_idx[wrb_class] = material_idx
                                logging.info(f"WRB class {wrb_class} ({material_name}) → material index {material_idx}")
                            else:
                                # Fallback to baresoil if material not found
                                logging.warning(f"Material {material_name} not found, using baresoil fallback")
                                fallback_idx = self.class_to_index.get(bare_vegetation_esa_class, default_material_index)
                                soil_to_material_idx[wrb_class] = fallback_idx
                                logging.warning(f"WRB class {wrb_class} ({material_name}) → fallback material index {fallback_idx}")
                                
                    # Check for any WRB values not covered by our mapping
                    unique_wrb_values = np.unique(wrb_values[bare_mask])
                    logging.info(f"WRB values found in bare areas: {unique_wrb_values}")
                    unmapped_values = [v for v in unique_wrb_values if v not in soil_to_material_idx]
                    if unmapped_values:
                        logging.warning(f"Unmapped WRB values (will remain as default): {unmapped_values}")
                    
                    # Apply soil-aware mapping to bare areas
                    for soil_class, target_material_idx in soil_to_material_idx.items():
                        soil_mask = bare_mask & (wrb_values == soil_class)
                        if np.any(soil_mask):
                            selection_texture[soil_mask] = target_material_idx
                            logging.info(
                                f"Mapped {np.sum(soil_mask)} bare pixels from WRB class {soil_class} to material index {target_material_idx}"
                            )
                    
                    logging.info("Soil-aware mapping applied successfully")
                    
                except Exception as e:
                    logging.warning(f"Error in soil-aware mapping: {e}, falling back to standard bare soil")
                    # Fallback to standard bare soil mapping
                    fallback_idx = self.class_to_index.get(bare_vegetation_esa_class, default_material_index)
                    selection_texture[bare_mask] = fallback_idx
            else:
                # No soil data available, use standard mapping
                fallback_idx = self.class_to_index.get(bare_vegetation_esa_class, default_material_index)
                selection_texture[bare_mask] = fallback_idx
                logging.info(f"Mapped {bare_pixel_count} bare pixels to standard baresoil material (index {fallback_idx})")

        if flip_vertical:
            selection_texture = np.flipud(selection_texture)
            logging.info("Applied vertical flip for rendering engine compatibility")

        self._save_selection_texture(selection_texture, output_path)

        logging.info(f"Soil-aware selection texture saved to {output_path}")
        return selection_texture

    def create_preview_texture(
        self,
        landcover_data: xr.DataArray,
        output_path: Path,
        flip_vertical: bool = True,
    ) -> np.ndarray:
        """
        Creates a color preview texture showing the actual material colors.

        Args:
            landcover_data: xarray DataArray containing land cover class values.
            output_path: Path where the preview PNG will be saved.
            flip_vertical: If True, flips the texture vertically.

        Returns:
            The preview texture as a numpy array with shape (height, width, 3).
        """
        logging.info("Creating color preview texture...")
        landcover_data.load()
        class_values = landcover_data.values

        height, width = class_values.shape
        color_texture = np.zeros((height, width, 3), dtype=np.uint8)

        for material in self.materials:
            esa_class = material["esa_class"]
            color = material["color_8bit"]
            mask = class_values == esa_class
            color_texture[mask] = color

        known_classes = set(mat["esa_class"] for mat in self.materials)
        unknown_mask = ~np.isin(class_values, list(known_classes))
        color_texture[unknown_mask] = (128, 128, 128)

        if flip_vertical:
            color_texture = np.flipud(color_texture)

        self._save_color_texture(color_texture, output_path)

        logging.info(f"Preview texture saved to {output_path}")
        return color_texture

    def _save_selection_texture(self, texture: np.ndarray, output_path: Path) -> None:
        """Save selection texture as a grayscale PNG."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(texture, mode="L")
        image.save(output_path)

    def _save_color_texture(self, texture: np.ndarray, output_path: Path) -> None:
        """Save color texture as RGB PNG."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(texture, mode="RGB")
        image.save(output_path)

    def get_material_info(self) -> Dict:
        """
        Returns information about the configured materials.

        Returns:
            Dictionary containing material configuration details.
        """
        return {
            "num_materials": len(self.materials),
            "materials": self.materials,
            "class_mapping": self.class_to_index,
        }

    def analyze_landcover_classes(self, landcover_data: xr.DataArray) -> Dict:
        """
        Analyzes the land cover data to show class distribution.

        Args:
            landcover_data: xarray DataArray containing land cover class values.

        Returns:
            Dictionary with class statistics.
        """
        landcover_data.load()
        class_values = landcover_data.values

        unique_classes, counts = np.unique(class_values, return_counts=True)
        total_pixels = class_values.size

        class_stats = {}
        for cls, count in zip(unique_classes, counts):
            percentage = (count / total_pixels) * 100
            material_name = "Unknown"

            for material in self.materials:
                if material["esa_class"] == cls:
                    material_name = material["name"]
                    break

            class_stats[int(cls)] = {
                "name": material_name,
                "count": int(count),
                "percentage": round(percentage, 2),
            }

        return {
            "total_pixels": total_pixels,
            "unique_classes": len(unique_classes),
            "class_distribution": class_stats,
        }

    def generate_textures_from_file(
        self,
        landcover_file_path: Path,
        output_dir: Path,
        base_name: str,
        create_preview: bool = True,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Complete pipeline: loads land cover from file and generates textures.

        Args:
            landcover_file_path: Path to the land cover NetCDF file.
            output_dir: Directory where textures will be saved.
            base_name: Base name for output files.
            create_preview: Whether to create a color preview texture.

        Returns:
            Tuple of (selection_texture_path, preview_texture_path).
        """
        logging.info(f"Loading land cover data from {landcover_file_path}")

        landcover_dataset = xr.open_zarr(landcover_file_path)
        landcover_data = landcover_dataset["landcover"]

        if isinstance(landcover_data, xr.Dataset):
            if "landcover" in landcover_data.data_vars:
                landcover_data = landcover_data["landcover"]
            else:
                landcover_data = landcover_data[
                    list(landcover_data.data_vars.keys())[0]
                ]

        selection_path = output_dir / f"{base_name}_selection.png"
        preview_path = (
            output_dir / f"{base_name}_preview.png" if create_preview else None
        )

        self.landcover_to_selection_texture(landcover_data, selection_path)

        if create_preview:
            self.create_preview_texture(landcover_data, preview_path)

        analysis = self.analyze_landcover_classes(landcover_data)
        logging.info(f"Land cover analysis: {analysis['unique_classes']} classes found")

        return selection_path, preview_path

    def generate_buffer_mask(
        self, mask_size: int, target_size: int, output_path: Path
    ) -> Path:
        """
        Generates a square buffer mask texture with center hole for target area.

        Args:
            mask_size: Total size of the mask in pixels (buffer area)
            target_size: Size of the center hole in pixels (target area)
            output_path: Path where the mask will be saved

        Returns:
            Path to the generated mask file
        """
        logging.info(
            f"Generating buffer mask texture {mask_size}x{mask_size} with {target_size}x{target_size} center hole"
        )

        mask = np.ones((mask_size, mask_size), dtype=np.uint8) * 255

        center = mask_size // 2
        half_target = target_size // 2

        start_y = center - half_target
        end_y = center + half_target
        start_x = center - half_target
        end_x = center + half_target

        mask[start_y:end_y, start_x:end_x] = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(mask, mode="L")
        image.save(output_path)
        logging.info(f"Buffer mask texture saved to {output_path}")

        return output_path
