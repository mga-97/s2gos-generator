"""WRB soil classification processor for s2gos scene generation."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
from shapely.geometry import Polygon

from .base_processor import BaseTileProcessor


# WRB soil class mapping for supported soil types
# Maps actual WRB class values from Gobabeb data to soil material indices
WRB_SOIL_MAPPING = {
    0: 0,    # Other/unknown soil -> 0 (fallback to baresoil)
    # Actual soil classes from Gobabeb WRB data
    4: 1,    # Arenosols (WRB class 4) -> arenosols material index  
    24: 2,   # Regosols (WRB class 24) -> regosols material index
    16: 3,   # Leptosols (WRB class 16) -> leptosols material index
    5: 4,    # Calcisols (WRB class 5) -> calcisols material index
    25: 5,   # Solonchaks (WRB class 25) -> solonchaks material index
}

# Default mapping for unsupported WRB classes to material indices
DEFAULT_WRB_TO_MATERIAL = {
    # Map all other WRB classes to 0 (baresoil fallback)
    i: 0 for i in range(30) if i not in WRB_SOIL_MAPPING
}
DEFAULT_WRB_TO_MATERIAL.update(WRB_SOIL_MAPPING)


class WRBSoilProcessor(BaseTileProcessor):
    """Processes WRB soil classification data for soil-aware texture generation."""

    def __init__(self, wrb_data_path: Union[Path, str]):
        """Initialize the WRB soil processor.
        
        Args:
            wrb_data_path: Path to WRB soil classification VRT file
        """
        self.wrb_data_path = Path(wrb_data_path)
        
        if not self.wrb_data_path.exists():
            raise FileNotFoundError(f"WRB data file not found: {wrb_data_path}")
            
        # Initialize without spatial index since we use a single VRT file
        self.data_description = "WRB soil classification"
        logging.info(f"WRBSoilProcessor initialized with data: {self.wrb_data_path}")

    @property
    def path_column(self) -> str:
        """Not used for single VRT file approach."""
        return "path"

    @property
    def data_variable_name(self) -> str:
        """Name of the data variable in the processed dataset."""
        return "wrb_soil_class"

    @property
    def default_interpolation_method(self) -> str:
        """Nearest neighbor for categorical soil data."""
        return "nearest"

    @property
    def data_type(self) -> Optional[str]:
        """Data type for soil classification."""
        return "uint8"

    @property
    def default_fill_value(self) -> Union[float, int]:
        """Default fill value for unknown soil classes."""
        return 0

    @property
    def use_context_manager(self) -> bool:
        """Use direct assignment for VRT files."""
        return False

    def _process_wrb_classes(self, data_array: xr.DataArray) -> xr.DataArray:
        """Map WRB classes to material indices for supported soil types.
        
        Args:
            data_array: Raw WRB classification data
            
        Returns:
            Processed data array with material indices
        """
        logging.info("Mapping WRB classes to material indices...")
        
        # Create output array initialized with fallback (baresoil = 0)
        processed = xr.zeros_like(data_array, dtype="uint8")
        
        # Map supported soil classes
        for wrb_class, material_idx in WRB_SOIL_MAPPING.items():
            if wrb_class != 0:  # Skip the default fallback
                mask = data_array == wrb_class
                processed = processed.where(~mask, material_idx)
                
        logging.info(f"WRB class mapping complete")
        return processed

    def generate_wrb_soil_data(
        self,
        aoi_polygon: Polygon,
        output_path: Path,
        target_resolution_m: float,
        center_lat: float,
        center_lon: float,
        aoi_size_km: float,
        fillna_value: Optional[float] = None,
    ) -> Path:
        """Generate WRB soil classification data for the AOI.

        Args:
            aoi_polygon: Area of interest polygon
            output_path: Path where the processed data will be saved
            target_resolution_m: Target resolution in meters
            center_lat: Center latitude for projection
            center_lon: Center longitude for projection  
            aoi_size_km: AOI size in kilometers
            fillna_value: Fill value for NaN values

        Returns:
            Path to the generated WRB soil data file
        """
        logging.info("=== Processing WRB Soil Classification Data ===")
        
        try:
            # Open the VRT file and extract data for AOI
            import rioxarray as rxr
            
            logging.info(f"Opening WRB data: {self.wrb_data_path}")
            wrb_da = rxr.open_rasterio(self.wrb_data_path, chunks={"x": 1024, "y": 1024})
            
            # Get AOI bounds and clip data
            bounds = aoi_polygon.bounds
            logging.info(f"Clipping to AOI bounds: {bounds}")
            
            # Clip to AOI with some buffer for reprojection
            buffer_deg = 0.1  # Add small buffer
            clipped = wrb_da.sel(
                x=slice(bounds[0] - buffer_deg, bounds[2] + buffer_deg),
                y=slice(bounds[3] + buffer_deg, bounds[1] - buffer_deg)
            ).isel(band=0, drop=True)
            
            # Rename coordinates and data variable  
            clipped = clipped.rename({"x": "lon", "y": "lat"}).rename(self.data_variable_name)
            
            # Process WRB classes to material indices
            processed_data = self._process_wrb_classes(clipped)
            
            # Create dataset
            dataset = xr.Dataset({self.data_variable_name: processed_data})
            
            # Apply fill values
            fill_value = fillna_value if fillna_value is not None else self.default_fill_value
            if fill_value is not None:
                dataset = dataset.fillna(fill_value)
            
            # Regrid to target projection and resolution
            logging.info("Regridding WRB data to target projection...")
            regridded_dataset = self._regrid_data(
                dataset=dataset,
                target_resolution_m=target_resolution_m,
                center_lat=center_lat,
                center_lon=center_lon,
                aoi_size_km=aoi_size_km,
                fillna_value=fill_value,
            )
            
            # Save the processed dataset
            self._save_dataset(regridded_dataset, output_path)
            
            logging.info(f"WRB soil data processing complete: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"WRB soil processing failed: {e}")
            raise

    def get_material_mapping_info(self) -> dict:
        """Get information about WRB to material mapping.
        
        Returns:
            Dictionary with mapping information
        """
        return {
            "supported_soil_types": list(WRB_SOIL_MAPPING.keys()),
            "material_indices": list(WRB_SOIL_MAPPING.values()),
            "fallback_material": 0,
            "description": "WRB soil class to material index mapping"
        }