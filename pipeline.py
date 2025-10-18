"""
Conversion Pipeline

Orchestrates dataset conversions through the intermediate format.
"""
import logging
from pathlib import Path
from typing import Type, Dict
import sys
sys.path.insert(0, '/tmp')

from base_dataset import BaseDataset
from idd3d import IDD3DDataset
from nuscenes import NuScenesDataset

logging.basicConfig(level=logging.INFO)


class DatasetRegistry:
    """Registry of available dataset implementations"""
    
    _datasets: Dict[str, Type[BaseDataset]] = {
        "IDD3D": IDD3DDataset,
        "nuScenes": NuScenesDataset,
        # Easy to add more: "KITTI": KITTIDataset, etc.
    }
    
    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseDataset]):
        """Register a new dataset implementation"""
        cls._datasets[name] = dataset_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseDataset]:
        """Get dataset class by name"""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._datasets.keys())}")
        return cls._datasets[name]
    
    @classmethod
    def list_datasets(cls):
        """List all available datasets"""
        return list(cls._datasets.keys())


class ConversionPipeline:
    """
    Main conversion pipeline
    
    Converts any source dataset to any target dataset through intermediate format.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ConversionPipeline")
    
    def convert(
        self,
        source_type: str,
        target_type: str,
        source_path: str,
        target_path: str,
        source_config: dict = None,
        target_config: dict = None
    ):
        """
        Convert from source dataset to target dataset
        
        Args:
            source_type: Source dataset type (e.g., "IDD3D")
            target_type: Target dataset type (e.g., "nuScenes")
            source_path: Path to source dataset
            target_path: Path to save target dataset
            source_config: Additional config for source (e.g., {"sequence_name": "seq_10"})
            target_config: Additional config for target (e.g., {"version": "v1.0-mini"})
        """
        self.logger.info(f"Starting conversion: {source_type} → {target_type}")
        
        source_config = source_config or {}
        target_config = target_config or {}
        
        # Get dataset classes
        SourceClass = DatasetRegistry.get(source_type)
        TargetClass = DatasetRegistry.get(target_type)
        
        # Step 1: Load source dataset
        self.logger.info(f"Step 1: Loading {source_type} from {source_path}")
        source_dataset = SourceClass(source_path, **source_config)
        source_dataset.load()
        
        # Step 2: Convert to intermediate format
        self.logger.info(f"Step 2: Converting {source_type} to intermediate format")
        intermediate = source_dataset.to_intermediate()
        
        # Step 3: Convert from intermediate to target format
        self.logger.info(f"Step 3: Converting intermediate to {target_type}")
        target_dataset = TargetClass(target_path, **target_config)
        target_dataset.from_intermediate(intermediate)
        
        # Step 4: Save target dataset
        self.logger.info(f"Step 4: Saving {target_type} to {target_path}")
        target_dataset.save(target_path)
        
        self.logger.info("✓ Conversion complete!")
        
        return {
            "source": source_type,
            "target": target_type,
            "stats": target_dataset.get_stats()
        }
    
    def list_available_datasets(self):
        """List all registered datasets"""
        return DatasetRegistry.list_datasets()
