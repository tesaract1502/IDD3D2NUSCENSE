"""
Backend API Interface

Simple interface for backend systems to trigger conversions.
"""
import sys
sys.path.insert(0, '/tmp')
from pipeline import ConversionPipeline, DatasetRegistry


class DatasetConverterAPI:
    """
    Simple API for dataset conversion
    
    Usage:
        api = DatasetConverterAPI()
        
        # List available datasets
        datasets = api.get_available_datasets()
        
        # Convert IDD3D to nuScenes
        result = api.convert_dataset(
            source="IDD3D",
            target="nuScenes",
            source_path="/path/to/idd3d",
            target_path="/path/to/output",
            source_params={"sequence_name": "20220118103308_seq_10"},
            target_params={"version": "v1.0-mini"}
        )
    """
    
    def __init__(self):
        self.pipeline = ConversionPipeline()
    
    def get_available_datasets(self):
        """
        Get list of supported datasets
        
        Returns:
            List of dataset names
        """
        return self.pipeline.list_available_datasets()
    
    def convert_dataset(
        self,
        source: str,
        target: str,
        source_path: str,
        target_path: str,
        source_params: dict = None,
        target_params: dict = None
    ):
        """
        Convert dataset from source to target format
        
        Args:
            source: Source dataset name (e.g., "IDD3D")
            target: Target dataset name (e.g., "nuScenes")
            source_path: Path to source dataset
            target_path: Path to save converted dataset
            source_params: Source-specific parameters (e.g., sequence_name)
            target_params: Target-specific parameters (e.g., version)
        
        Returns:
            Dictionary with conversion results and stats
        """
        try:
            result = self.pipeline.convert(
                source_type=source,
                target_type=target,
                source_path=source_path,
                target_path=target_path,
                source_config=source_params,
                target_config=target_params
            )
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Backend usage example
if __name__ == "__main__":
    api = DatasetConverterAPI()
    
    # List available datasets
    print("Available datasets:", api.get_available_datasets())
    
    # Convert IDD3D to nuScenes
    result = api.convert_dataset(
        source="IDD3D",
        target="nuScenes",
        source_path="/home/siddharthb9/Desktop/nuSceneses&IDD3D",
        target_path="/home/siddharthb9/Desktop/output_nuscenes",
        source_params={"sequence_name": "20220118103308_seq_10"},
        target_params={"version": "v1.0-mini"}
    )
    
    print("Conversion result:", result)
