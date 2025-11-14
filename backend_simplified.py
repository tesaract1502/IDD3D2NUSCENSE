# legacy_backend.py
# ----------------------
# LEGACY/DEPRECATED: Old granular converter classes
#
# This file preserves the original conversion algorithms from
# extensible_converter_backend.py for backward compatibility.
#
# ⚠️ WARNING: This approach is DEPRECATED!
# ⚠️ Please use the new Reader/Writer architecture instead:
#    - readers.py (for reading data)
#    - writers.py (for writing data)
#    - convert_cli.py (for CLI usage)
#
# This file is kept ONLY for:
# 1. Reference/comparison with new architecture
# 2. Legacy code that depends on granular converters
# 3. Migration validation (compare outputs)
#
# DO NOT USE FOR NEW PROJECTS!
# ----------------------

"""
MIGRATION GUIDE:
===============

Old Code (This File):
---------------------
from legacy_backend import IDD3DDataLoader, IDD3DLidarConverter
loader = IDD3DDataLoader(root, sequence)
converter = IDD3DLidarConverter()
converter.run(loader, log_handler)

New Code (Recommended):
-----------------------
from readers import Idd3dReader
from writers import NuScenesWriter

reader = Idd3dReader()
data = reader.read('/path/to/sequence')

writer = NuScenesWriter()
writer.write(data, '/path/to/output')
"""

import os
import json
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


# ============================================================================
# LEGACY CONVERTER FRAMEWORK
# ============================================================================

class LegacyLogHandler:
    """
    Legacy log handler for backward compatibility.
    In the new architecture, use standard Python logging instead.
    """
    def __init__(self):
        pass
    
    def log(self, message, log_type='info'):
        if log_type == 'error':
            log.error(message)
        elif log_type == 'warning':
            log.warning(message)
        elif log_type == 'success':
            log.info(f"✓ {message}")
        else:
            log.info(message)


class LegacyBaseConverter(ABC):
    """
    Legacy base converter class.
    
    DEPRECATED: Use readers.BaseReader or writers.BaseWriter instead.
    """
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run(self, data_loader, log_handler):
        pass


class LegacyIDD3DDataLoader:
    """
    Legacy data loader for IDD3D dataset.
    
    DEPRECATED: Use readers.Idd3dReader instead.
    
    This class only exists for backward compatibility.
    The new Idd3dReader does the same thing but returns
    an IntermediateData object instead.
    """
    def __init__(self, root: str, sequence: str = '20220118103308_seq_10'):
        log.warning("⚠️ LegacyIDD3DDataLoader is DEPRECATED! Use readers.Idd3dReader instead.")
        self.root = os.path.abspath(root)
        self.sequence = sequence
        self.seq_base = os.path.join(
            self.root,
            'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val',
            sequence
        )
        self.lidar_dir = os.path.join(self.seq_base, 'lidar')
        self.label_dir = os.path.join(self.seq_base, 'label')
        self.calib_dir = os.path.join(self.seq_base, 'calib')
        self.annot_json = os.path.join(self.seq_base, 'annot_data.json')
        
        self.out_data = os.path.join(self.root, 'Intermediate_format/data')
        self.annot_out = os.path.join(self.root, 'Intermediate_format/anotations')
        self.converted_lidar = os.path.join(self.out_data, 'converted_lidar')
    
    def validate(self) -> dict:
        """Validates the dataset structure."""
        if not os.path.exists(self.seq_base):
            return {'valid': False, 'error': f'Sequence path not found: {self.seq_base}'}
        
        required_dirs = ['lidar', 'label', 'calib']
        missing = []
        for dir_name in required_dirs:
            if not os.path.exists(os.path.join(self.seq_base, dir_name)):
                missing.append(dir_name)
        
        if missing:
            return {'valid': False, 'error': f'Missing directories: {", ".join(missing)}'}
        
        return {'valid': True, 'path': self.seq_base}
    
    def read_annotations(self):
        """Reads annot_data.json file."""
        if not os.path.exists(self.annot_json):
            return {}
        with open(self.annot_json, 'r') as f:
            return json.load(f)


# ============================================================================
# LEGACY NOTE ABOUT MISSING CONVERTERS
# ============================================================================

"""
The following converter classes were in the original extensible_converter_backend.py:

1. IDD3DLidarConverter - Now in writers.py → _process_sensor_files() + convert_lidar_pcd_to_bin()
2. IDD3DCameraConverter - Now in writers.py → _process_sensor_files() + convert_camera_to_jpg()
3. IDD3DCalibConverter - Now split between readers.py and writers.py
4. IDD3DAnnotationConverter - Now in readers.py → read() method
5. IDD3DCategoryConverter - Now in writers.py → _write_category()
6. IDD3DSampleConverter - Now in writers.py → _write_sample_and_ego_pose()
7. IDD3DSampleAnnotationConverter - Now in writers.py → _write_instance_and_annotation()
8. IDD3DInstanceConverter - Now in writers.py → _write_instance_and_annotation()
9. IDD3DObjectsJsonConverter - ⚠️ NOT MIGRATED (may need to add to writers.py)
10. IDD3DTimestampSyncConverter - Now handled automatically by _NuScenesTokenManager
11. IDD3DLogConverter - Now in writers.py → _write_log()
12. IDD3DEgoPoseConverter - Now split between readers.py and writers.py
13. IDD3DMapConverter - Now in writers.py → _write_map()
14. IDD3DSceneConverter - Now in writers.py → _write_sample_and_ego_pose()
15. IDD3DFileManifestConverter - Now in writers.py → _write_file_manifest()

These classes are NOT re-implemented here because:
- They are already available in the new architecture
- Re-implementing would create duplicate, unmaintained code
- The new architecture does the same thing more efficiently

If you ABSOLUTELY need the old granular converter classes, they are available
in the original extensible_converter_backend.py file (archived).
"""


# ============================================================================
# OBJECTS.JSON CONVERTER (The Only Missing Piece)
# ============================================================================

class IDD3DObjectsJsonConverter(LegacyBaseConverter):
    """
    Generates objects.json with bbox_3d format.
    
    NOTE: This is the ONLY converter not fully migrated to the new architecture.
    
    objects.json is similar to sample_annotation.json but with a different structure.
    It may be needed for compatibility with certain visualization tools.
    
    If you need this, consider adding it to writers.py as _write_objects_json()
    """
    
    def __init__(self):
        super().__init__("objects_json")
        log.warning("⚠️ IDD3DObjectsJsonConverter is LEGACY! Consider adding to NuScenesWriter.")
    
    def run(self, data_loader, log_handler):
        """
        Generates objects.json file.
        
        This creates a flattened list of all 3D bounding boxes with the structure:
        {
            "object_token": "...",
            "frame_token": "...",
            "instance_token": "...",
            "category_token": "...",
            "bbox_3d": {
                "center": [x, y, z],
                "size": [w, l, h],
                "rotation": [qw, qx, qy, qz]
            },
            "num_lidar_pts": 0
        }
        """
        log_handler.log("⚠️ Using legacy objects.json converter", 'warning')
        log_handler.log("Consider migrating this to writers.py → _write_objects_json()", 'warning')
        
        # This implementation is intentionally left as a stub
        # If you need it, copy from the original extensible_converter_backend.py
        # and adapt to use the new token manager
        
        log_handler.log("objects.json generation is not implemented in legacy mode", 'error')
        log_handler.log("Please use the new architecture or implement in writers.py", 'error')


# ============================================================================
# USAGE EXAMPLE (DEPRECATED)
# ============================================================================

def legacy_conversion_example():
    """
    Example of how the old backend worked.
    
    DO NOT USE THIS APPROACH FOR NEW CODE!
    Use convert_cli.py or the Reader/Writer classes instead.
    """
    log.warning("=" * 70)
    log.warning("RUNNING LEGACY CONVERSION (DEPRECATED)")
    log.warning("=" * 70)
    
    # Old way (granular converters)
    root = "/path/to/idd3d/data"
    sequence = "20220118103308_seq_10"
    
    loader = LegacyIDD3DDataLoader(root, sequence)
    validation = loader.validate()
    
    if not validation['valid']:
        log.error(f"Validation failed: {validation['error']}")
        return
    
    log_handler = LegacyLogHandler()
    
    # Run converters one by one
    # IDD3DLidarConverter().run(loader, log_handler)
    # IDD3DCameraConverter().run(loader, log_handler)
    # ... etc
    
    log.warning("=" * 70)
    log.warning("PLEASE MIGRATE TO NEW ARCHITECTURE!")
    log.warning("=" * 70)


# ============================================================================
# DEPRECATION NOTICE
# ============================================================================

if __name__ == '__main__':
    print("""
