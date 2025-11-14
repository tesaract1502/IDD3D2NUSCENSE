import os
import json
import shutil
import logging
import uuid
import re
import hashlib
from abc import ABC, abstractmethod
from PIL import Image
from datetime import datetime
from intermediate_format_enhanced import IntermediateData
from utils import TokenTimestampManager, append_to_json_list, json_file_lock, merge_and_overwrite_json_list

try:
    import numpy as np
    import pyarrow.feather as pf
    import pandas as pd
except ImportError:
    print("WARNING: numpy, pyarrow or pandas not found. Argoverse LiDAR conversion will fail.")
    print("Please run: pip install numpy pyarrow pandas")

try:
    import open3d as o3d
except ImportError:
    print("WARNING: open3d not found. IDD3D .pcd conversion will fail.")
    print("Please run: pip install open3d")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def convert_lidar_file(src_path, dst_path):
    try:
        if not os.path.exists(src_path):
            log.warning(f"Source LiDAR file not found: {src_path}")
            open(dst_path, 'wb').close()
            return
        pcd = o3d.io.read_point_cloud(src_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts = np.hstack((xyz, intensity))
        pts.astype(np.float32).tofile(dst_path)
    except Exception as e:
        log.error(f"Error converting {src_path} (is open3d installed?): {e}. Creating empty file.")
        open(dst_path, 'wb').close()

def convert_camera_file(src_path, dst_path):
    try:
        if not os.path.exists(src_path):
            log.warning(f"Source camera file not found: {src_path}")
            return
        img = Image.open(src_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dst_path, 'JPEG', quality=95)
    except ImportError:
        log.warning(f"PIL/Pillow not available. Skipping camera conversion for {src_path}")
    except Exception as e:
        log.error(f"Error converting {src_path} to {dst_path}: {e}")

def convert_feather_to_pcd_bin(src_path, dst_path):
    try:
        if not os.path.exists(src_path):
            log.warning(f"Source LiDAR file not found: {src_path}")
            open(dst_path, 'wb').close()
            return
        table = pf.read_feather(src_path)
        df = table.to_pandas()
        pts = df[['x', 'y', 'z', 'intensity']].values.astype(np.float32)
        pts.astype(np.float32).tofile(dst_path)
    except Exception as e:
        log.error(f"Error converting {src_path} (is pyarrow/pandas installed?): {e}. Creating empty file.")
        open(dst_path, 'wb').close()

class BaseWriter(ABC):
    @abstractmethod
    def write(self, data: IntermediateData, output_path: str):
        pass

class NuScenesWriter(BaseWriter):
    def __init__(self):
        self.token_manager = None
        self.output_path = None
        self.annot_out_dir = None
        self.samples_out_dir = None
        self.sweeps_out_dir = None
        self.maps_out_dir = None
        self.map_expansion_dir = None
        self.map_expansion_basemap_dir = None
        self.map_expansion_expansion_dir = None
        self.map_expansion_prediction_dir = None
        self.generated_log_tokens = []
    
    def write(self, data: IntermediateData, output_path: str):
        log.info(f"Initializing NuScenesWriter for output to: {output_path}")
        
        self.output_path = os.path.abspath(output_path)
        self.annot_out_dir = os.path.join(self.output_path, 'annotations')
        self.samples_out_dir = os.path.join(self.output_path, 'samples')
        self.sweeps_out_dir = os.path.join(self.output_path, 'sweeps')
        self.maps_out_dir = os.path.join(self.output_path, 'maps')
        os.makedirs(self.annot_out_dir, exist_ok=True)
        os.makedirs(self.samples_out_dir, exist_ok=True)
        os.makedirs(self.maps_out_dir, exist_ok=True)

        self.map_expansion_dir = os.path.join(self.output_path, 'idd3d_map_expansion')
        self.map_expansion_basemap_dir = os.path.join(self.map_expansion_dir, 'basemap')
        self.map_expansion_expansion_dir = os.path.join(self.map_expansion_dir, 'expansion')
        self.map_expansion_prediction_dir = os.path.join(self.map_expansion_dir, 'prediction')
        os.makedirs(self.map_expansion_basemap_dir, exist_ok=True)
        os.makedirs(self.map_expansion_expansion_dir, exist_ok=True)
        os.makedirs(self.map_expansion_prediction_dir, exist_ok=True)

        registry_path = os.path.join(self.annot_out_dir, 'token_registry.json')
        self.token_manager = TokenTimestampManager(registry_path=registry_path)
        self._pre_populate_categories()

        if not data.scenes:
            log.error("No scenes found in intermediate data. Cannot proceed.")
            return

        sequence_name = data.scenes[0].name
        log.info(f"Processing sequence: {sequence_name}")

        self._write_sensor_and_calib(data.calibrations)
        self._write_visibility()
        self._write_attribute()
        self._write_log(data.scenes)
        self._write_map()
        self._write_map_expansion()
        self._write_prediction(data.scenes, data.samples)
        self._write_file_manifest(data)
        self._write_sample_and_ego_pose(data.samples, data.ego_poses)
        self._write_sample_data(data.sensor_data, sequence_name)
        self._write_category(data.instances)
        self._write_instance_and_annotation(data.instances, data.annotations)

        log.info("Converting and copying physical sensor files...")
        self._process_sensor_files(data.sensor_data, data.sequence_path, sequence_name)

        log.info("Duplicating 'samples' directory to 'sweeps'...")
        self._duplicate_sweeps()

        log.info("Saving global token registry...")
        self.token_manager.save_registry(registry_path)

        log.info(f"--- NuScenes Write Complete ---")
        log.info(f"Output successfully written to: {self.output_path}")

    # ... Additional private methods _write_sensor_and_calib, _write_visibility, _write_attribute etc.
    # ... Also _process_sensor_files, _duplicate_sweeps, _pre_populate_categories, _write_log, _write_map_expansion, etc.

# Implement all helper methods mentioned above accordingly, following the pattern
# from the original code but replacing error swallowings with logs + exceptions,
# adding stub logging, and using token_manager consistently.

