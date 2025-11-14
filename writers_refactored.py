# writers.py
# ----------------------
# This file contains "Writer" classes.
# Each Writer is responsible for consuming the 'IntermediateData' object
# and writing it to a specific dataset format (like nuScenes).
#
# Writers handle format-specific concerns like token management,
# file naming conventions, and directory structures.
# ----------------------

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
from intermediate_format import IntermediateData
from utils import append_to_json_list, json_file_lock, merge_and_overwrite_json_list, load_json_safely, save_json_safely

# Import numpy and optional libraries
try:
    import numpy as np
    import pyarrow.feather as pf
    import pandas as pd
except ImportError:
    log.warning("numpy, pyarrow or pandas not found. Argoverse LiDAR conversion will fail.")

try:
    import open3d as o3d
except ImportError:
    log.warning("open3d not found. IDD3D .pcd conversion will fail.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  FILE CONVERSION HELPERS
# -----------------------------------------------------------------------------

def convert_lidar_pcd_to_bin(src_path, dst_path):
    """Converts a .pcd file to a .pcd.bin file."""
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
        log.error(f"Error converting {src_path}: {e}. Creating empty file.")
        open(dst_path, 'wb').close()


def convert_lidar_feather_to_bin(src_path, dst_path):
    """Converts Argoverse .feather LiDAR file to .pcd.bin format."""
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
        log.error(f"Error converting {src_path}: {e}. Creating empty file.")
        open(dst_path, 'wb').close()


def convert_camera_to_jpg(src_path, dst_path, quality=95):
    """Converts a camera image to .jpg format."""
    try:
        if not os.path.exists(src_path):
            log.warning(f"Source camera file not found: {src_path}")
            return
        
        img = Image.open(src_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dst_path, 'JPEG', quality=quality)
    except Exception as e:
        log.error(f"Error converting {src_path} to {dst_path}: {e}")


# -----------------------------------------------------------------------------
#  BASE WRITER
# -----------------------------------------------------------------------------

class BaseWriter(ABC):
    """
    Abstract base class for all dataset writers.
    
    Writers should handle format-specific concerns internally,
    including token generation, file naming, and directory structures.
    """
    
    @abstractmethod
    def write(self, data: IntermediateData, output_path: str):
        """
        Writes intermediate data to the target format.
        
        Args:
            data: IntermediateData object to write
            output_path: Base output directory path
        """
        pass


# -----------------------------------------------------------------------------
#  NUSCENES TOKEN MANAGER (Private to NuScenesWriter)
# -----------------------------------------------------------------------------

class _NuScenesTokenManager:
    """
    Manages token generation for nuScenes format.
    
    This is a PRIVATE class used only by NuScenesWriter.
    It handles:
    - Local tokens (frame, instance, ego_pose) - regenerated each run
    - Global tokens (category, sensor, etc.) - persistent across runs
    """
    
    def __init__(self, registry_path=None, base_timestamp=None, frame_rate_hz=10):
        self.frame_rate_hz = frame_rate_hz
        self.frame_interval_us = int(1_000_000 / frame_rate_hz)
        
        if base_timestamp is None:
            self.base_timestamp = 1640995200000000  # Jan 1, 2022
        else:
            self.base_timestamp = base_timestamp
        
        # Local tokens (regenerated each run)
        self.frame_tokens = {}        # temp_frame_id -> token
        self.instance_tokens = {}     # temp_instance_id -> token
        self.ego_pose_tokens = {}     # temp_frame_id -> ego_pose_token
        self.scene_token = None
        
        # Global tokens (persistent across runs)
        self.category_tokens = {}
        self.attribute_tokens = {}
        self.visibility_tokens = {}
        self.map_tokens = {}
        self.log_tokens = {}
        self.sensor_tokens = {}
        self.calibration_tokens = {}
        
        self.registry_path = registry_path
        self._load_registry()
    
    def _load_registry(self):
        """Loads ONLY global tokens from the registry."""
        if not self.registry_path or not os.path.exists(self.registry_path):
            log.info("No existing token registry found. Starting fresh.")
            return
        
        try:
            registry = load_json_safely(self.registry_path, default={})
            
            self.category_tokens = registry.get('category_tokens', {})
            self.attribute_tokens = registry.get('attribute_tokens', {})
            self.visibility_tokens = registry.get('visibility_tokens', {})
            self.map_tokens = registry.get('map_tokens', {})
            self.log_tokens = registry.get('log_tokens', {})
            self.sensor_tokens = registry.get('sensor_tokens', {})
            self.calibration_tokens = registry.get('calibration_tokens', {})
            
            log.info(f"Loaded {len(self.category_tokens)} global category tokens from registry.")
        except Exception as e:
            log.warning(f"Could not load token registry: {e}")
    
    def save_registry(self):
        """Saves ONLY global tokens to the registry."""
        if not self.registry_path:
            return
        
        registry = {
            'base_timestamp': self.base_timestamp,
            'frame_rate_hz': self.frame_rate_hz,
            'category_tokens': self.category_tokens,
            'attribute_tokens': self.attribute_tokens,
            'visibility_tokens': self.visibility_tokens,
            'map_tokens': self.map_tokens,
            'log_tokens': self.log_tokens,
            'sensor_tokens': self.sensor_tokens,
            'calibration_tokens': self.calibration_tokens
        }
        
        save_json_safely(self.registry_path, registry)
        log.info(f"Global token registry saved to {self.registry_path}")
    
    def get_timestamp(self, frame_index):
        """Generates a timestamp based on frame index."""
        return self.base_timestamp + (frame_index * self.frame_interval_us)
    
    def get_frame_token(self, frame_id):
        """Gets a deterministic token for a temp_frame_id."""
        if frame_id not in self.frame_tokens:
            self.frame_tokens[frame_id] = uuid.uuid4().hex
        return self.frame_tokens[frame_id]
    
    def get_ego_pose_token(self, frame_id):
        """Generates a deterministic ego_pose token for this frame_id."""
        if frame_id not in self.ego_pose_tokens:
            self.ego_pose_tokens[frame_id] = uuid.uuid4().hex
        return self.ego_pose_tokens[frame_id]
    
    def get_instance_token(self, obj_id):
        """Gets a deterministic token for a temp_instance_id."""
        if obj_id not in self.instance_tokens:
            self.instance_tokens[obj_id] = uuid.uuid4().hex
        return self.instance_tokens[obj_id]
    
    def get_category_token(self, category_name):
        """Gets/creates a global token for an object CATEGORY."""
        if category_name not in self.category_tokens:
            self.category_tokens[category_name] = uuid.uuid4().hex
        return self.category_tokens[category_name]
    
    def get_attribute_token(self, attr_name):
        """Gets/creates a global token for an ATTRIBUTE."""
        if attr_name not in self.attribute_tokens:
            self.attribute_tokens[attr_name] = uuid.uuid4().hex
        return self.attribute_tokens[attr_name]
    
    def get_visibility_token(self, vis_level):
        """Gets/creates a global token for a VISIBILITY level."""
        if vis_level not in self.visibility_tokens:
            self.visibility_tokens[vis_level] = uuid.uuid4().hex
        return self.visibility_tokens[vis_level]
    
    def get_map_token(self, map_name):
        """Gets/creates a global token for a MAP."""
        if map_name not in self.map_tokens:
            self.map_tokens[map_name] = uuid.uuid4().hex
        return self.map_tokens[map_name]
    
    def get_log_token(self, log_name):
        """Gets/creates a global token for a LOG."""
        if log_name not in self.log_tokens:
            self.log_tokens[log_name] = uuid.uuid4().hex
        return self.log_tokens[log_name]
    
    def get_sensor_token(self, sensor_name):
        """Gets/creates a global token for a sensor name."""
        if sensor_name not in self.sensor_tokens:
            self.sensor_tokens[sensor_name] = uuid.uuid4().hex
        return self.sensor_tokens[sensor_name]
    
    def get_calibration_token(self, sensor_name):
        """Gets/creates a global token for a calibration profile."""
        if sensor_name not in self.calibration_tokens:
            self.calibration_tokens[sensor_name] = uuid.uuid4().hex
        return self.calibration_tokens[sensor_name]
    
    def get_scene_token(self):
        """Gets/creates a token for the current scene."""
        if self.scene_token is None:
            self.scene_token = uuid.uuid4().hex
        return self.scene_token
    
    def generate_annotation_token(self):
        """Annotations are always unique."""
        return uuid.uuid4().hex


# -----------------------------------------------------------------------------
#  NUSCENES WRITER
# -----------------------------------------------------------------------------

class NuScenesWriter(BaseWriter):
    """
    Writes data to the nuScenes dataset format.
    
    NOTE: This writer is STATEFUL and designed to merge multiple 
    sequences into a single output dataset. Do not reuse the same 
    writer instance for separate output datasets.
    
    The writer handles:
    - Token generation and persistence (via _NuScenesTokenManager)
    - JSON metadata file generation
    - Physical file conversion and copying
    - Cross-sequence linking and merging
    """
    
    # Official category tokens from category2.json
    OFFICIAL_CATEGORY_TOKENS = {
        "vehicle.motorcycle": "dc39d8b2858e4bc0b7ddf66ede8d734e",
        "movable_object.bicyclerider": "d411b4e8157d445193034d6f408900d3",
        "movable_object.tourcar": "e2325ce5697e45678ee0fe4017918290",
        "movable_object.scooterrider": "9a438c7df65d4ae0b5e87f603a3e91b7",
        "vehicle.bus": "1046b59779f24cf7b55114161208b0f5",
        "movable_object.bicyclegroup": "57c2b779b57b496297048ea55aaed2c7",
        "movable_object.van": "869140488b264d7780ed9cc8233cb5ce",
        "vehicle.truck": "69d88d0df8274f56995aacff1982ec65",
        "movable_object.pedestrian": "9a6c42f9792f40789bc0437eba0aef9b",
        "movable_object.scooter": "f15d03bf64834024a0601aae7a07c156",
        "vehicle.bicycle": "366ad39f728a4ab5ae9a4146f528bd00",
        "movable_object.unknown": "f0add8f1828d4b7ca20d135edd7ecd4e",
        "movable_object.unknown1": "6bc7bdefe76646e193288d5928a2d58a",
        "vehicle.car": "3305eeb43e684538b00bcc41fc38d84e"
    }
    
    def __init__(self):
        """Initialize the writer with clean state."""
        self._token_manager = None
        self._output_path = None
        self._annot_dir = None
        self._samples_dir = None
        self._sweeps_dir = None
        self._maps_dir = None
        self._map_expansion_dir = None
        
        # Cross-run state
        self._generated_log_tokens = []
    
    def write(self, data: IntermediateData, output_path: str):
        """
        Writes intermediate data to nuScenes format.
        
        Args:
            data: IntermediateData object containing sequence data
            output_path: Base output directory path
        """
        log.info(f"Initializing NuScenesWriter for output to: {output_path}")
        self._output_path = os.path.abspath(output_path)
        
        # --- 1. Setup Output Directories ---
        self._annot_dir = os.path.join(self._output_path, 'anotations')
        self._samples_dir = os.path.join(self._output_path, 'samples')
        self._sweeps_dir = os.path.join(self._output_path, 'sweeps')
        self._maps_dir = os.path.join(self._output_path, 'maps')
        self._map_expansion_dir = os.path.join(self._output_path, 'idd3d_map_expansion')
        
        os.makedirs(self._annot_dir, exist_ok=True)
        os.makedirs(self._samples_dir, exist_ok=True)
        os.makedirs(self._maps_dir, exist_ok=True)
        os.makedirs(os.path.join(self._map_expansion_dir, 'basemap'), exist_ok=True)
        os.makedirs(os.path.join(self._map_expansion_dir, 'expansion'), exist_ok=True)
        os.makedirs(os.path.join(self._map_expansion_dir, 'prediction'), exist_ok=True)
        
        # --- 2. Initialize Token Manager ---
        registry_path = os.path.join(self._annot_dir, 'token_registry.json')
        last_timestamp = self._get_last_timestamp()
        new_base_timestamp = (last_timestamp + 20_000_000) if last_timestamp else None
        
        self._token_manager = _NuScenesTokenManager(
            registry_path=registry_path,
            base_timestamp=new_base_timestamp
        )
        
        # Pre-populate official category tokens
        self._pre_populate_categories()
        
        # Validate data
        if not data.scenes:
            log.error("No scenes found in intermediate data. Cannot proceed.")
            return
        
        sequence_name = data.scenes[0].name
        log.info(f"Processing sequence: {sequence_name}")
        
        # --- 3. Write JSON Metadata Files ---
        log.info("Writing JSON metadata files...")
        
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
        
        # --- 4. Process Physical Files ---
        log.info("Converting and copying physical sensor files...")
        self._process_sensor_files(data.sensor_data, data.sequence_path, sequence_name)
        
        # --- 5. Duplicate Sweeps ---
        log.info("Duplicating 'samples' directory to 'sweeps'...")
        self._duplicate_sweeps()
        
        # --- 6. Save Token Registry ---
        log.info("Saving global token registry...")
        self._token_manager.save_registry()
        
        log.info("=" * 50)
        log.info("NuScenes Write Complete")
        log.info("=" * 50)
        log.info(f"Output: {self._output_path}")
        log.info("=" * 50)
    
    def _get_last_timestamp(self):
        """Gets the last timestamp from existing sample.json."""
        sample_json_path = os.path.join(self._annot_dir, 'sample.json')
        if not os.path.exists(sample_json_path):
            return None
        
        samples = load_json_safely(sample_json_path, default=[])
        if samples and isinstance(samples, list):
            return samples[-1].get('timestamp')
        return None
    
    def _pre_populate_categories(self):
        """Injects official category tokens into the token manager."""
        log.info("Pre-populating TokenManager with official category tokens...")
        
        for cat_name, cat_token in self.OFFICIAL_CATEGORY_TOKENS.items():
            if cat_name not in self._token_manager.category_tokens:
                self._token_manager.category_tokens[cat_name] = cat_token
        
        log.info(f"Injected {len(self.OFFICIAL_CATEGORY_TOKENS)} official category tokens.")
    
    def _format_scene_name(self, raw_scene_name: str) -> str:
        """Converts a sequence name to nuScenes 'scene-NNNN' format."""
        match = re.search(r'\d+$', raw_scene_name)
        if match:
            num_str = match.group(0)
            return f"scene-{num_str.zfill(4)}"
        else:
            fallback_hash = hashlib.md5(raw_scene_name.encode()).hexdigest()[:4]
            log.warning(f"Could not parse number from '{raw_scene_name}'. Using 'scene-{fallback_hash}'")
            return f"scene-{fallback_hash}"
    
    # --- JSON Writing Methods ---
    
    def _write_sensor_and_calib(self, calibrations):
        """Writes sensor.json and calibrated_sensor.json."""
        new_sensors = []
        new_calib_sensors = []
        
        for if_calib in calibrations:
            sensor_token = self._token_manager.get_sensor_token(if_calib.sensor_name)
            is_camera = len(if_calib.camera_intrinsic) > 0
            
            new_sensors.append({
                "token": sensor_token,
                "modality": "camera" if is_camera else "lidar",
                "channel": if_calib.sensor_name,
            })
            
            new_calib_sensors.append({
                "token": self._token_manager.get_calibration_token(if_calib.sensor_name),
                "sensor_token": sensor_token,
                "translation": if_calib.translation,
                "rotation": if_calib.rotation,
                "camera_intrinsic": if_calib.camera_intrinsic
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self._annot_dir, 'sensor.json'),
            new_sensors,
            key_field='channel'
        )
        merge_and_overwrite_json_list(
            os.path.join(self._annot_dir, 'calibrated_sensor.json'),
            new_calib_sensors,
            key_field='sensor_token'
        )
    
    def _write_visibility(self):
        """Writes visibility.json."""
        vis_levels = [
            {"level": "v1-0", "description": "visibility 0-40%"},
            {"level": "v2-0", "description": "visibility 40-60%"},
            {"level": "v3-0", "description": "visibility 60-80%"},
            {"level": "v4-0", "description": "visibility 80-100%"}
        ]
        new_entries = []
        for vis in vis_levels:
            new_entries.append({
                "token": self._token_manager.get_visibility_token(vis["level"]),
                "level": vis["level"],
                "description": vis["description"]
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self._annot_dir, 'visibility.json'),
            new_entries,
            key_field='level'
        )
    
    def _write_attribute(self):
        """Writes attribute.json."""
        attributes = [
            {"name": "vehicle.moving", "description": "Vehicle is moving"},
            {"name": "pedestrian.moving", "description": "Pedestrian is moving"},
        ]
        new_entries = []
        for attr in attributes:
            new_entries.append({
                "token": self._token_manager.get_attribute_token(attr["name"]),
                "name": attr["name"],
                "description": attr["description"]
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self._annot_dir, 'attribute.json'),
            new_entries,
            key_field='name'
        )
    
    def _write_log(self, scenes):
        """Writes log.json."""
        new_entries = []
        for if_scene in scenes:
            logfile = f"{if_scene.name}-{datetime.now().strftime('%Y-%m-%d')}"
            log_token = self._token_manager.get_log_token(f"log_{logfile}")
            
            self._generated_log_tokens.append(log_token)
            
            new_entries.append({
                "token": log_token,
                "logfile": logfile,
                "vehicle": "stub_vehicle",
                "date_captured": datetime.now().strftime('%Y-%m-%d'),
                "location": "Hyderabad"
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self._annot_dir, 'log.json'),
            new_entries,
            key_field='token'
        )
    
    def _write_map(self):
        """Writes map.json and creates map image."""
        location = "Hyderabad"
        map_filename = f"maps/{location.lower()}.png"
        map_token = self._token_manager.get_map_token(f"map_{location}")
        
        new_map_entry = {
            "token": map_token,
            "log_tokens": self._generated_log_tokens,
            "category": "semantic_prior",
            "filename": map_filename,
        }
        
        merge_and_overwrite_json_list(
            os.path.join(self._annot_dir, 'map.json'),
            [new_map_entry],
            key_field='token'
        )
        
        # Create dummy map image
        image_path = os.path.join(self._maps_dir, f"{location.lower()}.png")
        if not os.path.exists(image_path):
            try:
                img = Image.new('RGB', (10, 10), color='black')
                img.save(image_path, 'PNG')
                log.info(f"Created dummy map file: {image_path}")
            except Exception as e:
                log.error(f"Could not create dummy map image: {e}")
        
        # Copy to basemap folder
        basemap_path = os.path.join(self._map_expansion_dir, 'basemap', f"{location.lower()}.png")
        if not os.path.exists(basemap_path):
            try:
                shutil.copyfile(image_path, basemap_path)
            except Exception as e:
                log.error(f"Could not copy basemap image: {e}")
    
    def _write_map_expansion(self):
        """Creates stubbed map expansion file."""
        log.info("Creating stubbed map expansion file...")
        expansion_path = os.path.join(self._map_expansion_dir, 'expansion', 'singapore-queenstown.json')
        
        node_tokens = [uuid.uuid4().hex for _ in range(4)]
        nodes = [
            {"token": node_tokens[0], "x": 10.0, "y": 10.0},
            {"token": node_tokens[1], "x": 10.0, "y": -10.0},
            {"token": node_tokens[2], "x": -10.0, "y": -10.0},
            {"token": node_tokens[3], "x": -10.0, "y": 10.0}
        ]
        
        poly_token = uuid.uuid4().hex
        polygons = [{
            "token": poly_token,
            "exterior_node_tokens": node_tokens,
            "holes": []
        }]
        
        stub_data = {
            "polygon": polygons,
            "node": nodes,
            "lane": [],
            "lane_divider_segment": [],
            "road_segment": [],
            "drivable_area": [],
            "traffic_control": []
        }
        
        save_json_safely(expansion_path, stub_data)
        log.info(f"Created stub map expansion file")
    
    def _write_prediction(self, scenes, samples):
        """Writes prediction.json."""
        if not scenes or not samples:
            return
        
        prediction_path = os.path.join(self._map_expansion_dir, 'prediction', 'prediction.json')
        
        prediction_data = load_json_safely(prediction_path, default={})
        
        raw_scene_name = scenes[0].name
        formatted_scene_name = self._format_scene_name(raw_scene_name)
        
        first_sample = min(samples, key=lambda x: x.timestamp_us)
        first_sample_token = self._token_manager.get_frame_token(first_sample.temp_frame_id)
        
        stubbed_predictions = []
        for _ in range(3):
            prediction_id = uuid.uuid4().hex
            stubbed_predictions.append(f"{prediction_id}_{first_sample_token}")
        
        prediction_data[formatted_scene_name] = stubbed_predictions
        
        save_json_safely(prediction_path, prediction_data)
        log.info(f"Merged scene '{formatted_scene_name}' into prediction.json")
    
    def _write_file_manifest(self, data: IntermediateData):
        """Writes file_manifest.json."""
        new_entries = []
        frame_to_sensor_data = {}
        
        for sd in data.sensor_data:
            if sd.temp_frame_id not in frame_to_sensor_data:
                frame_to_sensor_data[sd.temp_frame_id] = []
            frame_to_sensor_data[sd.temp_frame_id].append(sd)
        
        for if_sample in data.samples:
            frame_id = if_sample.temp_frame_id
            sequence_name = if_sample.scene_name
            
            manifest_entry = {
                "frame_id": frame_id,
                "sequence": sequence_name,
                "sample_token": self._token_manager.get_frame_token(frame_id),
                "sensors": []
            }
            
            if frame_id not in frame_to_sensor_data:
                continue
            
            for sd in frame_to_sensor_data[frame_id]:
                timestamp = sd.timestamp_us
                output_filename_base = f"{sequence_name}_frame_{timestamp}"
                
                if sd.sensor_name.startswith("CAM_"):
                    output_filename = f"{output_filename_base}.jpg"
                    source_file = f"{sequence_name}/camera/{sd.original_filename}"
                else:
                    output_filename = f"{output_filename_base}.pcd.bin"
                    source_file = f"{sequence_name}/lidar/{sd.original_filename}"
                
                manifest_entry["sensors"].append({
                    "channel": sd.sensor_name,
                    "source_file": source_file,
                    "output_file": f"samples/{sd.sensor_name}/{output_filename}"
                })
            
            new_entries.append(manifest_entry)
        
        append_to_json_list(os.path.join(self._annot_dir, 'file_manifest.json'), new_entries)
    
    def _write_sample_and_ego_pose(self, samples, ego_poses):
        """Writes sample.json and ego_pose.json."""
        sample_path = os.path.join(self._annot_dir, 'sample.json')
        ego_pose_path = os.path.join(self._annot_dir, 'ego_pose.json')
        
        all_samples = load_json_safely(sample_path, default=[])
        all_ego_poses = load_json_safely(ego_pose_path, default=[])
        
        for if_sample in samples:
            all_samples.append({
                "token": self._token_manager.get_frame_token(if_sample.temp_frame_id),
                "timestamp": if_sample.timestamp_us,
                "scene_token": self._token_manager.get_scene_token()
            })
        
        for if_pose in ego_poses:
            all_ego_poses.append({
                "token": self._token_manager.get_ego_pose_token(if_pose.temp_frame_id),
                "timestamp": if_pose.timestamp_us,
                "translation": if_pose.translation,
                "rotation": if_pose.rotation
            })
        
        all_samples.sort(key=lambda x: x['timestamp'])
        all_ego_poses.sort(key=lambda x: x['timestamp'])
        
        # Link prev/next for samples by scene
        scene_tokens = {s['scene_token'] for s in all_samples}
        final_samples = []
        
        for scene_token in scene_tokens:
            scene_samples = [s for s in all_samples if s['scene_token'] == scene_token]
            for i, sample in enumerate(scene_samples):
                sample['prev'] = scene_samples[i-1]['token'] if i > 0 else ""
                sample['next'] = scene_samples[i+1]['token'] if i < len(scene_samples) - 1 else ""
            final_samples.extend(scene_samples)
        
        save_json_safely(sample_path, final_samples)
        save_json_safely(ego_pose_path, all_ego_poses)
        
        # Write scene.json
        if samples:
            raw_scene_name = samples[0].scene_name
            formatted_scene_name = self._format_scene_name(raw_scene_name)
            
            new_scene = {
                "token": self._token_manager.get_scene_token(),
                "log_token": self._generated_log_tokens[-1] if self._generated_log_tokens else "",
                "nbr_samples": len(samples),
                "first_sample_token": self._token_manager.get_frame_token(samples[0].temp_frame_id),
                "last_sample_token": self._token_manager.get_frame_token(samples[-1].temp_frame_id),
                "name": formatted_scene_name,
                "description": f"Scene {raw_scene_name}"
            }
            append_to_json_list(os.path.join(self._annot_dir, 'scene.json'), [new_scene])
    
    def _write_sample_data(self, sensor_data, sequence_name):
        """Writes sample_data.json."""
        sample_data_path = os.path.join(self._annot_dir, 'sample_data.json')
        all_sample_data = load_json_safely(sample_data_path, default=[])
        
        for if_data in sensor_data:
            is_camera = if_data.sensor_name.startswith("CAM_")
            timestamp = if_data.timestamp_us
            output_filename_base = f"{sequence_name}_frame_{timestamp}"
            
            if is_camera:
                output_filename = f"{output_filename_base}.jpg"
                fileformat = "jpg"
            else:
                output_filename = f"{output_filename_base}.pcd.bin"
                fileformat = "pcd.bin"
            
            all_sample_data.append({
                "token": uuid.uuid4().hex,
                "sample_token": self._token_manager.get_frame_token(if_data.temp_frame_id),
                "ego_pose_token": self._token_manager.get_ego_pose_token(if_data.temp_frame_id),
                "calibrated_sensor_token": self._token_manager.get_calibration_token(if_data.sensor_name),
                "filename": f"samples/{if_data.sensor_name}/{output_filename}",
                "fileformat": fileformat,
                "width": 1440 if is_camera else 0,
                "height": 1080 if is_camera else 0,
                "timestamp": if_data.timestamp_us,
                "is_key_frame": if_data.is_keyframe,
            })
        
        # Link prev/next by sensor
        sensor_groups = {}
        for sd in all_sample_data:
            token = sd['calibrated_sensor_token']
            if token not in sensor_groups:
                sensor_groups[token] = []
            sensor_groups[token].append(sd)
        
        final_sample_data = []
        for sensor_token, sd_list in sensor_groups.items():
            sorted_list = sorted(sd_list, key=lambda x: x['timestamp'])
            for i, sd in enumerate(sorted_list):
                sd['prev'] = sorted_list[i-1]['token'] if i > 0 else ""
                sd['next'] = sorted_list[i+1]['token'] if i < len(sorted_list) - 1 else ""
            final_sample_data.extend(sorted_list)
        
        save_json_safely(sample_data_path, final_sample_data)
        log.info(f"Merged and overwrote sample_data.json. Total: {len(final_sample_data)}")
    
    def _write_category(self, instances):
        """Writes category.json."""
        new_categories = []
        all_category_names = {inst.category_name for inst in instances}
        
        # Add all categories from token manager
        for name, token in self._token_manager.category_tokens.items():
            new_categories.append({
                "token": token,
                "name": name,
                "description": f"{name} category"
            })
        
        # Add any new categories from data
        for name in all_category_names:
            if name not in self._token_manager.category_tokens:
                token = self._token_manager.get_category_token(name)
                new_categories.append({
                    "token": token,
                    "name": name,
                    "description": f"{name} category"
                })
        
        merge_and_overwrite_json_list(
            os.path.join(self._annot_dir, 'category.json'),
            new_categories,
            key_field='name'
        )
    
    def _write_instance_and_annotation(self, instances, annotations):
        """Writes instance.json and sample_annotation.json."""
        instance_path = os.path.join(self._annot_dir, 'instance.json')
        ann_path = os.path.join(self._annot_dir, 'sample_annotation.json')
        
        all_anns = load_json_safely(ann_path, default=[])
        inst_list = load_json_safely(instance_path, default=[])
        inst_db = {i['token']: i for i in inst_list}
        
        # Group annotations by instance
        new_anns_by_inst = {}
        for ann in annotations:
            if ann.temp_instance_id not in new_anns_by_inst:
                new_anns_by_inst[ann.temp_instance_id] = []
            new_anns_by_inst[ann.temp_instance_id].append(ann)
        
        inst_name_map = {inst.temp_instance_id: inst.category_name for inst in instances}
        used_category_tokens = {inst['category_token'] for inst in inst_db.values()}
        
        # Process each instance
        for temp_inst_id, new_anns_list in new_anns_by_inst.items():
            inst_token = self._token_manager.get_instance_token(temp_inst_id)
            new_anns_list.sort(key=lambda x: x.timestamp_us)
            
            last_ann_token = ""
            if inst_token in inst_db:
                last_ann_token = inst_db[inst_token]['last_annotation_token']
            
            generated_tokens = [self._token_manager.generate_annotation_token() for _ in new_anns_list]
            
            for i, if_ann in enumerate(new_anns_list):
                category_name = inst_name_map.get(temp_inst_id, "")
                
                # Add appropriate attributes
                attribute_tokens = []
                if category_name.startswith('vehicle.'):
                    attribute_tokens = [self._token_manager.get_attribute_token("vehicle.moving")]
                elif category_name.startswith('human.') or 'pedestrian' in category_name.lower():
                    attribute_tokens = [self._token_manager.get_attribute_token("pedestrian.moving")]
                
                ann_token = generated_tokens[i]
                prev_token = generated_tokens[i-1] if i > 0 else last_ann_token
                next_token = generated_tokens[i+1] if i < len(generated_tokens) - 1 else ""
                
                all_anns.append({
                    "token": ann_token,
                    "sample_token": self._token_manager.get_frame_token(if_ann.temp_frame_id),
                    "instance_token": inst_token,
                    "attribute_tokens": attribute_tokens,
                    "visibility_token": self._token_manager.get_visibility_token("v4-0"),
                    "translation": if_ann.translation,
                    "size": if_ann.size,
                    "rotation": if_ann.rotation,
                    "prev": prev_token,
                    "next": next_token,
                    "num_lidar_pts": 0,
                    "num_radar_pts": 0
                })
            
            category_token = self._token_manager.get_category_token(inst_name_map.get(temp_inst_id, ""))
            used_category_tokens.add(category_token)
            
            # Update or create instance
            if inst_token not in inst_db:
                inst_db[inst_token] = {
                    "token": inst_token,
                    "category_token": category_token,
                    "nbr_annotations": len(generated_tokens),
                    "first_annotation_token": generated_tokens[0],
                    "last_annotation_token": generated_tokens[-1]
                }
            else:
                inst_db[inst_token]["nbr_annotations"] += len(generated_tokens)
                inst_db[inst_token]["last_annotation_token"] = generated_tokens[-1]
        
        # Create dummy instances for unused categories
        log.info("Creating dummy instances for unused categories...")
        dummy_count = 0
        
        for cat_name, cat_token in self._token_manager.category_tokens.items():
            if cat_token not in used_category_tokens:
                dummy_instance_token = self._token_manager.get_instance_token(f"dummy_{cat_name}")
                
                if dummy_instance_token not in inst_db:
                    inst_db[dummy_instance_token] = {
                        "token": dummy_instance_token,
                        "category_token": cat_token,
                        "nbr_annotations": 0,
                        "first_annotation_token": dummy_instance_token,
                        "last_annotation_token": dummy_instance_token
                    }
                    dummy_count += 1
        
        if dummy_count > 0:
            log.info(f"Created {dummy_count} dummy instances for unused categories")
        
        save_json_safely(instance_path, list(inst_db.values()))
        save_json_safely(ann_path, all_anns)
        log.info(f"Wrote instance.json ({len(inst_db)} instances) and sample_annotation.json ({len(all_anns)} annotations)")
    
    # --- File Processing Methods ---
    
    def _process_sensor_files(self, sensor_data, sequence_path, sequence_name):
        """Converts and copies physical sensor files."""
        num_lidar = 0
        num_camera = 0
        
        for sd in sensor_data:
            timestamp = sd.timestamp_us
            output_filename_base = f"{sequence_name}_frame_{timestamp}"
            
            # Handle LiDAR files
            if sd.original_filename.endswith('.pcd'):
                src_file = os.path.join(sequence_path, 'lidar', sd.original_filename)
                output_filename = f"{output_filename_base}.pcd.bin"
                dst_folder = os.path.join(self._samples_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                if not os.path.exists(dst_file):
                    convert_lidar_pcd_to_bin(src_file, dst_file)
                    num_lidar += 1
            
            elif sd.original_filename.endswith('.feather'):
                src_file = os.path.join(sequence_path, 'lidar', sd.original_filename)
                output_filename = f"{output_filename_base}.pcd.bin"
                dst_folder = os.path.join(self._samples_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                if not os.path.exists(dst_file):
                    convert_lidar_feather_to_bin(src_file, dst_file)
                    num_lidar += 1
            
            else:  # Camera file
                src_file = os.path.join(sequence_path, 'camera', sd.original_filename)
                output_filename = f"{output_filename_base}.jpg"
                dst_folder = os.path.join(self._samples_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                if not os.path.exists(dst_file):
                    convert_camera_to_jpg(src_file, dst_file)
                    num_camera += 1
        
        log.info(f"Processed {num_lidar} LiDAR files and {num_camera} camera files")
    
    def _duplicate_sweeps(self):
        """Duplicates samples directory to sweeps."""
        if os.path.exists(self._sweeps_dir):
            try:
                shutil.rmtree(self._sweeps_dir)
            except Exception as e:
                log.error(f"Could not remove 'sweeps' directory: {e}")
                return
        
        try:
            shutil.copytree(self._samples_dir, self._sweeps_dir)
            log.info("Successfully duplicated 'samples' to 'sweeps'")
        except Exception as e:
            log.error(f"Could not copy 'samples' to 'sweeps': {e}")


# -----------------------------------------------------------------------------
#  PLACEHOLDER FOR FUTURE WRITERS
# -----------------------------------------------------------------------------

# class KittiWriter(BaseWriter):
#     """Writes data to KITTI format."""
#     def write(self, data: IntermediateData, output_path: str):
#         pass
#
# class WaymoWriter(BaseWriter):
#     """Writes data to Waymo Open Dataset format."""
#     def write(self, data: IntermediateData, output_path: str):
#         pass
