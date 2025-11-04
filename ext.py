from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from abc import ABC, abstractmethod
import os
import json
import threading
import shutil
from queue import Queue
from datetime import datetime
import logging
import uuid
import math

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for active conversions
conversion_state = {
    'active': False,
    'logs': Queue(),
    'progress': 0,
    'total_steps': 0,
    'current_step': 0
}

conversion_lock = threading.Lock()
# --- Lock for all JSON file I/O to prevent race conditions ---
json_file_lock = threading.Lock()


class LogHandler:
    """Handler to capture conversion logs and emit them"""
    
    def __init__(self, log_queue):
        self.queue = log_queue
    
    def log(self, message, log_type='info'):
        """Add a log entry to the queue"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'type': log_type
        }
        self.queue.put(log_entry)
        logger.info(f"[{log_type.upper()}] {message}")

# --- HELPER FUNCTION FOR APPENDING TO JSON LISTS ---
def append_to_json_list(file_path, new_data_list, log_handler):
    """
    Reads a JSON file (which is a list), appends new data, and writes it back.
    Uses a lock to prevent race conditions.
    """
    if not new_data_list:
        log_handler.log(f"No new data to append to {os.path.basename(file_path)}", 'info')
        return

    with json_file_lock:
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        log_handler.log(f"Warning: {file_path} is not a list. Overwriting.", 'warning')
                        existing_data = []
            except json.JSONDecodeError:
                log_handler.log(f"Warning: {file_path} is corrupted. Overwriting.", 'warning')
                existing_data = []
        
        final_data = existing_data + new_data_list
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            log_handler.log(f"Appended {len(new_data_list)} items to {os.path.basename(file_path)}. Total items: {len(final_data)}", 'success')
        except Exception as e:
            log_handler.log(f"FATAL: Could not write to {file_path}: {e}", 'error')
            raise

# TOKEN AND TIMESTAMP MANAGER (PERSISTENT)

class TokenTimestampManager:
    """
    Manages consistent token generation and timestamp synchronization
    across all converted files for the intermediate format.
    Loads from a registry path to be persistent across runs.
    """
    
    def __init__(self, registry_path=None, base_timestamp=None, frame_rate_hz=10):
        self.frame_rate_hz = frame_rate_hz
        self.frame_interval_us = int(1_000_000 / self.frame_rate_hz)
        
        if base_timestamp is None:
            self.base_timestamp = 1640995200000000
        else:
            self.base_timestamp = base_timestamp
        
        self.frame_tokens = {}
        self.instance_tokens = {}
        self.category_tokens = {}
        self.sensor_tokens = {}
        self.calibration_tokens = {}
        self.scene_token = None
        
        self.registry_path = registry_path
        self.load_registry()
        
    def load_registry(self):
        """Loads tokens from the registry path if it exists."""
        if self.registry_path and os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                
                self.frame_tokens = registry.get('frame_tokens', {})
                self.instance_tokens = registry.get('instance_tokens', {})
                self.category_tokens = registry.get('category_tokens', {})
                self.sensor_tokens = registry.get('sensor_tokens', {})
                self.calibration_tokens = registry.get('calibration_tokens', {})
                
                logger.info(f"Loaded {len(self.frame_tokens)} frame tokens from registry.")
                logger.info(f"Loaded {len(self.instance_tokens)} instance tokens from registry.")
                logger.info(f"Loaded {len(self.category_tokens)} category tokens from registry.")

            except Exception as e:
                logger.warning(f"Could not load token registry: {e}")
        else:
            logger.info("No existing token registry found. Starting fresh.")

    
    def get_timestamp(self, frame_index):
        return self.base_timestamp + (frame_index * self.frame_interval_us)
    
    def get_frame_token(self, frame_id):
        if frame_id not in self.frame_tokens:
            self.frame_tokens[frame_id] = uuid.uuid4().hex
        return self.frame_tokens[frame_id]
    
    def get_instance_token(self, obj_id):
        if obj_id not in self.instance_tokens:
            self.instance_tokens[obj_id] = uuid.uuid4().hex
        return self.instance_tokens[obj_id]
    
    def get_category_token(self, category_name):
        if category_name not in self.category_tokens:
            self.category_tokens[category_name] = uuid.uuid4().hex
        return self.category_tokens[category_name]
    
    def get_sensor_token(self, sensor_name):
        if sensor_name not in self.sensor_tokens:
            self.sensor_tokens[sensor_name] = uuid.uuid4().hex
        return self.sensor_tokens[sensor_name]
    
    def get_calibration_token(self, sensor_name):
        if sensor_name not in self.calibration_tokens:
            self.calibration_tokens[sensor_name] = uuid.uuid4().hex
        return self.calibration_tokens[sensor_name]
    
    def get_scene_token(self):
        if self.scene_token is None:
            self.scene_token = uuid.uuid4().hex
        return self.scene_token
    
    def generate_annotation_token(self):
        return uuid.uuid4().hex
    
    def save_registry(self, output_path):
        """Save the token registry to a JSON file for debugging."""
        registry = {
            'base_timestamp': self.base_timestamp,
            'frame_rate_hz': self.frame_rate_hz,
            'frame_interval_us': self.frame_interval_us,
            'scene_token': self.scene_token,
            'frame_tokens': self.frame_tokens,
            'instance_tokens': self.instance_tokens,
            'category_tokens': self.category_tokens,
            'sensor_tokens': self.sensor_tokens,
            'calibration_tokens': self.calibration_tokens
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Token registry saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save token registry: {e}")


# CONVERTER FRAMEWORK - Base Classes

class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    def __init__(self, root: str, sequence: str = None):
        self.root = os.path.abspath(root)
        self.sequence = sequence
    
    @abstractmethod
    def ensure_output_dirs(self):
        pass
    
    @abstractmethod
    def validate(self) -> dict:
        pass


class BaseConverter(ABC):
    """Abstract base converter class"""
    
    def __init__(self, name: str):
        self.name = name
        self.dry_run = False
    
    @abstractmethod
    def run(self, data_loader: BaseDataLoader, log_handler: LogHandler):
        pass


class DatasetConversionPipeline:
    """Orchestrates multi-step dataset conversions"""
    
    def __init__(self, source_format: str, target_format: str):
        self.source_format = source_format
        self.target_format = target_format
        self.converters = []
    
    def add_converter(self, converter: BaseConverter):
        self.converters.append(converter)
        return self
    
    def run(self, data_loader: BaseDataLoader, log_handler: LogHandler):
        if not self.converters:
            log_handler.log("No converters in pipeline", "warning")
            return
        
        for idx, converter in enumerate(self.converters):
            log_handler.log(
                f"\n[{idx+1}/{len(self.converters)}] Running {converter.name} converter...",
                'info'
            )
            try:
                converter.run(data_loader, log_handler)
                conversion_state['progress'] = ((idx + 1) / len(self.converters)) * 100
            except Exception as e:
                log_handler.log(f"{converter.name} failed: {str(e)}", 'error')
                raise


# --- MAPPING for new folder structure ---
IDD3D_TO_NUSCENES_CAM_MAP = {
    "cam0": "CAM_FRONT_LEFT",
    "cam1": "CAM_BACK_RIGHT",
    "cam2": "CAM_FRONT_RIGHT",
    "cam3": "CAM_FRONT",
    "cam4": "CAM_BACK_LEFT",
    "cam5": "CAM_BACK"
}
LIDAR_CHANNEL = "LIDAR"

# IDD3D IMPLEMENTATION

class IDD3DDataLoader(BaseDataLoader):
    """
    Loader for IDD3D dataset.
    Expects 'sequence_path' to be the full, absolute path to the sequence folder.
    All output paths now point to a single 'nuScenesFormat' directory.
    """
    
    def __init__(self, sequence_path: str):
        
        self.seq_base = os.path.abspath(sequence_path)
        root_path = os.path.dirname(self.seq_base)
        sequence_name = os.path.basename(self.seq_base)
        
        super().__init__(root_path, sequence_name)
        
        # --- INPUT PATHS (Based on self.seq_base) ---
        self.lidar_dir = os.path.join(self.seq_base, 'lidar')
        self.label_dir = os.path.join(self.seq_base, 'lable') 
        self.calib_dir = os.path.join(self.seq_base, 'calib')
        self.annot_json = os.path.join(self.seq_base, 'annot_data.json') 
        
        # --- UNIFIED OUTPUT PATHS ---
        self.output_base = os.path.join(self.root, 'nuScenesFormat')
        self.annot_out = os.path.join(self.output_base, 'anotations')
        
        # --- NEW: 'samples' folder ---
        self.samples_dir = os.path.join(self.output_base, 'samples')
        self.converted_lidar = os.path.join(self.samples_dir, LIDAR_CHANNEL)
        
        # Path for the persistent token registry
        self.token_registry_path = os.path.join(self.annot_out, 'token_registry.json')

    
    def ensure_output_dirs(self):
        # This function NO LONGER DELETES. It only ensures directories exist.
        
        # Create 'anotations' folder
        os.makedirs(self.annot_out, exist_ok=True)
        
        # Create 'samples' folder
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Create 'LIDAR' subfolder
        os.makedirs(self.converted_lidar, exist_ok=True)
        
        # Create all camera subfolders
        for cam_channel in IDD3D_TO_NUSCENES_CAM_MAP.values():
            os.makedirs(os.path.join(self.samples_dir, cam_channel), exist_ok=True)

    
    def validate(self) -> dict:
        if not os.path.exists(self.seq_base):
            return {'valid': False, 'error': f'Sequence path not found: {self.seq_base}'}
        
        required_dirs = ['lidar', 'lable', 'calib']
        missing = []
        for dir_name in required_dirs:
            dir_path = os.path.join(self.seq_base, dir_name)
            if not os.path.exists(dir_path):
                missing.append(dir_name)
        
        if missing:
            return {'valid': False, 'error': f'Missing directories: {", ".join(missing)} in {self.seq_base}'}
        
        if not os.path.exists(self.annot_json):
             return {'valid': False, 'error': f'Missing file: {self.annot_json}'}

        lidar_count = len([f for f in os.listdir(self.lidar_dir) 
                          if f.lower().endswith('.pcd')])
        label_count = len([f for f in os.listdir(self.label_dir) 
                          if f.lower().endswith('.json')])
        
        return {
            'valid': True,
            'path': self.seq_base,
            'lidar_files': lidar_count,
            'label_files': label_count
        }
    
    def list_lidar_files(self):
        if not os.path.exists(self.lidar_dir):
            return []
        # Return frame_ids (e.g., '00000', '00200') instead of full paths
        return sorted([os.path.splitext(f)[0] for f in os.listdir(self.lidar_dir) if f.lower().endswith('.pcd')])
    
    def read_annotations(self):
        if not os.path.exists(self.annot_json):
            return {}
        try:
            with open(self.annot_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {self.annot_json}: {e}")
            return {}


class IDD3DLidarConverter(BaseConverter):
    """
    Convert IDD3D PCD files to nuScenes .pcd.bin files.
    Reads annot_data.json to use the correct nuScenes-style filename.
    """
    
    def __init__(self):
        super().__init__('lidar')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        try:
            import numpy as np
            import open3d as o3d
            use_o3d = True
        except ImportError:
            use_o3d = False
            log_handler.log("Warning: open3d not available, creating placeholder files", 'warning')
        
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found, skipping LiDAR conversion.", 'warning')
            return
            
        frame_ids = sorted(annot_data.keys()) # e.g., "00000", "00200"
        
        dst_dir = data_loader.converted_lidar # .../samples/LIDAR
        src_dir = data_loader.lidar_dir       # .../idd3d_seq10/lidar
        
        converted = 0
        placeholders = 0
        overwritten = 0
        
        for frame_id in frame_ids:
            src_file = os.path.join(src_dir, f"{frame_id}.pcd")
            if not os.path.exists(src_file):
                log_handler.log(f"Source file not found: {src_file}", 'warning')
                continue

            if frame_id not in annot_data:
                log_handler.log(f"Skipping frame {frame_id}: not found in annot_data.json", 'warning')
                continue
            frame_data = annot_data[frame_id]
            
            dst_filename_base_pcd = os.path.basename(frame_data.get('lidar', f"{frame_id}.pcd"))
            dst_filename_base = os.path.splitext(dst_filename_base_pcd)[0]
            dst_filename = f"{dst_filename_base}.pcd.bin"
            
            dst_path = os.path.join(dst_dir, dst_filename)
            
            if os.path.exists(dst_path):
                overwritten += 1
            
            try:
                if use_o3d:
                    pcd = o3d.io.read_point_cloud(src_file)
                    xyz = np.asarray(pcd.points, dtype=np.float32)
                    intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
                    pts = np.hstack((xyz, intensity))
                    pts.astype(np.float32).tofile(dst_path)
                    converted += 1
                else:
                    open(dst_path, 'wb').close()
                    placeholders += 1
            except Exception as e:
                log_handler.log(f"Error converting {src_file}: {e}", 'error')
                open(dst_path, 'wb').close()
                placeholders += 1
        
        log_handler.log(f"LiDAR conversion complete: {converted} converted, {placeholders} placeholders.", 'success')
        if overwritten > 0:
            log_handler.log(f"Warning: {overwritten} existing LiDAR files were overwritten.", 'warning')
        log_handler.log(f"  Output: {dst_dir}", 'info')


class IDD3DCameraConverter(BaseConverter):
    """
    Convert IDD3D camera images from PNG to JPEG.
    Reads annot_data.json to use the correct nuScenes-style filename.
    """
    
    def __init__(self):
        super().__init__("camera")
    
    def run(self, dataloader: 'IDD3DDataLoader', loghandler: 'LogHandler'):
        try:
            from PIL import Image
            usepil = True
        except ImportError:
            usepil = False
            loghandler.log("PIL/Pillow not available, skipping camera conversion", "warning")
            return
        
        annot_data = dataloader.read_annotations()
        if not annot_data:
            loghandler.log("No annotations found, skipping Camera conversion.", 'warning')
            return
            
        frame_ids = sorted(annot_data.keys()) # e.g., "00000", "00200"
        
        src_camera_dir = os.path.join(dataloader.seq_base, "camera")
        dst_samples_dir = dataloader.samples_dir
        
        if not os.path.exists(src_camera_dir):
            loghandler.log("No camera directory found", "warning")
            return
        
        converted = 0
        errors = 0
        overwritten = 0

        for frame_id in frame_ids:
            if frame_id not in annot_data:
                loghandler.log(f"Skipping frame {frame_id}: not found in annot_data.json", 'warning')
                continue
            frame_data = annot_data[frame_id]
            
            for idd_cam, nu_cam in IDD3D_TO_NUSCENES_CAM_MAP.items(): # e.g., "cam0", "CAM_FRONT_LEFT"
                
                src_path = os.path.join(src_camera_dir, idd_cam, f"{frame_id}.png")
                if not os.path.exists(src_path):
                    loghandler.log(f"Source file not found: {src_path}", 'warning')
                    continue
                
                dst_filename_png = frame_data.get(idd_cam)
                if not dst_filename_png:
                    loghandler.log(f"No filename for {idd_cam} in frame {frame_id}", 'warning')
                    continue
                
                dst_filename_base = os.path.splitext(os.path.basename(dst_filename_png))[0]
                dst_filename = f"{dst_filename_base}.jpg"

                dst_path = os.path.join(dst_samples_dir, nu_cam, dst_filename)
                
                if os.path.exists(dst_path):
                    overwritten += 1
                
                try:
                    if usepil:
                        img = Image.open(src_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(dst_path, 'JPEG', quality=95)
                        converted += 1
                except Exception as e:
                    errors += 1
                    loghandler.log(f"Error converting {src_path} to {dst_path}: {str(e)}", "error")
        
        loghandler.log(f"Camera conversion complete: {converted} images converted, {errors} errors", "success")
        if overwritten > 0:
            loghandler.log(f"Warning: {overwritten} existing image files were overwritten.", 'warning')
        loghandler.log(f"  Output: {dst_samples_dir}", 'info')


class IDD3DCalibConverter(BaseConverter):
    """
    Generate calibration stubs for IDD3D.
    Saves to 'anotations' folder.
    Uses new channel names (e.g., 'CAM_FRONT') and persistent tokens.
    """
    
    def __init__(self, token_manager):
        super().__init__('calib')
        self.token_manager = token_manager
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        
        sensors = [LIDAR_CHANNEL] + list(IDD3D_TO_NUSCENES_CAM_MAP.values())
        
        calibrated_list = []
        sensors_j = []
        
        for s in sensors:
            sensor_token = self.token_manager.get_sensor_token(s)
            calib_token = self.token_manager.get_calibration_token(s)
            
            entry = {
                "token": calib_token,
                "sensor_token": sensor_token,
                "translation": [0.0, 0.0, 1.8] if s == LIDAR_CHANNEL else [0.0, 0.0, 1.6],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": []
            }
            calibrated_list.append(entry)
            sensors_j.append({
                "token": sensor_token,
                "modality": "lidar" if s == LIDAR_CHANNEL else "camera",
                "channel": s,
            })
        
        calib_path = os.path.join(data_loader.annot_out, 'calibrated_sensor.json')
        sensor_path = os.path.join(data_loader.annot_out, 'sensor.json')

        with open(calib_path, 'w') as f:
            json.dump(calibrated_list, f, indent=2)
        with open(sensor_path, 'w') as f:
            json.dump(sensors_j, f, indent=2)
        
        log_handler.log("Calibration stubs created/overwritten", 'success')
        log_handler.log(f"  Output: {data_loader.annot_out}", 'info')


class IDD3DLogConverter(BaseConverter):
    """
    Generates and merges a log.json entry for this sequence.
    """
    def __init__(self, token_manager, sequence_name):
        super().__init__('log')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
        
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        out_path = os.path.join(data_loader.annot_out, 'log.json')
        
        logfile = f"{data_loader.sequence}-{datetime.now().strftime('%Y-%m-%d')}"
        log_token = self.token_manager.get_category_token(f"log_{logfile}")
        
        new_log_entry = {
            "token": log_token,
            "logfile": logfile,
            "vehicle": "idd3d_stub_vehicle",
            "date_captured": datetime.now().strftime('%Y-%m-%d'),
            "location": "Hyderabad"
        }
        
        with json_file_lock:
            logs = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        logs = json.load(f)
                        if not isinstance(logs, list): logs = []
                except Exception as e:
                    log_handler.log(f"Could not read existing log.json: {e}", 'warning')
                    logs = []
            
            found = False
            for i, log in enumerate(logs):
                if log.get('logfile') == logfile:
                    logs[i] = new_log_entry
                    found = True
                    log_handler.log(f"Updating existing log entry for {logfile}", 'info')
                    break
            
            if not found:
                logs.append(new_log_entry)
                log_handler.log(f"Adding new log entry for {logfile}", 'info')

            try:
                with open(out_path, 'w') as f:
                    json.dump(logs, f, indent=2)
                log_handler.log(f"Unified log.json updated. Total logs: {len(logs)}", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to log.json: {e}", 'error')
                raise

class IDD3DEgoPoseConverter(BaseConverter):
    """
    Generates stubbed ego_pose.json entries for this sequence.
    """
    def __init__(self, token_manager):
        super().__init__('ego_pose')
        self.token_manager = token_manager
        
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found, skipping ego_pose", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        new_poses = []
        
        for i, frame_id in enumerate(frame_ids):
            timestamp = self.token_manager.get_timestamp(i)
            
            new_poses.append({
                "token": uuid.uuid4().hex,
                "timestamp": timestamp,
                "translation": [0.0, 0.0, 0.0], # Stubbed
                "rotation": [1.0, 0.0, 0.0, 0.0]  # Stubbed (identity quaternion)
            })
            
        out_path = os.path.join(data_loader.annot_out, 'ego_pose.json')
        append_to_json_list(out_path, new_poses, log_handler)

class IDD3DMapConverter(BaseConverter):
    """
    Generates and merges a single stubbed map.json entry for the location.
    Now links to all known logs for that location.
    """
    def __init__(self, token_manager):
        super().__init__('map')
        self.token_manager = token_manager
        
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        out_path = os.path.join(data_loader.annot_out, 'map.json')
        log_path = os.path.join(data_loader.annot_out, 'log.json')
        
        location = "Hyderabad"
        map_token = self.token_manager.get_category_token(f"map_{location}")
        
        with json_file_lock:
            all_log_tokens_for_location = []
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        all_logs = json.load(f)
                        if isinstance(all_logs, list):
                            for log in all_logs:
                                if log.get('location') == location and log.get('token'):
                                    all_log_tokens_for_location.append(log.get('token'))
                except Exception as e:
                    log_handler.log(f"Could not read log.json to link maps: {e}", 'warning')
            
            new_map_entry = {
                "token": map_token,
                "log_tokens": all_log_tokens_for_location,
                "category": "semantic_prior",
                "filename": f"maps/{location.lower()}.png",
            }
            
            maps = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        maps = json.load(f)
                        if not isinstance(maps, list): maps = []
                except Exception as e:
                    log_handler.log(f"Could not read existing map.json: {e}", 'warning')
                    maps = []
            
            found = False
            for i, map_entry in enumerate(maps):
                if map_entry.get('token') == map_token:
                    maps[i] = new_map_entry
                    found = True
                    log_handler.log(f"Updating existing map entry for {location}", 'info')
                    break
            
            if not found:
                maps.append(new_map_entry)
                log_handler.log(f"Adding new map entry for {location}", 'info')

            try:
                with open(out_path, 'w') as f:
                    json.dump(maps, f, indent=2)
                log_handler.log(f"Unified map.json updated. Total maps: {len(maps)}", 'success')
                log_handler.log(f"  Map '{location}' now linked to {len(all_log_tokens_for_location)} logs.", 'info')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to map.json: {e}", 'error')
                raise

class IDD3DSceneConverter(BaseConverter):
    """
    Generate and update a *shared* scene.json with scene metadata.
    """
    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('scene')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        scene_token = self.token_manager.get_scene_token()
        logfile = f"{data_loader.sequence}-{datetime.now().strftime('%Y-%m-%d')}"
        log_token = self.token_manager.get_category_token(f"log_{logfile}")
        
        first_sample_token = self.token_manager.get_frame_token(frame_ids[0])
        last_sample_token = self.token_manager.get_frame_token(frame_ids[-1])
        
        seq_num_str = self.sequence_name.split('_')[-1].replace('seq', '')
        formatted_num = seq_num_str.zfill(3)
        new_scene_name = f"scene-{formatted_num}"

        current_scene = {
            "token": scene_token,
            "log_token": log_token, 
            "nbr_samples": len(frame_ids),
            "first_sample_token": first_sample_token,
            "last_sample_token": last_sample_token,
            "name": new_scene_name, 
            "description": f"IDD3D sequence {self.sequence_name}"
        }
        
        out_path = os.path.join(data_loader.annot_out, 'scene.json')
        
        with json_file_lock:
            scenes = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        scenes = json.load(f)
                        if not isinstance(scenes, list): scenes = []
                except Exception as e:
                    log_handler.log(f"Could not read shared scene.json: {e}", 'warning')
                    scenes = []
            
            found = False
            for i, scene in enumerate(scenes):
                if scene.get('name') == new_scene_name:
                    log_handler.log(f"Updating existing scene: {new_scene_name}", 'info')
                    scenes[i] = current_scene
                    found = True
                    break
            
            if not found:
                log_handler.log(f"Adding new scene: {new_scene_name}", 'info')
                scenes.append(current_scene)
            
            try:
                with open(out_path, 'w') as f:
                    json.dump(scenes, f, indent=2)
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to shared scene.json: {e}", 'error')
                raise

        log_handler.log(f"Unified scene.json updated ({len(scenes)} total scenes)", 'success')
        log_handler.log(f"  Output: {out_path}", 'info')

class IDD3DSampleDataConverter(BaseConverter):
    """
    Generate sample_data.json. Appends to existing file.
    --- UPDATED ---
    Reads filenames from annot_data.json
    Uses correct 1440x1080 resolution.
    Correctly links ALL prev/next tokens on every run.
    """
    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('sample_data')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        new_sample_data_list = []
        
        for i, frame_id in enumerate(frame_ids):
            if frame_id not in annot_data:
                log_handler.log(f"Skipping frame {frame_id}: not found in annot_data.json", 'warning')
                continue
            frame_data = annot_data[frame_id]
            sample_token = self.token_manager.get_frame_token(frame_id)
            timestamp = self.token_manager.get_timestamp(i)
            
            # --- LiDAR Data ---
            lidar_filename_raw = frame_data.get('lidar', f'{frame_id}.pcd.bin')
            lidar_filename_base = os.path.splitext(os.path.basename(lidar_filename_raw))[0]
            lidar_filename = f"{lidar_filename_base}.pcd.bin"
            
            new_sample_data_list.append({
                "token": uuid.uuid4().hex, 
                "sample_token": sample_token,
                "calibrated_sensor_token": self.token_manager.get_calibration_token(LIDAR_CHANNEL),
                "filename": f"samples/{LIDAR_CHANNEL}/{lidar_filename}", 
                "fileformat": "pcd.bin", "width": 0, "height": 0, "timestamp": timestamp,
                "is_key_frame": True, "next": "", "prev": ""
            })
            
            # --- Camera Data ---
            for idd_cam, nu_cam in IDD3D_TO_NUSCENES_CAM_MAP.items(): 
                
                cam_filename_raw = frame_data.get(idd_cam, f'{frame_id}.jpg')
                cam_filename = os.path.basename(cam_filename_raw)
                
                new_sample_data_list.append({
                    "token": uuid.uuid4().hex,
                    "sample_token": sample_token,
                    "calibrated_sensor_token": self.token_manager.get_calibration_token(nu_cam),
                    "filename": f"samples/{nu_cam}/{cam_filename}", 
                    "fileformat": "jpg", 
                    "width": 1440,
                    "height": 1080, 
                    "timestamp": timestamp,
                    "is_key_frame": True, "next": "", "prev": ""
                })
        
        # --- PREV/NEXT LINKING LOGIC ---
        out_path = os.path.join(data_loader.annot_out, 'sample_data.json')
        
        with json_file_lock:
            existing_data = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list): existing_data = []
                except Exception as e:
                    log_handler.log(f"Warning: could not read existing sample_data.json: {e}", "warning")
                    existing_data = []

            all_data = existing_data + new_sample_data_list
            
            sensor_groups = {}
            for sd in all_data:
                token = sd['calibrated_sensor_token']
                if token not in sensor_groups: sensor_groups[token] = []
                sensor_groups[token].append(sd)

            log_handler.log(f"Linking prev/next tokens for {len(all_data)} total sample_data entries...", 'info')
            
            final_linked_list = []
            
            for sensor_token, sd_list in sensor_groups.items():
                sorted_list = sorted(sd_list, key=lambda x: x['timestamp'])
                for i, sd in enumerate(sorted_list):
                    sd['prev'] = sorted_list[i-1]['token'] if i > 0 else ""
                    sd['next'] = sorted_list[i+1]['token'] if i < len(sorted_list) - 1 else ""
                
                final_linked_list.extend(sorted_list)
            
            try:
                with open(out_path, 'w') as f:
                    json.dump(final_linked_list, f, indent=2)
                log_handler.log(f"Appended {len(new_sample_data_list)} items to sample_data.json. Total items: {len(final_linked_list)}", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to {out_path}: {e}", 'error')
                raise

class IDD3DCategoryConverter(BaseConverter):
    """
    Generate category.json. Merges with existing file.
    """
    
    def __init__(self, token_manager):
        super().__init__('category')
        self.token_manager = token_manager
    
    def run(self, data_loader, log_handler):
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car', 'Truck': 'vehicle.truck', 'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle', 'MotorcyleRider': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle', 'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult', 'Rider': 'human.pedestrian.rider',
            'Animal': 'animal', 'TrafficLight': 'static_object.traffic_light',
            'TrafficSign': 'static_object.traffic_sign', 'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other', 'Misc': 'movable_object.debris'
        }
        nuscenes_descriptions = {
            'vehicle.car': 'A car.', 'vehicle.truck': 'A truck.', 'vehicle.bus': 'A bus.',
            'vehicle.motorcycle': 'A motorcycle or motorcyclist.', 'vehicle.bicycle': 'A bicycle.',
            'vehicle.auto': 'An auto-rickshaw.', 'human.pedestrian.adult': 'An adult pedestrian.',
            'human.pedestrian.rider': 'A person riding a vehicle (e.g., bicycle).',
            'animal': 'An animal.', 'static_object.traffic_light': 'A traffic light.',
            'static_object.traffic_sign': 'A traffic sign.', 'static_object.pole': 'A pole.',
            'vehicle.other': 'Other vehicle types.', 'movable_object.debris': 'Miscellaneous debris or movable objects.'
        }
        
        out_path = os.path.join(data_loader.annot_out, 'category.json')
        
        with json_file_lock:
            existing_categories = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        cats = json.load(f)
                        existing_categories = {cat['name']: cat for cat in cats}
                except Exception as e:
                    log_handler.log(f"Could not read existing category.json: {e}", 'warning')

            new_cats_added = 0
            for idd_type, nuscenes_name in idd3d_to_nuscenes_categories.items():
                if nuscenes_name not in existing_categories:
                    token = self.token_manager.get_category_token(nuscenes_name)
                    existing_categories[nuscenes_name] = {
                        "token": token,
                        "name": nuscenes_name,
                        "description": nuscenes_descriptions.get(nuscenes_name, "")
                    }
                    new_cats_added += 1
            
            final_categories = list(existing_categories.values())
            try:
                with open(out_path, 'w') as f:
                    json.dump(final_categories, f, indent=2)
                log_handler.log(f"Category file merged. Added {new_cats_added} new. Total: {len(final_categories)}", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to category.json: {e}", 'error')
                raise

class IDD3DSampleConverter(BaseConverter):
    """
    Generate sample.json. Appends to existing file.
    """
    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('sample')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader, log_handler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        scene_token = self.token_manager.get_scene_token()
        samples = []
        
        for i, frame_id in enumerate(frame_ids):
            token = self.token_manager.get_frame_token(frame_id)
            timestamp = self.token_manager.get_timestamp(i)
            prev = self.token_manager.get_frame_token(frame_ids[i-1]) if i > 0 else ""
            next_token = self.token_manager.get_frame_token(frame_ids[i+1]) if i < len(frame_ids)-1 else ""
            
            sample = {
                "token": token,
                "timestamp": timestamp,
                "prev": prev,
                "next": next_token,
                "scene_token": scene_token
            }
            samples.append(sample)
        
        out_path = os.path.join(data_loader.annot_out, 'sample.json')
        append_to_json_list(out_path, samples, log_handler)

class IDD3DSampleAnnotationConverter(BaseConverter):
    """
    Generate sample_annotation.json. Appends to existing file.
    """
    
    def __init__(self, token_manager, sequence_name: str = 'seq'):
        super().__init__('sample_annotation')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        sample_annotations = []
        object_instances_in_this_run = {}
        
        for frame_id in frame_ids:
            label_path = os.path.join(data_loader.label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = obj.get("obj_id")
                    if not obj_id:
                        continue
                    
                    instance_token = self.token_manager.get_instance_token(obj_id)
                    
                    if obj_id not in object_instances_in_this_run:
                        object_instances_in_this_run[obj_id] = {'annotations': []}
                    
                    ann_token = self.token_manager.generate_annotation_token()
                    frame_token = self.token_manager.get_frame_token(frame_id)
                    
                    psr = obj.get("psr", {})
                    pos = psr.get("position", {})
                    rot = psr.get("rotation", {})
                    scl = psr.get("scale", {})
                    
                    translation = [pos.get("x",0), pos.get("y",0), pos.get("z",0)]
                    size = [scl.get("x",1), scl.get("y",1), scl.get("z",1)]
                    rotation_quat = [rot.get("x",0), rot.get("y",0), rot.get("z",0), 1.0]
                    
                    annotation = {
                        "token": ann_token,
                        "sample_token": frame_token,
                        "instance_token": instance_token,
                        "translation": translation, "size": size, "rotation": rotation_quat,
                        "prev": "", "next": "",
                        "num_lidar_pts": 0, "num_radar_pts": 0
                    }
                    object_instances_in_this_run[obj_id]['annotations'].append(annotation)
                    
            except Exception as e:
                log_handler.log(f"Error processing label {frame_id}: {str(e)}", 'warning')
        
        for obj_id, instance_data in object_instances_in_this_run.items():
            annotations = instance_data['annotations']
            for i, ann in enumerate(annotations):
                if i > 0:
                    ann['prev'] = annotations[i-1]['token']
                if i < len(annotations) - 1:
                    ann['next'] = annotations[i+1]['token']
                sample_annotations.append(ann)
        
        out_path = os.path.join(data_loader.annot_out, 'sample_annotation.json')
        append_to_json_list(out_path, sample_annotations, log_handler)

class IDD3DInstanceConverter(BaseConverter):
    """
    Generate instance.json. Appends to existing file.
    """
    
    def __init__(self, token_manager):
        super().__init__('instance')
        self.token_manager = token_manager
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car', 'Truck': 'vehicle.truck', 'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle', 'MotorcyleRider': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle', 'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult', 'Rider': 'human.pedestrian.rider',
            'Animal': 'animal', 'TrafficLight': 'static_object.traffic_light',
            'TrafficSign': 'static_object.traffic_sign', 'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other', 'Misc': 'movable_object.debris'
        }
        
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        instance_tracker = {} 
        
        for frame_id in frame_ids:
            label_path = os.path.join(data_loader.label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = obj.get("obj_id")
                    obj_type = obj.get("obj_type")
                    if not obj_id or not obj_type:
                        continue
                    
                    if obj_id not in instance_tracker:
                        instance_token = self.token_manager.get_instance_token(obj_id)
                        category_name = idd3d_to_nuscenes_categories.get(obj_type, f'movable_object.{obj_type.lower()}')
                        category_token = self.token_manager.get_category_token(category_name)
                        
                        instance_tracker[obj_id] = {
                            'instance_token': instance_token,
                            'category_token': category_token,
                            'obj_type': obj_type,
                            'first_ann_token': self.token_manager.generate_annotation_token(),
                            'last_ann_token': self.token_manager.generate_annotation_token()
                        }
                    else:
                        instance_tracker[obj_id]['last_ann_token'] = self.token_manager.generate_annotation_token()
                        
            except Exception as e:
                log_handler.log(f"Error processing label {frame_id}: {str(e)}", 'warning')
        
        new_instances = []
        for obj_id, data in instance_tracker.items():
            new_instances.append({
                "token": data['instance_token'],
                "category_token": data['category_token'],
                "nbr_annotations": None, 
                "first_annotation_token": data['first_ann_token'],
                "last_annotation_token": data['last_ann_token']
            })
        
        out_path = os.path.join(data_loader.annot_out, 'instance.json')
        
        with json_file_lock:
            existing_instances = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        insts = json.load(f)
                        existing_instances = {inst['token']: inst for inst in insts}
                except Exception as e:
                    log_handler.log(f"Could not read existing instance.json: {e}", 'warning')

            updated_count = 0
            new_count = 0
            for inst in new_instances:
                if inst['token'] in existing_instances:
                    existing_instances[inst['token']]['last_annotation_token'] = inst['last_annotation_token']
                    updated_count += 1
                else:
                    existing_instances[inst['token']] = inst
                    new_count += 1
            
            final_instances = list(existing_instances.values())
            try:
                with open(out_path, 'w') as f:
                    json.dump(final_instances, f, indent=2)
                log_handler.log(f"Instance file merged. Added {new_count}, updated {updated_count}. Total: {len(final_instances)}", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to instance.json: {e}", 'error')
                raise

# --- NEW FILE MANIFEST CONVERTER ---
class IDD3DFileManifestConverter(BaseConverter):
    """
    Generates a human-readable manifest linking source files to output files.
    Appends to the existing file_manifest.json.
    """
    def __init__(self, token_manager, sequence_name):
        super().__init__('manifest')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found, skipping manifest", 'warning')
            return
            
        frame_ids = sorted(annot_data.keys())
        new_manifest_entries = []

        for i, frame_id in enumerate(frame_ids):
            if frame_id not in annot_data:
                continue
            frame_data = annot_data[frame_id]
            
            manifest_entry = {
                "frame_id": frame_id,
                "sequence": self.sequence_name,
                "sample_token": self.token_manager.get_frame_token(frame_id),
                "sensors": []
            }

            # --- LiDAR Entry ---
            src_lidar_name = f"{frame_id}.pcd"
            dst_lidar_raw = frame_data.get('lidar', f"{frame_id}.pcd.bin")
            dst_lidar_name = f"{os.path.splitext(os.path.basename(dst_lidar_raw))[0]}.pcd.bin"
            
            manifest_entry["sensors"].append({
                "channel": LIDAR_CHANNEL,
                "source_file": f"lidar/{src_lidar_name}",
                "output_file": f"samples/{LIDAR_CHANNEL}/{dst_lidar_name}"
            })
            
            # --- Camera Entries ---
            for idd_cam, nu_cam in IDD3D_TO_NUSCENES_CAM_MAP.items():
                src_cam_name = f"{frame_id}.png"
                dst_cam_raw = frame_data.get(idd_cam, f"{frame_id}.jpg")
                dst_cam_name = os.path.basename(dst_cam_raw)
                
                manifest_entry["sensors"].append({
                    "channel": nu_cam,
                    "source_file": f"camera/{idd_cam}/{src_cam_name}",
                    "output_file": f"samples/{nu_cam}/{dst_cam_name}"
                })
            
            new_manifest_entries.append(manifest_entry)
        
        out_path = os.path.join(data_loader.annot_out, 'file_manifest.json')
        append_to_json_list(out_path, new_manifest_entries, log_handler)


class IDD3DTokenRegistrySaver(BaseConverter):
    """
    Saves the token registry at the end of the pipeline.
    """
    
    def __init__(self, token_manager, registry_path):
        super().__init__('save_registry')
        self.token_manager = token_manager
        self.registry_path = registry_path
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        try:
            self.token_manager.save_registry(self.registry_path)
            log_handler.log(f"Token registry saved to: {self.registry_path}", 'success')
        except Exception as e:
            log_handler.log(f"Warning: Could not save token registry: {str(e)}", 'warning')


# CONVERTER REGISTRY

class ConverterRegistry:
    """Registry for dataset conversions"""
    
    _conversions = {}
    
    @classmethod
    def register(cls, source: str, target: str, pipeline_builder):
        key = (source, target)
        cls._conversions[key] = pipeline_builder
    
    @classmethod
    def get_pipeline(cls, source: str, target: str, config: dict):
        key = (source, target)
        if key not in cls._conversions:
            raise ValueError(f"No conversion registered for {source} -> {target}")
        pipeline_builder = cls._conversions[key]
        return pipeline_builder(config)
    
    @classmethod
    def get_available_conversions(cls):
        return [{'source': s, 'target': t} for s, t in cls._conversions.keys()]


# REGISTER CONVERSIONS

def build_idd3d_to_nuscenes_pipeline(config: dict) -> DatasetConversionPipeline:
    """Build conversion pipeline for IDD3D -> nuScenes with TokenTimestampManager"""
    pipeline = DatasetConversionPipeline('idd3d', 'nuscenes')
    
    conversions = config.get('conversions', {})
    sequence_name = config.get('sequence_id', 'idd3d_seq10')
    root_path = config.get('root_path')
    
    if not root_path:
        raise ValueError("root_path is required in config to build pipeline")
        
    registry_path = os.path.join(root_path, 'nuScenesFormat', 'anotations', 'token_registry.json')
    
    # --- Calculate new base_timestamp ---
    last_timestamp = 1640995200000000 # Default start
    sample_json_path = os.path.join(root_path, 'nuScenesFormat', 'anotations', 'sample.json')
    
    with json_file_lock:
        if os.path.exists(sample_json_path):
            try:
                with open(sample_json_path, 'r') as f:
                    samples = json.load(f)
                    if samples and isinstance(samples, list):
                        last_timestamp = samples[-1].get('timestamp', last_timestamp)
            except Exception:
                pass 
                
    new_base_timestamp = last_timestamp + 20_000_000 # 20-second gap
    
    token_manager = TokenTimestampManager(
        frame_rate_hz=10, 
        registry_path=registry_path,
        base_timestamp=new_base_timestamp
    )
    
    # ---
    
    # PHASE 1: Data Conversion
    if conversions.get('lidar', False):
        pipeline.add_converter(IDD3DLidarConverter())
    if conversions.get('camera', False):
        pipeline.add_converter(IDD3DCameraConverter())
    if conversions.get('calib', False):
        pipeline.add_converter(IDD3DCalibConverter(token_manager))
    
    # PHASE 2: Taxonomy & Stubs (Merges)
    if conversions.get('category', False):
        pipeline.add_converter(IDD3DCategoryConverter(token_manager))
    
    if conversions.get('log', False):
        pipeline.add_converter(IDD3DLogConverter(token_manager, sequence_name))
    if conversions.get('map', False):
        pipeline.add_converter(IDD3DMapConverter(token_manager))
    if conversions.get('scene', False):
        pipeline.add_converter(IDD3DSceneConverter(token_manager, sequence_name))

    # PHASE 3: Core Data (Appends)
    if conversions.get('sample', False):
        pipeline.add_converter(IDD3DSampleConverter(token_manager, sequence_name))
    if conversions.get('sample_data', False):
        pipeline.add_converter(IDD3DSampleDataConverter(token_manager, sequence_name))
    if conversions.get('ego_pose', False):
        pipeline.add_converter(IDD3DEgoPoseConverter(token_manager))

    # PHASE 4: Annotations (Appends)
    if conversions.get('instance', False):
        pipeline.add_converter(IDD3DInstanceConverter(token_manager))
    if conversions.get('sample_annotation', False):
        pipeline.add_converter(IDD3DSampleAnnotationConverter(token_manager, sequence_name))
    
    # --- ADD MANIFEST (unconditional) ---
    if conversions.get('manifest', False):
        pipeline.add_converter(IDD3DFileManifestConverter(token_manager, sequence_name))

    # PHASE 5: Save Token Registry (runs last)
    pipeline.add_converter(IDD3DTokenRegistrySaver(token_manager, registry_path))
    
    return pipeline


ConverterRegistry.register('idd3d', 'nuscenes', build_idd3d_to_nuscenes_pipeline)


# FLASK API ENDPOINTS

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Dataset Converter API'})


@app.route('/api/conversions', methods=['GET'])
def get_conversions():
    """Get all available conversions"""
    conversions = ConverterRegistry.get_available_conversions()
    return jsonify({'conversions': conversions})


@app.route('/api/validate-paths', methods=['POST'])
def validate_paths():
    """Validate dataset paths"""
    data = request.json
    source = data.get('source', 'idd3d')
    sequence_path = data.get('sequence_path')
    
    if not sequence_path:
        return jsonify({'valid': False, 'error': 'Sequence Path is required'}), 400
    
    if not os.path.exists(sequence_path):
        return jsonify({'valid': False, 'error': f'Sequence Path does not exist: {sequence_path}'}), 400
    
    if source == 'idd3d':
        loader = IDD3DDataLoader(sequence_path)
        validation = loader.validate()
        return jsonify(validation)
    
    return jsonify({'valid': False, 'error': f'Unknown source dataset: {source}'}), 400


@app.route('/api/convert/stream', methods=['POST'])
def convert_stream():
    """Start conversion and stream logs via SSE"""
    with conversion_lock:
        if conversion_state['active']:
            return jsonify({'error': 'Conversion already in progress'}), 409
        conversion_state['active'] = True
    
    data = request.json
    source = data.get('source', 'idd3d')
    target = data.get('target', 'nuscenes')
    sequence_path = data.get('sequence_path')
    conversions = data.get('conversions', {})
    
    def generate():
        loader = None
        try:
            while not conversion_state['logs'].empty():
                conversion_state['logs'].get()
            
            log_handler = LogHandler(conversion_state['logs'])
            
            log_handler.log(f"Starting conversion: {source} → {target}", 'info')
            log_handler.log(f"Sequence Path: {sequence_path}", 'info')
            
            # Create data loader
            if source == 'idd3d':
                loader = IDD3DDataLoader(sequence_path)
            else:
                raise ValueError(f"Unknown source dataset: {source}")
            
            # This prepares the 'nuScenesFormat' directory
            loader.ensure_output_dirs()
            log_handler.log(f"Outputting to: {loader.output_base}", 'info')
            
            pipeline = ConverterRegistry.get_pipeline(
                source, target,
                {
                    'conversions': conversions, 
                    'root_path': loader.root, 
                    'sequence_id': loader.sequence
                }
            )
            conversion_state['total_steps'] = len(pipeline.converters)
            
            if conversion_state['total_steps'] == 0:
                log_handler.log("No conversion modules selected", 'warning')
            else:
                pipeline.run(loader, log_handler)
                log_handler.log("Conversion pipeline completed successfully!", 'success')
                log_handler.log(f"Unified output is in: {loader.output_base}", 'success')
        
        except Exception as e:
            log_handler.log(f"Conversion failed: {str(e)}", 'error')
            import traceback
            log_handler.log(traceback.format_exc(), 'error')
        
        finally:
            conversion_state['active'] = False
            while not conversion_state['logs'].empty():
                log_entry = conversion_state['logs'].get()
                yield f"data: {json.dumps(log_entry)}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


if __name__ == '__main__':
    print("Starting Extensible Dataset Converter API...")
    print("Registered conversions:")
    for conv in ConverterRegistry.get_available_conversions():
        print(f"  {conv['source']} → {conv['target']}")
    print("\nServer running on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
