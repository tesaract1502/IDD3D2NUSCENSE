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
from intermediate_format_enhanced import (
    IntermediateData, IFScene, IFSample, IFInstance, IFAnnotation,
    IFCalibration, IFEgoPose
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conversion_state = {
    'active': False,
    'logs': Queue(),
    'progress': 0,
    'total_steps': 0,
    'current_step': 0
}
conversion_lock = threading.Lock()
json_file_lock = threading.Lock()

class LogHandler:
    def __init__(self, log_queue):
        self.queue = log_queue
    def log(self, message, log_type='info'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'type': log_type
        }
        self.queue.put(log_entry)
        logger.info(f"[{log_type.upper()}] {message}")

def append_to_json_list(file_path, new_data_list, log_handler: LogHandler):
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

class TokenTimestampManager:
    def __init__(self, registry_path=None, base_timestamp=None, frame_rate_hz=10):
        self.frame_rate_hz = frame_rate_hz
        self.frame_interval_us = int(1_000_000 / self.frame_rate_hz)
        self.base_timestamp = base_timestamp or 1640995200000000
        self.frame_tokens = {}
        self.instance_tokens = {}
        self.scene_token = None
        self.ego_pose_tokens = {}
        self.category_tokens = {}
        self.sensor_tokens = {}
        self.calibration_tokens = {}
        self.registry_path = registry_path
        self.load_registry()

    def load_registry(self):
        if self.registry_path and os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                self.category_tokens = registry.get('category_tokens', {})
                self.sensor_tokens = registry.get('sensor_tokens', {})
                self.calibration_tokens = registry.get('calibration_tokens', {})
                logger.info(f"Loaded {len(self.category_tokens)} global category tokens from registry.")
                logger.info(f"Loaded {len(self.sensor_tokens)} global sensor tokens from registry.")
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

    def get_ego_pose_token(self, frame_id):
        if frame_id not in self.ego_pose_tokens:
            self.ego_pose_tokens[frame_id] = uuid.uuid4().hex
        return self.ego_pose_tokens[frame_id]

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
        registry = {
            'base_timestamp': self.base_timestamp,
            'frame_rate_hz': self.frame_rate_hz,
            'category_tokens': self.category_tokens,
            'sensor_tokens': self.sensor_tokens,
            'calibration_tokens': self.calibration_tokens
        }
        try:
            with open(output_path, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Global token registry saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save token registry: {e}")
            raise

class BaseDataLoader(ABC):
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
    def __init__(self, name: str):
        self.name = name
        self.dry_run = False

    @abstractmethod
    def run(self, data_loader: BaseDataLoader, log_handler: LogHandler):
        pass

class DatasetConversionPipeline:
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

# Pipeline builder can be added below to add converters with correct token manager

# ConverterRegistry and am example usage or registry omitted for brevity.

