# utils.py
# ----------------------
# Contains helper classes and functions, like the TokenTimestampManager
# and JSON appender, moved from the old backend script.
# ----------------------

import os
import json
import uuid
import logging
import threading
from datetime import datetime

log = logging.getLogger(__name__)

# This lock is used by append_to_json_list to prevent race conditions
json_file_lock = threading.Lock()

def append_to_json_list(file_path, new_data_list):
    """
    Safely appends a list of new data to an existing JSON file
    that is expected to contain a list.
    """
    if not new_data_list:
        log.info(f"No new data to append to {os.path.basename(file_path)}")
        return

    with json_file_lock:
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        log.warning(f"{file_path} is not a list. Overwriting.")
                        existing_data = []
            except json.JSONDecodeError:
                log.warning(f"{file_path} is corrupted. Overwriting.")
                existing_data = []
        
        final_data = existing_data + new_data_list
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            log.info(f"Appended {len(new_data_list)} items to {os.path.basename(file_path)}. Total items: {len(final_data)}")
        except Exception as e:
            log.error(f"FATAL: Could not write to {file_path}: {e}")
            raise

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
        
        # Local tokens (frame, instance) are ALWAYS new for each run.
        self.frame_tokens = {}        # Map: temp_frame_id -> token
        self.instance_tokens = {}     # Map: temp_instance_id -> token
        self.ego_pose_tokens = {}     # Map: temp_frame_id -> ego_pose_token
        self.scene_token = None
        
        # --- FIXED: Separate dictionaries for each token type ---
        self.category_tokens = {}
        self.attribute_tokens = {}
        self.visibility_tokens = {}
        self.map_tokens = {}
        self.log_tokens = {}
        self.sensor_tokens = {}
        self.calibration_tokens = {}
        
        self.registry_path = registry_path
        self.load_registry()
        
    def load_registry(self):
        """
        Loads ONLY global tokens from the registry path if it exists.
        """
        if self.registry_path and os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                
                # Load all token types
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
        else:
            log.info("No existing token registry found. Starting fresh.")
 
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
    
    # --- FIXED: Specific getters for each token type ---
    
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
    
    def save_registry(self, output_path):
        """
        Save ONLY global tokens to the registry.
        """
        registry = {
            'base_timestamp': self.base_timestamp,
            'frame_rate_hz': self.frame_rate_hz,
            # --- FIXED: Save all dictionaries ---
            'category_tokens': self.category_tokens,
            'attribute_tokens': self.attribute_tokens,
            'visibility_tokens': self.visibility_tokens,
            'map_tokens': self.map_tokens,
            'log_tokens': self.log_tokens,
            'sensor_tokens': self.sensor_tokens,
            'calibration_tokens': self.calibration_tokens
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(registry, f, indent=2)
            log.info(f"Global token registry saved to {output_path}")
        except Exception as e:
            log.error(f"Failed to save token registry: {e}")

# --- NEW FUNCTION ---
def merge_and_overwrite_json_list(file_path, new_entries, key_field):
    """
    Safely merges a list of new entries into a JSON file,
    using 'key_field' to check for uniqueness.
    
    This reads the file, updates a dictionary, and overwrites the file.
    """
    with json_file_lock:
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_list = json.load(f)
                    if isinstance(existing_list, list):
                        for entry in existing_list:
                            if key_field in entry:
                                existing_data[entry[key_field]] = entry
                    else:
                        log.warning(f"{file_path} is not a list. Overwriting.")
            except json.JSONDecodeError:
                log.warning(f"{file_path} is corrupted. Overwriting.")
        
        # Add new entries, overwriting duplicates
        for entry in new_entries:
            if key_field in entry:
                existing_data[entry[key_field]] = entry
        
        final_list = list(existing_data.values())
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_list, f, indent=2)
            log.info(f"Merged/overwrote {file_path}. Total items: {len(final_list)}")
        except Exception as e:
            log.error(f"FATAL: Could not write to {file_path}: {e}")
            raise
