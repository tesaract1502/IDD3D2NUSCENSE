# writers.py
# ----------------------
# This file contains "Writer" classes.
# Each Writer is responsible for consuming the 'IntermediateData' object
# and writing it to a specific dataset format (like nuScenes).
# ----------------------

import os
import json
import shutil
import logging
from abc import ABC, abstractmethod
from PIL import Image  # <-- Import for dummy map
from intermediate_format import IntermediateData
from utils import TokenTimestampManager, append_to_json_list, json_file_lock

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# --- File Conversion Helpers (from old IDD3DLidar/Camera Converters) ---

def convert_lidar_file(src_path, dst_path):
    """
    Converts a .pcd file to a .pcd.bin file.
    If open3d is not available, creates an empty placeholder file.
    """
    try:
        import numpy as np
        import open3d as o3d
        
        if not os.path.exists(src_path):
            log.warning(f"Source LiDAR file not found: {src_path}")
            open(dst_path, 'wb').close() # Create empty file
            return

        pcd = o3d.io.read_point_cloud(src_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        # nuScenes .pcd.bin format is [x, y, z, intensity]
        # We only have xyz, so we stub intensity.
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        
        # Combine xyz and intensity
        pts = np.hstack((xyz, intensity))
        
        # Save to binary file
        pts.astype(np.float32).tofile(dst_path)
        
    except ImportError:
        log.warning(f"open3d not available. Creating empty placeholder for {dst_path}")
        open(dst_path, 'wb').close()
    except Exception as e:
        log.error(f"Error converting {src_path}: {e}. Creating empty file.")
        open(dst_path, 'wb').close()

def convert_camera_file(src_path, dst_path):
    """
    Converts a .png file to a .jpg file.
    If PIL is not available, does nothing.
    """
    try:
        # from PIL import Image # No longer needed here, imported at top
        
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

# -----------------------------------------------------------------------------
#  BASE WRITER
# -----------------------------------------------------------------------------

class BaseWriter(ABC):
    """
    Abstract base class for all dataset writers.
    """
    @abstractmethod
    def write(self, data: IntermediateData, output_path: str):
        """
        Consumes the IntermediateData object and writes the dataset
        to 'output_path'.
        """
        pass

# -----------------------------------------------------------------------------
#  NUSCENES WRITER
# -----------------------------------------------------------------------------

class NuScenesWriter(BaseWriter):
    """
    Writes data to the nuScenes dataset format.
    """
    
    def __init__(self):
        self.token_manager = None
        self.output_path = None
        self.annot_out_dir = None
        self.samples_out_dir = None
        self.sweeps_out_dir = None
        self.maps_out_dir = None  # <-- ADDED FOR MAPS

    def write(self, data: IntermediateData, output_path: str):
        log.info(f"Initializing NuScenesWriter for output to: {output_path}")
        self.output_path = os.path.abspath(output_path)
        
        # --- 1. Setup Output Directories (from IDD3DDataLoader) ---
        self.annot_out_dir = os.path.join(self.output_path, 'anotations')
        self.samples_out_dir = os.path.join(self.output_path, 'samples')
        self.sweeps_out_dir = os.path.join(self.output_path, 'sweeps')
        self.maps_out_dir = os.path.join(self.output_path, 'maps')  # <-- ADDED FOR MAPS

        os.makedirs(self.annot_out_dir, exist_ok=True)
        os.makedirs(self.samples_out_dir, exist_ok=True)
        os.makedirs(self.maps_out_dir, exist_ok=True)  # <-- ADDED FOR MAPS

        # --- 2. Initialize TokenManager (from build_..._pipeline) ---
        registry_path = os.path.join(self.annot_out_dir, 'token_registry.json')
        
        # Find the last timestamp to ensure new timestamps are sequential
        last_timestamp = self._get_last_timestamp()
        new_base_timestamp = (last_timestamp + 20_000_000) if last_timestamp else None # 20-sec gap
        
        self.token_manager = TokenTimestampManager(
            registry_path=registry_path,
            base_timestamp=new_base_timestamp
        )

        # --- 3. Run Writing Tasks in Order ---
        log.info("Writing JSON metadata files...")
        
        self._write_calibrated_sensor(data.calibrations)
        self._write_sensor(data.calibrations) # Uses calibration data
        
        # These stubs must be written first so tokens are available
        self._write_visibility()
        self._write_attribute()
        self._write_map()  # <-- ADDED FOR MAPS
        
        # Main data
        self._write_scene(data.scenes, data.samples)
        self._write_sample(data.samples)
        self._write_ego_pose(data.ego_poses)
        self._write_sample_data(data.sensor_data)
        
        # Annotations
        self._write_instance(data.instances)
        self._write_category(data.instances)
        self._write_sample_annotation(data.annotations, data.instances)
        
        # --- 4. Process Physical Files ---
        log.info("Converting and copying physical sensor files...")
        self._process_sensor_files(data.sensor_data, data.sequence_path)
        
        # --- 5. Duplicate Sweeps (from IDD3DDuplicateSweepsConverter) ---
        log.info("Duplicating 'samples' directory to 'sweeps'...")
        self._duplicate_sweeps()
        
        # --- 6. Save Token Registry (from IDD3DTokenRegistrySaver) ---
        log.info("Saving global token registry...")
        self.token_manager.save_registry(registry_path)
        
        log.info(f"--- NuScenes Write Complete ---")
        log.info(f"Output successfully written to: {self.output_path}")

    def _get_last_timestamp(self):
        """Finds the last timestamp from sample.json to ensure continuity."""
        sample_json_path = os.path.join(self.annot_out_dir, 'sample.json')
        last_timestamp = None
        if os.path.exists(sample_json_path):
            with json_file_lock:
                try:
                    with open(sample_json_path, 'r') as f:
                        samples = json.load(f)
                        if samples and isinstance(samples, list):
                            last_timestamp = samples[-1].get('timestamp')
                except Exception as e:
                    log.warning(f"Could not read last timestamp: {e}")
        
        if last_timestamp:
            log.info(f"Found existing data. Last timestamp: {last_timestamp}")
        return last_timestamp

    # --- JSON Writing Methods (called by write()) ---

    def _write_calibrated_sensor(self, calibrations):
        new_entries = []
        for if_calib in calibrations:
            new_entries.append({
                "token": self.token_manager.get_calibration_token(if_calib.sensor_name),
                "sensor_token": self.token_manager.get_sensor_token(if_calib.sensor_name),
                "translation": if_calib.translation,
                "rotation": if_calib.rotation,
                "camera_intrinsic": if_calib.camera_intrinsic
            })
        append_to_json_list(os.path.join(self.annot_out_dir, 'calibrated_sensor.json'), new_entries)

    def _write_sensor(self, calibrations):
        new_entries = []
        for if_calib in calibrations:
            is_camera = len(if_calib.camera_intrinsic) > 0
            new_entries.append({
                "token": self.token_manager.get_sensor_token(if_calib.sensor_name),
                "modality": "camera" if is_camera else "lidar",
                "channel": if_calib.sensor_name,
            })
        append_to_json_list(os.path.join(self.annot_out_dir, 'sensor.json'), new_entries)

    def _write_scene(self, scenes, samples):
        new_entries = []
        for if_scene in scenes:
            samples_in_scene = [s for s in samples if s.scene_name == if_scene.name]
            if not samples_in_scene:
                continue
            
            samples_in_scene.sort(key=lambda x: x.timestamp_us)
            
            new_entries.append({
                "token": self.token_manager.get_scene_token(),
                "log_token": "", # Stubbed log token
                "nbr_samples": len(samples_in_scene),
                "first_sample_token": self.token_manager.get_frame_token(samples_in_scene[0].temp_frame_id),
                "last_sample_token": self.token_manager.get_frame_token(samples_in_scene[-1].temp_frame_id),
                "name": if_scene.name,
                "description": if_scene.description
            })
        append_to_json_list(os.path.join(self.annot_out_dir, 'scene.json'), new_entries)

    def _write_sample(self, samples):
        new_entries = []
        sorted_samples = sorted(samples, key=lambda x: x.timestamp_us)
        
        for i, if_sample in enumerate(sorted_samples):
            prev_token = ""
            if i > 0 and sorted_samples[i-1].scene_name == if_sample.scene_name:
                prev_token = self.token_manager.get_frame_token(sorted_samples[i-1].temp_frame_id)
            
            next_token = ""
            if i < len(sorted_samples) - 1 and sorted_samples[i+1].scene_name == if_sample.scene_name:
                next_token = self.token_manager.get_frame_token(sorted_samples[i+1].temp_frame_id)
            
            new_entries.append({
                "token": self.token_manager.get_frame_token(if_sample.temp_frame_id),
                "timestamp": if_sample.timestamp_us,
                "prev": prev_token,
                "next": next_token,
                "scene_token": self.token_manager.get_scene_token() # Assumes one scene per run
            })
        append_to_json_list(os.path.join(self.annot_out_dir, 'sample.json'), new_entries)

    def _write_ego_pose(self, ego_poses):
        new_entries = []
        for if_pose in ego_poses:
            new_entries.append({
                "token": self.token_manager.get_ego_pose_token(if_pose.temp_frame_id),
                "timestamp": if_pose.timestamp_us,
                "translation": if_pose.translation,
                "rotation": if_pose.rotation
            })
        append_to_json_list(os.path.join(self.annot_out_dir, 'ego_pose.json'), new_entries)

    def _write_sample_data(self, sensor_data):
        new_entries = []
        
        sensor_groups = {}
        for sd in sensor_data:
            if sd.sensor_name not in sensor_groups:
                sensor_groups[sd.sensor_name] = []
            sensor_groups[sd.sensor_name].append(sd)

        for sensor_name, data_list in sensor_groups.items():
            sorted_data = sorted(data_list, key=lambda x: x.timestamp_us)
            
            for i, if_data in enumerate(sorted_data):
                sd_token = uuid.uuid4().hex # sample_data tokens are always unique
                is_camera = sensor_name.startswith("CAM_")
                
                # Filename logic
                if is_camera:
                    filename_base = os.path.splitext(os.path.basename(if_data.original_filename))[0]
                    output_filename = f"{filename_base}.jpg"
                else:
                    filename_base = os.path.splitext(if_data.original_filename)[0]
                    output_filename = f"{filename_base}.pcd.bin"

                new_entries.append({
                    "token": sd_token,
                    "sample_token": self.token_manager.get_frame_token(if_data.temp_frame_id),
                    "ego_pose_token": self.token_manager.get_ego_pose_token(if_data.temp_frame_id),
                    "calibrated_sensor_token": self.token_manager.get_calibration_token(if_data.sensor_name),
                    "filename": f"samples/{if_data.sensor_name}/{output_filename}", 
                    "fileformat": "jpg" if is_camera else "pcd.bin",
                    "width": 1440 if is_camera else 0,
                    "height": 1080 if is_camera else 0,
                    "timestamp": if_data.timestamp_us,
                    "is_key_frame": if_data.is_keyframe,
                    "next": "", # Stubbed for this simple append-only logic
                    "prev": ""  # Stubbed for this simple append-only logic
                })
        
        log.warning("sample_data.json 'prev' and 'next' tokens are not linked in this version.")
        append_to_json_list(os.path.join(self.annot_out_dir, 'sample_data.json'), new_entries)

    def _write_instance(self, instances):
        new_entries = []
        for if_inst in instances:
            new_entries.append({
                "token": self.token_manager.get_instance_token(if_inst.temp_instance_id),
                "category_token": self.token_manager.get_category_token(if_inst.category_name),
                "nbr_annotations": 0, # Stubbed
                "first_annotation_token": "", # Stubbed
                "last_annotation_token": ""   # Stubbed
            })
        log.warning("instance.json nbr_annotations, first/last tokens are not linked.")
        append_to_json_list(os.path.join(self.annot_out_dir, 'instance.json'), new_entries)
        
    def _write_category(self, instances):
        new_entries = []
        all_categories = {inst.category_name for inst in instances}
        
        for cat_name in all_categories:
            new_entries.append({
                "token": self.token_manager.get_category_token(cat_name),
                "name": cat_name,
                "description": f"{cat_name} category"
            })
        append_to_json_list(os.path.join(self.annot_out_dir, 'category.json'), new_entries)

    def _write_sample_annotation(self, annotations, instances):
        new_entries = []
        inst_cat_map = {inst.temp_instance_id: inst.category_name for inst in instances}
        
        for if_ann in annotations:
            category_name = inst_cat_map.get(if_ann.temp_instance_id, "")
            attribute_tokens = []
            if category_name.startswith('vehicle.'):
                attribute_tokens = [self.token_manager.get_category_token("vehicle.moving")]
            elif category_name.startswith('human.'):
                attribute_tokens = [self.token_manager.get_category_token("pedestrian.moving")]
            
            new_entries.append({
                "token": self.token_manager.generate_annotation_token(),
                "sample_token": self.token_manager.get_frame_token(if_ann.temp_frame_id),
                "instance_token": self.token_manager.get_instance_token(if_ann.temp_instance_id),
                "attribute_tokens": attribute_tokens,
                "visibility_token": self.token_manager.get_category_token("v4-0"), # Stubbed: 80-100%
                "translation": if_ann.translation,
                "size": if_ann.size,
                "rotation": if_ann.rotation,
                "prev": "", "next": "", # Stubbed
                "num_lidar_pts": 0, "num_radar_pts": 0 # Stubbed
            })
        log.warning("sample_annotation.json 'prev' and 'next' tokens are not linked.")
        append_to_json_list(os.path.join(self.annot_out_dir, 'sample_annotation.json'), new_entries)

    def _write_visibility(self):
        """Writes a static visibility.json file."""
        vis_levels = [
            {"level": "v1-0", "description": "visibility 0-40%"},
            {"level": "v2-0", "description": "visibility 40-60%"},
            {"level": "v3-0", "description": "visibility 60-80%"},
            {"level": "v4-0", "description": "visibility 80-100%"}
        ]
        new_entries = []
        for vis in vis_levels:
            new_entries.append({
                "token": self.token_manager.get_category_token(vis["level"]),
                "level": vis["level"],
                "description": vis["description"]
            })
        out_path = os.path.join(self.annot_out_dir, 'visibility.json')
        with json_file_lock:
             with open(out_path, 'w') as f:
                json.dump(new_entries, f, indent=2)
        log.info(f"Overwrote {out_path} with {len(new_entries)} entries.")

    def _write_attribute(self):
        """Writes a static attribute.json file."""
        attributes = [
            {"name": "vehicle.moving", "description": "Vehicle is moving (default stub)"},
            {"name": "pedestrian.moving", "description": "Pedestrian is moving (default stub)"},
        ]
        new_entries = []
        for attr in attributes:
            new_entries.append({
                "token": self.token_manager.get_category_token(attr["name"]),
                "name": attr["name"],
                "description": attr["description"]
            })
        out_path = os.path.join(self.annot_out_dir, 'attribute.json')
        with json_file_lock:
             with open(out_path, 'w') as f:
                json.dump(new_entries, f, indent=2)
        log.info(f"Overwote {out_path} with {len(new_entries)} entries.")

    def _write_map(self):
        """
        Writes a static map.json file and a dummy map image.
        This logic is from the old IDD3DMapConverter.
        """
        location = "Hyderabad"
        map_filename = f"maps/{location.lower()}.png"
        map_token = self.token_manager.get_category_token(f"map_{location}")
        
        new_map_entry = {
            "token": map_token,
            "log_tokens": [], # We stub this as empty
            "category": "semantic_prior",
            "filename": map_filename,
        }
        
        # --- 1. Write the map.json file (OVERWRITE, not append) ---
        out_path = os.path.join(self.annot_out_dir, 'map.json')
        with json_file_lock:
            try:
                # We always overwrite map.json with a list containing this one map
                with open(out_path, 'w') as f:
                    json.dump([new_map_entry], f, indent=2)
                log.info(f"Overwrote {out_path} with static map entry for {location}.")
            except Exception as e:
                log.error(f"FATAL: Could not write to map.json: {e}")
                raise
        
        # --- 2. Create the dummy hyderabad.png image ---
        image_path = os.path.join(self.maps_out_dir, f"{location.lower()}.png")
        if not os.path.exists(image_path):
            try:
                # Create a simple 10x10 black PNG
                img = Image.new('RGB', (10, 10), color='black')
                img.save(image_path, 'PNG')
                log.info(f"Created dummy map file: {image_path}")
            except Exception as e:
                log.error(f"Could not create dummy map image: {e}")

    # --- File Processing Methods (called by write()) ---
    
    def _process_sensor_files(self, sensor_data, sequence_path):
        """
        Loops through sensor data, finds original files,
        converts, and saves them.
        """
        num_lidar = 0
        num_camera = 0
        
        for sd in sensor_data:
            # 1. Find Source Path
            if sd.sensor_name == "LIDAR_TOP":
                src_file = os.path.join(sequence_path, 'lidar', sd.original_filename)
                
                # 2. Define Destination Path
                filename_base = os.path.splitext(sd.original_filename)[0]
                output_filename = f"{filename_base}.pcd.bin"
                dst_folder = os.path.join(self.samples_out_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                # 3. Convert
                convert_lidar_file(src_file, dst_file)
                num_lidar += 1
            
            else: # It's a camera
                # 'original_filename' is e.g., "cam0/00000.png"
                src_file = os.path.join(sequence_path, 'camera', sd.original_filename)
                
                # 2. Define Destination Path
                filename_base = os.path.splitext(os.path.basename(sd.original_filename))[0]
                output_filename = f"{filename_base}.jpg"
                dst_folder = os.path.join(self.samples_out_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                # 3. Convert
                convert_camera_file(src_file, dst_file)
                num_camera += 1
                
        log.info(f"Processed {num_lidar} LiDAR files and {num_camera} camera files.")

    def _duplicate_sweeps(self):
        """
        Deletes old 'sweeps' dir and copies 'samples' to it.
        """
        if os.path.exists(self.sweeps_out_dir):
            try:
                shutil.rmtree(self.sweeps_out_dir)
                log.info(f"Removed old 'sweeps' directory.")
            except Exception as e:
                log.error(f"Could not remove 'sweeps' directory: {e}")
                return # Stop if we can't remove it

        try:
            shutil.copytree(self.samples_out_dir, self.sweeps_out_dir)
            log.info(f"Successfully duplicated 'samples' to 'sweeps'.")
        except Exception as e:
            log.error(f"FATAL: Could not copy 'samples' to 'sweeps': {e}")
            raise
