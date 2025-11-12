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
from PIL import Image
from datetime import datetime
from intermediate_format import IntermediateData
from utils import TokenTimestampManager, append_to_json_list, json_file_lock, merge_and_overwrite_json_list

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# --- File Conversion Helpers (from old IDD3DLidar/Camera Converters) ---
# ... (convert_lidar_file and convert_camera_file functions remain unchanged) ...
def convert_lidar_file(src_path, dst_path):
    try:
        import numpy as np
        import open3d as o3d
        if not os.path.exists(src_path):
            log.warning(f"Source LiDAR file not found: {src_path}")
            open(dst_path, 'wb').close(); return
        pcd = o3d.io.read_point_cloud(src_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts = np.hstack((xyz, intensity))
        pts.astype(np.float32).tofile(dst_path)
    except ImportError:
        log.warning(f"open3d not available. Creating empty placeholder for {dst_path}")
        open(dst_path, 'wb').close()
    except Exception as e:
        log.error(f"Error converting {src_path}: {e}. Creating empty file.")
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

# -----------------------------------------------------------------------------
#  BASE WRITER
# -----------------------------------------------------------------------------
class BaseWriter(ABC):
    @abstractmethod
    def write(self, data: IntermediateData, output_path: str):
        pass

# -----------------------------------------------------------------------------
#  NUSCENES WRITER
# -----------------------------------------------------------------------------
class NuScenesWriter(BaseWriter):
    """
    Writes data to the nuScenes dataset format.
    Handles merging and linking data across multiple runs.
    """
    
    def __init__(self):
        self.token_manager = None
        self.output_path = None
        self.annot_out_dir = None
        self.samples_out_dir = None
        self.sweeps_out_dir = None
        self.maps_out_dir = None
        
        # --- NEW: Holders for cross-run data ---
        self.generated_log_tokens = []
        self.all_sample_annotations = []
        self.instance_db = {} # Holds all instance data

    def write(self, data: IntermediateData, output_path: str):
        log.info(f"Initializing NuScenesWriter for output to: {output_path}")
        self.output_path = os.path.abspath(output_path)
        
        # --- 1. Setup Output Directories ---
        self.annot_out_dir = os.path.join(self.output_path, 'anotations')
        self.samples_out_dir = os.path.join(self.output_path, 'samples')
        self.sweeps_out_dir = os.path.join(self.output_path, 'sweeps')
        self.maps_out_dir = os.path.join(self.output_path, 'maps')

        os.makedirs(self.annot_out_dir, exist_ok=True)
        os.makedirs(self.samples_out_dir, exist_ok=True)
        os.makedirs(self.maps_out_dir, exist_ok=True)

        # --- 2. Initialize TokenManager ---
        registry_path = os.path.join(self.annot_out_dir, 'token_registry.json')
        last_timestamp = self._get_last_timestamp()
        new_base_timestamp = (last_timestamp + 20_000_000) if last_timestamp else None # 20-sec gap
        
        self.token_manager = TokenTimestampManager(
            registry_path=registry_path,
            base_timestamp=new_base_timestamp
        )

        if not data.scenes:
            log.error("No scenes found in intermediate data. Cannot proceed."); return
        sequence_name = data.scenes[0].name
        log.info(f"Processing sequence: {sequence_name}")

        # --- 3. Run Writing Tasks in Order ---
        log.info("Writing JSON metadata files...")
        
        # --- Dictionary Files (Merge & Overwrite) ---
        self._write_sensor_and_calib(data.calibrations)
        self._write_visibility()
        self._write_attribute()
        
        # --- Log Files (Append) ---
        self._write_log(data.scenes)
        self._write_map() # Must be after _write_log
        self._write_file_manifest(data) # Can be appended

        # --- Linked-List Files (Read, Merge, Re-link, Overwrite) ---
        self._write_sample_and_ego_pose(data.samples, data.ego_poses)
        self._write_sample_data(data.sensor_data, sequence_name)
        
        # --- Annotation & Instance (Most complex linking) ---
        self._write_category(data.instances) # Must be before instance/ann
        self._write_instance_and_annotation(data.instances, data.annotations)
        
        # --- 4. Process Physical Files ---
        log.info("Converting and copying physical sensor files...")
        self._process_sensor_files(data.sensor_data, data.sequence_path, sequence_name)
        
        # --- 5. Duplicate Sweeps ---
        log.info("Duplicating 'samples' directory to 'sweeps'...")
        self._duplicate_sweeps()
        
        # --- 6. Save Token Registry ---
        log.info("Saving global token registry...")
        self.token_manager.save_registry(registry_path)
        
        log.info(f"--- NuScenes Write Complete ---")
        log.info(f"Output successfully written to: {self.output_path}")

    def _get_last_timestamp(self):
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
        if last_timestamp: log.info(f"Found existing data. Last timestamp: {last_timestamp}")
        return last_timestamp

    # --- JSON Writing Methods (called by write()) ---

    def _write_sensor_and_calib(self, calibrations):
        new_sensors = []
        new_calib_sensors = []
        
        for if_calib in calibrations:
            sensor_token = self.token_manager.get_sensor_token(if_calib.sensor_name)
            is_camera = len(if_calib.camera_intrinsic) > 0
            
            new_sensors.append({
                "token": sensor_token,
                "modality": "camera" if is_camera else "lidar",
                "channel": if_calib.sensor_name,
            })
            
            new_calib_sensors.append({
                "token": self.token_manager.get_calibration_token(if_calib.sensor_name),
                "sensor_token": sensor_token,
                "translation": if_calib.translation,
                "rotation": if_calib.rotation,
                "camera_intrinsic": if_calib.camera_intrinsic
            })
        
        # Use merge_and_overwrite to prevent duplicates
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'sensor.json'), 
            new_sensors, 
            key_field='channel'
        )
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'calibrated_sensor.json'), 
            new_calib_sensors, 
            key_field='sensor_token'
        )

    def _write_sample_and_ego_pose(self, samples, ego_poses):
        # --- Read Existing Data ---
        sample_path = os.path.join(self.annot_out_dir, 'sample.json')
        ego_pose_path = os.path.join(self.annot_out_dir, 'ego_pose.json')
        
        all_samples = []
        all_ego_poses = []

        with json_file_lock:
            if os.path.exists(sample_path):
                try: all_samples = json.load(open(sample_path, 'r'))
                except: log.warning("sample.json corrupted. Overwriting.")
            if os.path.exists(ego_pose_path):
                try: all_ego_poses = json.load(open(ego_pose_path, 'r'))
                except: log.warning("ego_pose.json corrupted. Overwriting.")
        
        # --- Add New Data ---
        for if_sample in samples:
            all_samples.append({
                "token": self.token_manager.get_frame_token(if_sample.temp_frame_id),
                "timestamp": if_sample.timestamp_us,
                "scene_token": self.token_manager.get_scene_token() # Assumes one scene per run
            })
        
        for if_pose in ego_poses:
            all_ego_poses.append({
                "token": self.token_manager.get_ego_pose_token(if_pose.temp_frame_id),
                "timestamp": if_pose.timestamp_us,
                "translation": if_pose.translation,
                "rotation": if_pose.rotation
            })

        # --- Sort and Re-link ---
        all_samples.sort(key=lambda x: x['timestamp'])
        all_ego_poses.sort(key=lambda x: x['timestamp'])
        
        scene_tokens = {s['scene_token'] for s in all_samples}
        final_samples = []
        
        for scene_token in scene_tokens:
            scene_samples = [s for s in all_samples if s['scene_token'] == scene_token]
            for i, sample in enumerate(scene_samples):
                sample['prev'] = scene_samples[i-1]['token'] if i > 0 else ""
                sample['next'] = scene_samples[i+1]['token'] if i < len(scene_samples) - 1 else ""
            final_samples.extend(scene_samples)
        
        # --- Overwrite Files ---
        with json_file_lock:
            json.dump(final_samples, open(sample_path, 'w'), indent=2)
            log.info(f"Merged and overwrote sample.json. Total items: {len(final_samples)}")
            json.dump(all_ego_poses, open(ego_pose_path, 'w'), indent=2)
            log.info(f"Merged and overwrote ego_pose.json. Total items: {len(all_ego_poses)}")
        
        # --- Add Scene (Append is OK here) ---
        if samples:
            new_scene = {
                "token": self.token_manager.get_scene_token(),
                "log_token": self.generated_log_tokens[-1] if self.generated_log_tokens else "",
                "nbr_samples": len(samples),
                "first_sample_token": self.token_manager.get_frame_token(samples[0].temp_frame_id),
                "last_sample_token": self.token_manager.get_frame_token(samples[-1].temp_frame_id),
                "name": samples[0].scene_name,
                "description": f"Scene {samples[0].scene_name}"
            }
            append_to_json_list(os.path.join(self.annot_out_dir, 'scene.json'), [new_scene])


    def _write_sample_data(self, sensor_data, sequence_name):
        sample_data_path = os.path.join(self.annot_out_dir, 'sample_data.json')
        
        all_sample_data = []
        with json_file_lock:
            if os.path.exists(sample_data_path):
                try: all_sample_data = json.load(open(sample_data_path, 'r'))
                except: log.warning("sample_data.json corrupted. Overwriting.")

        for if_data in sensor_data:
            sd_token = uuid.uuid4().hex
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
                "token": sd_token,
                "sample_token": self.token_manager.get_frame_token(if_data.temp_frame_id),
                "ego_pose_token": self.token_manager.get_ego_pose_token(if_data.temp_frame_id),
                "calibrated_sensor_token": self.token_manager.get_calibration_token(if_data.sensor_name),
                "filename": f"samples/{if_data.sensor_name}/{output_filename}",
                "fileformat": fileformat,
                "width": 1440 if is_camera else 0,
                "height": 1080 if is_camera else 0,
                "timestamp": if_data.timestamp_us,
                "is_key_frame": if_data.is_keyframe,
            })
        
        # --- Group, Sort, and Re-link ---
        sensor_groups = {}
        for sd in all_sample_data:
            token = sd['calibrated_sensor_token']
            if token not in sensor_groups: sensor_groups[token] = []
            sensor_groups[token].append(sd)

        final_sample_data = []
        for sensor_token, sd_list in sensor_groups.items():
            sorted_list = sorted(sd_list, key=lambda x: x['timestamp'])
            for i, sd in enumerate(sorted_list):
                sd['prev'] = sorted_list[i-1]['token'] if i > 0 else ""
                sd['next'] = sorted_list[i+1]['token'] if i < len(sorted_list) - 1 else ""
            final_sample_data.extend(sorted_list)
        
        with json_file_lock:
            json.dump(final_sample_data, open(sample_data_path, 'w'), indent=2)
            log.info(f"Merged and overwrote sample_data.json. Total items: {len(final_sample_data)}")


    def _write_category(self, instances):
        new_categories = []
        all_categories = {inst.category_name for inst in instances}
        for cat_name in all_categories:
            new_categories.append({
                "token": self.token_manager.get_category_token(cat_name),
                "name": cat_name,
                "description": f"{cat_name} category"
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'category.json'), 
            new_categories, 
            key_field='name'
        )

    def _write_instance_and_annotation(self, instances, annotations):
        instance_path = os.path.join(self.annot_out_dir, 'instance.json')
        ann_path = os.path.join(self.annot_out_dir, 'sample_annotation.json')

        # --- Read Existing Data ---
        with json_file_lock:
            try:
                all_anns = json.load(open(ann_path, 'r')) if os.path.exists(ann_path) else []
            except: all_anns = []; log.warning("sample_annotation.json corrupted.")
            
            try:
                inst_list = json.load(open(instance_path, 'r')) if os.path.exists(instance_path) else []
                inst_db = {i['token']: i for i in inst_list} # key by token
            except: inst_db = {}; log.warning("instance.json corrupted.")

        # --- Process New Data ---
        new_anns_by_inst_id = {} # key by temp_instance_id
        for ann in annotations:
            if ann.temp_instance_id not in new_anns_by_inst_id:
                new_anns_by_inst_id[ann.temp_instance_id] = []
            new_anns_by_inst_id[ann.temp_instance_id].append(ann)

        inst_name_map = {inst.temp_instance_id: inst.category_name for inst in instances}

        for temp_inst_id, new_anns_list in new_anns_by_inst_id.items():
            inst_token = self.token_manager.get_instance_token(temp_inst_id)
            
            # Sort new annotations
            new_anns_list.sort(key=lambda x: x.timestamp_us)
            
            # Find link to existing data
            last_ann_token_from_existing = ""
            if inst_token in inst_db:
                last_ann_token_from_existing = inst_db[inst_token]['last_annotation_token']

            generated_tokens = [self.token_manager.generate_annotation_token() for _ in new_anns_list]
            
            for i, if_ann in enumerate(new_anns_list):
                category_name = inst_name_map.get(temp_inst_id, "")
                attribute_tokens = []
                if category_name.startswith('vehicle.'):
                    attribute_tokens = [self.token_manager.get_category_token("vehicle.moving")]
                elif category_name.startswith('human.'):
                    attribute_tokens = [self.token_manager.get_category_token("pedestrian.moving")]

                ann_token = generated_tokens[i]
                prev_token = generated_tokens[i-1] if i > 0 else last_ann_token_from_existing
                next_token = generated_tokens[i+1] if i < len(generated_tokens) - 1 else ""
                
                all_anns.append({
                    "token": ann_token,
                    "sample_token": self.token_manager.get_frame_token(if_ann.temp_frame_id),
                    "instance_token": inst_token,
                    "attribute_tokens": attribute_tokens,
                    "visibility_token": self.token_manager.get_category_token("v4-0"),
                    "translation": if_ann.translation,
                    "size": if_ann.size,
                    "rotation": if_ann.rotation,
                    "prev": prev_token, "next": next_token,
                    "num_lidar_pts": 0, "num_radar_pts": 0
                })

            # --- Create / Update Instance DB Entry ---
            if inst_token not in inst_db:
                inst_db[inst_token] = {
                    "token": inst_token,
                    "category_token": self.token_manager.get_category_token(inst_name_map.get(temp_inst_id, "")),
                    "nbr_annotations": len(generated_tokens),
                    "first_annotation_token": generated_tokens[0],
                    "last_annotation_token": generated_tokens[-1]
                }
            else:
                inst_db[inst_token]["nbr_annotations"] += len(generated_tokens)
                inst_db[inst_token]["last_annotation_token"] = generated_tokens[-1]
        
        # --- Overwrite Files ---
        with json_file_lock:
            json.dump(list(inst_db.values()), open(instance_path, 'w'), indent=2)
            log.info(f"Merged and overwrote instance.json. Total items: {len(inst_db)}")
            json.dump(all_anns, open(ann_path, 'w'), indent=2)
            log.info(f"Merged and overwrote sample_annotation.json. Total items: {len(all_anns)}")


    def _write_visibility(self):
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
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'visibility.json'),
            new_entries,
            key_field='level'
        )

    def _write_attribute(self):
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
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'attribute.json'),
            new_entries,
            key_field='name'
        )

    def _write_map(self):
        location = "Hyderabad"
        map_filename = f"maps/{location.lower()}.png"
        map_token = self.token_manager.get_category_token(f"map_{location}")
        
        # Link all logs generated so far
        new_map_entry = {
            "token": map_token,
            "log_tokens": self.generated_log_tokens, # <-- FIXED
            "category": "semantic_prior",
            "filename": map_filename,
        }
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'map.json'),
            [new_map_entry],
            key_field='token'
        )
        
        log.info(f"User is responsible for adding '{location.lower()}.png' to the '{self.maps_out_dir}' directory.")


    def _write_log(self, scenes):
        new_entries = []
        for if_scene in scenes:
            logfile = f"{if_scene.name}-{datetime.now().strftime('%Y-%m-%d')}"
            log_token = self.token_manager.get_category_token(f"log_{logfile}") 
            
            # Add to list for map.json
            self.generated_log_tokens.append(log_token) # <-- FIXED
            
            new_entries.append({
                "token": log_token,
                "logfile": logfile,
                "vehicle": "stub_vehicle",
                "date_captured": datetime.now().strftime('%Y-%m-%d'),
                "location": "Hyderabad"
            })
        
        # Logs can be appended, but merging is safer
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'log.json'),
            new_entries,
            key_field='token'
        )

    def _write_file_manifest(self, data: IntermediateData):
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
                "sample_token": self.token_manager.get_frame_token(frame_id),
                "sensors": []
            }
            if frame_id not in frame_to_sensor_data: continue
            for sd in frame_to_sensor_data[frame_id]:
                timestamp = sd.timestamp_us
                output_filename_base = f"{sequence_name}_frame_{timestamp}"
                if sd.sensor_name.startswith("CAM_"):
                    output_filename = f"{output_filename_base}.jpg"
                    source_file = f"{sequence_name}/camera/{sd.original_filename}"
                else: # LIDAR_TOP
                    output_filename = f"{output_filename_base}.pcd.bin"
                    source_file = f"{sequence_name}/lidar/{sd.original_filename}"
                manifest_entry["sensors"].append({
                    "channel": sd.sensor_name,
                    "source_file": source_file,
                    "output_file": f"samples/{sd.sensor_name}/{output_filename}"
                })
            new_entries.append(manifest_entry)
            
        # Manifest can be safely appended
        append_to_json_list(os.path.join(self.annot_out_dir, 'file_manifest.json'), new_entries)


    # --- File Processing Methods (called by write()) ---
    
    def _process_sensor_files(self, sensor_data, sequence_path, sequence_name):
        num_lidar = 0
        num_camera = 0
        
        for sd in sensor_data:
            timestamp = sd.timestamp_us
            output_filename_base = f"{sequence_name}_frame_{timestamp}"
            
            if sd.sensor_name == "LIDAR_TOP":
                src_file = os.path.join(sequence_path, 'lidar', sd.original_filename)
                output_filename = f"{output_filename_base}.pcd.bin"
                dst_folder = os.path.join(self.samples_out_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                # Only convert if it doesn't already exist
                if not os.path.exists(dst_file):
                    convert_lidar_file(src_file, dst_file)
                    num_lidar += 1
            
            else: # It's a camera
                src_file = os.path.join(sequence_path, 'camera', sd.original_filename)
                output_filename = f"{output_filename_base}.jpg"
                dst_folder = os.path.join(self.samples_out_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                # Only convert if it doesn't already exist
                if not os.path.exists(dst_file):
                    convert_camera_file(src_file, dst_file)
                    num_camera += 1
                
        log.info(f"Processed {num_lidar} new LiDAR files and {num_camera} new camera files.")

    def _duplicate_sweeps(self):
        if os.path.exists(self.sweeps_out_dir):
            try: shutil.rmtree(self.sweeps_out_dir)
            except Exception as e:
                log.error(f"Could not remove 'sweeps' directory: {e}")
                return
        try:
            shutil.copytree(self.samples_out_dir, self.sweeps_out_dir)
            log.info(f"Successfully duplicated 'samples' to 'sweeps'.")
        except Exception as e:
            log.error(f"FATAL: Could not copy 'samples' to 'sweeps': {e}")
            raise
