# readers.py
# ----------------------
# This file contains "Reader" classes.
# Each Reader is responsible for converting a specific dataset format
# (like IDD3D) INTO the common 'IntermediateData' format.
# ----------------------

import os
import json
import logging
from abc import ABC, abstractmethod
from intermediate_format import * # Import all our new classes

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

class BaseReader(ABC):
    """
    Abstract base class for all dataset readers.
    """
    @abstractabstractmethod
    def read(self, sequence_path: str) -> IntermediateData:
        """
        Reads a dataset sequence from 'sequence_path' and returns
        a populated IntermediateData object.
        """
        pass

# -----------------------------------------------------------------------------
#  IDD3D READER
# -----------------------------------------------------------------------------

class Idd3dReader(BaseReader):
    """
    Reads data from the IDD3D dataset format.
    """
    
    def __init__(self):
        # --- THIS MAPPING IS NOW UPDATED ---
        # It maps IDD3D categories to the valid names from category2.json
        self.idd3d_to_nuscenes_categories = {
            # Valid mappings from category2.json
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle',
            'Person': 'movable_object.pedestrian', # Was human.pedestrian.adult
            
            # Guesses for other valid mappings
            'Auto': 'movable_object.van', # Map Auto-rickshaw to 'van'
            'Rider': 'movable_object.bicyclerider', # Map generic 'Rider'
            'MotorcyleRider': 'vehicle.motorcycle', # Rider *on* a motorcycle
            
            # Mappings to UNKNOWN (since they aren't in category2.json)
            'Animal': 'movable_object.unknown',
            'TrafficLight': 'movable_object.unknown', # Was static_object.traffic_light
            'TrafficSign': 'movable_object.unknown', # Was static_object.traffic_sign
            'Pole': 'movable_object.unknown', # Was static_object.pole
            'OtherVehicle': 'movable_object.unknown', # Was vehicle.other
            'Misc': 'movable_object.unknown' # Was movable_object.debris
        }
        
        # This mapping is moved from the old IDD3DDataLoader
        self.IDD3D_TO_NUSCENES_CAM_MAP = {
            "cam0": "CAM_FRONT_LEFT",
            "cam1": "CAM_BACK_RIGHT",
            "cam2": "CAM_FRONT_RIGHT",
            "cam3": "CAM_FRONT",
            "cam4": "CAM_BACK_LEFT",
            "cam5": "CAM_BACK"
        }
        
        self.LIDAR_CHANNEL = "LIDAR_TOP"
        
        # Based on 1440x1080 resolution
        # fx=2916, fy=2916, cx=720, cy=540
        self.CAMERA_INTRINSIC = [
            [2916.0, 0.0, 720.0],
            [0.0, 2916.0, 540.0],
            [0.0, 0.0, 1.0]
        ]

    def read(self, sequence_path: str) -> IntermediateData:
        """
        Reads a specific IDD3D sequence and returns it in the
        IntermediateData format.
        """
        log.info(f"Initializing IDD3D reader for sequence: {sequence_path}")
        
        sequence_path = os.path.abspath(sequence_path)
        sequence_name = os.path.basename(sequence_path)
        
        # --- Define input paths (logic from old IDD3DDataLoader) ---
        annot_json_path = os.path.join(sequence_path, 'annot_data.json')
        label_dir = os.path.join(sequence_path, 'label')
        
        # --- Validation (logic from old IDD3DDataLoader.validate) ---
        if not os.path.exists(annot_json_path):
            log.error(f"annot_data.json not found at: {annot_json_path}")
            raise FileNotFoundError(f"annot_data.json not found at: {annot_json_path}")
        if not os.path.exists(label_dir):
            log.error(f"label directory not found at: {label_dir}")
            raise FileNotFoundError(f"label directory not found at: {label_dir}")
            
        # --- Load main annotation file (logic from old read_annotations) ---
        try:
            with open(annot_json_path, 'r') as f:
                annot_data = json.load(f)
            frame_ids = sorted(annot_data.keys())
            log.info(f"Found {len(frame_ids)} frames in annot_data.json.")
        except Exception as e:
            log.error(f"Failed to read or parse {annot_json_path}: {e}")
            raise
            
        # This is the main object we will build and return
        data = IntermediateData(sequence_path=sequence_path)
        
        # --- 1. Populate Scene (logic from IDD3DSceneConverter) ---
        data.scenes.append(IFScene(
            name=sequence_name,
            description=f"IDD3D sequence {sequence_name}"
        ))

        # --- 2. Populate Calibrations (logic from IDD3DCalibConverter) ---
        # LIDAR
        data.calibrations.append(IFCalibration(
            sensor_name=self.LIDAR_CHANNEL,
            translation=[0.0, 0.0, 1.8], # Stubbed
            rotation=[1.0, 0.0, 0.0, 0.0], # Identity quaternion
            camera_intrinsic=[]
        ))
        # CAMERAS
        for nu_cam_name in self.IDD3D_TO_NUSCENES_CAM_MAP.values():
            data.calibrations.append(IFCalibration(
                sensor_name=nu_cam_name,
                translation=[0.0, 0.0, 1.6], # Stubbed
                rotation=[1.0, 0.0, 0.0, 0.0],
                camera_intrinsic=self.CAMERA_INTRINSIC
            ))
            
        # --- 3. Loop frames to populate Samples, Poses, Data, Annotations ---
        
        # Readers are responsible for timestamps.
        # We'll use the old logic: 10Hz frequency.
        base_timestamp = 1640995200000000  # A fixed start time
        frame_interval_us = int(1_000_000 / 10) # 10Hz = 100,000 us
        
        instance_tracker = set() # To track unique instances
        
        for i, frame_id in enumerate(frame_ids):
            timestamp = base_timestamp + (i * frame_interval_us)
            
            # --- Populate Sample (from IDD3DSampleConverter) ---
            data.samples.append(IFSample(
                temp_frame_id=frame_id,
                timestamp_us=timestamp,
                scene_name=sequence_name
            ))
            
            # --- Populate EgoPose (from IDD3DEgoPoseConverter) ---
            data.ego_poses.append(IFEgoPose(
                temp_frame_id=frame_id,
                timestamp_us=timestamp,
                translation=[0.0, 0.0, 0.0], # Stubbed
                rotation=[1.0, 0.0, 0.0, 0.0]  # Stubbed
            ))

            # --- Populate SensorData (from IDD3DSampleDataConverter) ---
            # LIDAR
            data.sensor_data.append(IFSensorData(
                temp_frame_id=frame_id,
                sensor_name=self.LIDAR_CHANNEL,
                original_filename=f"{frame_id}.pcd",
                timestamp_us=timestamp,
                is_keyframe=True
            ))
            # CAMERAS
            for idd_cam, nu_cam in self.IDD3D_TO_NUSCENES_CAM_MAP.items():
                data.sensor_data.append(IFSensorData(
                    temp_frame_id=frame_id,
                    sensor_name=nu_cam,
                    original_filename=f"{idd_cam}/{frame_id}.png",
                    timestamp_us=timestamp,
                    is_keyframe=True
                ))
                
            # --- Populate Annotations & Instances ---
            # (from IDD3DSampleAnnotationConverter & IDD3DInstanceConverter)
            label_path = os.path.join(label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                log.warning(f"Label file not found, skipping annotations for frame: {frame_id}")
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = str(obj.get("obj_id")) # Ensure ID is a string
                    obj_type = obj.get("obj_type")
                    if not obj_id or not obj_type:
                        continue
                        
                    # --- Populate Instance (from IDD3DInstanceConverter) ---
                    if obj_id not in instance_tracker:
                        # Use the new, corrected mapping
                        category_name = self.idd3d_to_nuscenes_categories.get(
                            obj_type, 'movable_object.unknown' # Default to unknown
                        )
                        data.instances.append(IFInstance(
                            temp_instance_id=obj_id,
                            category_name=category_name
                        ))
                        instance_tracker.add(obj_id)
                        
                    # --- Populate Annotation (from IDD3DSampleAnnotation) ---
                    psr = obj.get("psr", {})
                    pos = psr.get("position", {})
                    rot = psr.get("rotation", {})
                    scl = psr.get("scale", {})
                    
                    translation = [pos.get("x",0), pos.get("y",0), pos.get("z",0)]
                    size = [scl.get("x",1), scl.get("y",1), scl.get("z",1)]
                    
                    # Convert Euler (x,y,z) to Quaternion (w,x,y,z)
                    rotation_quat = [1.0, 0.0, 0.0, 0.0] # Stubbed
                    
                    # Stub attributes
                    attributes = []
                    
                    data.annotations.append(IFAnnotation(
                        temp_instance_id=obj_id,
                        temp_frame_id=frame_id,
                        timestamp_us=timestamp,
                        translation=translation,
                        size=size,
                        rotation=rotation_quat,
                        attributes=attributes
                    ))
            
            except Exception as e:
                log.error(f"Error processing label file {label_path}: {e}")

        log.info(f"--- IDD3D Read Complete ---")
        log.info(f"Scenes:         {len(data.scenes)}")
        log.info(f"Samples:        {len(data.samples)}")
        log.info(f"Instances:      {len(data.instances)}")
        log.info(f"Annotations:    {len(data.annotations)}")
        log.info(f"SensorData:     {len(data.sensor_data)}")
        log.info(f"EgoPoses:       {len(data.ego_poses)}")
        log.info(f"Calibrations:   {len(data.calibrations)}")
        log.info(f"---------------------------")
        
        return data
