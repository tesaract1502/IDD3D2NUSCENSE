# readers.py
# ----------------------
# This file contains "Reader" classes.
# Each Reader is responsible for converting a specific dataset format
# (like IDD3D) INTO the common 'IntermediateData' format.
#
# Readers are STATELESS and format-agnostic - they only know about
# their source format and the intermediate representation.
# ----------------------

import os
import json
import logging
from abc import ABC, abstractmethod
from intermediate_format import *

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


class BaseReader(ABC):
    """
    Abstract base class for all dataset readers.
    
    Readers should be STATELESS - they convert one sequence at a time
    and don't maintain any state between calls.
    """
    
    @abstractmethod
    def read(self, sequence_path: str) -> IntermediateData:
        """
        Reads a dataset sequence from 'sequence_path' and returns
        a populated IntermediateData object.
        
        Args:
            sequence_path: Absolute path to the sequence directory
            
        Returns:
            IntermediateData object with all parsed data
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If data format is invalid
        """
        pass
    
    @abstractmethod
    def validate(self, sequence_path: str) -> dict:
        """
        Validates whether a directory is a valid sequence for this reader.
        
        Args:
            sequence_path: Path to check
            
        Returns:
            Dictionary with keys:
                - 'valid': bool
                - 'error': str (if valid=False)
                - 'warnings': list (optional)
                - 'info': dict (optional metadata)
        """
        pass


# -----------------------------------------------------------------------------
#  IDD3D READER
# -----------------------------------------------------------------------------

class Idd3dReader(BaseReader):
    """
    Reads data from the IDD3D dataset format.
    
    Expected structure:
        sequence_path/
            annot_data.json       # Frame metadata
            label/                # Per-frame object annotations
                00000.json
                00200.json
                ...
            lidar/                # LiDAR point clouds
                00000.pcd
                00200.pcd
                ...
            camera/               # Camera images
                cam0/
                    00000.png
                    ...
                cam1/
                ...
    """
    
    # IDD3D uses 10Hz frame rate
    FRAME_RATE_HZ = 10
    BASE_TIMESTAMP_US = 1640995200000000  # Jan 1, 2022, 00:00:00 UTC
    
    def __init__(self):
        # Map IDD3D categories to standard category names
        # These match the official category tokens from category2.json
        self.idd3d_to_standard_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle',
            'Person': 'movable_object.pedestrian',
            'Auto': 'movable_object.van',
            'Rider': 'movable_object.bicyclerider',
            'Animal': 'movable_object.unknown',
            'TrafficLight': 'movable_object.unknown',
            'TrafficSign': 'movable_object.unknown',
            'Pole': 'movable_object.unknown',
            'OtherVehicle': 'movable_object.unknown',
            'Misc': 'movable_object.unknown'
        }
        
        # Map IDD3D camera names to standard names
        self.idd3d_to_standard_cameras = {
            "cam0": "CAM_FRONT_LEFT",
            "cam1": "CAM_BACK_RIGHT",
            "cam2": "CAM_FRONT_RIGHT",
            "cam3": "CAM_FRONT",
            "cam4": "CAM_BACK_LEFT",
            "cam5": "CAM_BACK"
        }
        
        self.LIDAR_CHANNEL = "LIDAR_TOP"
        
        # IDD3D camera intrinsics (1440x1080 resolution)
        # fx=2916, fy=2916, cx=720, cy=540
        self.CAMERA_INTRINSIC = [
            [2916.0, 0.0, 720.0],
            [0.0, 2916.0, 540.0],
            [0.0, 0.0, 1.0]
        ]

    def validate(self, sequence_path: str) -> dict:
        """
        Validates an IDD3D sequence directory.
        """
        sequence_path = os.path.abspath(sequence_path)
        
        # Check if directory exists
        if not os.path.isdir(sequence_path):
            return {
                'valid': False,
                'error': f'Not a directory: {sequence_path}'
            }
        
        # Check for required files
        annot_json_path = os.path.join(sequence_path, 'annot_data.json')
        if not os.path.exists(annot_json_path):
            return {
                'valid': False,
                'error': f'Missing annot_data.json in {sequence_path}'
            }
        
        # Check for required directories
        required_dirs = ['label', 'lidar', 'camera']
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = os.path.join(sequence_path, dir_name)
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            return {
                'valid': False,
                'error': f'Missing directories: {", ".join(missing_dirs)}'
            }
        
        # Count files
        label_dir = os.path.join(sequence_path, 'label')
        lidar_dir = os.path.join(sequence_path, 'lidar')
        
        label_count = len([f for f in os.listdir(label_dir) if f.endswith('.json')])
        lidar_count = len([f for f in os.listdir(lidar_dir) if f.endswith('.pcd')])
        
        return {
            'valid': True,
            'info': {
                'sequence_name': os.path.basename(sequence_path),
                'label_files': label_count,
                'lidar_files': lidar_count
            }
        }

    def read(self, sequence_path: str) -> IntermediateData:
        """
        Reads an IDD3D sequence and returns it in the IntermediateData format.
        
        Args:
            sequence_path: Absolute path to the IDD3D sequence directory
            
        Returns:
            IntermediateData object containing all parsed data
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If data format is invalid
        """
        log.info(f"Reading IDD3D sequence: {sequence_path}")
        
        # Validate first
        validation = self.validate(sequence_path)
        if not validation['valid']:
            raise FileNotFoundError(validation['error'])
        
        sequence_path = os.path.abspath(sequence_path)
        sequence_name = os.path.basename(sequence_path)
        
        # Define input paths
        annot_json_path = os.path.join(sequence_path, 'annot_data.json')
        label_dir = os.path.join(sequence_path, 'label')
        
        # Load main annotation file
        try:
            with open(annot_json_path, 'r') as f:
                annot_data = json.load(f)
            frame_ids = sorted(annot_data.keys())
            log.info(f"Found {len(frame_ids)} frames in annot_data.json")
        except Exception as e:
            raise ValueError(f"Failed to read {annot_json_path}: {e}")
        
        # Initialize the intermediate data object
        data = IntermediateData(sequence_path=sequence_path)
        
        # --- 1. Populate Scene ---
        data.scenes.append(IFScene(
            name=sequence_name,
            description=f"IDD3D sequence {sequence_name}"
        ))
        
        # --- 2. Populate Calibrations ---
        # LIDAR calibration
        data.calibrations.append(IFCalibration(
            sensor_name=self.LIDAR_CHANNEL,
            translation=[0.0, 0.0, 1.8],  # Stubbed - 1.8m height
            rotation=[1.0, 0.0, 0.0, 0.0],  # Identity quaternion
            camera_intrinsic=[]
        ))
        
        # Camera calibrations
        for standard_cam_name in self.idd3d_to_standard_cameras.values():
            data.calibrations.append(IFCalibration(
                sensor_name=standard_cam_name,
                translation=[0.0, 0.0, 1.6],  # Stubbed - 1.6m height
                rotation=[1.0, 0.0, 0.0, 0.0],  # Identity quaternion
                camera_intrinsic=self.CAMERA_INTRINSIC
            ))
        
        # --- 3. Loop through frames to populate remaining data ---
        frame_interval_us = int(1_000_000 / self.FRAME_RATE_HZ)  # 100,000 us for 10Hz
        instance_tracker = set()  # Track unique object IDs
        
        for i, frame_id in enumerate(frame_ids):
            timestamp = self.BASE_TIMESTAMP_US + (i * frame_interval_us)
            
            # --- Populate Sample ---
            data.samples.append(IFSample(
                temp_frame_id=frame_id,
                timestamp_us=timestamp,
                scene_name=sequence_name
            ))
            
            # --- Populate EgoPose ---
            data.ego_poses.append(IFEgoPose(
                temp_frame_id=frame_id,
                timestamp_us=timestamp,
                translation=[0.0, 0.0, 0.0],  # Stubbed
                rotation=[1.0, 0.0, 0.0, 0.0]  # Stubbed - identity quaternion
            ))
            
            # --- Populate SensorData ---
            # LIDAR
            data.sensor_data.append(IFSensorData(
                temp_frame_id=frame_id,
                sensor_name=self.LIDAR_CHANNEL,
                original_filename=f"{frame_id}.pcd",
                timestamp_us=timestamp,
                is_keyframe=True
            ))
            
            # CAMERAS
            for idd_cam, standard_cam in self.idd3d_to_standard_cameras.items():
                data.sensor_data.append(IFSensorData(
                    temp_frame_id=frame_id,
                    sensor_name=standard_cam,
                    original_filename=f"{idd_cam}/{frame_id}.png",
                    timestamp_us=timestamp,
                    is_keyframe=True
                ))
            
            # --- Populate Annotations & Instances ---
            label_path = os.path.join(label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                log.warning(f"Label file not found: {label_path}")
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = str(obj.get("obj_id"))
                    obj_type = obj.get("obj_type")
                    
                    if not obj_id or not obj_type:
                        continue
                    
                    # --- Populate Instance (once per unique obj_id) ---
                    if obj_id not in instance_tracker:
                        category_name = self.idd3d_to_standard_categories.get(
                            obj_type, 
                            'movable_object.unknown'  # Default fallback
                        )
                        
                        data.instances.append(IFInstance(
                            temp_instance_id=obj_id,
                            category_name=category_name
                        ))
                        instance_tracker.add(obj_id)
                    
                    # --- Populate Annotation ---
                    psr = obj.get("psr", {})
                    pos = psr.get("position", {})
                    rot = psr.get("rotation", {})
                    scl = psr.get("scale", {})
                    
                    translation = [
                        pos.get("x", 0.0),
                        pos.get("y", 0.0),
                        pos.get("z", 0.0)
                    ]
                    
                    size = [
                        scl.get("x", 1.0),
                        scl.get("y", 1.0),
                        scl.get("z", 1.0)
                    ]
                    
                    # IDD3D provides Euler angles, but we store as quaternion
                    # For now, use identity quaternion (stub)
                    rotation_quat = [1.0, 0.0, 0.0, 0.0]
                    
                    # No attributes in IDD3D source data
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
        
        # --- Log summary ---
        log.info("=" * 50)
        log.info("IDD3D Read Complete")
        log.info("=" * 50)
        log.info(f"Sequence:       {sequence_name}")
        log.info(f"Scenes:         {len(data.scenes)}")
        log.info(f"Samples:        {len(data.samples)}")
        log.info(f"Instances:      {len(data.instances)}")
        log.info(f"Annotations:    {len(data.annotations)}")
        log.info(f"SensorData:     {len(data.sensor_data)}")
        log.info(f"EgoPoses:       {len(data.ego_poses)}")
        log.info(f"Calibrations:   {len(data.calibrations)}")
        log.info("=" * 50)
        
        return data


# -----------------------------------------------------------------------------
#  PLACEHOLDER FOR FUTURE READERS
# -----------------------------------------------------------------------------

# class ArgoverseReader(BaseReader):
#     """Reads Argoverse 2 dataset format."""
#     def read(self, sequence_path: str) -> IntermediateData:
#         pass
#
# class KittiReader(BaseReader):
#     """Reads KITTI dataset format."""
#     def read(self, sequence_path: str) -> IntermediateData:
#         pass
#
# class WaymoReader(BaseReader):
#     """Reads Waymo Open Dataset format."""
#     def read(self, sequence_path: str) -> IntermediateData:
#         pass
