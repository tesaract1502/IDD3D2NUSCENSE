# intermediate_format.py
# ----------------------
# This file defines the "Intermediate Format" (IF) for data exchange.
#
# ARCHITECTURE:
# - READERS (e.g., Idd3dReader, ArgoverseReader) CREATE instances of IntermediateData
# - WRITERS (e.g., NuScenesWriter, KittiWriter) CONSUME instances of IntermediateData
#
# DESIGN PRINCIPLES:
# 1. Format-agnostic: No dataset-specific concepts (tokens, file formats, etc.)
# 2. Self-contained: All necessary information is included
# 3. Standardized: Uses common category names, coordinate systems, units
# 4. Temporary IDs: 'temp_..._id' fields are Reader-assigned identifiers
#    that Writers convert to their own token/ID systems
#
# COORDINATE SYSTEM:
# - Translation: [x, y, z] in meters
# - Rotation: Quaternions [w, x, y, z] (Hamilton convention)
# - Timestamps: Microseconds (int64)
# ----------------------

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class IFCalibration:
    """
    Defines the calibration (extrinsics and intrinsics) for a single sensor.
    
    Extrinsics define the sensor's pose relative to the ego vehicle origin.
    Intrinsics (for cameras) define the projection from 3D to 2D.
    """
    sensor_name: str                      # Sensor channel/name (e.g., "CAM_FRONT", "LIDAR_TOP")
    translation: List[float]              # Extrinsic translation [x, y, z] in meters
    rotation: List[float]                 # Extrinsic rotation (quaternion [w, x, y, z])
    camera_intrinsic: List[List[float]] = field(default_factory=list)  # 3x3 matrix for cameras
    
    def __post_init__(self):
        """Validate calibration data."""
        assert len(self.translation) == 3, "Translation must be [x, y, z]"
        assert len(self.rotation) == 4, "Rotation must be quaternion [w, x, y, z]"
        if self.camera_intrinsic:
            assert len(self.camera_intrinsic) == 3, "Camera intrinsic must be 3x3 matrix"
            assert all(len(row) == 3 for row in self.camera_intrinsic), "Each row must have 3 elements"


@dataclass
class IFEgoPose:
    """
    Defines the ego vehicle's pose at a specific timestamp.
    
    This represents the vehicle's position and orientation in the global/world
    coordinate frame at a particular moment in time.
    """
    temp_frame_id: str                    # Temporary frame ID (Reader-assigned)
    timestamp_us: int                     # Timestamp in microseconds
    translation: List[float]              # [x, y, z] in global coordinates (meters)
    rotation: List[float]                 # Quaternion [w, x, y, z] in global coordinates
    
    def __post_init__(self):
        """Validate ego pose data."""
        assert len(self.translation) == 3, "Translation must be [x, y, z]"
        assert len(self.rotation) == 4, "Rotation must be quaternion [w, x, y, z]"
        assert self.timestamp_us >= 0, "Timestamp must be non-negative"


@dataclass
class IFInstance:
    """
    Defines a unique object instance (a tracked object across multiple frames).
    
    An instance represents a single real-world object that may appear in
    multiple frames with different annotations.
    """
    temp_instance_id: str                 # Reader's unique ID for this track (e.g., "obj_123")
    category_name: str                    # Standardized category (e.g., "vehicle.car", "movable_object.pedestrian")
    
    # Optional: Additional metadata about the instance
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate instance data."""
        assert self.temp_instance_id, "Instance ID cannot be empty"
        assert self.category_name, "Category name cannot be empty"


@dataclass
class IFAnnotation:
    """
    Defines a single 3D bounding box annotation for one instance in one frame.
    
    This represents where an object is located, its size, orientation, and
    any additional attributes at a specific point in time.
    """
    temp_instance_id: str                 # ID of the instance being annotated
    temp_frame_id: str                    # ID of the frame this annotation belongs to
    timestamp_us: int                     # Timestamp in microseconds
    translation: List[float]              # Center position [x, y, z] in global/ego frame (meters)
    size: List[float]                     # Bounding box size [width, length, height] in meters
    rotation: List[float]                 # Orientation quaternion [w, x, y, z]
    attributes: List[str] = field(default_factory=list)  # Standardized attributes (e.g., "vehicle.moving")
    
    # Optional fields
    num_lidar_pts: int = 0                # Number of LiDAR points inside the bbox
    num_radar_pts: int = 0                # Number of radar points inside the bbox
    visibility: float = 1.0               # Visibility score [0.0-1.0], where 1.0 = fully visible
    
    def __post_init__(self):
        """Validate annotation data."""
        assert len(self.translation) == 3, "Translation must be [x, y, z]"
        assert len(self.size) == 3, "Size must be [width, length, height]"
        assert len(self.rotation) == 4, "Rotation must be quaternion [w, x, y, z]"
        assert self.timestamp_us >= 0, "Timestamp must be non-negative"
        assert 0.0 <= self.visibility <= 1.0, "Visibility must be in range [0.0, 1.0]"


@dataclass
class IFSensorData:
    """
    Represents a single piece of sensor data (e.g., one LiDAR scan, one camera image).
    
    This links a sensor reading to a specific frame and provides the information
    needed by the Writer to locate and convert the physical file.
    """
    temp_frame_id: str                    # Frame ID this sensor data belongs to
    sensor_name: str                      # Sensor channel (e.g., "LIDAR_TOP", "CAM_FRONT")
    original_filename: str                # ORIGINAL filename from source dataset
                                          # Examples: "00000.pcd", "cam0/00000.png", "lidar/00000.feather"
                                          # Writer uses this to find the source file
    timestamp_us: int                     # Timestamp in microseconds
    is_keyframe: bool = True              # Whether this is a keyframe (true for most samples)
    
    # Optional: Image dimensions (for cameras)
    width: Optional[int] = None
    height: Optional[int] = None
    
    def __post_init__(self):
        """Validate sensor data."""
        assert self.temp_frame_id, "Frame ID cannot be empty"
        assert self.sensor_name, "Sensor name cannot be empty"
        assert self.original_filename, "Original filename cannot be empty"
        assert self.timestamp_us >= 0, "Timestamp must be non-negative"


@dataclass
class IFSample:
    """
    Represents a "keyframe" or "sample" in time.
    
    A sample links all sensor data and annotations at a single timestamp,
    representing a complete snapshot of the world at that moment.
    """
    temp_frame_id: str                    # Reader's unique ID for this frame (e.g., "00000", "frame_123")
    timestamp_us: int                     # Timestamp in microseconds
    scene_name: str                       # Name of the scene this sample belongs to
    
    # Optional: Additional sample metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate sample data."""
        assert self.temp_frame_id, "Frame ID cannot be empty"
        assert self.timestamp_us >= 0, "Timestamp must be non-negative"
        assert self.scene_name, "Scene name cannot be empty"


@dataclass
class IFScene:
    """
    Defines a single "scene" or "log" (a continuous driving sequence).
    
    A scene represents a contiguous recording session, typically containing
    multiple samples captured over a period of time.
    """
    name: str                             # Unique name for this scene (e.g., "idd3d_seq10", "scene-0001")
    description: str                      # Human-readable description
    
    # Optional: Scene metadata
    location: Optional[str] = None        # Location name (e.g., "Hyderabad", "Singapore")
    date_captured: Optional[str] = None   # Date in YYYY-MM-DD format
    weather: Optional[str] = None         # Weather conditions (e.g., "clear", "rainy")
    time_of_day: Optional[str] = None     # Time of day (e.g., "day", "night", "dawn")
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate scene data."""
        assert self.name, "Scene name cannot be empty"


@dataclass
class IntermediateData:
    """
    This is the main "package" of data passed from Reader to Writer.
    
    It contains all the processed information from a source dataset sequence,
    organized in a format-agnostic way that any Writer can consume.
    
    USAGE:
        # Reader creates and populates
        data = IntermediateData(sequence_path="/path/to/data")
        data.scenes.append(IFScene(...))
        data.samples.append(IFSample(...))
        # ... etc
        
        # Writer consumes
        writer = NuScenesWriter()
        writer.write(data, "/output/path")
    """
    # --- Core Data Lists ---
    scenes: List[IFScene] = field(default_factory=list)
    samples: List[IFSample] = field(default_factory=list)
    sensor_data: List[IFSensorData] = field(default_factory=list)
    annotations: List[IFAnnotation] = field(default_factory=list)
    instances: List[IFInstance] = field(default_factory=list)
    calibrations: List[IFCalibration] = field(default_factory=list)
    ego_poses: List[IFEgoPose] = field(default_factory=list)
    
    # --- File Path Information ---
    # The Reader MUST populate this so the Writer knows where to find
    # the original physical files (e.g., .pcd, .png, .feather).
    sequence_path: str = ""               # Absolute path to the sequence root directory
    
    # --- Optional: Dataset-Specific Extensions ---
    # Use this for dataset-specific data that doesn't fit the standard schema
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate intermediate data."""
        if self.samples and not self.scenes:
            raise ValueError("Cannot have samples without at least one scene")
    
    def validate(self) -> bool:
        """
        Performs comprehensive validation of the intermediate data.
        
        Returns:
            True if all data is valid, raises ValueError otherwise
        """
        # Check for required data
        if not self.scenes:
            raise ValueError("IntermediateData must contain at least one scene")
        
        if not self.samples:
            raise ValueError("IntermediateData must contain at least one sample")
        
        if not self.sequence_path:
            raise ValueError("sequence_path must be set")
        
        # Validate relationships
        scene_names = {scene.name for scene in self.scenes}
        for sample in self.samples:
            if sample.scene_name not in scene_names:
                raise ValueError(f"Sample references unknown scene: {sample.scene_name}")
        
        # Validate frame ID consistency
        sample_frame_ids = {sample.temp_frame_id for sample in self.samples}
        
        for sensor_data in self.sensor_data:
            if sensor_data.temp_frame_id not in sample_frame_ids:
                raise ValueError(f"SensorData references unknown frame: {sensor_data.temp_frame_id}")
        
        for annotation in self.annotations:
            if annotation.temp_frame_id not in sample_frame_ids:
                raise ValueError(f"Annotation references unknown frame: {annotation.temp_frame_id}")
        
        # Validate instance ID consistency
        instance_ids = {inst.temp_instance_id for inst in self.instances}
        
        for annotation in self.annotations:
            if annotation.temp_instance_id not in instance_ids:
                raise ValueError(f"Annotation references unknown instance: {annotation.temp_instance_id}")
        
        return True
    
    def summary(self) -> str:
        """
        Returns a human-readable summary of the data.
        
        Returns:
            Formatted string with statistics
        """
        return f"""
IntermediateData Summary:
========================
Sequence Path:  {self.sequence_path}
Scenes:         {len(self.scenes)}
Samples:        {len(self.samples)}
Instances:      {len(self.instances)}
Annotations:    {len(self.annotations)}
Sensor Data:    {len(self.sensor_data)}
Calibrations:   {len(self.calibrations)}
Ego Poses:      {len(self.ego_poses)}

Scene Names:    {', '.join(scene.name for scene in self.scenes)}
Sensors:        {', '.join(sorted(set(cal.sensor_name for cal in self.calibrations)))}
Categories:     {', '.join(sorted(set(inst.category_name for inst in self.instances)))}
"""


# --- Helper Functions ---

def create_empty_intermediate_data(sequence_path: str) -> IntermediateData:
    """
    Factory function to create an empty IntermediateData object.
    
    Args:
        sequence_path: Absolute path to the sequence directory
        
    Returns:
        Empty IntermediateData object ready to be populated
    """
    return IntermediateData(sequence_path=sequence_path)


def validate_category_name(category_name: str) -> bool:
    """
    Validates that a category name follows the standard naming convention.
    
    Standard format: "{group}.{specific}" (e.g., "vehicle.car", "movable_object.pedestrian")
    
    Args:
        category_name: The category name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not category_name or '.' not in category_name:
        return False
    
    parts = category_name.split('.')
    if len(parts) != 2:
        return False
    
    group, specific = parts
    
    # Common valid groups
    valid_groups = {
        'vehicle', 'human', 'movable_object', 
        'static_object', 'animal', 'flat'
    }
    
    return group in valid_groups and specific.strip() != ''
