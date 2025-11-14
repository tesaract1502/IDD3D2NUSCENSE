from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class IFCalibration:
    sensor_name: str
    translation: List[float]  # length 3
    rotation: List[float]     # quaternion length 4 (w, x, y, z)
    camera_intrinsic: List[List[float]] = field(default_factory=list)  # 3x3 matrix for cameras

    def __post_init__(self):
        assert len(self.translation) == 3, "Translation must be length 3"
        assert len(self.rotation) == 4, "Rotation must be quaternion length 4"
        if self.camera_intrinsic:
            assert len(self.camera_intrinsic) == 3, "Camera intrinsic must be 3x3"
            for row in self.camera_intrinsic:
                assert len(row) == 3, "Camera intrinsic each row must be length 3"

@dataclass
class IFEgoPose:
    temp_frame_id: str
    timestamp_us: int
    translation: List[float]  # length 3
    rotation: List[float]     # quaternion length 4

    def __post_init__(self):
        assert len(self.translation) == 3, "Translation must be length 3"
        assert len(self.rotation) == 4, "Rotation must be quaternion length 4"

@dataclass
class IFInstance:
    temp_instance_id: str
    category_name: str
    attributes: Optional[List[str]] = field(default_factory=list)

@dataclass
class IFAnnotation:
    temp_instance_id: str
    temp_frame_id: str
    timestamp_us: int
    translation: List[float]  # length 3
    size: List[float]         # length 3 (width, length, height)
    rotation: List[float]     # quaternion length 4
    attributes: Optional[List[str]] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.translation) == 3, "Translation must be length 3"
        assert len(self.size) == 3, "Size must be length 3"
        assert len(self.rotation) == 4, "Rotation must be quaternion length 4"

@dataclass
class IFSample:
    temp_frame_id: str
    timestamp_us: int
    scene_name: str

@dataclass
class IFSensorData:
    temp_frame_id: str
    sensor_name: str
    original_filename: str
    timestamp_us: int
    is_keyframe: bool = True

@dataclass
class IFScene:
    name: str
    description: Optional[str] = ""

@dataclass
class IntermediateData:
    sequence_path: str = ""
    scenes: List[IFScene] = field(default_factory=list)
    samples: List[IFSample] = field(default_factory=list)
    instances: List[IFInstance] = field(default_factory=list)
    annotations: List[IFAnnotation] = field(default_factory=list)
    sensor_data: List[IFSensorData] = field(default_factory=list)
    ego_poses: List[IFEgoPose] = field(default_factory=list)
    calibrations: List[IFCalibration] = field(default_factory=list)

    def validateself(self):
        if not self.scenes:
            raise ValueError("No scenes defined")
        if not self.samples:
            raise ValueError("No samples defined")
        if not self.calibrations:
            raise ValueError("No calibrations defined")
        # Check references
        sample_frame_ids = {s.temp_frame_id for s in self.samples}
        instance_ids = {inst.temp_instance_id for inst in self.instances}

        for anno in self.annotations:
            if anno.temp_frame_id not in sample_frame_ids:
                raise ValueError(f"Annotation with frame id {anno.temp_frame_id} references unknown sample")
            if anno.temp_instance_id not in instance_ids:
                raise ValueError(f"Annotation with instance id {anno.temp_instance_id} references unknown instance")

        for ego_pose in self.ego_poses:
            if ego_pose.temp_frame_id not in sample_frame_ids:
                raise ValueError(f"EgoPose with frame id {ego_pose.temp_frame_id} references unknown sample")

        for sensor_data in self.sensor_data:
            if sensor_data.temp_frame_id not in sample_frame_ids:
                raise ValueError(f"SensorData with frame id {sensor_data.temp_frame_id} references unknown sample")

        # Add more validation rules as required

