"""
Extensible Flask backend for Dataset Converter UI
Supports multiple dataset conversions via pluggable converter classes
"""

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from abc import ABC, abstractmethod
import os
import json
import threading
from queue import Queue
from datetime import datetime
import logging
import math
import uuid
from typing import Dict, List

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


class LogHandler:
    """Handler to capture conversion logs and emit them"""
    
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


# ============================================================================ #
# BASE CLASSES
# ============================================================================ #

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
            log_handler.log(f"\n[{idx+1}/{len(self.converters)}] Running {converter.name} converter...", 'info')
            try:
                converter.run(data_loader, log_handler)
                conversion_state['progress'] = ((idx + 1) / len(self.converters)) * 100
            except Exception as e:
                log_handler.log(f"✗ {converter.name} failed: {str(e)}", 'error')
                raise


# ============================================================================ #
# IDD3D IMPLEMENTATION
# ============================================================================ #

class IDD3DDataLoader(BaseDataLoader):
    def __init__(self, root: str, sequence: str = '20220118103308_seq_10'):
        super().__init__(root, sequence)
        self.seq_base = os.path.join(
            self.root,
            'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val',
            sequence
        )
        self.lidar_dir = os.path.join(self.seq_base, 'lidar')
        self.label_dir = os.path.join(self.seq_base, 'label')
        self.calib_dir = os.path.join(self.seq_base, 'calib')
        self.annot_json = os.path.join(self.seq_base, 'annot_data.json')
        
        self.out_data = os.path.join(self.root, 'Intermediate_format/data')
        self.annot_out = os.path.join(self.root, 'Intermediate_format/anotations')
        self.converted_lidar = os.path.join(self.out_data, 'converted_lidar')
    
    def ensure_output_dirs(self):
        os.makedirs(self.out_data, exist_ok=True)
        os.makedirs(self.annot_out, exist_ok=True)
        os.makedirs(self.converted_lidar, exist_ok=True)
    
    def validate(self) -> dict:
        if not os.path.exists(self.seq_base):
            return {'valid': False, 'error': f'Sequence path not found: {self.seq_base}'}
        
        required_dirs = ['lidar', 'label', 'calib']
        missing = []
        for dir_name in required_dirs:
            dir_path = os.path.join(self.seq_base, dir_name)
            if not os.path.exists(dir_path):
                missing.append(dir_name)
        
        if missing:
            return {'valid': False, 'error': f'Missing directories: {", ".join(missing)}'}
        
        lidar_count = len([f for f in os.listdir(self.lidar_dir) if f.lower().endswith('.pcd')])
        label_count = len([f for f in os.listdir(self.label_dir) if f.lower().endswith('.json')])
        
        return {
            'valid': True,
            'path': self.seq_base,
            'lidar_files': lidar_count,
            'label_files': label_count
        }

    def list_lidar_files(self):
        if not os.path.exists(self.lidar_dir):
            return []
        return [os.path.join(self.lidar_dir, f) for f in sorted(os.listdir(self.lidar_dir)) if f.lower().endswith('.pcd')]

    def read_annotations(self):
        if not os.path.exists(self.annot_json):
            return {}
        with open(self.annot_json, 'r') as f:
            return json.load(f)


# Existing converters (LiDAR, Camera, Calib, Annotations) are unchanged here...
# --- [omitted for brevity: your existing IDD3DLidarConverter, IDD3DCameraConverter, IDD3DCalibConverter, IDD3DAnnotationConverter classes remain intact] ---


# ============================================================================ #
# NEW CONVERTER: IDD3D -> nuScenes JSON EXPORT
# ============================================================================ #

def annot_timestamp_from_frame_id(frame_id: str) -> int:
    try:
        return int(frame_id) * 100_000
    except Exception:
        return int(datetime.now().timestamp() * 1_000_000)


class IDD3DToNuScenesJsonConverter(BaseConverter):
    """Generate nuScenes-like JSON files from IDD3D annot_data + label files."""

    def __init__(self):
        super().__init__('json_export')

    def _tok(self):
        return uuid.uuid4().hex

    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        log_handler.log("Starting nuScenes JSON export...", "info")

        out_dir = os.path.join(data_loader.out_data, 'nuScenes_json')
        os.makedirs(out_dir, exist_ok=True)

        try:
            with open(data_loader.annot_json, 'r') as f:
                annot_data = json.load(f)
        except Exception as e:
            log_handler.log(f"Failed to read annot_data.json: {e}", "error")
            return

        categories_map, instances_map, samples_map = {}, {}, {}
        category_list, instance_list = [], []
        sample_list, sample_data_list, sample_ann_list, log_list = [], [], [], []

        def ensure_category(obj_type):
            if obj_type not in categories_map:
                tok = self._tok()
                categories_map[obj_type] = tok
                category_list.append({
                    "token": tok,
                    "name": obj_type.lower().replace(" ", "."),
                    "description": f"Auto category for {obj_type}"
                })
            return categories_map[obj_type]

        def ensure_instance(obj_id, cat_tok):
            if obj_id not in instances_map:
                tok = self._tok()
                instances_map[obj_id] = tok
                instance_list.append({
                    "token": tok,
                    "category_token": cat_tok,
                    "nbr_annotations": 0,
                    "first_annotation_token": None,
                    "last_annotation_token": None
                })
            return instances_map[obj_id]

        frame_ids = sorted(annot_data.keys())

        for idx, fid in enumerate(frame_ids):
            st = self._tok()
            samples_map[fid] = st
            ts = annot_timestamp_from_frame_id(fid)
            sample_list.append({
                "token": st,
                "timestamp": ts,
                "prev": "" if idx == 0 else samples_map[frame_ids[idx - 1]],
                "next": "" if idx == len(frame_ids) - 1 else "",
                "scene_token": None
            })
        for i in range(len(sample_list) - 1):
            sample_list[i]["next"] = sample_list[i + 1]["token"]

        fake_calib = self._tok()
        fake_pose = self._tok()
        inst_anns = {}

        for fid in frame_ids:
            st = samples_map[fid]
            entry = annot_data[fid]

            for k, path in entry.items():
                if k in ("bag_id", "session_id"): continue
                ext = os.path.splitext(path)[-1].lower().strip('.')
                fmt = "jpg" if ext in ("jpg", "jpeg", "png") else "pcd"
                sd = {
                    "token": self._tok(),
                    "sample_token": st,
                    "ego_pose_token": fake_pose,
                    "calibrated_sensor_token": fake_calib,
                    "timestamp": annot_timestamp_from_frame_id(fid),
                    "fileformat": fmt,
                    "is_key_frame": True,
                    "height": 0,
                    "width": 0,
                    "filename": path.replace("\\", "/"),
                    "prev": "",
                    "next": ""
                }
                sample_data_list.append(sd)

            label_path = os.path.join(data_loader.label_dir, f"{fid}.json")
            objs = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    try:
                        objs = json.load(f)
                    except Exception:
                        pass

            for o in objs:
                obj_id = str(o.get("obj_id", ""))
                obj_type = o.get("obj_type", "unknown")
                psr = o.get("psr", {})
                pos, rot, sc = psr.get("position", {}), psr.get("rotation", {}), psr.get("scale", {})

                cat_tok = ensure_category(obj_type)
                inst_tok = ensure_instance(obj_id, cat_tok)
                ann_tok = self._tok()

                ann = {
                    "token": ann_tok,
                    "sample_token": st,
                    "instance_token": inst_tok,
                    "visibility_token": "4",
                    "attribute_tokens": [],
                    "translation": [pos.get("x",0), pos.get("y",0), pos.get("z",0)],
                    "size": [sc.get("x",0), sc.get("y",0), sc.get("z",0)],
                    "rotation": self._rot_to_quat(rot),
                    "prev": "",
                    "next": "",
                    "num_lidar_pts": 0,
                    "num_radar_pts": 0
                }
                sample_ann_list.append(ann)
                inst_anns.setdefault(inst_tok, []).append(ann_tok)

        for inst in instance_list:
            tok = inst["token"]
            anns = inst_anns.get(tok, [])
            inst["nbr_annotations"] = len(anns)
            inst["first_annotation_token"] = anns[0] if anns else None
            inst["last_annotation_token"] = anns[-1] if anns else None

        log_list.append({
            "token": self._tok(),
            "logfile": os.path.basename(data_loader.root),
            "vehicle": "vehicle_stub",
            "date_captured": datetime.now().strftime("%Y-%m-%d"),
            "location": "location_stub"
        })

        def save(name, obj):
            with open(os.path.join(out_dir, name), 'w') as f:
                json.dump(obj, f, indent=2)
            log_handler.log(f"Wrote {name} ({len(obj)})", "info")

        save("category.json", category_list)
        save("instance.json", instance_list)
        save("log.json", log_list)
        save("sample.json", sample_list)
        save("sample_data.json", sample_data_list)
        save("sample_annotation.json", sample_ann_list)

        log_handler.log("nuScenes JSON export complete.", "success")

    def _rot_to_quat(self, rot):
        try:
            if isinstance(rot, dict) and 'z' in rot:
                yaw = rot['z']
                return [
                    math.cos(yaw/2.0),
                    0.0,
                    0.0,
                    math.sin(yaw/2.0)
                ]
        except Exception:
            pass
        return [1.0, 0.0, 0.0, 0.0]


# ============================================================================ #
# CONVERTER REGISTRY + PIPELINE BUILDER
# ============================================================================ #

class ConverterRegistry:
    _conversions = {}

    @classmethod
    def register(cls, source, target, builder):
        cls._conversions[(source, target)] = builder

    @classmethod
    def get_pipeline(cls, source, target, cfg):
        key = (source, target)
        if key not in cls._conversions:
            raise ValueError(f"No conversion for {source}->{target}")
        return cls._conversions[key](cfg)

    @classmethod
    def get_available_conversions(cls):
        return [{'source': s, 'target': t} for s, t in cls._conversions.keys()]


def build_idd3d_to_nuscenes_pipeline(config: dict) -> DatasetConversionPipeline:
    pipeline = DatasetConversionPipeline('idd3d', 'nuscenes')
    conv = config.get('conversions', {})

    if conv.get('lidar'): pipeline.add_converter(IDD3DLidarConverter())
    if conv.get('camera'): pipeline.add_converter(IDD3DCameraConverter())
    if conv.get('calib'): pipeline.add_converter(IDD3DCalibConverter())
    if conv.get('annot'): pipeline.add_converter(IDD3DAnnotationConverter(config.get('sequence_id', 'seq_10')))
    # 🔹 NEW LINE — registers our JSON export converter
    if conv.get('json_export') or conv.get('json'):
        pipeline.add_converter(IDD3DToNuScenesJsonConverter())

    return pipeline


ConverterRegistry.register('idd3d', 'nuscenes', build_idd3d_to_nuscenes_pipeline)

# ============================================================================ #
# API ENDPOINTS (unchanged)
# ============================================================================ #
# [keep all your Flask routes here exactly as in your existing file]
