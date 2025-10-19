"""
Flask backend for Dataset Converter UI
Wraps the OOP converter classes and provides REST API endpoints
with real-time log streaming via Server-Sent Events (SSE)
"""

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from idd3d_converter_oop import (
    DataLoader, LidarConverter, CalibStubConverter, 
    AnnotationConverter
)
import os
import json
import threading
from queue import Queue
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
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
        """Add a log entry to the queue"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'type': log_type
        }
        self.queue.put(log_entry)
        logger.info(f"[{log_type.upper()}] {message}")


def wrap_converter_with_logging(converter, log_handler):
    """Wrap converter run method to capture logs"""
    original_run = converter.run
    
    def run_with_logging(data_loader=None):
        try:
            # Patch print statements by redirecting stdout temporarily
            import sys
            from io import StringIO
            
            # Store original stdout
            old_stdout = sys.stdout
            sys.stdout = log_capture = StringIO()
            
            try:
                original_run(data_loader)
                # Capture any print output
                output = log_capture.getvalue()
                if output:
                    log_handler.log(output.strip(), 'info')
            finally:
                sys.stdout = old_stdout
            
            log_handler.log(f"✓ {converter.name} conversion completed", 'success')
        except Exception as e:
            log_handler.log(f"✗ {converter.name} conversion failed: {str(e)}", 'error')
            raise
    
    converter.run = run_with_logging


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Dataset Converter API'})


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Return available datasets and their specs"""
    datasets = {
        'idd3d': {
            'name': 'IDD3D (Indian Dataset)',
            'description': '6 cameras, 1 LiDAR, GPS - 10Hz camera, 10Hz LiDAR',
            'sensors': '6 RGB cameras, 64-channel LiDAR, GPS',
            'format': 'PCD (lidar), PNG (camera), JSON (annotations)',
            'available': True
        },
        'nuscenes': {
            'name': 'nuScenes',
            'description': '6 cameras, 1 LiDAR, 5 RADARs, GPS/IMU - 12Hz camera, 20Hz LiDAR',
            'sensors': '6 cameras, Velodyne HDL-32E, 5 RADARs, GPS/IMU',
            'format': 'JPEG (camera), .pcd.bin (lidar), JSON (metadata)',
            'available': True
        }
    }
    return jsonify(datasets)


@app.route('/api/validate-paths', methods=['POST'])
def validate_paths():
    """Validate that required paths exist"""
    data = request.json
    root_path = data.get('root_path')
    sequence_id = data.get('sequence_id', '20220118103308_seq_10')
    
    if not root_path or not os.path.exists(root_path):
        return jsonify({
            'valid': False,
            'error': f'Root path does not exist: {root_path}'
        }), 400
    
    # Check for IDD3D sequence structure
    seq_base = os.path.join(
        root_path,
        'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val',
        sequence_id
    )
    
    required_dirs = ['lidar', 'label', 'calib', 'camera']
    missing = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(seq_base, dir_name)
        if not os.path.exists(dir_path):
            missing.append(dir_name)
    
    if missing:
        return jsonify({
            'valid': False,
            'error': f'Missing directories: {", ".join(missing)}',
            'expected_path': seq_base
        }), 400
    
    # Count files
    lidar_count = len([f for f in os.listdir(os.path.join(seq_base, 'lidar')) 
                       if f.lower().endswith('.pcd')])
    label_count = len([f for f in os.listdir(os.path.join(seq_base, 'label')) 
                       if f.lower().endswith('.json')])
    
    return jsonify({
        'valid': True,
        'path': seq_base,
        'lidar_files': lidar_count,
        'label_files': label_count,
        'intermediate_output': os.path.join(root_path, 'Intermediate_format')
    })


@app.route('/api/convert/stream', methods=['POST'])
def convert_stream():
    """
    Start conversion and stream logs via Server-Sent Events (SSE)
    """
    with conversion_lock:
        if conversion_state['active']:
            return jsonify({'error': 'Conversion already in progress'}), 409
        conversion_state['active'] = True
    
    data = request.json
    root_path = data.get('root_path')
    sequence_id = data.get('sequence_id', '20220118103308_seq_10')
    conversions = data.get('conversions', {})
    
    # Validate paths
    seq_base = os.path.join(
        root_path,
        'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val',
        sequence_id
    )
    
    if not os.path.exists(seq_base):
        conversion_state['active'] = False
        return jsonify({'error': f'Sequence path not found: {seq_base}'}), 400
    
    def generate():
        """Generator function for SSE stream"""
        try:
            # Clear previous logs
            while not conversion_state['logs'].empty():
                conversion_state['logs'].get()
            
            log_handler = LogHandler(conversion_state['logs'])
            
            log_handler.log(f"Starting conversion: IDD3D → nuScenes", 'info')
            log_handler.log(f"Root path: {root_path}", 'info')
            log_handler.log(f"Sequence ID: {sequence_id}", 'info')
            
            # Initialize DataLoader
            dl = DataLoader(root_path, sequence=sequence_id)
            dl.ensure_output_dirs()
            
            # Create list of converters to run
            converters_to_run = []
            
            if conversions.get('lidar', False):
                lidar_src = os.path.join(seq_base, 'lidar')
                lidar_dst = os.path.join(root_path, 'Intermediate_format/data/converted_lidar')
                conv = LidarConverter(lidar_src, lidar_dst)
                wrap_converter_with_logging(conv, log_handler)
                converters_to_run.append(('lidar', conv))
            
            if conversions.get('calib', False):
                calib_dir = os.path.join(seq_base, 'calib')
                out_data = os.path.join(root_path, 'Intermediate_format/data')
                conv = CalibStubConverter(calib_dir, out_data)
                wrap_converter_with_logging(conv, log_handler)
                converters_to_run.append(('calib', conv))
            
            if conversions.get('annot', False):
                annot_json = os.path.join(seq_base, 'annot_data.json')
                label_folder = os.path.join(seq_base, 'label')
                out_dir = os.path.join(root_path, 'Intermediate_format/anotations')
                conv = AnnotationConverter(annot_json, label_folder, out_dir, 
                                          sequence_name=sequence_id)
                wrap_converter_with_logging(conv, log_handler)
                converters_to_run.append(('annot', conv))
            
            # Check for unavailable features
            if conversions.get('maps', False):
                log_handler.log("⚠ Map generation skipped (data not available)", 'warning')
            if conversions.get('egoPose', False):
                log_handler.log("⚠ Ego pose generation skipped (data not available)", 'warning')
            if conversions.get('timeSync', False):
                log_handler.log("⚠ Time synchronization skipped (data not available)", 'warning')
            
            conversion_state['total_steps'] = len(converters_to_run)
            
            if len(converters_to_run) == 0:
                log_handler.log("✗ No conversion modules selected", 'warning')
            else:
                log_handler.log(f"Running {len(converters_to_run)} conversion modules...", 'info')
            
            # Run each converter
            for idx, (name, converter) in enumerate(converters_to_run):
                log_handler.log(f"\n[{idx+1}/{len(converters_to_run)}] Running {name} converter...", 'info')
                conversion_state['current_step'] = idx + 1
                
                converter.run(dl)
                conversion_state['progress'] = ((idx + 1) / len(converters_to_run)) * 100
            
            if len(converters_to_run) > 0:
                log_handler.log("\n✓ Conversion pipeline completed successfully!", 'success')
                log_handler.log(f"Output directory: {root_path}/Intermediate_format/", 'info')
        
        except Exception as e:
            log_handler.log(f"✗ Conversion failed: {str(e)}", 'error')
            import traceback
            log_handler.log(traceback.format_exc(), 'error')
        
        finally:
            conversion_state['active'] = False
            # Send final status
            while not conversion_state['logs'].empty():
                log_entry = conversion_state['logs'].get()
                yield f"data: {json.dumps(log_entry)}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/convert/status', methods=['GET'])
def conversion_status():
    """Get current conversion status"""
    return jsonify({
        'active': conversion_state['active'],
        'progress': conversion_state['progress'],
        'current_step': conversion_state['current_step'],
        'total_steps': conversion_state['total_steps']
    })


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get all current logs (non-streaming fallback)"""
    logs = []
    temp_queue = Queue()
    
    while not conversion_state['logs'].empty():
        log_entry = conversion_state['logs'].get()
        logs.append(log_entry)
        temp_queue.put(log_entry)
    
    # Restore logs to main queue
    while not temp_queue.empty():
        conversion_state['logs'].put(temp_queue.get())
    
    return jsonify({'logs': logs})


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500


if __name__ == '__main__':
    print("Starting Dataset Converter API...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/datasets - Get available datasets")
    print("  POST /api/validate-paths - Validate dataset paths")
    print("  POST /api/convert/stream - Start conversion with SSE logs")
    print("  GET  /api/convert/status - Get conversion status")
    print("  GET  /api/logs - Get all logs (fallback)")
    print("\nServer running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)