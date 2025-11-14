# extensible_backend.py
# ----------------------
# DEPRECATED: This Flask backend is being phased out in favor of convert_cli.py
# 
# This file is kept for backwards compatibility but should not be used for new projects.
# Please use the CLI tool instead: python convert_cli.py --help
# ----------------------

"""
NOTE: This backend has been DEPRECATED in favor of the modular CLI approach.

The new architecture uses:
- convert_cli.py: Main entry point
- readers.py: Dataset-specific readers (IDD3D, Argoverse, etc.)
- writers.py: Format-specific writers (nuScenes, KITTI, etc.)
- intermediate_format.py: Common data representation
- utils.py: Shared utilities

To convert datasets, use:
    python convert_cli.py --reader idd3d --writer nuscenes --input /path/to/data --output /path/to/output

This Flask backend remains here only for legacy support and should be removed in future versions.
"""

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import os
import json
import logging
from queue import Queue
from datetime import datetime

# Import the new modular components
try:
    from readers import Idd3dReader
    from writers import NuScenesWriter
    from intermediate_format import IntermediateData
except ImportError:
    print("ERROR: Could not import new modular components.")
    print("Please ensure readers.py, writers.py, and intermediate_format.py are in the same directory.")
    exit(1)

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for active conversions
conversion_state = {
    'active': False,
    'logs': Queue(),
    'progress': 0,
    'total_steps': 0
}


class LogHandler:
    """Handler to capture conversion logs and emit them via SSE."""
    
    def __init__(self, log_queue):
        self.queue = log_queue
    
    def log(self, message, log_type='info'):
        """Add a log entry to the queue."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'type': log_type
        }
        self.queue.put(log_entry)
        logger.info(f"[{log_type.upper()}] {message}")


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Dataset Converter API (DEPRECATED)',
        'message': 'Please use convert_cli.py instead'
    })


@app.route('/api/conversions', methods=['GET'])
def get_conversions():
    """Get all available conversions."""
    return jsonify({
        'conversions': [
            {'source': 'idd3d', 'target': 'nuscenes'}
        ],
        'warning': 'This API is deprecated. Please use convert_cli.py'
    })


@app.route('/api/validate-paths', methods=['POST'])
def validate_paths():
    """Validate dataset paths."""
    data = request.json
    source = data.get('source', 'idd3d')
    sequence_path = data.get('sequence_path')
    
    if not sequence_path:
        return jsonify({'valid': False, 'error': 'Sequence Path is required'}), 400
    
    if not os.path.exists(sequence_path):
        return jsonify({'valid': False, 'error': f'Path does not exist: {sequence_path}'}), 400
    
    try:
        if source == 'idd3d':
            reader = Idd3dReader()
            validation = reader.validate(sequence_path)
            return jsonify(validation)
        else:
            return jsonify({'valid': False, 'error': f'Unknown source: {source}'}), 400
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 500


@app.route('/api/convert/stream', methods=['POST'])
def convert_stream():
    """
    Start conversion and stream logs via SSE.
    
    WARNING: This endpoint is DEPRECATED.
    Use convert_cli.py for better performance and maintainability.
    """
    if conversion_state['active']:
        return jsonify({'error': 'Conversion already in progress'}), 409
    
    conversion_state['active'] = True
    
    data = request.json
    source = data.get('source', 'idd3d')
    target = data.get('target', 'nuscenes')
    sequence_path = data.get('sequence_path')
    output_path = data.get('output_path')
    
    def generate():
        log_handler = LogHandler(conversion_state['logs'])
        
        try:
            # Clear old logs
            while not conversion_state['logs'].empty():
                conversion_state['logs'].get()
            
            log_handler.log(f"DEPRECATION WARNING: Please use convert_cli.py instead of this API", 'warning')
            log_handler.log(f"Starting conversion: {source} → {target}", 'info')
            log_handler.log(f"Sequence: {sequence_path}", 'info')
            log_handler.log(f"Output: {output_path}", 'info')
            
            # Initialize reader
            if source == 'idd3d':
                reader = Idd3dReader()
            else:
                raise ValueError(f"Unknown source: {source}")
            
            log_handler.log("Reading source data...", 'info')
            intermediate_data = reader.read(sequence_path)
            
            conversion_state['progress'] = 50
            
            # Initialize writer
            if target == 'nuscenes':
                writer = NuScenesWriter()
            else:
                raise ValueError(f"Unknown target: {target}")
            
            log_handler.log("Writing to target format...", 'info')
            writer.write(intermediate_data, output_path)
            
            conversion_state['progress'] = 100
            log_handler.log("✓ Conversion completed successfully!", 'success')
            
        except Exception as e:
            log_handler.log(f"✗ Conversion failed: {str(e)}", 'error')
            import traceback
            log_handler.log(traceback.format_exc(), 'error')
        
        finally:
            conversion_state['active'] = False
            while not conversion_state['logs'].empty():
                log_entry = conversion_state['logs'].get()
                yield f"data: {json.dumps(log_entry)}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'This API is deprecated. Please use convert_cli.py'
    }), 404


if __name__ == '__main__':
    print("=" * 70)
    print("WARNING: This Flask backend is DEPRECATED")
    print("=" * 70)
    print("Please use the CLI tool instead:")
    print("  python convert_cli.py --reader idd3d --writer nuscenes \\")
    print("    --input /path/to/sequences --output /path/to/output")
    print("=" * 70)
    print("\nStarting deprecated backend on http://localhost:5001")
    print("This will be removed in future versions.\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
