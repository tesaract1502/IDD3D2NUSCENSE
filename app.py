from flask import Flask, send_from_directory
from flask_cors import CORS
import os
import threading
import extensible_converter_backend as backend  # your backend file

app = Flask(__name__, static_folder='.')
CORS(app)

# Serve the HTML UI
@app.route('/')
def serve_ui():
    return send_from_directory('.', 'converter_html_unterface_1.html')

def run_backend():
    """Run the backend API (from extensible_converter_backend) on port 5001"""
    backend.app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)

if __name__ == '__main__':
    # Start backend server in a separate thread
    threading.Thread(target=run_backend, daemon=True).start()

    # Run UI on main thread
    print("Dataset Converter Web App running at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
