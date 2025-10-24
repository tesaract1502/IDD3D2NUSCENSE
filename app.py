
"""
app.py - Simple web server to host the Dataset Converter HTML interface
Run this file to access the converter through your browser
"""

from flask import Flask, send_file, redirect
from flask_cors import CORS
import os

# Create Flask app for serving the HTML interface
app = Flask(__name__)
CORS(app)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    """Serve the main HTML interface"""
    html_path = os.path.join(SCRIPT_DIR, 'converter.html')
    
    if not os.path.exists(html_path):
        return """
        <html>
        <head><title>Error</title></head>
        <body>
        <h1>Error: converter.html not found</h1>
        <p>Please make sure 'converter.html' exists in the same directory as app.py</p>
        <p>Current directory: {}</p>
        </body>
        </html>
        """.format(SCRIPT_DIR), 404
    
    return send_file(html_path)

@app.route('/health')
def health():
    """Simple health check"""
    return {'status': 'ok', 'service': 'Dataset Converter Web Interface'}

@app.errorhandler(404)
def not_found(error):
    return """
    <html>
    <head><title>404 Not Found</title></head>
    <body>
    <h1>404 - Page Not Found</h1>
    <p><a href="/">Return to Converter</a></p>
    </body>
    </html>
    """, 404


if __name__ == '__main__':
    print("=" * 70)
    print("Dataset Converter Web Interface")
    print("=" * 70)
    print("\nStarting web server...")
    print("\nIMPORTANT: You need TWO servers running:")
    print("   1. Backend API (extensible_converter_backend.py) on port 5001")
    print("   2. This web interface (app.py) on port 3000")
    print("\nTo start the backend API (in another terminal):")
    print("   python extensible_converter_backend.py")
    print("\n Web Interface will be available at:")
    print("   http://localhost:3000")
    print("   http://127.0.0.1:3000")
    print("\nMake sure 'converter.html' is in the same directory as this file")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    # Check if HTML file exists
    html_path = os.path.join(SCRIPT_DIR, 'converter.html')
    if not os.path.exists(html_path):
        print(" WARNING: converter.html not found in current directory!")
        print(f"Looking for: {html_path}")
        print("Please create or copy converter.html to this location\n")
    else:
        print(f"✓ Found converter.html at: {html_path}\n")
    
    # Start the Flask server
    try:
        app.run(
            host='0.0.0.0',
            port=3000,
            debug=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\nShutting down web server...")
        print("Goodbye!")
    except Exception as e:
        print(f"\n\n Error starting server: {e}")
        print("Make sure port 3000 is not already in use")
