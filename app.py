import os
import logging
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, Response
import torch
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import time
import mediapipe as mp

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "yolov5-detection-app")

# Configure paths and settings
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'static/models'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MODEL_PATH'] = MODEL_PATH
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Global variables for the model
model = None
class_names = []

# Initialize MediaPipe with a check for availability
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    HAS_MEDIAPIPE = True
except Exception as e:
    logger.error(f"MediaPipe initialization failed: {e}")
    HAS_MEDIAPIPE = False

def load_model():
    """Load the YOLOv5 model"""
    global model, class_names
    try:
        logger.info("Loading model from yolov5s.pt")
        # Load pre-trained YOLOv5s model from Torch Hub
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        class_names = model.names
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gen_frames():
    """Video streaming generator function with AI Proctoring logic"""
    global model
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 1. Run YOLOv5 for object detection
        if model is not None:
            results = model(frame)
            
            # CRITICAL FIX: results.render() returns a read-only array.
            # We make a deep copy to allow cv2.putText to modify the image.
            frame = results.render()[0].copy() 
            
            # Proctoring Logic: Check for prohibited objects (cell phone)
            df = results.pandas().xyxy[0]
            if not df[df['name'] == 'cell phone'].empty:
                cv2.putText(frame, "CHEATING ALERT: PHONE", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 2. Run MediaPipe for Behavior/Gaze Tracking
        if HAS_MEDIAPIPE:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = face_mesh.process(rgb_frame)

            if mp_results.multi_face_landmarks:
                for landmarks in mp_results.multi_face_landmarks:
                    # Target the nose tip landmark for head orientation
                    nose = landmarks.landmark[1]
                    
                    # Flag if the student looks too far left or right
                    if nose.x < 0.35 or nose.x > 0.65:
                        cv2.putText(frame, "WARNING: LOOKING AWAY", (20, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, "STATUS: FOCUSED", (20, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 3. Encode and stream to browser
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Streaming route for the live webcam feed"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Home page with upload and live proctoring options"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{str(uuid.uuid4())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            if model is None:
                load_model()
                
            try:
                results = model(filepath)
                results_filename = f"res_{unique_filename}"
                results.save(save_dir=app.config['RESULTS_FOLDER'])
                
                session['result_data'] = {
                    'original': unique_filename,
                    'result': results_filename
                }
                return redirect(url_for('results'))
            except Exception as e:
                logger.error(f"Error during detection: {e}")
                return redirect(url_for('index'))
            
    return render_template('index.html')

@app.route('/results')
def results():
    """Display detection results for uploaded images"""
    result_data = session.get('result_data', None)
    if result_data is None:
        flash('No detection results available', 'warning')
        return redirect(url_for('index'))
    return render_template('results.html', result_data=result_data)

@app.route('/live')
def live():
    """Render the live proctoring dashboard"""
    return render_template('live.html')

@app.before_request
def initialize():
    """Ensure the model is loaded before handling the first request"""
    global _model_initialized
    if not globals().get('_model_initialized', False):
        logger.info("Initializing YOLOv5 model...")
        load_model()
        globals()['_model_initialized'] = True

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)