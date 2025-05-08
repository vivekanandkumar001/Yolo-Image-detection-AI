import os
import logging
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import torch
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import time

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path='yolov5s.pt'):
    """Load YOLOv5 model from path or download pre-trained model if not available"""
    global model
    try:
        # Always use torch hub for compatibility
        logger.info(f"Loading model from {model_path}")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True, trust_repo=True)
        
        # Set model parameters
        if hasattr(model, 'conf'):
            model.conf = 0.25  # Confidence threshold
        if hasattr(model, 'iou'):
            model.iou = 0.45   # IoU threshold
        if hasattr(model, 'classes'):
            model.classes = None  # All classes
        if hasattr(model, 'max_det'):
            model.max_det = 1000  # Maximum detections per image
        if hasattr(model, 'eval'):
            model.eval()  # Set to evaluation mode
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """Route for object detection"""
    if request.method == 'POST':
        # Check if model is loaded
        global model
        if model is None:
            if not load_model():
                flash('Failed to load model', 'danger')
                return redirect(url_for('index'))
        
        # Check if file part is in request
        if 'file' not in request.files:
            flash('No file part', 'warning')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file
        if file.filename == '':
            flash('No selected file', 'warning')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            base_name, extension = os.path.splitext(filename)
            unique_filename = f"{base_name}_{unique_id}{extension}"
            
            # Save uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Perform detection
            try:
                logger.info(f"Performing detection on {file_path}")
                start_time = time.time()
                
                # Make sure model is loaded
                if model is None:
                    load_model()
                
                # Run detection
                results = model(file_path)
                inference_time = time.time() - start_time
                
                # Save results
                results_filename = f"result_{unique_id}{extension}"
                results_path = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
                
                # Handle different model output formats (ultralytics YOLO vs torch hub)
                if hasattr(results, 'render'):
                    results.render()  # Updates results.imgs with boxes and labels
                    
                    # Save result image
                    if hasattr(results, 'ims'):
                        for img in results.ims:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(img_rgb)
                            img_pil.save(results_path)
                    else:
                        # Handle case where results.ims might not exist
                        img = Image.open(file_path)
                        img.save(results_path)
                
                # Get detection data
                try:
                    # For torch hub YOLOv5
                    if hasattr(results, 'pandas'):
                        detection_results = results.pandas().xyxy[0].to_dict(orient="records")
                    # For ultralytics YOLO
                    elif hasattr(results, 'boxes') and hasattr(results.boxes, 'data'):
                        detection_results = []
                        for i, box in enumerate(results.boxes.data):
                            x1, y1, x2, y2, conf, cls = box.tolist()
                            class_id = int(cls)
                            class_name = results.names[class_id] if hasattr(results, 'names') else f"class_{class_id}"
                            detection_results.append({
                                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                                'confidence': conf, 'class': class_id, 'name': class_name
                            })
                    else:
                        # Fallback
                        detection_results = []
                except Exception as e:
                    logger.error(f"Error parsing detection results: {e}")
                    detection_results = []
                
                # Prepare results for display
                result_data = {
                    'original_image': os.path.join('uploads', unique_filename),
                    'result_image': os.path.join('results', results_filename),
                    'detections': detection_results,
                    'inference_time': f"{inference_time:.4f}",
                    'total_objects': len(detection_results)
                }
                
                # Save to session
                session['result_data'] = result_data
                
                return redirect(url_for('results'))
            
            except Exception as e:
                logger.error(f"Error during detection: {e}")
                flash(f'Error during detection: {str(e)}', 'danger')
                return redirect(url_for('index'))
        else:
            flash('File type not allowed', 'warning')
            return redirect(request.url)
            
    return render_template('index.html')

@app.route('/results')
def results():
    """Show detection results"""
    result_data = session.get('result_data', None)
    if result_data is None:
        flash('No detection results available', 'warning')
        return redirect(url_for('index'))
    
    return render_template('results.html', result_data=result_data)

@app.route('/train')
def train():
    """Training page"""
    return render_template('train.html')

@app.route('/evaluate')
def evaluate():
    """Evaluation page"""
    return render_template('evaluate.html')

# Initialize the model on startup
# Flask 2.0+ doesn't support @app.before_first_request anymore
# Using @app.before_request with a flag to initialize once
_model_initialized = False

@app.before_request
def initialize():
    """Initialize model before the first request"""
    global _model_initialized
    if not _model_initialized:
        logger.info("Initializing YOLOv5 model...")
        load_model()
        logger.info("Model initialization complete")
        _model_initialized = True

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
