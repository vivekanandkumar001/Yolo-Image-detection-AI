import os
import uuid
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import logging
import base64
import yaml
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_unique_filename(filename):
    """
    Generate a unique filename by adding a UUID
    
    Args:
        filename: Original filename
        
    Returns:
        str: Unique filename
    """
    base_name, extension = os.path.splitext(filename)
    unique_id = str(uuid.uuid4())
    return f"{base_name}_{unique_id}{extension}"

def ensure_dir(directory):
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_image(image_path):
    """
    Load an image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy.ndarray: Image as numpy array
    """
    try:
        img = Image.open(image_path)
        img_np = np.array(img)
        return img_np
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def save_image(image, save_path):
    """
    Save an image to file
    
    Args:
        image: Image as numpy array
        save_path: Path to save image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if necessary (OpenCV uses BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return False
            
        image_pil.save(save_path)
        return True
    except Exception as e:
        logger.error(f"Error saving image to {save_path}: {e}")
        return False

def plot_detection_results(image, detections, save_path=None):
    """
    Plot detection results on image
    
    Args:
        image: Image as numpy array
        detections: List of detection dictionaries with xmin, ymin, xmax, ymax, confidence, class
        save_path: Path to save result image
        
    Returns:
        numpy.ndarray: Image with detections plotted if save_path is None, otherwise None
    """
    try:
        # Make a copy of the image to avoid modifying the original
        img_copy = image.copy()
        
        # Create figure and axis
        fig, ax = plt.subplots(1)
        ax.imshow(img_copy)
        
        # Define colors for different classes (can be extended with more colors)
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        
        # Plot each detection
        for i, det in enumerate(detections):
            # Extract bounding box coordinates
            xmin, ymin, xmax, ymax = det['xmin'], det['ymin'], det['xmax'], det['ymax']
            width = xmax - xmin
            height = ymax - ymin
            
            # Determine color based on class
            class_id = det.get('class', 0)
            color = colors[class_id % len(colors)]
            
            # Create rectangle patch
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            class_name = det.get('name', f"Class {class_id}")
            confidence = det.get('confidence', 0)
            label = f"{class_name}: {confidence:.2f}"
            ax.text(xmin, ymin - 5, label, color=color, fontsize=12, backgroundcolor='white')
        
        # Remove axis
        ax.axis('off')
        
        # Save or return the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            return None
        else:
            # Convert plot to image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buffer.seek(0)
            img_pil = Image.open(buffer)
            return np.array(img_pil)
    
    except Exception as e:
        logger.error(f"Error plotting detection results: {e}")
        return None

def create_data_yaml(train_path, val_path, test_path=None, class_names=None, nc=None, yaml_path='data.yaml'):
    """
    Create data.yaml file for YOLOv5
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data (optional)
        class_names: List of class names
        nc: Number of classes
        yaml_path: Path to save yaml file
        
    Returns:
        str: Path to created yaml file
    """
    try:
        # Create data dictionary
        data = {
            'train': train_path,
            'val': val_path,
            'nc': nc if nc is not None else len(class_names),
            'names': class_names
        }
        
        if test_path:
            data['test'] = test_path
        
        # Write to file
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        return yaml_path
    
    except Exception as e:
        logger.error(f"Error creating data.yaml: {e}")
        return None

def convert_to_yolo_format(annotations, image_width, image_height, class_mapping=None):
    """
    Convert bounding box annotations to YOLOv5 format
    
    Args:
        annotations: List of annotation dictionaries with xmin, ymin, xmax, ymax, class_name
        image_width: Width of the image
        image_height: Height of the image
        class_mapping: Dictionary mapping class names to class IDs
        
    Returns:
        list: List of YOLOv5 format annotations (class_id, x_center, y_center, width, height)
    """
    yolo_annotations = []
    
    for ann in annotations:
        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
        
        # Convert to YOLOv5 format (x_center, y_center, width, height) normalized
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        # Get class ID
        if class_mapping and 'class_name' in ann:
            class_id = class_mapping.get(ann['class_name'], 0)
        else:
            class_id = ann.get('class_id', 0)
        
        # Create YOLOv5 format annotation
        yolo_ann = [class_id, x_center, y_center, width, height]
        yolo_annotations.append(yolo_ann)
    
    return yolo_annotations

def image_to_base64(image):
    """
    Convert image to base64 string
    
    Args:
        image: Image as numpy array or PIL Image
        
    Returns:
        str: Base64 encoded image
    """
    try:
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if necessary (OpenCV uses BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return None
            
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        img_bytes = buffer.read()
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return base64_str
    
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def get_available_models(models_dir="static/models"):
    """
    Get list of available models
    
    Args:
        models_dir: Directory to search for models
        
    Returns:
        list: List of available model paths
    """
    try:
        ensure_dir(models_dir)
        models = []
        
        # Find .pt files in models directory
        for file in os.listdir(models_dir):
            if file.endswith(".pt"):
                models.append(os.path.join(models_dir, file))
        
        return models
    
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []

def xyxy_to_xywh(box, img_width, img_height):
    """
    Convert xyxy format (xmin, ymin, xmax, ymax) to xywh format (x_center, y_center, width, height)
    
    Args:
        box: List or tuple of xmin, ymin, xmax, ymax
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        tuple: (x_center, y_center, width, height) normalized
    """
    xmin, ymin, xmax, ymax = box
    
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return (x_center, y_center, width, height)

def xywh_to_xyxy(box, img_width, img_height):
    """
    Convert xywh format (x_center, y_center, width, height) to xyxy format (xmin, ymin, xmax, ymax)
    
    Args:
        box: List or tuple of x_center, y_center, width, height (normalized)
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        tuple: (xmin, ymin, xmax, ymax)
    """
    x_center, y_center, width, height = box
    
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    
    return (xmin, ymin, xmax, ymax)
