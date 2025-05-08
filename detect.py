import os
import torch
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """YOLOv5 object detector class"""
    
    def __init__(self, model_path=None, conf_thres=0.25, iou_thres=0.45):
        """
        Initialize the YOLOv5 model
        
        Args:
            model_path: Path to the model weights file
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
        """
        self.model = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No model path provided, using pre-trained YOLOv5s")
            self.load_pretrained_model()
    
    def load_model(self, model_path):
        """
        Load YOLOv5 model from a file
        
        Args:
            model_path: Path to the model weights file
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.conf = self.conf_thres
            self.model.iou = self.iou_thres
            self.model.to(self.device)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_pretrained_model(self, model_name='yolov5s'):
        """
        Load pre-trained YOLOv5 model
        
        Args:
            model_name: Name of the pre-trained model (yolov5s, yolov5m, yolov5l, yolov5x)
        """
        try:
            logger.info(f"Loading pre-trained {model_name}")
            self.model = torch.hub.load('ultralytics/yolov5', model_name)
            self.model.conf = self.conf_thres
            self.model.iou = self.iou_thres
            self.model.to(self.device)
            logger.info("Pre-trained model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {e}")
            return False
    
    def detect(self, image_path, save_path=None):
        """
        Perform object detection on an image
        
        Args:
            image_path: Path to the input image
            save_path: Path to save the detection result image
            
        Returns:
            Dictionary with detection results
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # Perform inference
            results = self.model(image_path)
            
            # Save results if path is provided
            if save_path:
                results.render()  # Adds boxes and labels to images
                for img in results.imgs:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(save_path, img_rgb)
            
            # Convert results to dictionary
            detections = results.pandas().xyxy[0].to_dict(orient="records")
            
            return {
                'detections': detections,
                'results_object': results
            }
        
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return None
    
    def detect_video(self, video_path, output_path=None):
        """
        Perform object detection on a video
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the detection result video
            
        Returns:
            Path to the output video
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Error opening video file")
                return None
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video writer if output_path is provided
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Perform detection on the frame
                results = self.model(frame)
                
                # Draw results on the frame
                annotated_frame = results.render()[0]
                
                # Write frame to output video
                if output_path:
                    out.write(annotated_frame)
                
                frame_idx += 1
                if frame_idx % 10 == 0:
                    logger.info(f"Processed {frame_idx}/{frame_count} frames")
            
            # Release resources
            cap.release()
            if output_path:
                out.release()
                
            logger.info(f"Video processing complete. Output saved to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error during video detection: {e}")
            return None

# Example usage
if __name__ == "__main__":
    detector = ObjectDetector()
    detector.load_pretrained_model()
    results = detector.detect("path/to/image.jpg", "path/to/output.jpg")
    print(results)
