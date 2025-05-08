import os
import torch
import yaml
import shutil
import logging
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOTrainer:
    """Class for training YOLOv5 models"""
    
    def __init__(self, data_yaml=None, model_type="yolov5s", img_size=416, batch_size=16, epochs=50, device="cpu"):
        """
        Initialize YOLOTrainer
        
        Args:
            data_yaml: Path to data.yaml file
            model_type: YOLOv5 model type (yolov5s, yolov5m, yolov5l, yolov5x)
            img_size: Image size for training
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use (cuda, cpu)
        """
        self.data_yaml = data_yaml
        self.model_type = model_type
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Check if CUDA is available if device is set to cuda
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU instead")
            self.device = "cpu"
        else:
            self.device = device
        
        # Set up paths
        self.yolov5_dir = "yolov5"
        self.weights_dir = os.path.join("static", "models")
        
        # Create weights directory if it doesn't exist
        os.makedirs(self.weights_dir, exist_ok=True)
        
    def setup_yolov5(self):
        """
        Set up YOLOv5 repository
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Check if YOLOv5 directory already exists
            if os.path.exists(self.yolov5_dir):
                logger.info("YOLOv5 directory already exists")
                return True
                
            # Clone YOLOv5 repository
            logger.info("Cloning YOLOv5 repository")
            clone_cmd = ["git", "clone", "https://github.com/ultralytics/yolov5.git"]
            subprocess.run(clone_cmd, check=True)
            
            # Install requirements
            logger.info("Installing YOLOv5 requirements")
            req_cmd = [sys.executable, "-m", "pip", "install", "-r", os.path.join(self.yolov5_dir, "requirements.txt")]
            subprocess.run(req_cmd, check=True)
            
            logger.info("YOLOv5 setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up YOLOv5: {e}")
            return False
    
    def create_data_yaml(self, train_path, val_path, test_path=None, class_names=None, nc=None):
        """
        Create data.yaml file for training
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data (optional)
            class_names: List of class names
            nc: Number of classes
            
        Returns:
            str: Path to created data.yaml file
        """
        # Create data.yaml content
        data = {
            'train': train_path,
            'val': val_path,
            'nc': nc if nc is not None else len(class_names),
            'names': class_names
        }
        
        if test_path:
            data['test'] = test_path
        
        # Write to file
        yaml_path = 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        self.data_yaml = yaml_path
        return yaml_path
    
    def train(self, project_name="yolov5_training", name=None):
        """
        Train YOLOv5 model
        
        Args:
            project_name: Project name for saving results
            name: Run name for saving results (default: YOLOv5 generates a name)
            
        Returns:
            str: Path to best weights file if successful, None otherwise
        """
        if not self.data_yaml:
            logger.error("data.yaml not provided")
            return None
        
        try:
            # Set up YOLOv5 repository if not already set up
            if not self.setup_yolov5():
                return None
            
            # Prepare training command
            cmd = [
                sys.executable,
                os.path.join(self.yolov5_dir, "train.py"),
                "--img", str(self.img_size),
                "--batch", str(self.batch_size),
                "--epochs", str(self.epochs),
                "--data", self.data_yaml,
                "--weights", f"{self.model_type}.pt",
                "--project", project_name,
                "--device", self.device
            ]
            
            if name:
                cmd.extend(["--name", name])
            
            # Execute training
            logger.info(f"Starting training with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Get path to best weights file
            if name:
                run_dir = os.path.join(project_name, name)
            else:
                # Find the most recent run directory
                runs = list(Path(project_name).glob("exp*"))
                runs.sort(key=os.path.getmtime)
                if not runs:
                    logger.error("No training results found")
                    return None
                run_dir = str(runs[-1])
            
            best_weights = os.path.join(run_dir, "weights", "best.pt")
            if not os.path.exists(best_weights):
                logger.error(f"Best weights file {best_weights} not found")
                return None
            
            # Copy best weights to models directory
            target_path = os.path.join(self.weights_dir, f"{os.path.basename(run_dir)}_best.pt")
            shutil.copy(best_weights, target_path)
            logger.info(f"Best weights copied to {target_path}")
            
            return target_path
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return None
    
    def export_model(self, weights_path, export_format="onnx"):
        """
        Export trained model to different formats
        
        Args:
            weights_path: Path to weights file
            export_format: Format to export to (onnx, torchscript, tflite, etc.)
            
        Returns:
            str: Path to exported model if successful, None otherwise
        """
        if not os.path.exists(weights_path):
            logger.error(f"Weights file {weights_path} not found")
            return None
        
        try:
            # Set up YOLOv5 repository if not already set up
            if not self.setup_yolov5():
                return None
            
            # Prepare export command
            cmd = [
                sys.executable,
                os.path.join(self.yolov5_dir, "export.py"),
                "--weights", weights_path,
                "--include", export_format
            ]
            
            # Execute export
            logger.info(f"Exporting model with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Get path to exported model
            export_path = os.path.splitext(weights_path)[0] + f".{export_format}"
            if not os.path.exists(export_path):
                logger.error(f"Exported model {export_path} not found")
                return None
            
            logger.info(f"Model exported to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error during model export: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Example class names for COCO dataset
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                 'hair drier', 'toothbrush']
    
    trainer = YOLOTrainer(model_type="yolov5s", img_size=416, batch_size=16, epochs=50, device="cpu")
    
    # Create data.yaml for a sample dataset
    data_yaml = trainer.create_data_yaml(
        train_path="path/to/train/images",
        val_path="path/to/val/images",
        class_names=class_names
    )
    
    # Train model
    weights_path = trainer.train(project_name="yolov5_training", name="sample_run")
    
    if weights_path:
        # Export model to ONNX
        exported_path = trainer.export_model(weights_path, export_format="onnx")
        print(f"Exported model: {exported_path}")
