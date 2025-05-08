import os
import torch
import yaml
import pandas as pd
import numpy as np
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOEvaluator:
    """Class for evaluating YOLOv5 models"""
    
    def __init__(self, model_path=None, data_yaml=None, img_size=416, batch_size=16, device="cpu"):
        """
        Initialize YOLOEvaluator
        
        Args:
            model_path: Path to model weights file
            data_yaml: Path to data.yaml file
            img_size: Image size for evaluation
            batch_size: Batch size for evaluation
            device: Device to use (cuda, cpu)
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Check if CUDA is available if device is set to cuda
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU instead")
            self.device = "cpu"
        else:
            self.device = device
        
        # Set up paths
        self.yolov5_dir = "yolov5"
        
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
            
    def evaluate(self):
        """
        Evaluate YOLOv5 model
        
        Returns:
            dict: Evaluation results
        """
        if not self.model_path:
            logger.error("Model path not provided")
            return None
            
        if not self.data_yaml:
            logger.error("data.yaml not provided")
            return None
        
        try:
            # Set up YOLOv5 repository if not already set up
            if not self.setup_yolov5():
                return None
            
            # Prepare evaluation command
            cmd = [
                sys.executable,
                os.path.join(self.yolov5_dir, "val.py"),
                "--img", str(self.img_size),
                "--batch", str(self.batch_size),
                "--data", self.data_yaml,
                "--weights", self.model_path,
                "--task", "val",
                "--device", self.device,
                "--save-json"
            ]
            
            # Execute evaluation
            logger.info(f"Starting evaluation with command: {' '.join(cmd)}")
            start_time = time.time()
            subprocess.run(cmd, check=True)
            eval_time = time.time() - start_time
            
            # Find the results file
            results_dir = Path(self.yolov5_dir) / "runs" / "val"
            results_dirs = list(results_dir.glob("exp*"))
            if not results_dirs:
                logger.error("No evaluation results found")
                return None
                
            # Sort by modification time to get the most recent
            results_dirs.sort(key=os.path.getmtime, reverse=True)
            latest_results_dir = results_dirs[0]
            
            # Parse results from summary file
            results_file = latest_results_dir / "results.csv"
            if not results_file.exists():
                logger.error(f"Results file {results_file} not found")
                return None
                
            # Read results CSV
            results_df = pd.read_csv(results_file)
            
            # Extract metrics
            metrics = {}
            if len(results_df) > 0:
                row = results_df.iloc[0]
                metrics = {
                    "mAP_50": float(row["metrics/mAP_0.5"]),
                    "mAP_50_95": float(row["metrics/mAP_0.5:0.95"]),
                    "precision": float(row["metrics/precision"]),
                    "recall": float(row["metrics/recall"]),
                    "inference_time": float(eval_time),
                    "results_path": str(latest_results_dir)
                }
                
            logger.info(f"Evaluation complete: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return None
            
    def evaluate_image_performance(self, num_iterations=100):
        """
        Evaluate model performance on a single image (inference speed)
        
        Args:
            num_iterations: Number of inference iterations
            
        Returns:
            dict: Performance metrics
        """
        if not self.model_path:
            logger.error("Model path not provided")
            return None
        
        try:
            # Load model
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
            model.to(self.device)
            
            # Create a dummy image
            img = torch.zeros((3, self.img_size, self.img_size), device=self.device)
            
            # Warm-up
            logger.info("Warming up model...")
            for _ in range(10):
                _ = model(img)
            
            # Measure inference time
            logger.info(f"Measuring inference time over {num_iterations} iterations...")
            start_time = time.time()
            for _ in range(num_iterations):
                _ = model(img)
            
            total_time = time.time() - start_time
            avg_time = total_time / num_iterations
            fps = num_iterations / total_time
            
            metrics = {
                "total_time": total_time,
                "avg_inference_time": avg_time,
                "fps": fps,
                "iterations": num_iterations,
                "device": self.device,
                "img_size": self.img_size
            }
            
            logger.info(f"Performance evaluation complete: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during performance evaluation: {e}")
            return None
            
    def compare_models(self, model_paths, num_iterations=50):
        """
        Compare multiple models in terms of performance
        
        Args:
            model_paths: List of model paths to compare
            num_iterations: Number of inference iterations
            
        Returns:
            dict: Comparison results
        """
        results = {}
        
        for model_path in model_paths:
            logger.info(f"Evaluating model: {model_path}")
            
            try:
                # Load model
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                model.to(self.device)
                
                # Create a dummy image
                img = torch.zeros((3, self.img_size, self.img_size), device=self.device)
                
                # Warm-up
                for _ in range(5):
                    _ = model(img)
                
                # Measure inference time
                start_time = time.time()
                for _ in range(num_iterations):
                    _ = model(img)
                
                total_time = time.time() - start_time
                avg_time = total_time / num_iterations
                fps = num_iterations / total_time
                
                # Get model size
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                
                # Store results
                model_name = os.path.basename(model_path)
                results[model_name] = {
                    "avg_inference_time": avg_time,
                    "fps": fps,
                    "model_size_mb": model_size_mb
                }
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_path}: {e}")
                results[os.path.basename(model_path)] = {"error": str(e)}
        
        logger.info(f"Model comparison complete: {results}")
        return results

# Example usage
if __name__ == "__main__":
    # Example for evaluating a model
    evaluator = YOLOEvaluator(
        model_path="path/to/yolov5s.pt",
        data_yaml="data.yaml",
        img_size=416,
        batch_size=16,
        device="cpu"
    )
    
    # Evaluate model
    metrics = evaluator.evaluate()
    print(f"Evaluation metrics: {metrics}")
    
    # Evaluate performance
    performance = evaluator.evaluate_image_performance(num_iterations=100)
    print(f"Performance metrics: {performance}")
