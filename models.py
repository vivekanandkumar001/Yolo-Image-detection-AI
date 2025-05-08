from app import db
from datetime import datetime

class DetectionResult(db.Model):
    """Model to store detection results"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    result_path = db.Column(db.String(255), nullable=False)
    inference_time = db.Column(db.Float, nullable=False)
    objects_detected = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<DetectionResult {self.filename}>'

class TrainingJob(db.Model):
    """Model to store training job information"""
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(64), nullable=False)
    epochs = db.Column(db.Integer, nullable=False)
    batch_size = db.Column(db.Integer, nullable=False)
    image_size = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(32), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    model_path = db.Column(db.String(255), nullable=True)
    
    def __repr__(self):
        return f'<TrainingJob {self.dataset_name}>'

class EvaluationResult(db.Model):
    """Model to store model evaluation results"""
    id = db.Column(db.Integer, primary_key=True)
    model_path = db.Column(db.String(255), nullable=False)
    mAP_50 = db.Column(db.Float, nullable=True)
    mAP_50_95 = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    inference_time = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<EvaluationResult {self.model_path}>'
