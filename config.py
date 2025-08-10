import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'static/uploads'
    AUDIO_FOLDER = 'static/audio'
    MODEL_PATH = 'models/best.pt'
    
    # YOLOv8 Settings
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.4
    
    # CORS Settings
    CORS_ORIGINS = ['*']  # Restrict in production
