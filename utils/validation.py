import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_upload_file(file):
    """Validate uploaded file"""
    if not file:
        raise ValueError("No file provided")
    
    if file.filename == '':
        raise ValueError("No file selected")
    
    # Check file extension
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type {file_ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)")
    
    return True
