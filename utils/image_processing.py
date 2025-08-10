import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_staff_lines_enhanced(image):
    """Improved staff line detection using multiple techniques"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 2
        )
        
        # Morphological operations to clean up staff lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # HoughLinesP for better line detection
        lines = cv2.HoughLinesP(
            cleaned, 1, np.pi/180, threshold=100,
            minLineLength=image.shape[1] * 0.3,  # Minimum 30% of image width
            maxLineGap=10
        )
        
        # Group and validate staff lines
        staff_groups = group_staff_lines(lines, image.shape[0])
        
        logger.info(f"Detected {len(staff_groups)} staff systems")
        return staff_groups
        
    except Exception as e:
        logger.error(f"Staff line detection failed: {str(e)}")
        return []

def group_staff_lines(lines, image_height):
    """Group individual lines into 5-line staff systems"""
    if lines is None:
        return []
    
    # Sort lines by y-coordinate
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10:  # Nearly horizontal
            horizontal_lines.append((y1 + y2) // 2)
    
    horizontal_lines.sort()
    
    # Group into staff systems (5 lines each)
    staff_systems = []
    i = 0
    while i < len(horizontal_lines) - 4:
        staff_group = horizontal_lines[i:i+5]
        # Validate staff spacing (should be roughly equal)
        if validate_staff_spacing(staff_group):
            staff_systems.append(staff_group)
            i += 5
        else:
            i += 1
    
    return staff_systems

def validate_staff_spacing(staff_lines):
    """Validate that staff lines have consistent spacing"""
    if len(staff_lines) != 5:
        return False
    
    spacings = [staff_lines[i+1] - staff_lines[i] for i in range(4)]
    avg_spacing = sum(spacings) / len(spacings)
    
    # Check if all spacings are within 20% of average
    for spacing in spacings:
        if abs(spacing - avg_spacing) > avg_spacing * 0.2:
            return False
    
    return True

def process_detections_enhanced(results, staff_lines):
    """Process YOLOv8 detections and convert to musical symbols"""
    symbols = []
    
    if hasattr(results, 'boxes') and results.boxes is not None:
        boxes = results.boxes
        
        for i in range(len(boxes)):
            # Extract detection data
            bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            
            # Convert to required format
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            center_y = (y1 + y2) / 2
            
            # Get symbol class name
            from app import MUSICAL_SYMBOL_CLASSES
            symbol_class = MUSICAL_SYMBOL_CLASSES.get(class_id, 'unknown')
            
            # Calculate pitch for note symbols
            pitch = "C4"  # Default
            if 'note' in symbol_class:
                pitch = calculate_pitch_from_staff_position(center_y, staff_lines)
            
            # Get duration
            from app import SYMBOL_DURATIONS
            duration = SYMBOL_DURATIONS.get(symbol_class, 1.0)
            
            symbol = {
                'class': symbol_class,
                'confidence': confidence,
                'box': [int(x1), int(y1), int(width), int(height)],
                'pitch': pitch,
                'duration': duration
            }
            
            symbols.append(symbol)
    
    return symbols

def calculate_pitch_from_staff_position(y_pos, staff_lines):
    """Calculate pitch based on position relative to staff lines"""
    if not staff_lines:
        return "C4"
    
    # Define pitch sequence (treble clef)
    pitches = ['F5', 'E5', 'D5', 'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4']
    
    # Calculate relative position
    staff_top = min(staff_lines)
    staff_bottom = max(staff_lines)
    staff_height = staff_bottom - staff_top
    line_spacing = staff_height / 4
    
    # Determine pitch index based on position
    relative_pos = (y_pos - staff_top) / line_spacing
    pitch_index = int(relative_pos)
    
    if 0 <= pitch_index < len(pitches):
        return pitches[pitch_index]
    
    return "C4"
