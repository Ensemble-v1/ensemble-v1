from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from logging.handlers import RotatingFileHandler
import hashlib
from functools import lru_cache
from utils.image_processing import detect_staff_lines_enhanced, process_detections_enhanced
from utils.midi_generation import generate_midi_file
from utils.validation import validate_upload_file
from config import Config

app = Flask(__name__)
CORS(app)

# Load configuration
app.config.from_object(Config)

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    handlers=[RotatingFileHandler('logs/ensemble.log', maxBytes=100000, backupCount=10)],
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)

# Load YOLOv8 model
try:
    model = YOLO('models/best.pt')
    logging.info("YOLOv8 model loaded successfully")
except Exception as e:
    logging.warning(f"Could not load YOLOv8 model: {e}")
    model = None

# Musical symbol classes (47 classes)
MUSICAL_SYMBOL_CLASSES = {
    0: 'whole_note', 1: 'half_note', 2: 'quarter_note', 3: 'eighth_note',
    4: 'sixteenth_note', 5: 'thirty_second_note', 6: 'sixty_fourth_note',
    7: 'whole_rest', 8: 'half_rest', 9: 'quarter_rest', 10: 'eighth_rest',
    11: 'sixteenth_rest', 12: 'thirty_second_rest', 13: 'sixty_fourth_rest',
    14: 'treble_clef', 15: 'bass_clef', 16: 'alto_clef', 17: 'tenor_clef',
    18: 'sharp', 19: 'flat', 20: 'natural', 21: 'double_sharp', 22: 'double_flat',
    23: 'time_signature_2_4', 24: 'time_signature_3_4', 25: 'time_signature_4_4',
    26: 'time_signature_6_8', 27: 'time_signature_9_8', 28: 'time_signature_12_8',
    29: 'common_time', 30: 'cut_time', 31: 'bar_line', 32: 'double_bar_line',
    33: 'repeat_start', 34: 'repeat_end', 35: 'tie', 36: 'slur',
    37: 'beam', 38: 'dot', 39: 'staccato', 40: 'accent',
    41: 'fermata', 42: 'trill', 43: 'mordent', 44: 'turn',
    45: 'grace_note', 46: 'chord'
}

SYMBOL_DURATIONS = {
    'whole_note': 4.0, 'half_note': 2.0, 'quarter_note': 1.0,
    'eighth_note': 0.5, 'sixteenth_note': 0.25, 'thirty_second_note': 0.125,
    'sixty_fourth_note': 0.0625, 'whole_rest': 4.0, 'half_rest': 2.0,
    'quarter_rest': 1.0, 'eighth_rest': 0.5, 'sixteenth_rest': 0.25,
    'thirty_second_rest': 0.125, 'sixty_fourth_rest': 0.0625
}

@lru_cache(maxsize=100)
def get_cached_analysis(image_hash):
    """Cache analysis results to avoid reprocessing identical images"""
    return None  # Implement actual caching logic

def calculate_image_hash(image_path):
    """Calculate hash for image to enable caching"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def save_uploaded_file(file):
    """Save uploaded file and return path"""
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    filename = f"sheet_music_{hash(file.filename + str(os.urandom(16)))}.jpg"
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)
    return filepath

def calculate_pitch_from_position(y_pos, staff_lines, clef_type='treble'):
    """Calculate pitch based on vertical position and clef"""
    if not staff_lines:
        return "C4"
    
    # Define pitch mappings for different clefs
    treble_pitches = ['E5', 'D5', 'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4', 'B3']
    bass_pitches = ['G3', 'F3', 'E3', 'D3', 'C3', 'B2', 'A2', 'G2', 'F2', 'E2', 'D2']
    alto_pitches = ['G4', 'F4', 'E4', 'D4', 'C4', 'B3', 'A3', 'G3', 'F3', 'E3', 'D3']
    
    pitch_mapping = {
        'treble': treble_pitches,
        'bass': bass_pitches,
        'alto': alto_pitches
    }
    
    pitches = pitch_mapping.get(clef_type, treble_pitches)
    
    # Calculate position relative to staff
    staff_center = sum(staff_lines) / len(staff_lines)
    staff_spacing = (max(staff_lines) - min(staff_lines)) / 4
    
    # Determine pitch index based on position
    relative_pos = (y_pos - staff_center) / staff_spacing
    pitch_index = int(5 + relative_pos)  # 5 is middle of staff
    
    if 0 <= pitch_index < len(pitches):
        return pitches[pitch_index]
    
    return "C4"  # Default fallback

def apply_accidentals_to_notes(symbols):
    """Apply accidentals (sharp, flat, natural) to nearby notes"""
    accidentals = []
    notes = []
    
    # Separate accidentals and notes
    for symbol in symbols:
        if symbol['class'] in ['sharp', 'flat', 'natural']:
            accidentals.append(symbol)
        elif 'note' in symbol['class']:
            notes.append(symbol)
    
    # Apply accidentals to notes
    for note in notes:
        note_x = note['box'][0]
        note_y = note['box'][1]
        
        # Find closest accidental to the left
        closest_accidental = None
        min_distance = float('inf')
        
        for accidental in accidentals:
            acc_x = accidental['box'][0]
            acc_y = accidental['box'][1]
            
            # Check if accidental is to the left and roughly same height
            if acc_x < note_x and abs(acc_y - note_y) < 20:
                distance = note_x - acc_x
                if distance < min_distance:
                    min_distance = distance
                    closest_accidental = accidental
        
        # Apply accidental
        if closest_accidental:
            note['accidental'] = closest_accidental['class']
            # Modify pitch based on accidental
            pitch = note['pitch']
            if closest_accidental['class'] == 'sharp':
                note['pitch'] = pitch.replace('4', '#4') if '4' in pitch else pitch + '#'
            elif closest_accidental['class'] == 'flat':
                note['pitch'] = pitch.replace('4', 'b4') if '4' in pitch else pitch + 'b'
    
    return [s for s in symbols if s['class'] not in ['sharp', 'flat', 'natural']]

def group_symbols_by_measure(symbols, staff_lines):
    """Group symbols into measures for proper timing"""
    measures = []
    
    # Sort symbols by x-coordinate (left to right)
    sorted_symbols = sorted(symbols, key=lambda s: s['box'][0])
    
    # Detect measure boundaries (bar lines)
    bar_lines = [s for s in sorted_symbols if s['class'] in ['bar_line', 'double_bar_line']]
    
    if not bar_lines:
        # If no bar lines detected, create single measure
        return [sorted_symbols]
    
    # Group symbols between bar lines
    measures = []
    measure_start = 0
    
    for bar_line in bar_lines:
        bar_x = bar_line['box'][0]
        measure_symbols = [
            s for s in sorted_symbols 
            if measure_start <= s['box'][0] < bar_x and 'bar_line' not in s['class']
        ]
        if measure_symbols:
            measures.append(measure_symbols)
        measure_start = bar_x
    
    # Add final measure after last bar line
    final_symbols = [
        s for s in sorted_symbols 
        if s['box'][0] > bar_lines[-1]['box'][0] and 'bar_line' not in s['class']
    ]
    if final_symbols:
        measures.append(final_symbols)
    
    return measures

def analyze_sheet_music(image_path):
    """Enhanced analysis with better error handling and confidence scoring"""
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        logging.info(f"Analyzing image: {image_path}")
        
        # Detect staff lines
        staff_lines = detect_staff_lines_enhanced(image)
        if not staff_lines:
            logging.warning("No staff lines detected")
            staff_lines = [[100, 150, 200, 250, 300]]  # Mock staff lines
        
        # YOLOv8 detection
        if model:
            results = model.predict(
                image_path, 
                conf=0.25,  # Minimum confidence
                iou=0.4,    # IoU threshold for NMS
                save=False,
                verbose=False
            )
            symbols = process_detections_enhanced(results[0], staff_lines[0])
        else:
            # Mock data for testing
            logging.warning("Using mock detection data")
            symbols = [
                {
                    'class': 'quarter_note',
                    'confidence': 0.85,
                    'box': [150, 180, 20, 25],
                    'pitch': 'E4',
                    'duration': 1.0
                },
                {
                    'class': 'half_note',
                    'confidence': 0.90,
                    'box': [200, 160, 20, 25],
                    'pitch': 'G4',
                    'duration': 2.0
                }
            ]
        
        # Apply accidentals
        symbols = apply_accidentals_to_notes(symbols)
        
        # Group into measures
        measures = group_symbols_by_measure(symbols, staff_lines)
        
        # Extract musical information
        time_signature = "4/4"  # Default, can be detected from symbols
        key_signature = "C major"  # Default, can be detected from key signature symbols
        bpm = 120  # Default tempo
        
        # Generate MIDI file
        midi_filename = f"transcription_{hash(image_path)}.mid"
        midi_path = os.path.join('static/audio', midi_filename)
        
        if not os.path.exists('static/audio'):
            os.makedirs('static/audio')
        
        generate_midi_file(symbols, midi_path, bpm)
        
        # Prepare response
        result = {
            "status": "success",
            "original_image_url": f"http://localhost:5000/{image_path}",
            "midi_url": f"http://localhost:5000/{midi_path}",
            "analysis": {
                "bpm": bpm,
                "time_signature": time_signature,
                "key_signature": key_signature,
                "notes": [
                    {
                        "pitch": symbol['pitch'],
                        "duration": "quarter" if symbol['duration'] == 1.0 else "half",
                        "box": symbol['box']
                    }
                    for symbol in symbols if 'note' in symbol['class']
                ],
                "measures": len(measures),
                "symbols_detected": len(symbols)
            }
        }
        
        logging.info(f"Analysis complete: {len(symbols)} symbols detected")
        return result
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}"
        }

@app.route('/api/analyze', methods=['POST'])
def analyze_sheet_music_endpoint():
    """Main API endpoint for sheet music analysis"""
    try:
        # Validate file
        file = request.files.get('sheet_music')
        validate_upload_file(file)
        
        # Save and hash file
        image_path = save_uploaded_file(file)
        image_hash = calculate_image_hash(image_path)
        
        logging.info(f"Processing file: {file.filename}, hash: {image_hash}")
        
        # Check cache first
        cached_result = get_cached_analysis(image_hash)
        if cached_result:
            logging.info(f"Returning cached result for {image_hash}")
            return jsonify(cached_result)
        
        # Process image
        result = analyze_sheet_music(image_path)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Analysis endpoint error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logging.error(f"Unhandled exception: {str(error)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'message': 'An internal error occurred'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
