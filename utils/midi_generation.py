from midiutil import MIDIFile
import logging

logger = logging.getLogger(__name__)

def generate_midi_file(symbols, output_path, bpm=120):
    """Generate MIDI file from detected musical symbols"""
    try:
        # Create MIDI file
        midi = MIDIFile(1)  # One track
        track = 0
        channel = 0
        
        # Add track name and tempo
        midi.addTrackName(track, 0, "Transcribed Music")
        midi.addTempo(track, 0, bpm)
        
        # Convert notes to MIDI
        current_time = 0
        
        for symbol in symbols:
            if 'note' in symbol['class']:
                pitch_num = pitch_to_midi_number(symbol['pitch'])
                duration = symbol['duration']
                velocity = 100  # Default velocity
                
                midi.addNote(track, channel, pitch_num, current_time, duration, velocity)
                current_time += duration
        
        # Write MIDI file
        with open(output_path, 'wb') as output_file:
            midi.writeFile(output_file)
        
        logger.info(f"MIDI file generated: {output_path}")
        
    except Exception as e:
        logger.error(f"MIDI generation failed: {str(e)}")
        raise

def pitch_to_midi_number(pitch):
    """Convert pitch string (e.g., 'C4') to MIDI note number"""
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    try:
        # Parse pitch (e.g., 'C4', 'F#3')
        if len(pitch) == 2:
            note, octave = pitch[0], int(pitch[1])
        elif len(pitch) == 3:
            note, octave = pitch[:2], int(pitch[2])
        else:
            return 60  # Default to C4
        
        midi_number = (octave + 1) * 12 + note_map.get(note, 0)
        return max(0, min(127, midi_number))  # Ensure valid MIDI range
        
    except:
        return 60  # Default to C4 if parsing fails
