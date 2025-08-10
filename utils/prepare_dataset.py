import os
import json
import shutil
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from class names to class IDs
CLASS_MAPPING = {
    'whole_note': 0, 'half_note': 1, 'quarter_note': 2, 'eighth_note': 3,
    'sixteenth_note': 4, 'thirty_second_note': 5, 'sixty_fourth_note': 6,
    'whole_rest': 7, 'half_rest': 8, 'quarter_rest': 9, 'eighth_rest': 10,
    'sixteenth_rest': 11, 'thirty_second_rest': 12, 'sixty_fourth_rest': 13,
    'treble_clef': 14, 'bass_clef': 15, 'alto_clef': 16, 'tenor_clef': 17,
    'sharp': 18, 'flat': 19, 'natural': 20, 'double_sharp': 21, 'double_flat': 22,
    'time_signature_2_4': 23, 'time_signature_3_4': 24, 'time_signature_4_4': 25,
    'time_signature_6_8': 26, 'time_signature_9_8': 27, 'time_signature_12_8': 28,
    'common_time': 29, 'cut_time': 30, 'bar_line': 31, 'double_bar_line': 32,
    'repeat_start': 33, 'repeat_end': 34, 'tie': 35, 'slur': 36,
    'beam': 37, 'dot': 38, 'staccato': 39, 'accent': 40,
    'fermata': 41, 'trill': 42, 'mordent': 43, 'turn': 44,
    'grace_note': 45, 'chord': 46
}

def convert_to_yolo_format(img_width, img_height, bbox):
    """Convert bounding box to YOLO format."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height

def process_dataset(json_path, output_dir, image_source_dir):
    """Process a single JSON file and create YOLO dataset."""
    logger.info(f"Processing {json_path}")
    
    # Create output directories
    img_output_dir = os.path.join(output_dir, 'images')
    label_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    
    # Load the dataset
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    images = data['images']
    annotations = data['annotations']
    
    # Create a map of image_id to image info
    image_map = {img['id']: img for img in images}
    
    # Group annotations by image_id
    ann_by_img = {}
    for ann in annotations:
        if 'image_id' in ann:
            img_id = ann['image_id']
            if img_id not in ann_by_img:
                ann_by_img[img_id] = []
            ann_by_img[img_id].append(ann)
        
    # Process each image
    for img_info in images:
        img_id = img_info['id']
        if 'file_name' not in img_info:
            logger.warning(f"Image info missing 'file_name': {img_info}")
            continue
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Copy image
        source_img_path = os.path.join(image_source_dir, img_filename)
        dest_img_path = os.path.join(img_output_dir, os.path.basename(img_filename))
        logger.info(f"Source: {source_img_path}")
        logger.info(f"Destination: {dest_img_path}")
        if os.path.exists(source_img_path):
            command = f"cp '{source_img_path}' '{dest_img_path}'"
            print(f"Executing command: {command}")
            os.system(command)
        else:
            logger.warning(f"Image not found: {source_img_path}")
            continue

        # Create label file (if annotations exist)
        label_filename = os.path.splitext(os.path.basename(img_filename))[0] + '.txt'
        label_path = os.path.join(label_output_dir, label_filename)

        if img_id in ann_by_img:
            anns = ann_by_img[img_id]
            with open(label_path, 'w') as f:
                for ann in anns:
                    cat_id = str(ann['category_id'])
                    if cat_id in data['categories']:
                        class_name = data['categories'][cat_id]['name']
                        if class_name in CLASS_MAPPING:
                            class_id = CLASS_MAPPING[class_name]
                            bbox = ann['bbox']  # [x_min, y_min, width, height]

                            # Convert COCO bbox to [x_min, y_min, x_max, y_max]
                            x_min, y_min, width, height = bbox
                            x_max = x_min + width
                            y_max = y_min + height

                            yolo_bbox = convert_to_yolo_format(img_width, img_height, [x_min, y_min, x_max, y_max])

                            f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
        else:
            # Create an empty label file if no annotations exist
            open(label_path, 'w').close()

def main():
    """Main function to prepare the dataset."""
    logger.info("Starting dataset preparation...")
    
    # Define paths
    raw_data_dir = 'datasets/raw/ds2_dense'
    train_json = os.path.join(raw_data_dir, 'deepscores_train.json')
    test_json = os.path.join(raw_data_dir, 'deepscores_test.json')
    image_source_dir = os.path.join(raw_data_dir, 'images')
    
    train_output_dir = 'datasets/train'
    val_output_dir = 'datasets/val'
    
    # Process training data
    logger.info("Processing training data...")
    process_dataset(train_json, train_output_dir, image_source_dir)
    
    # Process validation data
    logger.info("Processing validation data...")
    process_dataset(test_json, val_output_dir, image_source_dir)
    
    logger.info("Dataset preparation finished.")

if __name__ == '__main__':
    main()
