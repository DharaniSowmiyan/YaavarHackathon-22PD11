import os
import shutil
from config import config

def validate_data_structure():
    """Validate the input folder structure"""
    required_folders = [config.IMAGE_FOLDER, config.METADATA_FOLDER]
    
    for folder in required_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        if not os.listdir(folder):
            raise ValueError(f"Folder is empty: {folder}")
    
    # Check matching files
    image_files = set(os.path.splitext(f)[0] for f in os.listdir(config.IMAGE_FOLDER))
    meta_files = set(os.path.splitext(f)[0] for f in os.listdir(config.METADATA_FOLDER))
    
    missing_meta = image_files - meta_files
    if missing_meta:
        raise ValueError(f"Missing metadata for images: {list(missing_meta)[:5]}...")
    
    print("Data structure validation passed.")

def cleanup_output_folder():
    """Clear existing output folder"""
    if os.path.exists(config.OUTPUT_FOLDER):
        shutil.rmtree(config.OUTPUT_FOLDER)
    os.makedirs(config.OUTPUT_FOLDER)