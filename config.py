import os

class Config:
    # Paths
    IMAGE_FOLDER = "img_folder"
    METADATA_FOLDER = "metadata_folder"
    OUTPUT_FOLDER = "output_folder"
    MODEL_SAVE_PATH = "blip_finetuning/trained_model"
    
    # Model
    MODEL_NAME = "Salesforce/blip-image-captioning-base"
    MAX_LENGTH = 100
    NUM_BEAMS = 4
    
    # Training
    BATCH_SIZE = 4  # Reduced for memory
    EPOCHS = 5
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    
    # Image processing
    IMAGE_SIZE = 384
    IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
    IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
    
    # Caption colors
    CONCISE_COLOR = (0, 0, 255)  # Blue
    DETAILED_COLOR = (255, 0, 0)  # Red
    LOW_CONFIDENCE_COLOR = (255, 255, 0)  # Yellow
    
    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.5
    
    # Verifiability settings
    MIN_KEYWORD_MATCH = 0.3  # Minimum ratio of keywords that should match between caption and metadata
    LOG_INCONSISTENCIES = True  # Whether to log inconsistencies to file
    HIGHLIGHT_LOW_CONFIDENCE = True  # Whether to highlight low confidence captions
    UNDERLINE_LOW_CONFIDENCE = True  # Whether to underline low confidence captions

config = Config()