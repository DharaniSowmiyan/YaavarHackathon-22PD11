import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from torch.utils.data import DataLoader
from data_loader import ImageCaptioningDataset
from config import config
import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=f'caption_verification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

nltk.download('punkt')

def calculate_semantic_similarity(reference, candidate):
    try:
        if not reference or not candidate:
            return 0.0
            
        reference = str(reference).strip()
        candidate = str(candidate).strip()
        
        if not reference or not candidate:
            return 0.0
            
        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        
        # Calculate BLEU score
        reference_tokens = [nltk.word_tokenize(reference)]
        candidate_tokens = nltk.word_tokenize(candidate)
        bleu_score = nltk.translate.bleu_score.sentence_bleu(reference_tokens, candidate_tokens)
        
        # Combined score
        similarity_score = (
            0.4 * rouge_scores['rouge1'].fmeasure + 
            0.4 * rouge_scores['rougeL'].fmeasure + 
            0.2 * bleu_score
        )
        
        return max(0.0, min(1.0, similarity_score))
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0

def check_caption_consistency(caption, metadata):
    """Check if generated caption is consistent with metadata"""
    inconsistencies = []
    
    # Check section header consistency
    if metadata.get("section_header"):
        section_keywords = set(metadata["section_header"].lower().split())
        caption_keywords = set(caption.lower().split())
        if not section_keywords.intersection(caption_keywords):
            inconsistencies.append(f"Caption may not align with section header: {metadata['section_header']}")
    
    # Check caption consistency with existing caption
    if metadata.get("caption"):
        existing_caption = metadata["caption"].lower()
        if not any(word in existing_caption for word in caption.lower().split()):
            inconsistencies.append(f"Generated caption differs significantly from existing caption: {metadata['caption']}")
    
    # Check footnote consistency
    if metadata.get("footnote"):
        footnote_keywords = set(metadata["footnote"].lower().split())
        if not footnote_keywords.intersection(caption.lower().split()):
            inconsistencies.append(f"Caption may not reflect footnote information: {metadata['footnote']}")
    
    return inconsistencies

def generate_captions():
    # Load model and processor
    processor = BlipProcessor.from_pretrained(config.MODEL_SAVE_PATH)
    model = BlipForConditionalGeneration.from_pretrained(config.MODEL_SAVE_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Create dataset
    eval_dataset = ImageCaptioningDataset(processor, mode="eval")
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    # Prepare output
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    captions_data = {}
    low_confidence_count = 0
    inconsistency_count = 0
    
    model.eval()
    with torch.no_grad():
        for batch, image_file, context in tqdm(eval_dataloader, desc="Generating captions"):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            image_file = image_file[0]
            context = str(context[0]) if context else ""
            
            # Get metadata for consistency checking
            metadata = eval_dataset.metadata[eval_dataset.image_files.index(image_file)]
            
            # Generate concise caption
            concise_output = model.generate(
                **batch,
                max_new_tokens=30,
                num_beams=config.NUM_BEAMS,
                no_repeat_ngram_size=2
            )
            concise_caption = processor.decode(concise_output[0], skip_special_tokens=True)
            
            # Generate detailed caption
            detailed_output = model.generate(
                **batch,
                max_new_tokens=config.MAX_LENGTH,
                num_beams=config.NUM_BEAMS
            )
            detailed_caption = processor.decode(detailed_output[0], skip_special_tokens=True)
            
            # Calculate confidence
            concise_conf = calculate_semantic_similarity(context, concise_caption)
            detailed_conf = calculate_semantic_similarity(context, detailed_caption)
            
            # Check for inconsistencies
            concise_inconsistencies = check_caption_consistency(concise_caption, metadata)
            detailed_inconsistencies = check_caption_consistency(detailed_caption, metadata)
            
            # Log low confidence and inconsistencies
            if concise_conf < config.CONFIDENCE_THRESHOLD or detailed_conf < config.CONFIDENCE_THRESHOLD:
                low_confidence_count += 1
                logging.warning(f"Low confidence for {image_file}: Concise={concise_conf:.2f}, Detailed={detailed_conf:.2f}")
            
            if concise_inconsistencies or detailed_inconsistencies:
                inconsistency_count += 1
                logging.warning(f"Inconsistencies found in {image_file}:")
                for inc in concise_inconsistencies:
                    logging.warning(f"  Concise: {inc}")
                for inc in detailed_inconsistencies:
                    logging.warning(f"  Detailed: {inc}")
            
            # Save results
            captions_data[image_file] = {
                "concise_caption": concise_caption,
                "detailed_caption": detailed_caption,
                "concise_confidence": float(concise_conf),
                "detailed_confidence": float(detailed_conf),
                "context": context,
                "concise_inconsistencies": concise_inconsistencies,
                "detailed_inconsistencies": detailed_inconsistencies
            }
            
            # Save annotated image
            image_path = os.path.join(config.IMAGE_FOLDER, image_file)
            annotated_image = create_annotated_image(
                image_path,
                concise_caption,
                detailed_caption,
                concise_conf,
                detailed_conf,
                concise_inconsistencies,
                detailed_inconsistencies
            )
            output_path = os.path.join(config.OUTPUT_FOLDER, f"annotated_{image_file}")
            annotated_image.save(output_path)
    
    # Log summary statistics
    logging.info(f"Processing complete. Found {low_confidence_count} low confidence captions and {inconsistency_count} inconsistencies.")
    
    # Save JSON
    with open(os.path.join(config.OUTPUT_FOLDER, "captions.json"), 'w') as f:
        json.dump(captions_data, f, indent=2)
    
    print(f"Results saved to {config.OUTPUT_FOLDER}")

def create_annotated_image(image_path, concise_caption, detailed_caption, concise_conf, detailed_conf, concise_inconsistencies, detailed_inconsistencies):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Text parameters
    font_scale = 0.7
    thickness = 2
    margin = 20
    
    # Concise caption (top)
    concise_color = config.CONCISE_COLOR
    if concise_conf < config.CONFIDENCE_THRESHOLD or concise_inconsistencies:
        concise_color = config.LOW_CONFIDENCE_COLOR
    
    # Add underline for low confidence
    if concise_conf < config.CONFIDENCE_THRESHOLD:
        text_size = cv2.getTextSize(f"Concise: {concise_caption}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.line(
            cv_image,
            (margin, margin + 35),
            (margin + text_size[0], margin + 35),
            concise_color,
            2
        )
    
    cv2.putText(
        cv_image,
        f"Concise: {concise_caption} ({concise_conf:.2f})",
        (margin, margin + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        concise_color,
        thickness,
        cv2.LINE_AA
    )
    
    # Detailed caption (bottom)
    detailed_color = config.DETAILED_COLOR
    if detailed_conf < config.CONFIDENCE_THRESHOLD or detailed_inconsistencies:
        detailed_color = config.LOW_CONFIDENCE_COLOR
    
    # Split into multiple lines
    max_line_length = 60
    detailed_lines = [detailed_caption[i:i+max_line_length] 
                     for i in range(0, len(detailed_caption), max_line_length)]
    
    y_position = img_height - margin - (len(detailed_lines) * 30)
    
    # Add underline for low confidence
    if detailed_conf < config.CONFIDENCE_THRESHOLD:
        for i, line in enumerate(detailed_lines):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.line(
                cv_image,
                (margin, y_position + (i * 30) + 5),
                (margin + text_size[0], y_position + (i * 30) + 5),
                detailed_color,
                2
            )
    
    for i, line in enumerate(detailed_lines):
        cv2.putText(
            cv_image,
            line,
            (margin, y_position + (i * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            detailed_color,
            thickness,
            cv2.LINE_AA
        )
    
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    generate_captions()