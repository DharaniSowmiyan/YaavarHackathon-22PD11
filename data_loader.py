import os
import re
from PIL import Image
from torch.utils.data import Dataset
from transformers import BlipProcessor
from config import config
import torchvision.transforms as transforms

class ImageCaptioningDataset(Dataset):
    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode
        self.image_folder = config.IMAGE_FOLDER
        self.metadata_folder = config.METADATA_FOLDER
        
        # Basic image transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        ])
        
        # Get all image files
        self.image_files = [f for f in os.listdir(self.image_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # Parse metadata
        self.metadata = []
        for img_file in self.image_files:
            metadata_file = os.path.join(self.metadata_folder, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(metadata_file):
                metadata = self._parse_metadata_file(metadata_file)
                self.metadata.append({
                    "image_file": img_file,
                    **metadata
                })
    
    def _parse_metadata_file(self, file_path):
        metadata = {
            "section_header": "",
            "above_text": "",
            "caption": "",
            "picture_id": "",
            "footnote": "",
            "below_text": ""
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for field in metadata.keys():
            pattern = re.compile(rf"{field}:\s*(.*?)(?=\n\w+:|$)", re.DOTALL)
            match = pattern.search(content)
            if match:
                value = match.group(1).strip()
                metadata[field] = value if value.lower() != "null" and value else ""
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = os.path.join(self.image_folder, item["image_file"])
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Initialize inputs dictionary
        inputs = {}
        
        # Ensure the image is unsqueezed to add the batch dimension
        inputs["pixel_values"] = image # Shape: (1, 3, height, width)
        
        # Prepare text input
        context = ""
        if item["section_header"]:
            context += f"concise: {item['section_header']}. "
        if item["caption"]:
            context += f"detailed: {item['caption']}"
        
        # For training, use caption as target if available
        if self.mode == "train":
            target_text = item["caption"] if item["caption"] else "Describe the image content"
        else:
            target_text = ""
        
        processed = self.processor(
        text=context if context else "Describe this image:",
        return_tensors="pt",
        padding="max_length",
        max_length=config.MAX_LENGTH,
        truncation=True
        )

        # Remove batch dimension from processor outputs
        for k in processed:
            inputs[k] = processed[k].squeeze(0)

        
        
        if self.mode == "train":
            labels = self.processor(
                text=target_text,
                return_tensors="pt",
                padding="max_length",
                max_length=config.MAX_LENGTH,
                truncation=True
            ).input_ids.squeeze(0)
            inputs["labels"] = labels
        
        return inputs, item["image_file"], context