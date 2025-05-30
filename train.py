import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from torch.utils.data import DataLoader, random_split
from data_loader import ImageCaptioningDataset
from config import config
import os
from tqdm import tqdm
import gc

def train():
    # Initialize processor with proper settings
    processor = BlipProcessor.from_pretrained(
        config.MODEL_NAME,
        size=config.IMAGE_SIZE,
        do_resize=True,
        do_normalize=True,
        image_mean=config.IMAGE_MEAN,
        image_std=config.IMAGE_STD,
        padding_side="left"
    )
    
    model = BlipForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    
    # Create dataset and split
    full_dataset = ImageCaptioningDataset(processor, mode="train")
    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Training loop
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for batch, _, _ in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            
            for k, v in batch.items():
                print(f"{k}: {v.shape}")

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    
    # Save model with memory optimizations
    try:
        print("Saving model...")
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        
        # Save in FP16 to reduce size
        model.half()
        model.save_pretrained(
            config.MODEL_SAVE_PATH,
            safe_serialization=False  # Disable safetensors to save memory
        )
        processor.save_pretrained(config.MODEL_SAVE_PATH)
        model.float()  # Convert back to FP32
        
        print(f"Model saved to {config.MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
        print("Attempting sharded save...")
        
        # Fallback: Save state dict directly
        torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, "pytorch_model.bin"))
        model.config.save_pretrained(config.MODEL_SAVE_PATH)
        processor.save_pretrained(config.MODEL_SAVE_PATH)
        
        print(f"Model saved using fallback method to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()