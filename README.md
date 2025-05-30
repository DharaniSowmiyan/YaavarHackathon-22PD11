# YaavarHackathon-22PD11
# Image Captioning using VLM (BLIP)
This project implements an image captioning pipeline using the BLIP (Bootstrapped Language Image Pretraining) model. It supports training, evaluation, and inference on custom datasets with metadata for verifiable and context-aware caption generation.

# why Salesforce/blip-image-captioning-base?

one of the most used and trending models for image captioning.

Pretrained on large image-text datasets, enabling strong captioning out-of-the-box.

Specifically optimized for image captioning with high-quality, natural language outputs.

Lightweight "base" model offers a good trade-off between accuracy and speed.

Easily fine-tunable for domain-specific tasks using Hugging Face Transformers.

# 1.Architecture overview
![image](https://github.com/user-attachments/assets/8c1b218d-5512-4205-8bf7-9d3dd30b3ac6)

# 2. Component Breakdown
A. Data Preparation Layer
![image](https://github.com/user-attachments/assets/8d90858f-6e06-4e38-9e39-db0b244829be)

B. Model Training Layer
![image](https://github.com/user-attachments/assets/b6e7446b-ad2a-4fb4-8ba6-2d77c495a46a)

C. Inference Layer
![image](https://github.com/user-attachments/assets/1d413349-dc7e-4797-91e5-d8c00b68b6f1)

# 3. Data Flow
![image](https://github.com/user-attachments/assets/56b42824-183e-4485-8d0f-e79e48f67537)

# Features
BLIP-based image captioning (Salesforce/blip-image-captioning-base)

Custom dataset support with metadata (section headers, captions, etc.)

Training and inference scripts

Caption verifiability checks (semantic similarity, consistency)

Annotated output images with generated captions and confidence scores

# Salesforce/blip-image-captioning-base model without fine-tuning
![Screenshot 2025-05-30 215938](https://github.com/user-attachments/assets/85773ac3-d267-4b9d-a769-c930b0369f40)

 
# Folder Structure
![image](https://github.com/user-attachments/assets/b865e2c3-3869-45b3-831d-db0317d1f7f0)

# Images for fine-tuning(circuits,graphs,logos,charts)
![Screenshot 2025-05-29 121054](https://github.com/user-attachments/assets/7c0c38ba-1df2-4395-8836-c598172ee26b)


# Metadata File Format
![image](https://github.com/user-attachments/assets/14fb7c32-0184-4ea6-a93a-6ac183864337)

# Training
![Screenshot 2025-05-30 101203](https://github.com/user-attachments/assets/0f965372-5c6e-4ed3-bb29-afa94c09cfe7)


# output
![Screenshot 2025-05-30 131655](https://github.com/user-attachments/assets/33c7a239-be73-4885-a66d-1c386bbc980c)
![Screenshot 2025-05-30 083942](https://github.com/user-attachments/assets/7d684264-01bb-4e09-9e76-e01524d4eca3)

# low confidence outputs are underlined
![Screenshot 2025-05-30 144656](https://github.com/user-attachments/assets/8f2a1556-eb3b-49e7-96fe-6d83e832e4dc)

# Setup
1.Clone the repository
   git clone https://github.com/DharaniSowmiyan/YaavarHackathon-22PD11
   
   cd your-repo

2.Install dependencies:
   pip install -r requirements.txt

3.Prepare your data:
Place images in img_folder/

Place corresponding metadata .txt files in metadata_folder/

# Resources:

Creating an image dataset:

https://huggingface.co/docs/datasets/image_dataset


Fine-tune BLIP on an image captioning dataset:

https://colab.research.google.com/drive/1lbqiSiA0sDF7JDWPeS0tccrM85LloVha?usp=sharing#scrollTo=AFGnjCgDoLIJ

# Various other models explored:

1.Florence-2

![Screenshot 2025-05-28 215659](https://github.com/user-attachments/assets/edb98b68-f73a-4b9f-a8f5-9a58145dcfc8)
![Screenshot 2025-05-28 215537](https://github.com/user-attachments/assets/a3fd556d-7b74-46a7-8818-6dbe6eba140f)


https://blog.roboflow.com/fine-tune-florence-2-object-detection/

Why it did not work:

Florence-2 requires flash-attn, which needs:

GPUs like A100, L4, or H100.

CUDA 11.8+ and PyTorch 2.1+.

2.Salesforce/blip2-opt-2.7b

https://huggingface.co/Salesforce/blip2-opt-2.7b

3.Salesforce/blip2-flan-t5-xl

https://huggingface.co/Salesforce/blip2-flan-t5-x

(2,3)Why it did not work: while running the model-out of memory since my gpu VRAM is 4GB

4.huggingface.co/yifeihu/TF-ID-base

https://huggingface.co/yifeihu/TF-ID-base

Why it did not work: it detects tables/figures image captioning not supported.

# Future work:

Making them(image+context and caption) better Relevant: How well the caption aligns with both the image and the provided context.







