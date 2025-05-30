# YaavarHackathon-22PD11
# Image Captioning using VLM (BLIP)
This project implements an image captioning pipeline using the BLIP (Bootstrapped Language Image Pretraining) model. It supports training, evaluation, and inference on custom datasets with metadata for verifiable and context-aware caption generation.

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

# Folder Structure
![image](https://github.com/user-attachments/assets/b865e2c3-3869-45b3-831d-db0317d1f7f0)

# Images
![Screenshot 2025-05-29 121054](https://github.com/user-attachments/assets/7c0c38ba-1df2-4395-8836-c598172ee26b)


# Metadata File Format
![image](https://github.com/user-attachments/assets/14fb7c32-0184-4ea6-a93a-6ac183864337)

# Training
![Screenshot 2025-05-30 101203](https://github.com/user-attachments/assets/0f965372-5c6e-4ed3-bb29-afa94c09cfe7)


# output
![Screenshot 2025-05-30 131655](https://github.com/user-attachments/assets/33c7a239-be73-4885-a66d-1c386bbc980c)


# Setup
1.Clone the repository
   git clone https://github.com/DharaniSowmiyan/YaavarHackathon-22PD11
   
   cd your-repo

2.Install dependencies:
   pip install -r requirements.txt

3.Prepare your data:
Place images in img_folder/

Place corresponding metadata .txt files in metadata_folder/







