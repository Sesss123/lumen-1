# Google Colab Training Guide

This guide explains how to run the `train_mistral_fast.py` script on Google Colab for free or using Colab Pro.

## Step 1: Upload Your Code to Colab
1. Zip your `lumen-1` folder (or push it to GitHub).
2. Open Google Colab and create a New Notebook.
3. Change Runtime to **T4 GPU** or **A100 GPU** (Runtime -> Change runtime type).

## Step 2: Install Requirements
Run this cell in Colab:
```python
!pip install -q -U torch transformers datasets accelerate peft trl bitsandbytes
```

## Step 3: Clone or Extract Your Project
Run this to unzip your project (assuming you uploaded `lumen-1.zip`):
```python
!unzip lumen-1.zip
%cd lumen-1
```

## Step 4: Run the Training Script
Run the Python training script directly from the Colab cell:
```python
!python scripts/train_mistral_fast.py
```

## Step 5: Save Model to Google Drive (Optional but Recommended)
To prevent losing your trained weights when Colab disconnects, mount your drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy the checkpoints to your Google Drive
!cp -r checkpoints/lumen_mistral_finetuned /content/drive/MyDrive/lumen_model/
```
