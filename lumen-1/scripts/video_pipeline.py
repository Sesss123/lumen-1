#!/usr/bin/env python3
"""
Lumen-1 Video & Audio Multimodal Pipeline.
Extracts keyframes and audio from video files, runs them through 
ViT-SigLIP and Conformer-S encoders, and runs autoregressive decoding 
to detect hazards and describe the video in Sinhala and English.
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from lumen.model.lumen_model import LumenForCausalLM
from lumen.model.config import LumenConfig, ModelSize
from lumen.tokenizer.lumen_tokenizer import LumenTokenizer

# Optional dependencies
OPENCV_AVAILABLE = False
LIBROSA_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("⚠️ opencv-python (cv2) is not installed. Video extraction will use mock frames.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ librosa is not installed. Audio extraction will use mock audio bins.")


def extract_video_keyframes(video_path: str, max_frames: int = 16, img_size: int = 384) -> torch.Tensor:
    """
    Extracts keyframes from a video file and processes them for the Vision Encoder.
    Returns tensor of shape (1, T, 3, img_size, img_size).
    """
    if not OPENCV_AVAILABLE or not os.path.exists(video_path):
        print(f"🔄 Using simulated video data (shape: 1, {max_frames}, 3, {img_size}, {img_size}).")
        return torch.randn(1, max_frames, 3, img_size, img_size)

    # Check if input path is an image
    ext = os.path.splitext(video_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
        print(f"📸 Reading photo/image from {video_path}...")
        frame = cv2.imread(video_path)
        if frame is not None:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to model input size
            frame = cv2.resize(frame, (img_size, img_size))
            # Normalize
            frame_data = frame.astype(np.float32) / 255.0
            frame_data = (frame_data - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            # HWC to CHW
            frame_data = np.transpose(frame_data, (2, 0, 1))
            # Shape: (1, 1, 3, img_size, img_size)
            frames_tensor = torch.tensor(frame_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            print(f"✅ Loaded image. Shape: {frames_tensor.shape}")
            return frames_tensor
        else:
            print("❌ cv2.imread failed to load the image. Falling back to mock frames.")

    print(f"🎬 Extracting up to {max_frames} keyframes from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video file. Falling back to mock frames.")
        return torch.randn(1, max_frames, 3, img_size, img_size)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"📊 Video Details: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds.")

    # Sample frames evenly
    num_to_sample = min(max_frames, total_frames) if total_frames > 0 else max_frames
    indices = np.linspace(0, max(0, total_frames - 1), num_to_sample, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        frame = cv2.resize(frame, (img_size, img_size))
        # Normalize to [0, 1] and standard ImageNet values
        frame_data = frame.astype(np.float32) / 255.0
        frame_data = (frame_data - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        # HWC to CHW
        frame_data = np.transpose(frame_data, (2, 0, 1))
        frames.append(frame_data)

    cap.release()

    if len(frames) == 0:
        print("⚠️ Failed to extract any valid frames. Using mock frames.")
        return torch.randn(1, max_frames, 3, img_size, img_size)

    frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0)
    print(f"✅ Extracted {frames_tensor.shape[1]} frames. Shape: {frames_tensor.shape}")
    return frames_tensor


def extract_audio_mel(video_path: str, sample_rate: int = 16000, n_mels: int = 80) -> torch.Tensor:
    """
    Extracts the audio track and computes log-mel spectrogram.
    Returns tensor of shape (1, audio_len, n_mels).
    """
    if os.path.exists(video_path):
        ext = os.path.splitext(video_path)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
            print("📸 Image input has no audio track. Providing empty audio representations.")
            return torch.zeros(1, 1, n_mels)

    if not LIBROSA_AVAILABLE or not os.path.exists(video_path):
        # 50 tokens/sec * 10 seconds = 500 length representation
        print(f"🔄 Using simulated audio data (shape: 1, 500, {n_mels}).")
        return torch.randn(1, 500, n_mels)

    print(f"🎵 Extracting audio from {video_path}...")
    try:
        y, sr = librosa.load(video_path, sr=sample_rate, mono=True)
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=400, hop_length=160)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Normalize log-mel
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        # Shape: (1, time_steps, n_mels)
        mel_tensor = torch.tensor(log_mel.T, dtype=torch.float32).unsqueeze(0)
        print(f"✅ Audio spectrogram computed. Shape: {mel_tensor.shape}")
        return mel_tensor
    except Exception as e:
        print(f"⚠️ Audio extraction failed: {e}. Falling back to mock audio.")
        return torch.randn(1, 500, n_mels)


def main():
    parser = argparse.ArgumentParser(description="Lumen-1 Video & Audio Multimodal Pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--prompt", type=str, default="Analyze this travel clip for safety issues or wildlife hazards in Sinhala.", help="User instruction prompt")
    parser.add_argument("--model_size", type=str, default="1b", choices=["1b", "3b", "7b"], help="Lumen model size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to custom model checkpoint (.pt)")
    parser.add_argument("--max_frames", type=int, default=8, help="Max frames to extract")
    args = parser.parse_args()

    # Load Tokenizer
    tokenizer_path = os.path.join(project_root, "tokenizer", "lumen_tokenizer.model")
    if os.path.exists(tokenizer_path):
        print(f"📖 Loading tokenizer from {tokenizer_path}...")
        tokenizer = LumenTokenizer(tokenizer_path)
    else:
        print("⚠️ Tokenizer model not found at standard path. Initializing placeholder tokenizer.")
        tokenizer = LumenTokenizer()

    # Load Model Configuration & Model
    print(f"🤖 Initializing Lumen-1-{args.model_size} Model...")
    config = LumenConfig.from_size(ModelSize(args.model_size))
    model = LumenForCausalLM(config)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📂 Loading weights from {args.checkpoint}...")
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}. Running with uninitialized weights.")
    else:
        print("💡 Checkpoint not provided or not found. Running inference with initialization weights.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Process Inputs
    pixel_values = extract_video_keyframes(args.video, max_frames=args.max_frames, img_size=config.vision_image_size)
    mel_spectrograms = extract_audio_mel(args.video, sample_rate=config.audio_sample_rate, n_mels=config.audio_num_mel_bins)

    pixel_values = pixel_values.to(device)
    mel_spectrograms = mel_spectrograms.to(device)

    # 1. vision patches = T * 576 (ViT-B/16 patches)
    num_frames = pixel_values.shape[1]
    patches_per_frame = (config.vision_image_size // config.vision_patch_size) ** 2
    num_vision_tokens = num_frames * patches_per_frame

    # 2. audio tokens = spectrogram len (or projected rate)
    num_audio_tokens = mel_spectrograms.shape[1]

    print(f"📝 Prompt: {args.prompt}")
    print(f"🔋 Embedding tokens mapping: {num_vision_tokens} vision pads, {num_audio_tokens} audio pads.")

    # Create special chat formatted prompt with pads and thinking/reasoning instructions
    # (හොඳම පිළිතුර ලබාදීමට පෙර සිතා බලා (think/reason) ක්‍රියාත්මක වන ලෙස system prompt එක සකසයි)
    chat_messages = [
        {"role": "system", "content": "You are Lumen-1, an advanced AI travel assistant for Sri Lanka. When answering questions, think deeply and structure your reasoning step-by-step to provide the best, most accurate, and safest output."},
        {"role": "user", "content": f"Video Input: <image> Audio Input: <audio>\n\nInstruction: {args.prompt}"}
    ]
    chat_text = tokenizer.apply_chat_template(chat_messages, add_generation_prompt=True)
    
    # Split text to inject vision pads where <image> is and audio pads where <audio> is
    text_parts = chat_text.split("<image>")
    # We want to replace "<image>" with vision pads and "<audio>" with audio pads.
    # Since apply_chat_template outputs flat string, let's encode and build ID sequence manually
    input_ids = []
    
    # Build the input ID sequence with placeholder pad tokens inserted correctly
    # (ලැබෙන input sequence එකට placeholder pad tokens නිවැරදිව එකතු කරගන්නවා)
    input_ids = tokenizer.build_multimodal_sequence(
        chat_text,
        num_vision_tokens=num_vision_tokens,
        num_audio_tokens=num_audio_tokens
    )
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"🧬 Total Token Sequence Length: {input_ids_tensor.shape[1]}")

    print("\n🚀 Executing forward pass and generating output...")
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids_tensor,
            pixel_values=pixel_values,
            mel_spectrograms=mel_spectrograms,
            vision_placeholder_id=tokenizer.vision_pad_id,
            audio_placeholder_id=tokenizer.audio_pad_id,
            max_new_tokens=150,
            temperature=0.7,
            eos_token_id=tokenizer.token_to_id(tokenizer.EOS_TOKEN)
        )

    # Decode answer
    response_tokens = generated_ids[0][input_ids_tensor.shape[1]:].tolist()
    response_text = tokenizer.decode(response_tokens)

    print("\n✨=== LUMEN-1 PIPELINE ANALYSIS ===✨")
    print(response_text)
    print("=====================================\n")


if __name__ == "__main__":
    main()
