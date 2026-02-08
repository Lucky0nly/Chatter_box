
import logging
logging.basicConfig(filename='debug_reader.log', level=logging.INFO)
import torchaudio as ta
import sounddevice as sd
import torch
import time
from chatterbox.tts_turbo import ChatterboxTurboTTS
from huggingface_hub import snapshot_download

print("Debug: Starting...")
device = "cpu" # Force CPU for debugging
print(f"Debug: Device {device}")

try:
    print("Debug: Downloading...")
    model_path = snapshot_download(
        repo_id="ResembleAI/chatterbox-turbo", 
        token=False, 
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
    )
    print("Debug: Loading model...")
    try:
        model = ChatterboxTurboTTS.from_local(model_path, device=device)
        print("Debug: Model Loaded!")
    except RuntimeError as e:
        print(f"Debug: Caught RuntimeError: {e}")
except Exception as e:
    print(f"Debug: Caught Exception: {e}")

print("Debug: Finished.")
