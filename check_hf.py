from huggingface_hub import snapshot_download
import os

try:
    print("Attempting download with token=False...")
    path = snapshot_download(repo_id="ResembleAI/chatterbox-turbo", token=False, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"])
    print(f"Success! Path: {path}")
except Exception as e:
    print(f"Failed: {e}")
