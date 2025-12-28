import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from huggingface_hub import snapshot_download
import sounddevice as sd
import numpy as np
import time

# List of expressive tokens found in the codebase
EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

def main():
    print("Initializing Chatterbox Token Demo...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model... (Please wait)")
    model_path = snapshot_download(
        repo_id="ResembleAI/chatterbox-turbo", 
        token=False, 
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
    )
    model = ChatterboxTurboTTS.from_local(model_path, device=device)
    print("Model loaded!")
    
    print("-" * 50)
    print("Demonstrating Expressive Tokens:")
    print("-" * 50)

    for tag in EVENT_TAGS:
        text = f"Here is a {tag} sound for you."
        print(f"Generating: '{text}'")
        
        # Generate
        wav_tensor = model.generate(text)
        
        # Play
        wav_numpy = wav_tensor.squeeze().detach().cpu().numpy()
        sample_rate = model.sr
        sd.play(wav_numpy, sample_rate)
        sd.wait()
        
        time.sleep(0.5) # Short pause between examples

    print("-" * 50)
    print("Demo complete!")

if __name__ == "__main__":
    main()
