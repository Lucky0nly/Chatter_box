import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from huggingface_hub import snapshot_download
import sounddevice as sd
import numpy as np
import os
import platform
import sys

def main():
    print("Initializing Chatterbox Agent...")
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: GPU not detected. Generation might be slow.")
    else:
        print("GPU detected! Generation should be fast.")

    try:
        print("Ensuring model is downloaded (token=False)...")
        model_path = snapshot_download(
            repo_id="ResembleAI/chatterbox-turbo", 
            token=False, 
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )
        # print(f"Model path: {model_path}")

        print("Loading model... (This may take ~20 seconds, please wait)")
        model = ChatterboxTurboTTS.from_local(model_path, device=device)
        # Cache the default (female) conditionals so we can restore them later
        default_conds = model.conds
        print("Model loaded! Ready to speak.")
        print("-" * 50)
        print("Type your text and press Enter. Type 'exit' to quit.")
        print("-" * 50)
        print("Type your text and press Enter. Type 'exit' to quit.")
        print("To change voice: Drop a 'male_voice.wav' file here and type 'reload_voice'")
        print("To switch back to female: type 'reload_voice_integrated'")
        print("-" * 50)

        current_voice_path = None
        if os.path.exists("male_voice.wav"):
            print("Found 'male_voice.wav', using it!")
            current_voice_path = "male_voice.wav"


        while True:
            try:
                text = input("\nYOU: ")
                if text.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if not text.strip():
                    continue

                if text.lower() == 'reload_voice':
                    if os.path.exists("male_voice.wav"):
                        current_voice_path = "male_voice.wav"
                        print("Voice updated to 'male_voice.wav'")
                    else:
                        current_voice_path = None
                        print("No 'male_voice.wav' found. Reverted to default voice.")
                    continue

                if text.lower() in ['reset_voice', 'reload_voice_integrated']:
                    current_voice_path = None
                    # RESTORE the original female voice conditionals
                    model.conds = default_conds
                    print("Reverted to default integrated voice.")
                    continue

                print(f"AGENT: [Generating audio...] (Voice: {'Custom' if current_voice_path else 'Default'})")
                
                # Generate audio
                wav_tensor = model.generate(text, audio_prompt_path=current_voice_path)
                
                # Convert to numpy for playback
                wav_numpy = wav_tensor.squeeze().detach().cpu().numpy()
                sample_rate = model.sr
                
                # Save specifically for debug/history, but not strictly needed for playback
                output_file = "agent_output.wav"
                ta.save(output_file, wav_tensor, sample_rate)
                
                # Play the audio file directly from memory
                sd.play(wav_numpy, sample_rate)
                sd.wait() # Wait until file is done playing
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
