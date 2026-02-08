
import os
import time
import torch
import keyboard
import pyperclip
import sounddevice as sd
import numpy as np
import logging
import threading
from chatterbox.tts_turbo import ChatterboxTurboTTS
from huggingface_hub import snapshot_download

# Configure Logging
logging.basicConfig(
    filename='reader_bot.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def play_audio(wav_numpy, sample_rate):
    try:
        sd.play(wav_numpy, sample_rate)
        sd.wait()
    except Exception as e:
        logging.error(f"Audio playback error: {e}")

def main():
    logging.info("Initializing Select and Read Bot...")
    
    # 1. Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        # 2. Load Model
        logging.info("Ensuring model is downloaded...")
        model_path = snapshot_download(
            repo_id="ResembleAI/chatterbox-turbo", 
            token=False, 
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        logging.info("Loading model... (This may take ~20 seconds)")
        try:
            model = ChatterboxTurboTTS.from_local(model_path, device=device)
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                logging.warning(f"CUDA Error: {e}. Falling back to CPU.")
                device = "cpu"
                torch.cuda.empty_cache()
                model = ChatterboxTurboTTS.from_local(model_path, device="cpu")
            else:
                raise e

        logging.info(f"Model loaded successfully on {device}!")
        
        # Test Audio on Startup
        logging.info("Testing audio output...")
        try:
            startup_text = "Reader bot ready."
            wav = model.generate(startup_text)
            play_audio(wav.squeeze().detach().cpu().numpy(), model.sr)
            logging.info("Audio test complete.")
        except Exception as e:
            logging.error(f"Startup audio test failed: {e}")

        print("\n" + "="*50)
        print(f"BOT IS READY! Select text and press 'Ctrl+Alt+R' to read.")
        print("Check 'reader_bot.log' for details if issues occur.")
        print("Press 'Esc' to quit.")
        print("="*50 + "\n")

        # 3. Define Trigger Function
        def on_trigger():
            logging.info("Hotkey detected! Starting capture...")
            try:
                old_clipboard = pyperclip.paste()
                
                # Release 'alt' if held down, as it interferes with Ctrl+C
                keyboard.release('alt')
                keyboard.release('ctrl')
                time.sleep(0.1)
                
                # Send Ctrl+C twice to be sure
                keyboard.send('ctrl+c')
                time.sleep(0.1)
                keyboard.send('ctrl+c')
                
                # Wait up to 1 second for clipboard to update
                text = ""
                for attempt in range(10): 
                    time.sleep(0.1)
                    text = pyperclip.paste()
                    if text and text != old_clipboard:
                        break
                
                logging.info(f"Clipboard content after capture attempt: '{text[:20]}...'")

                if not text or not text.strip():
                    logging.warning("Clipboard is empty or no text selected.")
                    return

                if text == old_clipboard:
                    logging.info("Clipboard content might be unchanged (re-reading old text or copy failed).")

                logging.info(f"Text captured ({len(text)} chars): {text[:50]}...")
                
                # Generate
                logging.info("Generating audio...")
                t0 = time.time()
                try:
                    wav = model.generate(text)
                except Exception as e:
                     logging.error(f"Generation failed: {e}")
                     return

                t1 = time.time()
                logging.info(f"Generation took {t1-t0:.2f}s")
                
                # Play
                logging.info("Playing audio...")
                try:
                    play_audio(wav.squeeze().detach().cpu().numpy(), model.sr)
                except Exception as e:
                     logging.error(f"Playback failed: {e}")
                logging.info("Playback finished.")
                
            except Exception as e:
                logging.error(f"Error during trigger execution: {e}")
                import traceback
                logging.error(traceback.format_exc())

        # 4. Set up Hotkey
        keyboard.add_hotkey('ctrl+alt+r', on_trigger)
        logging.info("Hotkey 'ctrl+alt+r' registered.")
        
        # 5. Heartbeat & Keep running
        logging.info("Entering main loop...")
        while True:
            if keyboard.is_pressed('esc'):
                logging.info("Esc pressed. Exiting...")
                break
            time.sleep(0.1)

    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        import traceback
        logging.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
