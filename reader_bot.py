
import os
import time
import torch
import keyboard
import pyperclip
import sounddevice as sd
import numpy as np
import logging
import asyncio
import re
import socket
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Configure Logging
logging.basicConfig(
    filename='reader_bot.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Global State
generation_executor = ThreadPoolExecutor(max_workers=1)
playback_executor = ThreadPoolExecutor(max_workers=1)
current_stream = None # Track the active stream for aborting
stop_signal = asyncio.Event()

# IPC Configuration (Socket)
IPC_HOST = "127.0.0.1"
IPC_PORT = 5678
ipc_trigger_callback = None  # Will be set after on_trigger is defined

# HTTP Server Configuration (Browser Extension)
HTTP_PORT = 5679
http_text_callback = None  # Will be set to handle text directly
main_loop = None  # Reference to asyncio loop for thread-safe scheduling

def split_into_chunks(text, max_chars=120):
    """Splits text into micro-chunks for instant start."""
    text = text.strip()
    if not text:
        return []
        
    chunks = []
    
    # 1. Split by sentence delimiters first
    raw_sentences = re.split(r'(?<=[.!?;\n])\s+', text)
    
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        # 2. If sentence is small enough, add it
        if len(sent) <= max_chars:
            chunks.append(sent)
            continue
            
        # 3. If too long, hard split by comma or space
        while len(sent) > max_chars:
            # Try splitting by comma first
            split_point = sent.rfind(",", 0, max_chars)
            if split_point == -1:
                # Try splitting by space
                split_point = sent.rfind(" ", 0, max_chars)
            
            if split_point == -1:
                # Hard split if no spaces (rare)
                split_point = max_chars
                
            chunks.append(sent[:split_point + 1]) # Include the delimiter
            sent = sent[split_point + 1:].strip()
        
        if sent:
            chunks.append(sent)
            
    return chunks

def generate_audio(model, chunk, device):
    """Runs synchronous model generation (GPU safe)."""
    try:
        t_gen_start = time.time()
        with torch.inference_mode():
            if device == "cuda":
                logging.debug(f"Pre-generation GPU Mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    wav = model.generate(chunk, norm_loudness=False)
            else:
                wav = model.generate(chunk, norm_loudness=False)
        
        elapsed = time.time() - t_gen_start
        logging.info(f"Generation time for chunk ({len(chunk)} chars): {elapsed:.4f}s")
        wav = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        return np.clip(wav, -1.0, 1.0)
    except Exception as e:
        logging.error(f"Generation error on chunk '{chunk[:20]}...': {e}")
        return None

async def generate_chunks(model, text, queue, device):
    """Async Producer: Generates chunks and puts them in queue."""
    loop = asyncio.get_running_loop()
    stop_signal.clear() # Reset stop signal
    
    chunks = split_into_chunks(text, max_chars=120)
    logging.info(f"Text split into {len(chunks)} chunks: {[len(c) for c in chunks]}")

    for i, chunk in enumerate(chunks):
        if stop_signal.is_set():
            logging.info("Generation stopped by user.")
            break
            
        t0 = time.time()
        logging.info(f"Generating chunk {i+1} ({len(chunk)} chars): '{chunk[:20]}...'")
        
        # Check stop before potentially expensive generation
        if stop_signal.is_set(): break

        wav = await loop.run_in_executor(
            generation_executor,
            lambda: generate_audio(model, chunk, device)
        )
        
        if stop_signal.is_set(): break
        
        t_gen = time.time() - t0
        logging.info(f"Chunk {i+1}/{len(chunks)} generated in {t_gen:.2f}s")
        
        if wav is not None:
            await queue.put(wav)

    await queue.put(None) # Sentinel

async def play_audio(queue, samplerate):
    """Async Consumer: Plays chunks from queue using OutputStream."""
    global current_stream
    loop = asyncio.get_running_loop()
    
    # Create OutputStream
    stream = sd.OutputStream(
        samplerate=samplerate,
        channels=1,
        dtype='float32'
    )
    current_stream = stream
    stream.start()
    logging.info("Audio stream started.")

    try:
        while True:
            if stop_signal.is_set():
                break
                
            wav = await queue.get()
            if wav is None:
                break
            
            # Write to stream (blocking call, so run in playback_executor)
            if not stop_signal.is_set():
                await loop.run_in_executor(playback_executor, lambda: stream.write(wav))
            
    except Exception as e:
        logging.error(f"Playback error: {e}")
    finally:
        # Use abort() for immediate silence if stopped, otherwise stop() triggers a polite end
        if stop_signal.is_set():
            stream.abort()
        else:
            stream.stop()
        stream.close()
        current_stream = None
        logging.info("Audio stream finished.")

async def stream_tts(model, text, device):
    """Orchestrator for streaming."""
    # Cancel previous run if any (handled by stop_signal mostly, but good to be clean)
    stop_signal.clear()
    
    queue = asyncio.Queue()
    t0 = time.time()
    
    producer = asyncio.create_task(generate_chunks(model, text, queue, device))
    consumer = asyncio.create_task(play_audio(queue, model.sr))
    
    await asyncio.gather(producer, consumer)
    logging.info(f"Total session time: {time.time()-t0:.2f}s")
    
def stop_active_playback():
    """Immediately stops generation and playback without exiting process."""
    logging.info("Stopping active playback...")
    stop_signal.set()
    if current_stream:
        try:
            current_stream.abort() 
        except:
            pass

# ==================== IPC SERVER ====================
def handle_ipc_connection(conn):
    """Handle incoming IPC connection."""
    global ipc_trigger_callback
    try:
        data = conn.recv(1024).decode("utf-8")
        if not data:
            return

        message = json.loads(data)

        if message.get("action") == "read":
            logging.info("[IPC] READ command received.")
            if ipc_trigger_callback:
                ipc_trigger_callback()
            else:
                logging.warning("[IPC] Trigger callback not set yet.")

    except Exception as e:
        logging.error(f"[IPC ERROR] {e}")
    finally:
        conn.close()


def start_ipc_server():
    """Start IPC server in background thread."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((IPC_HOST, IPC_PORT))
        server.listen(5)
        logging.info(f"[IPC] Listening on {IPC_HOST}:{IPC_PORT}")

        while True:
            conn, addr = server.accept()
            threading.Thread(
                target=handle_ipc_connection,
                args=(conn,),
                daemon=True
            ).start()
    except OSError as e:
        logging.error(f"[IPC] Could not start server: {e}")
# ====================================================

# ==================== HTTP SERVER (Browser Extension) ====================
def start_http_server():
    """Start Flask HTTP server for browser extension."""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)  # Allow browser to connect
    
    # Disable Flask's default logging to reduce noise
    import logging as flask_logging
    flask_logging.getLogger('werkzeug').setLevel(flask_logging.ERROR)
    
    @app.route("/read", methods=["POST"])
    def read_text():
        global http_text_callback, main_loop
        try:
            data = request.json
            text = data.get("text", "")
            
            if text and text.strip() and http_text_callback and main_loop:
                logging.info(f"[HTTP] Browser request: {len(text)} chars")
                # Schedule TTS on the main asyncio loop
                asyncio.run_coroutine_threadsafe(
                    http_text_callback(text),
                    main_loop
                )
                return jsonify({"status": "ok", "message": "Reading..."})
            else:
                return jsonify({"status": "error", "message": "No text or bot not ready"})
        except Exception as e:
            logging.error(f"[HTTP ERROR] {e}")
            return jsonify({"status": "error", "message": str(e)})
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "running"})
    
    logging.info(f"[HTTP] Starting server on {IPC_HOST}:{HTTP_PORT}")
    app.run(host=IPC_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)
# =========================================================================

def main():
    logging.info("Initializing Async Reader Bot...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load Model
    try:
        logging.info("Loading model...")
        model_path = snapshot_download(
            repo_id="ResembleAI/chatterbox-turbo", 
            token=False, 
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )
        
        try:
            model = ChatterboxTurboTTS.from_local(model_path, device=device)
        except RuntimeError as e:
             if "out of memory" in str(e):
                logging.warning("OOM, fallback to CPU")
                device = "cpu"
                model = ChatterboxTurboTTS.from_local(model_path, device="cpu")
             else:
                raise e
        
        # Explicit Precision Setup: T3 (Fast) + S3Gen (Stable)
        if device == "cuda":
            logging.info("Optimizing model precision...")
            try:
                model.t3 = model.t3.to("cuda").half()       # Transformer -> FP16 (Speed)
                model.s3gen = model.s3gen.to("cuda").float() # Vocoder -> FP32 (Quality)
                logging.info("T3 -> FP16 | S3Gen -> FP32 conversion successful.")
            except Exception as e:
                logging.error(f"Failed to verify model submodules: {e}")

        # Warmup and Announce
        logging.info("Warming up audio...")
        wav = generate_audio(model, "Reader bot is ready", device)
        sd.play(wav, model.sr)
        sd.wait()
        logging.info("Bot Ready! Press Ctrl+Alt+R to Read. Press Ctrl+Alt+X to STOP.")

        # Main Event Loop Logic
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def on_trigger():
            stop_active_playback()
            logging.info("==== TRIGGER PRESSED ====")
            start_total = time.time()
            
            # Clipboard timing
            t_clip = time.time()
            old_clipboard = pyperclip.paste()
            keyboard.release('alt')
            keyboard.release('ctrl')
            time.sleep(0.1)
            keyboard.send('ctrl+c')
            time.sleep(0.15)
            
            text = pyperclip.paste()
            if not text or text == old_clipboard:
                # Retry once
                time.sleep(0.1)
                text = pyperclip.paste()

            logging.info(f"Clipboard capture time: {time.time() - t_clip:.4f}s")

            if text and text.strip():
                logging.info(f"Text length: {len(text)}")
                logging.info(f"Text: '{text[:50]}...'")
                
                # Check GPU status
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    logging.info(f"GPU Memory Allocated: {mem_alloc:.2f} GB")
                    try:
                        logging.info(f"T3 device: {next(model.t3.parameters()).device}")
                        logging.info(f"T3 dtype: {next(model.t3.parameters()).dtype}")
                        logging.info(f"S3Gen device: {next(model.s3gen.parameters()).device}")
                        logging.info(f"S3Gen dtype: {next(model.s3gen.parameters()).dtype}")
                    except Exception as e:
                        logging.error(f"Could not access internal model attributes: {e}")
                        logging.info(f"Fallback Check - Current Device: {next(model.parameters()).device}")

                # Run the async stream in the loop
                future = asyncio.run_coroutine_threadsafe(stream_tts(model, text, device), loop)
            else:
                logging.warning("No text captured.")

        def on_stop_hotkey():
            stop_active_playback()

        keyboard.add_hotkey('ctrl+alt+r', on_trigger)
        keyboard.add_hotkey('ctrl+alt+x', on_stop_hotkey)

        # Start IPC Server for socket-based triggers
        global ipc_trigger_callback
        ipc_trigger_callback = on_trigger
        threading.Thread(target=start_ipc_server, daemon=True).start()

        # Start HTTP Server for browser extension
        global http_text_callback, main_loop
        main_loop = loop
        
        async def handle_browser_text(text):
            """Handle text received from browser extension."""
            stop_active_playback()
            logging.info(f"[Browser] Processing {len(text)} chars...")
            await stream_tts(model, text, device)
        
        http_text_callback = handle_browser_text
        threading.Thread(target=start_http_server, daemon=True).start()

        # Run asyncio loop forever
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt caught. Stopping...")
            stop_active_playback()
            import os
            os._exit(0)
        finally:
            loop.close()

    except Exception as e:
        logging.critical(f"Detailed Error: {e}")
        import traceback
        logging.critical(traceback.format_exc())
    finally:
        pass # Don't exit here unless fatal error caught above

if __name__ == "__main__":
    main()
