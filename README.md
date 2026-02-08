# ✨ Chatterbox TTS

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/PyPI-chatterbox--tts-green.svg)](https://pypi.org/project/chatterbox-tts/)
[![GitHub Stars](https://img.shields.io/github/stars/resemble-ai/chatterbox.svg?style=social)](https://github.com/resemble-ai/chatterbox)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-7289DA.svg)](https://discord.gg/rJq9cRJBJ6)

> **State-of-the-art, open-source Text-to-Speech models by Resemble AI**

🚀 High-quality speech synthesis with ultra-low latency. Perfect for voice agents, narration, and creative workflows.

---

## 🌟 Overview

**Chatterbox** is a family of cutting-edge, open-source TTS models designed for modern AI applications. The flagship **Chatterbox-Turbo** model delivers exceptional speech quality with minimal computational requirements—powered by a streamlined 350M parameter architecture.

This fork adds **production-ready tools** for real-world use:
- 🎙️ **Reader Bot** — Select any text, press a hotkey, hear it spoken instantly
- 🤖 **Agent REPL** — Interactive command-line TTS with voice switching
- ⚡ **Optimized Async Streaming** — GPU-accelerated, non-blocking audio pipeline

### Why Chatterbox?

✅ **Production-Ready** — Sub-200ms latency for real-time voice agents  
✅ **Efficient** — 350M parameters, lower VRAM requirements  
✅ **Expressive** — Native support for paralinguistic tags ([laugh], [cough], etc.)  
✅ **Multilingual** — 23+ languages supported  
✅ **Zero-Shot Voice Cloning** — Generate voices without fine-tuning  
✅ **Open Source** — MIT Licensed, fully customizable  
✅ **Watermarked** — Built-in responsible AI watermarking  

---

## ⚡ Model Zoo

| Model | Size | Languages | Key Features | Best For |
|-------|------|-----------|--------------|----------|
| **Chatterbox-Turbo** | 350M | English | Paralinguistic tags, Lower compute | Voice agents, Production |
| **Chatterbox-Multilingual** | 500M | 23+ | Zero-shot cloning | Global apps, Localization |
| **Chatterbox** | 500M | English | CFG & Exaggeration tuning | Creative control, General use |

---

## 🚀 Quick Start Tools

### 📖 Reader Bot (Select & Speak)

**The fastest way to hear any text on your screen.**

1. Run the bot:
   ```bash
   run_bot.bat
   ```

2. Wait for: `"Reader bot is ready"` (audio confirmation)

3. **Usage:**
   - Select any text on your screen
   - Press `Ctrl + Alt + R` → Instant TTS playback
   - Press `Ctrl + Alt + X` → Stop playback immediately
   - Press `Ctrl + C` → Exit the bot

**Features:**
- ⚡ Async streaming (audio starts before full generation completes)
- 🎯 Smart text chunking for instant response
- 🛑 Immediate stop on hotkey
- 🔊 Optimized precision: Transformer (FP16) + Vocoder (FP32)

---

### 🤖 Agent REPL (Interactive CLI)

**A conversational TTS agent with voice switching.**

```bash
python run_agent.py
```

**Commands:**
| Command | Action |
|---------|--------|
| `Type any text` | Generate and play TTS |
| `reload_voice` | Switch to custom voice (`male_voice.wav`) |
| `reset_voice` | Revert to default female voice |
| `exit` | Quit the agent |

**Custom Voice:**
- Drop a `male_voice.wav` file in the project root
- Type `reload_voice` to activate it
- Must be 5+ seconds of clean speech

---

## 📦 Installation

### Prerequisites

- **Python 3.10** (required for PyTorch CUDA compatibility)
- **NVIDIA GPU** with CUDA support (RTX 4050+ recommended)
- **Windows 10/11** (for `run_bot.bat`)

### Quick Setup

```bash
# Clone the repo
git clone https://github.com/Lucky0nly/Chatter_box.git
cd Chatter_box

# Create virtual environment
python -m venv venv_py310
venv_py310\Scripts\activate

# Install dependencies
pip install -e .
pip install keyboard pyperclip sounddevice

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Alternative: Automated Setup

```bash
setup_and_run.bat
```

---

## 🎯 Code Examples

### Chatterbox-Turbo (Programmatic)

```python
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with paralinguistic tags
text = "Hi there, Sarah calling back [chuckle], have you got a minute?"
wav = model.generate(
    text,
    audio_prompt_path="reference_clip.wav"
)

# Save output
ta.save("output.wav", wav, model.sr)
```

### Multilingual Support

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# French
wav_fr = model.generate(
    "Bonjour, comment ça va?",
    language_id="fr"
)

# Chinese
wav_zh = model.generate(
    "你好，今天天气真不错",
    language_id="zh"
)
```

More examples in `example_tts.py`, `example_vc.py`, and `example_tts_turbo.py`.

---

## 📂 Project Structure

```
chatterbox/
├── src/chatterbox/
│   ├── tts_turbo.py          # Turbo model implementation
│   ├── tts.py                # Standard TTS model
│   ├── mtl_tts.py            # Multilingual model
│   └── models/
│       ├── t3/               # Transformer backbone
│       └── s3gen/            # Vocoder
├── reader_bot.py             # 🆕 Async streaming TTS bot
├── run_bot.bat               # 🆕 Windows launcher for Reader Bot
├── run_agent.py              # 🆕 Interactive REPL agent
├── setup_and_run.bat         # 🆕 Automated setup script
├── gradio_tts_app.py         # Gradio web UI
├── gradio_tts_turbo_app.py   # Turbo Gradio app
├── multilingual_app.py       # Multilingual demo app
├── pyproject.toml            # Dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## ⚙️ Performance Optimization

### GPU Precision Settings

The Reader Bot uses optimized mixed-precision for best speed/quality balance:

| Component | Precision | Reason |
|-----------|-----------|--------|
| Transformer (T3) | FP16 | 2x faster inference |
| Vocoder (S3Gen) | FP32 | Stable audio quality |

This is configured automatically in `reader_bot.py`:

```python
model.t3 = model.t3.to("cuda").half()       # Speed
model.s3gen = model.s3gen.to("cuda").float() # Quality
```

### Latency Expectations

| Text Length | Generation Time | Notes |
|-------------|-----------------|-------|
| ~50 chars | ~1-2s | Single chunk |
| ~200 chars | ~4-6s | Multiple chunks, streaming |
| ~500 chars | ~8-12s | Long form, fully streamed |

*Tested on RTX 4050 6GB*

---

## 🗣️ Supported Languages

Arabic • Danish • German • Greek • English • Spanish • Finnish • French • Hebrew • Hindi • Italian • Japanese • Korean • Malay • Dutch • Norwegian • Polish • Portuguese • Russian • Swedish • Swahili • Turkish • Chinese

---

## 💡 Pro Tips

### Configuration Best Practices

**General Use:**
- Default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most cases
- Ensure reference clip language matches the target language
- Set `cfg_weight=0` to ignore reference speaker characteristics

**Fast Speakers:**
- Lower `cfg_weight` to ~0.3 to improve pacing

**Expressive Speech:**
- Try lower `cfg_weight` (~0.3) with higher `exaggeration` (0.7+)
- Note: Higher exaggeration speeds up speech; reduce `cfg_weight` to compensate

---

## 🛡️ Built-in Responsible AI Watermarking

Every generated audio includes **Resemble AI's Perth Watermarker**—imperceptible neural watermarks that survive compression and editing while maintaining 100% detection accuracy.

### Watermark Detection

```python
import perth
import librosa

audio, sr = librosa.load("generated_audio.wav", sr=None)
watermarker = perth.PerthImplicitWatermarker()
watermark = watermarker.get_watermark(audio, sample_rate=sr)

print(f"Watermark detected: {watermark}")  # 0.0 (no) or 1.0 (yes)
```

---

## 🚀 Roadmap

- [x] Chatterbox-Turbo (350M) release
- [x] Multilingual support (23+ languages)
- [x] Paralinguistic tag support
- [x] Zero-shot voice cloning
- [x] **Real-time streaming (Reader Bot)**
- [x] **Async pipeline optimization**
- [x] **Hotkey controls**
- [ ] Fine-tuning toolkit
- [ ] Multi-speaker models
- [ ] Emotion control features

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributing & Community

**Issues & Discussions:** [GitHub Issues](https://github.com/resemble-ai/chatterbox/issues)  
**Discord Community:** [Join our Discord](https://discord.gg/rJq9cRJBJ6)  
**Twitter:** [@ResembleAI](https://twitter.com/resembleai)

---

## 🙏 Acknowledgments

Built with inspiration from:
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFi-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

---

## 📚 Citation

If you use Chatterbox in your research, please cite:

```bibtex
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS: State-of-the-art Open-source Text-to-Speech}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```

---

## ⚠️ Disclaimer

Use Chatterbox responsibly. Do not use for:
- Impersonation or fraud
- Spreading misinformation
- Creating non-consensual deepfakes
- Violating privacy laws

---

<div align="center">

**Made with ❤️ by [Resemble AI](https://resemble.ai)**

[🌐 Website](https://resemble.ai) · [🎙️ Demo](https://huggingface.co/spaces/ResembleAI/chatterbox-turbo-demo) · [📖 Docs](https://github.com/resemble-ai/chatterbox) · [💬 Discord](https://discord.gg/rJq9cRJBJ6)

</div>
