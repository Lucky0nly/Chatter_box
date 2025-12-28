import librosa
import soundfile as sf
import numpy as np

fname = "male_voice.wav"
print(f"Fixing {fname}...")

# Load
y, sr = librosa.load(fname, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Original Duration: {duration:.2f}s")

if duration < 5.0:
    print("Audio is too short. Looping to extend...")
    # Concatenate to make it longer
    y_new = np.concatenate([y, y])
    new_duration = librosa.get_duration(y=y_new, sr=sr)
    
    # Save back
    sf.write(fname, y_new, sr)
    print(f"Fixed! New Duration: {new_duration:.2f}s")
else:
    print("Audio is already long enough.")
