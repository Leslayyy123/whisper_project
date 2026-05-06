import os
import wave
import pandas as pd

data = []
base_path = r"C:\Users\luvi\Documents\SANDRA\whisper_project"

def get_label(folder_name):
    folder = folder_name.lower()
    # Whisper (live only)
    if folder.startswith("live_whisper"):
        return "whisper"

    # Room noise (live only)
    if folder.startswith("live_roomnoise"):
        return "room_noise"

    # Distractor (live only)
    if folder.startswith("live_distractor"):
        return "distractor"

    # Normal speech (live only)
    if folder.startswith("live_normalspeech"):
        return "normal_speech"
    return "unknown"

def get_duration(filepath):
    try:
        with wave.open(filepath, 'r') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return round(frames / float(rate), 2)
    except:
        return 0.0

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(folder_path):
        continue
    
    label = get_label(folder)
    if label == "unknown":
        continue
    
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            filepath = os.path.join(folder_path, file)
            duration = get_duration(filepath)
            data.append({
                "file_path": filepath,
                "folder": folder,
                "label": label,
                "duration_seconds": duration
            })

df = pd.DataFrame(data)
df.to_csv(os.path.join(base_path, "metadata.csv"), index=False)

# Print summary
print(f"\n✅ Created metadata.csv with {len(df)} files\n")
print("=== FILES PER CATEGORY ===")
print(df.groupby("label")["file_path"].count().to_string())
print("\n=== FILES PER FOLDER ===")
print(df.groupby("folder")["file_path"].count().to_string())