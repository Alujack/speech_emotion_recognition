"""
Quick Dataset Downloader for Speech Emotion Recognition
Downloads and prepares public datasets for training
"""

import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import gdown

def download_file(url, output_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def download_ravdess():
    """
    Download RAVDESS dataset (RECOMMENDED)
    - 1,440 audio files
    - 8 emotions
    - Professional actors
    - Free to use
    """
    print("📥 Downloading RAVDESS Dataset...")
    print("  This is one of the best emotion datasets!")
    print("  Size: ~650MB")
    print()
    
    output_dir = "emotion_dataset_ravdess"
    os.makedirs(output_dir, exist_ok=True)
    
    # RAVDESS Zenodo links
    parts = [
        ("https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip", "ravdess.zip")
    ]
    
    for url, filename in parts:
        print(f"Downloading {filename}...")
        try:
            download_file(url, filename)
            
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            os.remove(filename)
            print("✅ Done!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n📖 Manual download:")
            print(f"  Go to: {url}")
            print(f"  Extract to: {output_dir}")
    
    print(f"\n✅ RAVDESS downloaded to: {output_dir}")
    print("📖 Now organize files by emotion using organize_ravdess()")

def organize_ravdess(input_dir="emotion_dataset_ravdess", output_dir="emotion_dataset"):
    """
    Organize RAVDESS files by emotion
    
    RAVDESS filename format: 03-01-05-01-01-01-12.wav
    Position 3 (5th position) is emotion:
    01 = neutral
    02 = calm
    03 = happy
    04 = sad
    05 = angry
    06 = fearful
    07 = disgust
    08 = surprised
    """
    print("📁 Organizing RAVDESS dataset by emotion...")
    
    emotion_map = {
        '01': 'neutral',
        '02': 'neutral',  # Treat calm as neutral
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgusted',
        '08': 'surprised'
    }
    
    # Create output directories
    for emotion in emotion_map.values():
        os.makedirs(f"{output_dir}/{emotion}", exist_ok=True)
    
    # Process all wav files
    count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                # Parse filename
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        emotion = emotion_map[emotion_code]
                        
                        # Copy file
                        src = os.path.join(root, file)
                        dst = os.path.join(output_dir, emotion, f"ravdess_{count:04d}.wav")
                        shutil.copy(src, dst)
                        count += 1
    
    print(f"✅ Organized {count} files into {output_dir}/")
    print("\n📊 Files per emotion:")
    for emotion in set(emotion_map.values()):
        num_files = len(os.listdir(f"{output_dir}/{emotion}"))
        print(f"  {emotion}: {num_files} files")

def download_tess():
    """
    Download TESS dataset
    - 2,800 audio files
    - 7 emotions
    - Female speakers
    """
    print("📥 Downloading TESS Dataset...")
    print("  Size: ~350MB")
    
    # TESS is on Google Drive
    url = "https://drive.google.com/uc?id=1w0pF9Jqvhfkp6p8TpAkLqAqwdXqXkLks"
    output = "tess.zip"
    output_dir = "emotion_dataset_tess"
    
    try:
        print("Downloading from Google Drive...")
        gdown.download(url, output, quiet=False)
        
        print("Extracting...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        os.remove(output)
        print(f"✅ TESS downloaded to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📖 Manual download:")
        print("  Go to: https://tspace.library.utoronto.ca/handle/1807/24487")
        print(f"  Extract to: {output_dir}")

def quick_setup():
    """Quick setup for beginners"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║        Quick Dataset Setup for Training                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("📚 This will download RAVDESS dataset and organize it for you!")
    print()
    
    choice = input("Download RAVDESS dataset? (y/n): ").lower()
    
    if choice == 'y':
        # Download
        download_ravdess()
        print()
        
        # Organize
        organize_ravdess()
        
        print("\n🎉 Dataset ready!")
        print(f"📁 Location: ./emotion_dataset/")
        print("\n📖 Next step:")
        print("  python train_custom_model.py")
    else:
        print("\n📖 Manual setup:")
        print("1. Download dataset from:")
        print("   - RAVDESS: https://zenodo.org/record/1188976")
        print("   - CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D")
        print("   - TESS: https://tspace.library.utoronto.ca/handle/1807/24487")
        print()
        print("2. Organize into folders:")
        print("   emotion_dataset/")
        print("     angry/")
        print("     happy/")
        print("     sad/")
        print("     ...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "ravdess":
            download_ravdess()
        elif command == "tess":
            download_tess()
        elif command == "organize":
            organize_ravdess()
        elif command == "setup":
            quick_setup()
        else:
            print(f"Unknown command: {command}")
    else:
        # Interactive mode
        quick_setup()

