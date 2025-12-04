"""
Custom Wav2Vec2 Training Script for Speech Emotion Recognition
Train your own model with your own dataset!
"""

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import soundfile as sf
from scipy import signal
from typing import Dict, List
import json
from dataclasses import dataclass
from tqdm import tqdm

# ===========================
# Configuration
# ===========================

@dataclass
class Config:
    """Training configuration"""
    # Dataset paths
    dataset_path: str = "./emotion_dataset"  # Your dataset folder
    output_dir: str = "./trained_model"      # Where to save the model
    
    # Model settings
    base_model: str = "facebook/wav2vec2-base"  # Pre-trained base model
    num_emotions: int = 7  # Number of emotion classes
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 20
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Audio settings
    target_sample_rate: int = 16000
    max_audio_length: int = 16000 * 10  # 10 seconds max
    
    # Emotion labels
    emotion_labels: List[str] = None
    
    def __post_init__(self):
        if self.emotion_labels is None:
            self.emotion_labels = [
                "angry",
                "disgusted", 
                "fearful",
                "happy",
                "neutral",
                "sad",
                "surprised"
            ]


# ===========================
# Dataset Class
# ===========================

class EmotionDataset(Dataset):
    """
    Custom dataset for emotion recognition
    
    Expected folder structure:
    dataset_path/
        angry/
            audio1.wav
            audio2.wav
            ...
        happy/
            audio1.wav
            audio2.wav
            ...
        ... (other emotions)
    """
    
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[int],
        processor: Wav2Vec2Processor,
        config: Config,
        augment: bool = False
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.config = config
        self.augment = augment
    
    def __len__(self):
        return len(self.audio_paths)
    
    def load_audio(self, path: str) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = sf.read(path, dtype='float32')
            
            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.config.target_sample_rate:
                duration = len(audio) / sr
                num_samples = int(duration * self.config.target_sample_rate)
                audio = signal.resample(audio, num_samples)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Trim silence
            threshold = 0.01
            mask = np.abs(audio) > threshold
            if mask.any():
                audio = audio[mask]
            
            return audio
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return silence if error
            return np.zeros(self.config.target_sample_rate)
    
    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        if not self.augment:
            return audio
        
        # Random choice of augmentation
        aug_type = np.random.choice(['noise', 'stretch', 'shift', 'none'])
        
        if aug_type == 'noise':
            # Add random noise
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        
        elif aug_type == 'stretch':
            # Time stretch (slow down or speed up)
            rate = np.random.uniform(0.9, 1.1)
            new_length = int(len(audio) * rate)
            audio = signal.resample(audio, new_length)
        
        elif aug_type == 'shift':
            # Time shift
            shift = np.random.randint(-len(audio)//10, len(audio)//10)
            audio = np.roll(audio, shift)
        
        return audio
    
    def __getitem__(self, idx):
        # Load audio
        audio = self.load_audio(self.audio_paths[idx])
        
        # Apply augmentation (only during training)
        audio = self.augment_audio(audio)
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            audio,
            sampling_rate=self.config.target_sample_rate,
            return_tensors="pt",
            padding=True,
            max_length=self.config.max_audio_length,
            truncation=True
        )
        
        # Get input values
        input_values = inputs.input_values.squeeze()
        
        return {
            "input_values": input_values,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ===========================
# Data Preparation
# ===========================

def prepare_dataset(config: Config):
    """
    Prepare dataset from folder structure
    
    Expected structure:
    dataset_path/
        emotion1/
            *.wav
        emotion2/
            *.wav
        ...
    """
    print("📂 Preparing dataset...")
    
    audio_paths = []
    labels = []
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(config.emotion_labels)}
    
    # Scan dataset folder
    for emotion in config.emotion_labels:
        emotion_path = os.path.join(config.dataset_path, emotion)
        
        if not os.path.exists(emotion_path):
            print(f"⚠️  Warning: Folder not found: {emotion_path}")
            continue
        
        # Get all audio files
        audio_files = [
            os.path.join(emotion_path, f)
            for f in os.listdir(emotion_path)
            if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))
        ]
        
        print(f"  {emotion}: {len(audio_files)} files")
        
        audio_paths.extend(audio_files)
        labels.extend([emotion_to_id[emotion]] * len(audio_files))
    
    print(f"\n✅ Total samples: {len(audio_paths)}")
    print(f"📊 Class distribution:")
    for emotion, idx in emotion_to_id.items():
        count = labels.count(idx)
        print(f"  {emotion}: {count} ({count/len(labels)*100:.1f}%)")
    
    return audio_paths, labels, emotion_to_id


# ===========================
# Training Functions
# ===========================

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def train_model(config: Config):
    """Main training function"""
    
    print("🚀 Starting Training Pipeline")
    print("=" * 60)
    
    # 1. Prepare dataset
    audio_paths, labels, emotion_to_id = prepare_dataset(config)
    
    if len(audio_paths) == 0:
        print("❌ No audio files found! Please check your dataset path.")
        return
    
    # 2. Split dataset
    print("\n📊 Splitting dataset...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        audio_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    
    # 3. Load processor and model
    print(f"\n🤖 Loading base model: {config.base_model}")
    processor = Wav2Vec2Processor.from_pretrained(config.base_model)
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=config.num_emotions,
        problem_type="single_label_classification"
    )
    
    # Freeze feature extractor (optional - speeds up training)
    model.freeze_feature_encoder()
    
    print("✅ Model loaded!")
    
    # 4. Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = EmotionDataset(
        train_paths,
        train_labels,
        processor,
        config,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = EmotionDataset(
        val_paths,
        val_labels,
        processor,
        config,
        augment=False  # No augmentation for validation
    )
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,  # Keep only 3 best checkpoints
        fp16=False,  # Set to True if you have GPU with FP16 support
        dataloader_num_workers=0,  # Increase if you have multiple CPU cores
    )
    
    # 6. Create trainer
    print("\n🏋️ Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 7. Train!
    print("\n🔥 Starting training...")
    print("=" * 60)
    trainer.train()
    
    # 8. Evaluate
    print("\n📈 Final evaluation...")
    metrics = trainer.evaluate()
    print(f"\n✅ Final Metrics:")
    print(f"  Accuracy: {metrics['eval_accuracy']*100:.2f}%")
    print(f"  F1 Score: {metrics['eval_f1']:.4f}")
    
    # 9. Save model
    print(f"\n💾 Saving model to {config.output_dir}...")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    
    # Save emotion mapping
    with open(f"{config.output_dir}/emotion_mapping.json", 'w') as f:
        json.dump(emotion_to_id, f, indent=2)
    
    print("\n🎉 Training complete!")
    print(f"📁 Model saved to: {config.output_dir}")
    print(f"🎯 Accuracy: {metrics['eval_accuracy']*100:.2f}%")
    
    return model, processor, emotion_to_id


# ===========================
# Testing Function
# ===========================

def test_trained_model(model_path: str, test_audio_path: str):
    """Test your trained model on a single audio file"""
    
    print(f"🧪 Testing model from: {model_path}")
    
    # Load model and processor
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    # Load emotion mapping
    with open(f"{model_path}/emotion_mapping.json", 'r') as f:
        emotion_to_id = json.load(f)
    id_to_emotion = {v: k for k, v in emotion_to_id.items()}
    
    # Load and preprocess audio
    audio, sr = sf.read(test_audio_path, dtype='float32')
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        duration = len(audio) / sr
        num_samples = int(duration * 16000)
        audio = signal.resample(audio, num_samples)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Process
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get results
    predicted_id = torch.argmax(probs, dim=-1).item()
    predicted_emotion = id_to_emotion[predicted_id]
    confidence = probs[0][predicted_id].item() * 100
    
    print(f"\n🎯 Prediction:")
    print(f"  Emotion: {predicted_emotion}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"\n📊 All probabilities:")
    for idx, prob in enumerate(probs[0]):
        emotion = id_to_emotion[idx]
        print(f"  {emotion}: {prob.item()*100:.2f}%")


# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Custom Wav2Vec2 Training for Emotion Recognition        ║
║                  Train Your Own Model!                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create configuration
    config = Config(
        dataset_path="./emotion_dataset",  # CHANGE THIS to your dataset path
        output_dir="./my_emotion_model",   # Where to save your model
        num_epochs=20,                     # Increase for better accuracy
        batch_size=8,                      # Decrease if out of memory
    )
    
    print("📋 Training Configuration:")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Emotions: {', '.join(config.emotion_labels)}")
    print()
    
    # Check if dataset exists
    if not os.path.exists(config.dataset_path):
        print(f"❌ Dataset path not found: {config.dataset_path}")
        print("\n📖 Please create your dataset with this structure:")
        print("""
        emotion_dataset/
            angry/
                audio1.wav
                audio2.wav
                ...
            happy/
                audio1.wav
                audio2.wav
                ...
            sad/
                ...
            ... (etc for all emotions)
        """)
        exit(1)
    
    # Start training
    try:
        model, processor, emotion_mapping = train_model(config)
        print("\n✅ SUCCESS! Your model is ready to use!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

