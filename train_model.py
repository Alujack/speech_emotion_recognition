"""
Custom Wav2Vec2 Training Script for Speech Emotion Recognition
Train your own model using your dataset!

Usage:
    python train_model.py

Make sure your dataset is organized as:
    dataset/
    ├── train/
    │   ├── angry/
    │   ├── happy/
    │   └── ...
    └── test/
        ├── angry/
        ├── happy/
        └── ...
"""

import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Optional
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import json


# ==================== CONFIGURATION ====================
# EDIT THESE TO MATCH YOUR DATASET!

class Config:
    """Training configuration - CUSTOMIZE THIS!"""
    
    # Dataset paths - CHANGE THESE TO YOUR PATHS!
    TRAIN_DIR = "dataset/train"  # Path to training data
    TEST_DIR = "dataset/test"     # Path to test data
    OUTPUT_DIR = "./my_emotion_model"  # Where to save your trained model
    
    # Model configuration
    BASE_MODEL = "facebook/wav2vec2-base"  # Base Wav2Vec2 model
    SAMPLING_RATE = 16000  # Audio sample rate
    
    # Training hyperparameters
    BATCH_SIZE = 8  # Reduce to 4 or 2 if you get memory errors
    EPOCHS = 20     # More epochs = better accuracy (but takes longer)
    LEARNING_RATE = 3e-5
    WARMUP_STEPS = 500
    
    # Emotions - CUSTOMIZE TO YOUR DATASET!
    # List all emotion folders you have in your dataset
    EMOTIONS = [
        "angry",
        "happy", 
        "sad",
        "neutral",
        "fearful",
        "surprised",
        "disgusted"
    ]


# ==================== HELPER FUNCTIONS ====================

def load_dataset_from_folder(data_dir: str) -> Dataset:
    """
    Load audio dataset from folder structure
    
    Expected structure:
        data_dir/
        ├── emotion1/
        │   ├── audio1.wav
        │   └── audio2.wav
        ├── emotion2/
        └── ...
    """
    print(f"\n📂 Loading dataset from: {data_dir}")
    
    audio_files = []
    labels = []
    
    # Load files for each emotion
    for emotion_idx, emotion in enumerate(Config.EMOTIONS):
        emotion_dir = os.path.join(data_dir, emotion)
        
        if not os.path.exists(emotion_dir):
            print(f"⚠️  Warning: Directory not found: {emotion_dir}")
            continue
        
        # Get all audio files
        files = [
            f for f in os.listdir(emotion_dir) 
            if f.endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))
        ]
        
        for file in files:
            file_path = os.path.join(emotion_dir, file)
            audio_files.append(file_path)
            labels.append(emotion_idx)
        
        print(f"  ✓ {emotion:12s}: {len(files):4d} files")
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {data_dir}! Please check your dataset path.")
    
    print(f"✅ Total: {len(audio_files)} audio files loaded\n")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict({
        "audio": audio_files,
        "label": labels
    })
    
    # Cast audio column to load and resample audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=Config.SAMPLING_RATE))
    
    return dataset


def preprocess_function(examples, feature_extractor):
    """Preprocess audio for Wav2Vec2 input"""
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000 * 10,  # Max 10 seconds
        truncation=True,
        padding=True,
    )
    
    return inputs


@dataclass
class DataCollatorWithPadding:
    """Data collator that pads inputs and labels"""
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: bool = True
    max_length: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]
        
        # Pad inputs
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Add labels to batch
        batch["labels"] = torch.tensor(label_features, dtype=torch.long)
        
        return batch


def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    accuracy_metric = evaluate.load("accuracy")
    
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    
    # Compute accuracy
    acc = accuracy_metric.compute(predictions=predictions, references=references)
    
    # Compute detailed metrics
    report = classification_report(
        references, 
        predictions,
        target_names=Config.EMOTIONS,
        output_dict=True,
        zero_division=0
    )
    
    return {
        "accuracy": acc["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }


# ==================== MAIN TRAINING FUNCTION ====================

def train_model():
    """Main training function"""
    
    print("\n" + "="*70)
    print("🚀 TRAINING CUSTOM WAV2VEC2 EMOTION RECOGNITION MODEL")
    print("="*70 + "\n")
    
    # Step 1: Load datasets
    print("📊 STEP 1: Loading Datasets")
    print("-" * 70)
    
    try:
        train_dataset = load_dataset_from_folder(Config.TRAIN_DIR)
        test_dataset = load_dataset_from_folder(Config.TEST_DIR)
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nPlease make sure:")
        print(f"  1. Training data is in: {Config.TRAIN_DIR}")
        print(f"  2. Test data is in: {Config.TEST_DIR}")
        print(f"  3. Each folder contains emotion subfolders: {Config.EMOTIONS}")
        return
    
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    print(f"📋 Dataset Summary:")
    print(f"  • Train samples: {len(train_dataset)}")
    print(f"  • Test samples: {len(test_dataset)}")
    print(f"  • Total samples: {len(train_dataset) + len(test_dataset)}")
    print(f"  • Number of emotions: {len(Config.EMOTIONS)}")
    print(f"  • Emotions: {', '.join(Config.EMOTIONS)}\n")
    
    # Step 2: Load model
    print("📊 STEP 2: Loading Base Model")
    print("-" * 70)
    print(f"Loading: {Config.BASE_MODEL}\n")
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Config.BASE_MODEL)
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        Config.BASE_MODEL,
        num_labels=len(Config.EMOTIONS),
        problem_type="single_label_classification"
    )
    
    # Freeze feature extractor (only train classifier layers)
    model.freeze_feature_encoder()
    
    print("✅ Model loaded successfully!")
    print(f"  • Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  • Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Step 3: Preprocess
    print("📊 STEP 3: Preprocessing Audio")
    print("-" * 70)
    print("This may take a few minutes...\n")
    
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing",
        num_proc=4  # Use multiple processes for speed
    )
    
    print("✅ Preprocessing complete!\n")
    
    # Step 4: Training setup
    print("📊 STEP 4: Setting Up Training")
    print("-" * 70)
    
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        warmup_steps=Config.WARMUP_STEPS,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    data_collator = DataCollatorWithPadding(feature_extractor=feature_extractor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("✅ Trainer initialized!")
    print(f"  • Batch size: {Config.BATCH_SIZE}")
    print(f"  • Epochs: {Config.EPOCHS}")
    print(f"  • Learning rate: {Config.LEARNING_RATE}")
    print(f"  • Device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    
    # Step 5: Train!
    print("📊 STEP 5: Training Model")
    print("="*70)
    print("🎓 TRAINING STARTED - This will take a while!")
    print("   Go grab some coffee ☕ or tea 🍵")
    print("="*70 + "\n")
    
    try:
        train_result = trainer.train()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTry:")
        print("  • Reducing BATCH_SIZE in Config")
        print("  • Using fewer epochs")
        print("  • Checking your GPU memory")
        return
    
    # Step 6: Evaluate
    print("📊 STEP 6: Evaluating Model")
    print("-" * 70)
    
    metrics = trainer.evaluate()
    
    print("\n📈 FINAL RESULTS:")
    print(f"  • Accuracy:      {metrics['eval_accuracy']*100:.2f}%")
    print(f"  • F1 (macro):    {metrics['eval_f1_macro']*100:.2f}%")
    print(f"  • F1 (weighted): {metrics['eval_f1_weighted']*100:.2f}%\n")
    
    # Detailed evaluation
    print("🔍 Detailed Per-Emotion Performance")
    print("-" * 70)
    
    predictions = trainer.predict(encoded_dataset["test"])
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    # Classification report
    report = classification_report(
        labels,
        preds,
        target_names=Config.EMOTIONS,
        digits=4,
        zero_division=0
    )
    
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\n🔢 Confusion Matrix:")
    print("Rows = True, Columns = Predicted\n")
    
    # Header
    print("           ", end="")
    for emotion in Config.EMOTIONS:
        print(f"{emotion[:8]:>9}", end="")
    print()
    
    # Matrix
    for i, emotion in enumerate(Config.EMOTIONS):
        print(f"{emotion[:10]:>10}", end=" ")
        for j in range(len(Config.EMOTIONS)):
            print(f"{cm[i][j]:>9}", end="")
        print()
    print()
    
    # Step 7: Save model
    print("📊 STEP 7: Saving Model")
    print("-" * 70)
    
    trainer.save_model(Config.OUTPUT_DIR)
    feature_extractor.save_pretrained(Config.OUTPUT_DIR)
    
    # Save training info
    config_info = {
        "emotions": Config.EMOTIONS,
        "num_labels": len(Config.EMOTIONS),
        "sampling_rate": Config.SAMPLING_RATE,
        "accuracy": float(metrics['eval_accuracy']),
        "f1_macro": float(metrics['eval_f1_macro']),
        "f1_weighted": float(metrics['eval_f1_weighted']),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "epochs": Config.EPOCHS,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.LEARNING_RATE,
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"✅ Model saved to: {Config.OUTPUT_DIR}")
    print(f"  • Model weights: pytorch_model.bin")
    print(f"  • Config: config.json")
    print(f"  • Training info: training_info.json\n")
    
    # Final summary
    print("="*70)
    print("🎉 TRAINING COMPLETE! 🎉")
    print("="*70 + "\n")
    
    print("📊 Summary:")
    print(f"  • Your model: {Config.OUTPUT_DIR}")
    print(f"  • Accuracy: {metrics['eval_accuracy']*100:.2f}%")
    print(f"  • Training time: {train_result.metrics['train_runtime']:.0f} seconds\n")
    
    print("📝 Next Steps:")
    print("  1. Test your model with new audio files")
    print("  2. Update model_simple.py to use your custom model:")
    print(f"     model_name = '{Config.OUTPUT_DIR}'")
    print("  3. Restart your API server")
    print("  4. Enjoy REAL emotion recognition! 🚀\n")


# ==================== MAIN ====================

if __name__ == "__main__":
    # Check prerequisites
    print("\n🔍 Checking Prerequisites...")
    
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"❌ Training directory not found: {Config.TRAIN_DIR}")
        print("\nPlease create your dataset with this structure:")
        print("  dataset/")
        print("  ├── train/")
        print("  │   ├── angry/")
        print("  │   ├── happy/")
        print("  │   └── ...")
        print("  └── test/")
        print("      ├── angry/")
        print("      ├── happy/")
        print("      └── ...\n")
        exit(1)
    
    # GPU check
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  No GPU detected - training will use CPU (slower)")
        print("   Consider using Google Colab or a GPU server for faster training")
    
    print("\n✅ All checks passed! Starting training...\n")
    
    # Train!
    train_model()
