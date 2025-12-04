# AI Model Implementation Guide
## Speech Emotion Recognition with Wav2Vec2

This guide explains how to implement a real AI model for speech emotion recognition using Wav2Vec2, achieving high accuracy (targeting 99%+).

## 🎯 Overview

**Current Status**: The API returns static/random results for demonstration.

**Goal**: Implement a real emotion recognition model using Wav2Vec2, a state-of-the-art transformer model for audio processing.

## 🔬 Why Wav2Vec2?

Wav2Vec2 is a powerful pre-trained model by Facebook AI that:
- Learns representations from raw audio waveforms
- Pre-trained on 960 hours of speech data
- Achieves state-of-the-art results on speech tasks
- Can be fine-tuned for emotion recognition
- Supports transfer learning with limited data

## 📋 Implementation Steps

### Step 1: Install Required Packages

Update `requirements.txt`:

```txt
fastapi==0.115.5
uvicorn[standard]==0.32.1
python-multipart==0.0.18

# AI/ML Libraries
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
librosa>=0.10.0
numpy>=1.24.0
soundfile>=0.12.0
```

Install:
```bash
pip install torch torchaudio transformers librosa numpy soundfile
```

### Step 2: Choose a Pre-trained Model

**Option A: Fine-tuned Wav2Vec2 for Emotion Recognition**

Use existing fine-tuned models from HuggingFace:

1. **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**
   - Trained on multiple emotion datasets
   - Supports 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise
   - Good accuracy (~85-90%)

2. **superb/wav2vec2-base-superb-er**
   - SUPERB benchmark model
   - Emotion recognition task
   - High quality, well-tested

3. **harshit345/xlsr-wav2vec-speech-emotion-recognition**
   - Multi-lingual support
   - Good generalization

**Option B: Train Your Own Model (For 99%+ Accuracy)**

Use datasets:
- **RAVDESS**: Acted emotions, clean audio
- **CREMA-D**: 7,442 clips, 6 emotions
- **TESS**: Toronto Emotional Speech Set
- **EmoDB**: German emotional database
- **IEMOCAP**: Interactive emotional dyadic motion capture

### Step 3: Implementation Code

Create a new file `model.py`:

```python
import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from typing import Dict, List
import soundfile as sf

class EmotionRecognitionModel:
    def __init__(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initialize the emotion recognition model
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels (adjust based on your model)
        self.emotion_labels = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise"
        }
        
        # Map to our API emotion names
        self.emotion_mapping = {
            "angry": "angry",
            "disgust": "disgusted",
            "fear": "fearful",
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
            "surprise": "surprised"
        }
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Preprocessed audio array
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio
    
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Extract audio features for metadata
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Dictionary of audio features
        """
        features = {}
        
        # Duration
        features['duration'] = len(audio) / sr
        
        # Sample rate
        features['sample_rate'] = float(sr)
        
        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
        
        # Energy/RMS
        rms = librosa.feature.rms(y=audio)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = float(tempo)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        return features
    
    def predict_emotion(self, audio_path: str) -> Dict:
        """
        Predict emotion from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with emotion predictions and features
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)
        
        # Extract features for metadata
        audio_features = self.extract_features(audio)
        
        # Prepare input for model
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get probabilities for all emotions
        probs = probabilities[0].cpu().numpy()
        
        # Create emotion scores
        emotion_scores = []
        for idx, prob in enumerate(probs):
            emotion_name = self.emotion_labels.get(idx, f"emotion_{idx}")
            mapped_emotion = self.emotion_mapping.get(emotion_name, emotion_name)
            
            emotion_scores.append({
                "emotion": mapped_emotion,
                "score": float(prob * 100),  # Convert to percentage
            })
        
        # Sort by score
        emotion_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Get dominant emotion
        dominant = emotion_scores[0]
        
        return {
            "dominant_emotion": dominant["emotion"],
            "confidence": dominant["score"],
            "emotion_scores": emotion_scores,
            "audio_features": audio_features
        }

# Global model instance (loaded once at startup)
model = None

def load_model():
    """Load the model at startup"""
    global model
    if model is None:
        print("Loading emotion recognition model...")
        model = EmotionRecognitionModel()
        print("Model loaded successfully!")
    return model

def get_model():
    """Get the loaded model instance"""
    global model
    if model is None:
        model = load_model()
    return model
```

### Step 4: Update `main.py`

Replace the static emotion generation with real model predictions:

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import os
import tempfile
import shutil

# Import the AI model
from model import get_model

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="AI-powered API for analyzing emotions from speech audio using Wav2Vec2",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Emotion metadata
EMOTIONS = {
    "happy": {"label": "Happy", "color": "#22c55e", "emoji": "😊"},
    "sad": {"label": "Sad", "color": "#3b82f6", "emoji": "😢"},
    "angry": {"label": "Angry", "color": "#ef4444", "emoji": "😠"},
    "neutral": {"label": "Neutral", "color": "#6b7280", "emoji": "😐"},
    "fearful": {"label": "Fearful", "color": "#a855f7", "emoji": "😨"},
    "surprised": {"label": "Surprised", "color": "#f59e0b", "emoji": "😮"},
    "disgusted": {"label": "Disgusted", "color": "#84cc16", "emoji": "🤢"},
}

class EmotionScore(BaseModel):
    emotion: str
    label: str
    score: float
    color: str
    emoji: str

class AnalysisResult(BaseModel):
    success: bool
    filename: str
    timestamp: str
    dominant_emotion: str
    confidence: float
    emotion_scores: List[EmotionScore]
    audio_features: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    """Load the model when the app starts"""
    get_model()

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file using AI model and return emotion recognition results
    """
    # Validate file type
    allowed_extensions = [".wav", ".mp3", ".ogg", ".webm", ".m4a"]
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        suffix = os.path.splitext(file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Write uploaded content
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Get model
        model = get_model()
        
        # Predict emotion
        prediction = model.predict_emotion(temp_file.name)
        
        # Format emotion scores with metadata
        emotion_scores = []
        for score in prediction["emotion_scores"]:
            emotion = score["emotion"]
            emotion_info = EMOTIONS.get(emotion, {
                "label": emotion.capitalize(),
                "color": "#6b7280",
                "emoji": "😐"
            })
            
            emotion_scores.append(EmotionScore(
                emotion=emotion,
                label=emotion_info["label"],
                score=round(score["score"], 2),
                color=emotion_info["color"],
                emoji=emotion_info["emoji"]
            ))
        
        result = AnalysisResult(
            success=True,
            filename=file.filename,
            timestamp=datetime.now().isoformat(),
            dominant_emotion=prediction["dominant_emotion"],
            confidence=round(prediction["confidence"], 2),
            emotion_scores=emotion_scores,
            audio_features=prediction["audio_features"]
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
```

### Step 5: Train Custom Model for 99%+ Accuracy

To achieve 99%+ accuracy, you need to:

#### A. Prepare Dataset

```python
# dataset_preparation.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset():
    """
    Prepare dataset for training
    Combine multiple datasets for better accuracy
    """
    datasets = {
        'RAVDESS': 'path/to/ravdess',
        'CREMA-D': 'path/to/cremad',
        'TESS': 'path/to/tess',
    }
    
    # Load and label all audio files
    data = []
    for dataset_name, path in datasets.items():
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.wav'):
                    emotion = extract_emotion_from_filename(file)
                    data.append({
                        'path': os.path.join(root, file),
                        'emotion': emotion,
                        'dataset': dataset_name
                    })
    
    df = pd.DataFrame(data)
    
    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['emotion'])
    
    return train_df, test_df
```

#### B. Fine-tune Wav2Vec2

```python
# train_model.py
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from transformers import TrainingArguments, Trainer
import torch

def train_emotion_model():
    """
    Fine-tune Wav2Vec2 for emotion recognition
    """
    # Load base model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=7,  # 7 emotions
    )
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./emotion_model",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=30,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained("./final_emotion_model")
    processor.save_pretrained("./final_emotion_model")
```

#### C. Data Augmentation for Higher Accuracy

```python
import librosa
import numpy as np

def augment_audio(audio, sr):
    """
    Apply data augmentation techniques
    """
    augmentations = [
        add_noise,
        time_stretch,
        pitch_shift,
        time_shift,
    ]
    
    augmented = []
    for aug_func in augmentations:
        augmented.append(aug_func(audio, sr))
    
    return augmented

def add_noise(audio, sr, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_stretch(audio, sr, rate=1.1):
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_shift(audio, sr, shift_max=0.2):
    shift = np.random.randint(int(sr * shift_max))
    return np.roll(audio, shift)
```

## 🎯 Achieving 99%+ Accuracy

To reach 99%+ accuracy:

1. **Use Ensemble Methods**: Combine multiple models
2. **Large Training Data**: 50,000+ labeled samples
3. **Data Augmentation**: 5-10x augmentation per sample
4. **Cross-validation**: 10-fold cross-validation
5. **Hyperparameter Tuning**: Use Optuna or Ray Tune
6. **Multi-task Learning**: Train on related tasks simultaneously
7. **Attention Mechanisms**: Add custom attention layers
8. **Context Window**: Use longer audio segments (3-5 seconds)

## 📊 Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader):
    predictions = []
    labels = []
    
    for batch in test_loader:
        with torch.no_grad():
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
```

## 🚀 Deployment

### Docker with GPU Support

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Download model at build time (optional)
RUN python3 -c "from model import load_model; load_model()"

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Performance Optimization

1. **Model Quantization**: Reduce model size by 4x
2. **ONNX Runtime**: Faster inference
3. **Batch Processing**: Process multiple files simultaneously
4. **Caching**: Cache model in memory
5. **GPU Acceleration**: Use CUDA if available

## 🔧 Troubleshooting

**Issue**: Low accuracy
- **Solution**: More training data, better preprocessing, data augmentation

**Issue**: Slow inference
- **Solution**: Use quantization, ONNX, or smaller model variant

**Issue**: Memory errors
- **Solution**: Reduce batch size, use gradient checkpointing

## 📚 Resources

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [HuggingFace Wav2Vec2 Docs](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [SUPERB Benchmark](https://superbbenchmark.org/)
- [Speech Emotion Recognition Tutorial](https://huggingface.co/blog/audio-transformers)

## 🎓 Citation

If using Wav2Vec2:
```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Henry and Mohamed, Abdelrahman and Auli, Michael},
  journal={arXiv preprint arXiv:2006.11477},
  year={2020}
}
```

## ✅ Next Steps

1. Install ML dependencies
2. Choose or train a model
3. Replace static generation with real predictions
4. Test with real audio files
5. Monitor and improve accuracy
6. Deploy to production

---

**Note**: Achieving exactly 99.99% accuracy is extremely challenging and may not be realistic for all datasets and scenarios. State-of-the-art models typically achieve 85-95% accuracy on standard benchmarks. The implementation above will get you very high accuracy with proper training and data.

