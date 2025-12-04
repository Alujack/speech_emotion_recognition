"""
Simple Speech Emotion Recognition Model using Wav2Vec2 (No Librosa)
This module provides AI model for analyzing emotions in speech without librosa dependency
"""

import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from typing import Dict
import warnings
import soundfile as sf
from scipy import signal
from scipy.fft import rfft, rfftfreq

warnings.filterwarnings('ignore')


class SimpleEmotionModel:
    """
    Simplified Wav2Vec2-based emotion recognition model
    Works without librosa dependency
    """
    
    def __init__(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """Initialize the emotion recognition model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Initializing model on device: {self.device}")
        
        try:
            print(f"📥 Loading model: {model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            raise
        
        # Emotion labels
        self.emotion_labels = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise"
        }
        
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
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = sf.read(audio_path, dtype='float32')
            
            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != target_sr:
                duration = len(audio) / sr
                num_samples = int(duration * target_sr)
                audio = signal.resample(audio, num_samples)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Simple trim
            threshold = 0.01
            mask = np.abs(audio) > threshold
            if mask.any():
                audio = audio[mask]
            
            return audio
            
        except Exception as e:
            raise ValueError(f"Error preprocessing audio: {str(e)}")
    
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Extract basic audio features"""
        features = {}
        
        try:
            features['duration'] = round(len(audio) / sr, 2)
            features['sample_rate'] = float(sr)
            
            # Energy
            rms = np.sqrt(np.mean(audio**2))
            features['energy_mean'] = round(float(rms), 4)
            features['energy_std'] = round(float(np.std(audio)), 4)
            
            # Zero crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
            features['zero_crossing_rate'] = round(float(zcr), 4)
            
            # Spectral features
            fft = rfft(audio)
            freqs = rfftfreq(len(audio), 1/sr)
            magnitude = np.abs(fft)
            
            spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
            features['spectral_centroid'] = round(float(spectral_centroid), 2)
            
            # Pitch estimation
            correlation = np.correlate(audio, audio, mode='full')
            correlation = correlation[len(correlation)//2:]
            d = np.diff(correlation)
            start = np.where(d > 0)[0][0] if len(np.where(d > 0)[0]) > 0 else 1
            peak = np.argmax(correlation[start:]) + start
            
            if peak > 0:
                pitch = sr / peak
                if 50 < pitch < 400:
                    features['pitch_mean'] = round(float(pitch), 2)
                else:
                    features['pitch_mean'] = 150.0
            else:
                features['pitch_mean'] = 150.0
            
            features['pitch_std'] = round(features['pitch_mean'] * 0.15, 2)
            
            # Tempo estimation
            hop_length = 512
            frame_length = 2048
            num_frames = 1 + (len(audio) - frame_length) // hop_length
            
            energy_envelope = []
            for i in range(num_frames):
                start_idx = i * hop_length
                end_idx = start_idx + frame_length
                if end_idx <= len(audio):
                    frame_energy = np.sqrt(np.mean(audio[start_idx:end_idx]**2))
                    energy_envelope.append(frame_energy)
            
            if len(energy_envelope) > 0:
                energy_envelope = np.array(energy_envelope)
                peaks = signal.find_peaks(energy_envelope)[0]
                
                if len(peaks) > 1:
                    avg_peak_distance = np.mean(np.diff(peaks))
                    tempo = 60.0 / (avg_peak_distance * hop_length / sr)
                    features['tempo'] = round(float(np.clip(tempo, 60, 180)), 2)
                else:
                    features['tempo'] = 120.0
            else:
                features['tempo'] = 120.0
            
        except Exception as e:
            print(f"⚠️  Warning: Error extracting features: {e}")
            features.update({
                'duration': round(len(audio) / sr, 2),
                'sample_rate': float(sr),
                'pitch_mean': 150.0,
                'pitch_std': 20.0,
                'energy_mean': 0.1,
                'energy_std': 0.05,
                'tempo': 120.0,
                'spectral_centroid': 2000.0
            })
        
        return features
    
    def predict_emotion(self, audio_path: str) -> Dict:
        """Predict emotion from audio file"""
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            
            # Extract features
            audio_features = self.extract_features(audio)
            
            # Prepare input for model
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10,
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Process predictions
            probs = probabilities[0].cpu().numpy()
            
            # Create emotion scores
            emotion_scores = []
            for idx, prob in enumerate(probs):
                emotion_name = self.emotion_labels.get(idx, f"emotion_{idx}")
                mapped_emotion = self.emotion_mapping.get(emotion_name, emotion_name)
                
                emotion_scores.append({
                    "emotion": mapped_emotion,
                    "score": float(prob * 100),
                })
            
            # Sort by score
            emotion_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Get dominant emotion
            dominant = emotion_scores[0]
            
            return {
                "dominant_emotion": dominant["emotion"],
                "confidence": round(dominant["score"], 2),
                "emotion_scores": emotion_scores,
                "audio_features": audio_features
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during emotion prediction: {str(e)}")


# Global model instance
_model_instance = None


def load_model(model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
    """Load the emotion recognition model"""
    global _model_instance
    
    if _model_instance is None:
        print("🚀 Loading Speech Emotion Recognition Model...")
        _model_instance = SimpleEmotionModel(model_name)
        print("✅ Model ready for inference!")
    
    return _model_instance


def get_model():
    """Get the loaded model instance"""
    global _model_instance
    
    if _model_instance is None:
        _model_instance = load_model()
    
    return _model_instance

