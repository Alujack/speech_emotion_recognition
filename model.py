"""
Speech Emotion Recognition Model using Wav2Vec2
This module provides the AI model for analyzing emotions in speech
"""

import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from typing import Dict, List
import warnings
import soundfile as sf
from scipy import signal
from scipy.fft import rfft, rfftfreq
import os

warnings.filterwarnings('ignore')


class EmotionRecognitionModel:
    """
    Wav2Vec2-based emotion recognition model
    Achieves high accuracy through transfer learning and fine-tuning
    """

    def __init__(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initialize the emotion recognition model

        Args:
            model_name: HuggingFace model identifier
                      Default uses a pre-trained emotion recognition model
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Initializing model on device: {self.device}")

        try:
            # Load processor and model from HuggingFace
            print(f"📥 Loading model: {model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print(
                "💡 Using fallback mode - install transformers, torch, and librosa for AI features")
            raise

        # Emotion labels mapping (adjust based on your specific model)
        self.emotion_labels = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise"
        }

        # Map model outputs to our API emotion names
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
        Load and preprocess audio file for model input

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (Wav2Vec2 expects 16kHz)

        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio file using soundfile
            audio, sr = sf.read(audio_path, dtype='float32')

            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if necessary
            if sr != target_sr:
                # Calculate new length
                duration = len(audio) / sr
                num_samples = int(duration * target_sr)
                audio = signal.resample(audio, num_samples)

            # Normalize audio to [-1, 1] range
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

            # Simple silence removal (trim low energy sections)
            energy = np.abs(audio)
            threshold = 0.01
            mask = energy > threshold
            if mask.any():
                audio = audio[mask]

            return audio

        except Exception as e:
            raise ValueError(f"Error preprocessing audio: {str(e)}")

    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Extract audio features for metadata and analysis

        Args:
            audio: Audio waveform numpy array
            sr: Sample rate

        Returns:
            Dictionary containing various audio features
        """
        features = {}

        try:
            # Duration (seconds)
            features['duration'] = round(len(audio) / sr, 2)
            features['sample_rate'] = float(sr)

            # Pitch analysis (F0)
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=sr, fmin=50, fmax=400)
            pitch_values = []

            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features['pitch_mean'] = round(float(np.mean(pitch_values)), 2)
                features['pitch_std'] = round(float(np.std(pitch_values)), 2)
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0

            # Energy (RMS)
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = round(float(np.mean(rms)), 4)
            features['energy_std'] = round(float(np.std(rms)), 4)

            # Tempo (BPM)
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = round(
                float(tempo), 2) if not np.isnan(tempo) else 120.0

            # Spectral centroid (brightness of sound)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[
                0]
            features['spectral_centroid'] = round(
                float(np.mean(spectral_centroids)), 2)

            # Zero crossing rate (indicator of noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zero_crossing_rate'] = round(float(np.mean(zcr)), 4)

            # MFCC statistics (timbre)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = round(float(np.mean(mfccs)), 4)
            features['mfcc_std'] = round(float(np.std(mfccs)), 4)

        except Exception as e:
            print(f"⚠️  Warning: Error extracting some features: {e}")
            # Return basic features if extraction fails
            features.update({
                'duration': round(len(audio) / sr, 2),
                'sample_rate': float(sr),
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'energy_mean': 0.0,
                'energy_std': 0.0,
                'tempo': 120.0,
                'spectral_centroid': 0.0
            })

        return features

    def predict_emotion(self, audio_path: str) -> Dict:
        """
        Predict emotion from audio file using Wav2Vec2 model

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing:
                - dominant_emotion: Most likely emotion
                - confidence: Confidence score (0-100)
                - emotion_scores: List of all emotion probabilities
                - audio_features: Extracted audio features
        """
        try:
            # Step 1: Preprocess audio
            audio = self.preprocess_audio(audio_path)

            # Step 2: Extract audio features for metadata
            audio_features = self.extract_features(audio)

            # Step 3: Prepare input for Wav2Vec2 model
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10,  # Max 10 seconds
                truncation=True
            )

            # Move tensors to device (GPU or CPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Step 4: Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Step 5: Process predictions
            probs = probabilities[0].cpu().numpy()

            # Create emotion scores list
            emotion_scores = []
            for idx, prob in enumerate(probs):
                emotion_name = self.emotion_labels.get(idx, f"emotion_{idx}")
                mapped_emotion = self.emotion_mapping.get(
                    emotion_name, emotion_name)

                emotion_scores.append({
                    "emotion": mapped_emotion,
                    "score": float(prob * 100),  # Convert to percentage
                })

            # Sort by score (highest first)
            emotion_scores.sort(key=lambda x: x['score'], reverse=True)

            # Get dominant emotion and confidence
            dominant = emotion_scores[0]

            return {
                "dominant_emotion": dominant["emotion"],
                "confidence": round(dominant["score"], 2),
                "emotion_scores": emotion_scores,
                "audio_features": audio_features
            }

        except Exception as e:
            raise RuntimeError(f"Error during emotion prediction: {str(e)}")


# Global model instance (singleton pattern)
_model_instance = None


def load_model(model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
    """
    Load the emotion recognition model (called once at startup)

    Args:
        model_name: HuggingFace model identifier

    Returns:
        EmotionRecognitionModel instance
    """
    global _model_instance

    if _model_instance is None:
        print("🚀 Loading Speech Emotion Recognition Model...")
        _model_instance = EmotionRecognitionModel(model_name)
        print("✅ Model ready for inference!")

    return _model_instance


def get_model():
    """
    Get the loaded model instance

    Returns:
        EmotionRecognitionModel instance
    """
    global _model_instance

    if _model_instance is None:
        _model_instance = load_model()

    return _model_instance


# For testing the model independently
if __name__ == "__main__":
    print("🧪 Testing Emotion Recognition Model")
    print("=" * 50)

    try:
        # Load model
        model = load_model()

        # Test with a sample audio file (if available)
        test_audio = "test_audio.wav"

        if os.path.exists(test_audio):
            print(f"\n📊 Analyzing: {test_audio}")
            result = model.predict_emotion(test_audio)

            print(f"\n🎯 Results:")
            print(f"   Dominant Emotion: {result['dominant_emotion']}")
            print(f"   Confidence: {result['confidence']:.2f}%")
            print(f"\n📈 All Emotions:")
            for score in result['emotion_scores']:
                print(f"   {score['emotion']}: {score['score']:.2f}%")

            print(f"\n🔊 Audio Features:")
            for key, value in result['audio_features'].items():
                print(f"   {key}: {value}")
        else:
            print(f"⚠️  No test audio file found at: {test_audio}")
            print("   Model loaded successfully! Ready for predictions.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have installed:")
        print("   pip install torch torchaudio transformers librosa numpy soundfile")
