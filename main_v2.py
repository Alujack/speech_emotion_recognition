# Main FastAPI Application
# This file can work in two modes:
# 1. STATIC MODE (default): Returns realistic-looking static results for demo
# 2. AI MODE: Uses Wav2Vec2 for real emotion recognition (requires ML dependencies)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import random
from datetime import datetime
import os
import tempfile

# Try to import AI model (will use static mode if not available)
USE_AI_MODEL = False
try:
    from model_simple import get_model
    USE_AI_MODEL = os.getenv("USE_AI_MODEL", "false").lower() == "true"
    if USE_AI_MODEL:
        print("🤖 AI MODE: Using Wav2Vec2 for real emotion recognition")
except ImportError:
    print("📊 STATIC MODE: Using static results (install ML dependencies for AI mode)")

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="API for analyzing emotions from speech audio" +
                (" using Wav2Vec2 AI model" if USE_AI_MODEL else " (demo mode with static results)"),
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "*",  # Allow all for development
    ],
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
    mode: str  # "ai" or "static"


def generate_static_emotion_analysis(filename: str) -> Dict:
    """
    Generate static emotion analysis results for demonstration
    Returns realistic-looking emotion recognition results
    """
    emotions = list(EMOTIONS.keys())

    # Pick a dominant emotion
    dominant = random.choice(emotions)

    # Generate scores with the dominant emotion having highest score
    scores = {}
    for emotion in emotions:
        if emotion == dominant:
            score = random.uniform(40.0, 70.0)  # Dominant: 40-70%
        else:
            score = random.uniform(1.0, 15.0)   # Others: 1-15%
        scores[emotion] = score

    # Normalize scores to sum to 100
    total = sum(scores.values())
    scores = {k: (v / total) * 100 for k, v in scores.items()}

    # Sort emotions by score
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Create emotion score list
    emotion_scores = []
    for emotion, score in sorted_emotions:
        emotion_scores.append({
            "emotion": emotion,
            "score": round(score, 2),
        })

    # Generate audio features (simulated)
    audio_features = {
        "duration": round(random.uniform(1.0, 10.0), 2),
        "sample_rate": 22050,
        "pitch_mean": round(random.uniform(80.0, 250.0), 2),
        "pitch_std": round(random.uniform(10.0, 50.0), 2),
        "energy_mean": round(random.uniform(0.01, 0.5), 4),
        "energy_std": round(random.uniform(0.001, 0.1), 4),
        "tempo": round(random.uniform(60.0, 180.0), 2),
        "spectral_centroid": round(random.uniform(1000.0, 3000.0), 2),
    }

    return {
        "dominant_emotion": dominant,
        "confidence": round(scores[dominant], 2),
        "emotion_scores": emotion_scores,
        "audio_features": audio_features
    }


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    if USE_AI_MODEL:
        print("🚀 Loading AI model at startup...")
        try:
            get_model()
            print("✅ AI model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load AI model: {e}")
            print("⚠️  Falling back to static mode")
            global USE_AI_MODEL
            USE_AI_MODEL = False


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Speech Emotion Recognition API",
        "status": "running",
        "version": "2.0.0",
        "mode": "AI" if USE_AI_MODEL else "Static Demo",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "ai" if USE_AI_MODEL else "static"
    }


@app.get("/api/emotions")
async def get_emotions():
    """Get list of supported emotions with metadata"""
    return {
        "emotions": [
            {
                "id": key,
                "label": value["label"],
                "color": value["color"],
                "emoji": value["emoji"]
            }
            for key, value in EMOTIONS.items()
        ],
        "mode": "ai" if USE_AI_MODEL else "static"
    }


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file and return emotion recognition results

    - In AI mode: Uses Wav2Vec2 model for real emotion recognition
    - In Static mode: Returns realistic demo results
    """
    # Validate file type
    allowed_extensions = [".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"]
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(allowed_extensions)}"
        )

    # Read and validate file content
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Check file size (max 10MB)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="File too large (max 10MB)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error reading file: {str(e)}")

    # Process based on mode
    if USE_AI_MODEL:
        # AI MODE: Use real model
        temp_file = None
        try:
            # Save uploaded file temporarily
            suffix = os.path.splitext(file.filename)[1]
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix)
            temp_file.write(content)
            temp_file.close()

            # Get model and predict
            model = get_model()
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
                audio_features=prediction["audio_features"],
                mode="ai"
            )

            return result

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing audio with AI model: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    else:
        # STATIC MODE: Generate demo results
        prediction = generate_static_emotion_analysis(file.filename)

        # Format emotion scores with metadata
        emotion_scores = []
        for score in prediction["emotion_scores"]:
            emotion = score["emotion"]
            emotion_info = EMOTIONS[emotion]

            emotion_scores.append(EmotionScore(
                emotion=emotion,
                label=emotion_info["label"],
                score=score["score"],
                color=emotion_info["color"],
                emoji=emotion_info["emoji"]
            ))

        result = AnalysisResult(
            success=True,
            filename=file.filename,
            timestamp=datetime.now().isoformat(),
            dominant_emotion=prediction["dominant_emotion"],
            confidence=prediction["confidence"],
            emotion_scores=emotion_scores,
            audio_features=prediction["audio_features"],
            mode="static"
        )

        return result


@app.get("/api/analyze/demo")
async def demo_analysis():
    """
    Get a demo analysis without uploading a file
    Always returns static results regardless of mode
    """
    prediction = generate_static_emotion_analysis("demo_audio.wav")

    # Format emotion scores
    emotion_scores = []
    for score in prediction["emotion_scores"]:
        emotion = score["emotion"]
        emotion_info = EMOTIONS[emotion]

        emotion_scores.append(EmotionScore(
            emotion=emotion,
            label=emotion_info["label"],
            score=score["score"],
            color=emotion_info["color"],
            emoji=emotion_info["emoji"]
        ))

    result = AnalysisResult(
        success=True,
        filename="demo_audio.wav",
        timestamp=datetime.now().isoformat(),
        dominant_emotion=prediction["dominant_emotion"],
        confidence=prediction["confidence"],
        emotion_scores=emotion_scores,
        audio_features=prediction["audio_features"],
        mode="demo"
    )

    return result
