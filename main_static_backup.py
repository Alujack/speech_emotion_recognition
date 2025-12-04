from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import random
from datetime import datetime
import os

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="API for analyzing emotions from speech audio",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js development
        "http://localhost:3001",      # Alternative port
        "*",                          # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported emotions with descriptions
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


def generate_static_emotion_analysis(filename: str) -> AnalysisResult:
    """
    Generate static emotion analysis results
    Returns realistic-looking emotion recognition results
    """
    # Generate random but realistic emotion scores that sum to ~100%
    emotions = list(EMOTIONS.keys())

    # Pick a dominant emotion
    dominant = random.choice(emotions)

    # Generate scores with the dominant emotion having highest score
    scores = {}
    remaining = 100.0

    for emotion in emotions:
        if emotion == dominant:
            # Dominant emotion gets 40-70% confidence
            score = random.uniform(40.0, 70.0)
        else:
            # Other emotions get smaller shares
            score = random.uniform(1.0, 15.0)
        scores[emotion] = score

    # Normalize scores to sum to 100
    total = sum(scores.values())
    scores = {k: (v / total) * 100 for k, v in scores.items()}

    # Create emotion score objects
    emotion_scores = [
        EmotionScore(
            emotion=emotion,
            label=EMOTIONS[emotion]["label"],
            score=round(scores[emotion], 2),
            color=EMOTIONS[emotion]["color"],
            emoji=EMOTIONS[emotion]["emoji"]
        )
        for emotion in emotions
    ]

    # Sort by score descending
    emotion_scores.sort(key=lambda x: x.score, reverse=True)

    # Generate audio features (static/random values)
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

    return AnalysisResult(
        success=True,
        filename=filename,
        timestamp=datetime.now().isoformat(),
        dominant_emotion=dominant,
        confidence=round(scores[dominant], 2),
        emotion_scores=emotion_scores,
        audio_features=audio_features
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Speech Emotion Recognition API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


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
        ]
    }


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file and return emotion recognition results
    This returns static/random results for demonstration
    """
    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg",
                     "audio/mp3", "audio/ogg", "audio/webm"]

    if file.content_type not in allowed_types:
        # Also check file extension as a fallback
        allowed_extensions = [".wav", ".mp3", ".ogg", ".webm", ".m4a"]
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported types: {', '.join(allowed_extensions)}"
            )

    # Read file content (we won't use it for static results, but good for validation)
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Check file size (max 10MB)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="File too large (max 10MB)")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error reading file: {str(e)}")

    # Generate static emotion analysis
    result = generate_static_emotion_analysis(file.filename)

    return result


@app.get("/api/analyze/demo")
async def demo_analysis():
    """
    Get a demo analysis without uploading a file
    """
    result = generate_static_emotion_analysis("demo_audio.wav")
    return result
