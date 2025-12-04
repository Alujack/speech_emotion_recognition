#!/bin/bash
# Start Speech Emotion Recognition API with AI mode enabled

echo "🚀 Starting Speech Emotion Recognition API with AI Mode..."

cd /opt/school-project/Speech_Emotion_Recognition/speech_emotion_recognition

# Activate virtual environment
source venv/bin/activate

# Enable AI mode
export USE_AI_MODEL=true

echo "✅ AI Mode: ENABLED"
echo "📡 Starting server on http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "The model will download on first use (~400MB, one-time)"
echo "Please wait..."
echo ""

# Start server
uvicorn main:app --reload --port 8000 --host 0.0.0.0

