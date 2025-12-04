# Speech Emotion Recognition - Backend API

FastAPI backend service for Speech Emotion Recognition system with static emotion analysis results.

> **Note:** This is a separate repository from the frontend. The frontend Next.js application is maintained in a different repository.

## Features

- RESTful API built with FastAPI
- CORS enabled for Next.js frontend integration
- Audio file upload endpoint
- Static emotion recognition results (for demonstration)
- Comprehensive emotion analysis with confidence scores
- Audio feature extraction (simulated)
- Health check endpoint

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:

   On macOS/Linux:

   ```bash
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the development server with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- API: http://localhost:8000
- Interactive API docs (Swagger): http://localhost:8000/docs
- Alternative API docs (ReDoc): http://localhost:8000/redoc

## API Endpoints

### Root Endpoint

- **GET** `/` - Welcome message and API status

### Health Check

- **GET** `/health` - Returns API health status with timestamp

### Emotions

- **GET** `/api/emotions` - Returns list of supported emotions with metadata (label, color, emoji)

### Analyze Audio

- **POST** `/api/analyze` - Upload audio file and get emotion analysis
  - Accepts: WAV, MP3, OGG, WebM, M4A files
  - Max size: 10MB
  - Returns: Emotion scores, dominant emotion, confidence, audio features

### Demo Analysis

- **GET** `/api/analyze/demo` - Get a demo analysis without uploading a file

## Response Format

The `/api/analyze` endpoint returns:

```json
{
  "success": true,
  "filename": "audio.wav",
  "timestamp": "2025-12-04T10:30:00",
  "dominant_emotion": "happy",
  "confidence": 65.42,
  "emotion_scores": [
    {
      "emotion": "happy",
      "label": "Happy",
      "score": 65.42,
      "color": "#22c55e",
      "emoji": "😊"
    },
    ...
  ],
  "audio_features": {
    "duration": 5.2,
    "sample_rate": 22050,
    "pitch_mean": 180.5,
    "pitch_std": 25.3,
    "energy_mean": 0.15,
    "energy_std": 0.03,
    "tempo": 120.0,
    "spectral_centroid": 2000.0
  }
}
```

## Supported Emotions

- Happy 😊
- Sad 😢
- Angry 😠
- Neutral 😐
- Fearful 😨
- Surprised 😮
- Disgusted 🤢

## Project Structure

```
speech_emotion_recognition/
├── main.py              # FastAPI application with all endpoints
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## CORS Configuration

The API is configured to accept requests from `http://localhost:3000` (Next.js default), `http://localhost:3001`, and all origins (for development).

**For production**, modify the `allow_origins` list in `main.py` to include only your frontend domain:

```python
allow_origins=[
    "https://your-frontend-domain.com",
]
```

## Integration with Frontend

The Next.js frontend (separate repository) makes API calls to this backend:

```typescript
const formData = new FormData();
formData.append('file', audioFile);

const response = await fetch("http://localhost:8000/api/analyze", {
  method: "POST",
  body: formData,
});
const data = await response.json();
```

## Current Implementation

This backend currently returns **static/random emotion analysis results** for demonstration purposes. The results are generated to simulate realistic emotion recognition output.

### To Implement Real AI Model:

1. Install ML libraries: `librosa`, `tensorflow`/`pytorch`, `numpy`
2. Load pre-trained emotion recognition model
3. Replace `generate_static_emotion_analysis()` function with real analysis
4. Extract actual audio features from uploaded files
5. Run inference through the ML model

## Deployment

### Option 1: Docker (Recommended)
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 2: Traditional Hosting
Deploy to services like:
- **Railway**: Easy Python deployment with auto-detection
- **Render**: Free tier with auto-deploy from git
- **Heroku**: Classic PaaS platform
- **DigitalOcean App Platform**: Container-based deployment
- **AWS EC2/Lambda**: For more control

### Environment Variables
For production, use environment variables for configuration:
```python
import os
from fastapi.middleware.cors import CORSMiddleware

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
```

## Testing

1. Start the server
2. Visit http://localhost:8000/docs
3. Try the `/api/analyze/demo` endpoint
4. Or upload an audio file via `/api/analyze`

## Contributing

When adding new features:

1. Update this README
2. Add appropriate error handling
3. Document new endpoints in docstrings
4. Test endpoints using `/docs`

## License

[Add your license here]
