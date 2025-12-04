# Speech Emotion Recognition - Backend API

FastAPI backend service for Speech Emotion Recognition system.

> **Note:** This is a separate repository from the frontend. The frontend Next.js application is maintained in a different repository.

## Features

- RESTful API built with FastAPI
- CORS enabled for Next.js frontend integration
- Health check endpoint
- Emotion classification endpoints (ready to integrate with ML models)

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

- **GET** `/health` - Returns API health status

### Emotions

- **GET** `/api/emotions` - Returns list of supported emotions

## Project Structure

```
speech_emotion_recognition/
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## Development

### Adding New Endpoints

Edit `main.py` to add new routes:

```python
@app.post("/api/analyze")
async def analyze_audio(file: UploadFile):
    # Your audio processing logic here
    return {"emotion": "happy", "confidence": 0.95}
```

### CORS Configuration

The API is configured to accept requests from `http://localhost:3000` (Next.js default).
To add more origins, modify the `allow_origins` list in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    # ...
)
```

## Integration with Frontend

The Next.js frontend (separate repository) can make API calls to this backend:

```typescript
const response = await fetch("http://localhost:8000/api/emotions");
const data = await response.json();
```

## Next Steps

1. Implement audio file upload endpoint
2. Integrate ML model for emotion recognition
3. Add data preprocessing pipeline
4. Implement authentication (if needed)
5. Add database support for storing results

## Contributing

When adding new features:

1. Update this README
2. Add appropriate error handling
3. Document new endpoints in docstrings
4. Test endpoints using `/docs`

## License

[Add your license here]
