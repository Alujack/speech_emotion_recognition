from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="API for analyzing emotions from speech audio",
    version="1.0.0"
)

# Configure CORS
# Update these origins based on your frontend deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js development
        "http://localhost:3001",      # Alternative port
        # Add your production frontend URL here
        # "https://your-frontend-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Speech Emotion Recognition API",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/emotions")
async def get_emotions():
    """Get list of supported emotions"""
    return {
        "emotions": [
            "happy",
            "sad",
            "angry",
            "neutral",
            "fearful",
            "surprised"
        ]
    }
