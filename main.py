import openai
from fastapi import FastAPI, WebSocket, Request, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests
import whisper
import asyncio
import logging
import sounddevice as sd
import numpy as np
import wave
from functions.database import store_messages, reset_messages, get_recent_messages
from functions.openai_requests import get_chat_response, convert_audio_to_text
from functions.text_to_speech import convert_text_to_speech
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta

# Directly set the API keys
OPENAI_API_KEY = "sk-proj-5HLvyDuG0hZB5k7LXyg4T3BlbkFJz9lW9LZ08y0fQW66KhAz"
ELEVEN_LABS_API_KEY = "98ea649722fe058e3dca3c3ab897d84a"

# Set the API key for OpenAI
openai.api_key = OPENAI_API_KEY
MODEL_ID = "base"  # Set the Whisper model you want to use

app = FastAPI()

# Load Whisper model
model = whisper.load_model(MODEL_ID)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    text: str

class TimerResponse(BaseModel):
    message: str
    time_left: int

# Scoreboard initialization
score_board = {
    "correct": 0,
    "partial": 0,
    "incorrect": 0
}

# Usage statistics initialization
usage_stats = {
    "requests_count": 0,
    "success_count": 0,
    "failure_count": 0
}

class TimerResponse(BaseModel):
    timer: int

@app.get("/timer/{question_type}", response_model=TimerResponse, tags=["Timer"])
async def timer(question_type: str):
    """
    Returns the appropriate timer duration based on the question type.

    Args:
        question_type: The type of question ("calculation" or other).

    Returns:
        A JSON response containing the timer duration in seconds.
    """
    timer_duration = 30 if question_type == "calculation" else 15  
    return {"timer": timer_duration}

@app.post("/update_score/{result}", tags=["Scoreboard"])
async def update_score(result: str):
    """
    Updates the scoreboard based on the provided result.

    Args:
        result: The result of the question ("correct", "partial", or "incorrect").

    Returns:
        A JSON response containing the updated scoreboard.

    Raises:
        HTTPException: If an invalid result is provided.
    """
    valid_results = ["correct", "partial", "incorrect"]
    if result not in valid_results:
        raise HTTPException(status_code=400, detail="Invalid result")
    
    if result == "correct":
        score_board["correct"] += 3
    elif result == "partial":
        score_board["partial"] += 1
    elif result == "incorrect":
        score_board["incorrect"] += 1

    return {"score_board": score_board}

@app.get("/score_board", tags=["Scoreboard"])
async def get_score_board():
    """
    Retrieves the current scoreboard.

    Returns:
        A JSON response containing the scoreboard data.
    """
    return {"score_board": score_board}

@app.get("/usage_stats", tags=["Usage Statistics"])
async def get_usage_stats():
    """
    Retrieves usage statistics for the API.

    Returns:
        A JSON response containing the usage statistics data.
    """
    usage_stats["requests_count"] += 1
    return {"usage_stats": usage_stats}

# Check Health
@app.get("/health/")
async def check_health():
    return {"message": "healthy"}

# Reset messages
@app.get("/reset_conversation/")
async def reset_conversation():
    reset_messages()
    return {"message": "conversation reset"}

# Post Audio
@app.post("/post-audio/")
async def post_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        audio_path = f"uploaded_{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # Convert speech to text
        transcript = convert_audio_to_text(audio_path)
        if not transcript:
            raise HTTPException(status_code=400, detail="Failed to decode audio")

        # Get chat response
        response = get_chat_response(transcript)
        if not response:
            raise HTTPException(status_code=400, detail="Failed to get chat response")

        # Store messages
        store_messages(transcript, response)

        # Convert chat response to audio using ElevenLabs
        output_audio_path = "response.mp3"
        convert_text_to_speech(response, output_audio_path)

        # Create a generator that yields chunks of data
        def iterfile():
            with open(output_audio_path, "rb") as f:
                yield from f

        return StreamingResponse(iterfile(), media_type="audio/mpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




     




 




    




















