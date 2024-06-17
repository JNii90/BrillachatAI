import os
import json
import random
import logging
import tempfile
import subprocess
import torch
import soundfile as sf
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyngrok import ngrok
import nest_asyncio
import uvicorn
from TTS.api import TTS
from pydub import AudioSegment

# Initialize the FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables for API keys
ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN", "2hE7mlcJMmUFSPDEYEh3tWAdEJv_4HLwxGnDCe3DEb3f5rdPv")

# Initialize Mistral model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.to(device)

# Device and dtype setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the DistilWhisper model and processor
model_id = "distil-whisper/distil-large-v2"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

# Initialize Coqui TTS API
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

nest_asyncio.apply()

# Scoreboard
score_board = {"correct": 3, "partial": 1, "incorrect": 0}

# Load questions from JSON file
QUESTIONS_FILE = "/mnt/data/sample_questions.json"  

def load_questions():
    try:
        with open(QUESTIONS_FILE, "r") as file:
            questions = json.load(file)
            if not questions:
                raise ValueError("No questions found in the file.")
            return questions
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error loading questions from {QUESTIONS_FILE}: {e}")
        return [] 

# Load questions on application start-up
questions = load_questions()
if not questions:
    logger.error("No questions loaded. Please check the JSON file.")

# Functions for audio processing and transcription
def convert_audio_to_wav(audio_file_path):
    try:
        logger.info(f"Processing audio file: {audio_file_path}")
        wav_file_path = tempfile.mktemp(suffix=".wav")

        # Convert audio file to WAV format if it's not already in WAV format
        if audio_file_path.endswith('.mp3'):
            logger.info(f"Converting MP3 to WAV")
            audio = AudioSegment.from_mp3(audio_file_path)
            audio.export(wav_file_path, format="wav")
        elif audio_file_path.endswith('.wav'):
            wav_file_path = audio_file_path
        else:
            raise ValueError("Unsupported audio format")

        logger.info(f"Using WAV file: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        logger.error(f"Error in convert_audio_to_wav: {e}")
        return ""

def transcribe_audio(wav_file_path):
    try:
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(wav_file_path)
        with audio_file as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        logger.error(f"Error in transcribing audio: {e}")
        return ""

# Conversation History Management
RECENT_MESSAGES_FILE = "recent_messages.json"
MAX_RECENT_MESSAGES = 5

def get_recent_messages():
    messages = []
    try:
        if os.path.exists(RECENT_MESSAGES_FILE):  # Check if file exists
            with open(RECENT_MESSAGES_FILE, "r") as file:
                messages = json.load(file)[-MAX_RECENT_MESSAGES:]
    except json.JSONDecodeError:
        logger.warning(f"Error decoding JSON from {RECENT_MESSAGES_FILE}. Starting fresh conversation.")
    return messages

# Prompt engineering function
def generate_prompt(messages):
    learn_instruction = {
        "role": "system",
        "content": (
            "You are a dedicated science and maths teacher for final year students preparing for a national science "
            "and maths quiz. Your task is to ask the students questions in science and maths and time their responses. "
            "If the question requires calculation, students will have 30 seconds to answer; otherwise, they will have "
            "15 seconds. For correct answers, award 3 points. For partially correct answers, award 1 point. Incorrect "
            "answers receive 0 points. For incorrect or partially correct answers, provide an explanation of the correct answer."
        )
    }

    # Add dynamic content to learn_instruction
    x = random.uniform(0, 1)
    if x < 0.2:
        learn_instruction["content"] += " Your response will include a useful study tip."
    elif x < 0.5:
        learn_instruction["content"] += " Your response will share an interesting science or maths trivia fact."
    else:
        learn_instruction["content"] += " Your response will recommend a practice exercise or problem."

    return [learn_instruction] + messages

def get_chat_response(message):
    messages = get_recent_messages() or [{"role": "system", "content": "You are a dedicated science and maths teacher for final year students preparing for a national science and maths quiz."}]
    messages.append({"role": "user", "content": message})

    # Generate the prompt with the recent messages
    prompt = generate_prompt(messages)

    # Get model's response
    encoded_inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)
    output_sequences = model.generate(encoded_inputs, max_new_tokens=1000)
    response = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]

    # Ensure correct role alternation
    if messages and messages[-1]["role"] == "user":
        messages.append({"role": "assistant", "content": response})

    # Save conversation history to file
    with open(RECENT_MESSAGES_FILE, "w") as file:
        json.dump(messages[-MAX_RECENT_MESSAGES:], file)  # Save only recent messages

    return response

# Text-to-Speech conversion using Coqui TTS
def convert_text_to_speech(text):
    try:
        wav = tts.tts(text)
        output_file_path = tempfile.mktemp(suffix=".wav")
        with open(output_file_path, "wb") as f:
            f.write(wav)
        return output_file_path
    except Exception as e:
        logger.error(f"Error in convert_text_to_speech: {e}")
        return ""

# Pydantic models for request/response validation
class TimerResponse(BaseModel):
    timer: int

class UserResponse(BaseModel):
    response: str

# API endpoints
@app.get("/timer/{question_type}", response_model=TimerResponse, tags=["Timer"])
async def timer(question_type: str):
    timer_duration = 30 if question_type == "calculation" else 15
    return {"timer": timer_duration}

@app.post("/update_score/{result}", tags=["Scoreboard"])
async def update_score(result: str):
    valid_results = ["correct", "partial", "incorrect"]
    if result not in valid_results:
        raise HTTPException(status_code=400, detail="Invalid result")
    score_board[result] += 3 if result == "correct" else 1 if result == "partial" else 0
    return {"score_board": score_board}

@app.get("/score_board", tags=["Scoreboard"])
async def get_score_board():
    return {"score_board": score_board}

@app.get("/health", tags=["Health"])
async def check_health():
    return {"message": "healthy"}

@app.post("/quiz_interaction/")
async def quiz_interaction(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        # Save the uploaded file to a temporary location
        suffix = '.mp3' if file.filename.endswith('.mp3') else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        logger.info(f"File saved to: {tmp_path}")

        # Convert audio to WAV if needed
        wav_file_path = convert_audio_to_wav(tmp_path)
        if not wav_file_path:
            raise HTTPException(status_code=500, detail="Failed to convert audio to WAV.")

        # Transcribe audio
        student_answer = transcribe_audio(wav_file_path)
        if not student_answer:
            raise HTTPException(status_code=500, detail="Failed to decode audio to text.")
        logger.info(f"Student Answer: {student_answer}")

        messages = get_recent_messages() or [{"role": "system", "content": "You are a dedicated science and maths teacher for final year students preparing for a national science and maths quiz."}]
        messages.append({"role": "user", "content": student_answer})

        # Get the next question
        if not questions:
            raise HTTPException(status_code=503, detail="No more questions available")
        next_question = random.choice(questions)
        questions.remove(next_question)  # Remove the question to avoid repetition

        # Generate Mistral's response (question or feedback)
        mistral_response = get_chat_response(next_question['question'])

        # Evaluate the student's answer
        is_correct = student_answer.lower() == next_question["answer"].lower()

        # Update messages and save conversation history
        if is_correct:
            mistral_response += " That's correct!"
            score_board["correct"] += 1
        else:
            mistral_response += f" The correct answer is {next_question['answer']}."
            score_board["incorrect"] += 1

        with open(RECENT_MESSAGES_FILE, "w") as file:
            json.dump(messages[-MAX_RECENT_MESSAGES:], file)

        logger.info(f"Mistral Response: {mistral_response}")

        output_audio_path = convert_text_to_speech(mistral_response)
        if not output_audio_path:
            raise HTTPException(status_code=500, detail="Failed to convert text to speech.")
        
        logger.info(f"Audio response saved to: {output_audio_path}")

        # Return audio
        def iterfile():
            with open(output_audio_path, "rb") as f:
                yield from f

        return StreamingResponse(iterfile(), media_type="application/octet-stream")

    except Exception as e:
        logger.error(f"Exception in /quiz_interaction/: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_conversation", tags=["Conversation"])
async def start_conversation(user_name: str = Form(...)):
    initial_message = (
        "Hi there, I will be your tutor today to help you prepare for the National Science and Math Quiz. "
        "My name is Jaycharl. What is your name and I hope you are ready to start."
    )
    return {"message": initial_message}

if __name__ == "__main__":
    ngrok.set_auth_token(ngrok_auth_token)
    public_url = ngrok.connect(8000).public_url
    print("Public URL:", public_url)

    uvicorn.run(app, host="0.0.0.0", port=8000)

