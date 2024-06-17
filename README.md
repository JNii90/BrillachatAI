AI-Powered Math & Science Quiz Tutor

This project is a FastAPI-based web application that provides an interactive quiz experience for students preparing for math and science quizzes. It uses the Mistral 7B language model to generate questions and provide feedback, DistilWhisper for speech-to-text transcription, and Coqui TTS for text-to-speech conversion.

Features
Interactive Quiz: Engages students in a conversational quiz format.
Speech Recognition: Accepts student answers via audio input (using DistilWhisper).
Text-to-Speech: Provides questions and feedback in audio format (using Coqui TTS).
Conversation History: Maintains context throughout the quiz session.
Question Variety: Loads questions from a JSON file.
Score Tracking: (Optional) Can be implemented to track student scores.


Requirements
Python Libraries:

fastapi
uvicorn[standard]
transformers
nest-asyncio
requests
soundfile
torch
TTS (Coqui-ai/TTS)
pydub
pyngrok (if using ngrok)
FFmpeg (for audio conversion)
NGROK_AUTH_TOKEN 

Hardware (Recommended):

GPU : For faster model inference.



Download the Coqui TTS model you want to use and place it in the appropriate directory (tts_models/...). You can find models in the Coqui repository.
Running the Application

Bash
uvicorn main:app --reload


If you're using ngrok to expose your localhost, run:


ngrok.set_auth_token(ngrok_auth_token)
public_url = ngrok.connect(8000).public_url
print("Public URL:", public_url)



Start Conversation:

Send a POST request to /start_conversation (optional, to provide the user's name).

Quiz Interaction:

Send a POST request to /quiz_interaction/ with an audio file (multipart/form-data) containing the student's answer.

The API will return an audio file (application/octet-stream) with the tutor's question or feedback.


Other Endpoints:

/timer/{question_type}: Get the timer duration for the given question type.
/update_score/{result}: Update the scoreboard (not fully implemented in this example).
/score_board/: Get the current score (not fully implemented).
/health/: Check if the API is running.


Additional Notes:

Make sure you have FFmpeg installed on your system.
If you don't want to use ngrok, you'll need to deploy this API on a public server for external access.
