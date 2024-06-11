import openai
from decouple import config
from functions.database import get_recent_messages  

# Directly set the API keys
openai.api_key = "sk-proj-5HLvyDuG0hZB5k7LXyg4T3BlbkFJz9lW9LZ08y0fQW66KhAz"
# Fine-tuned model ID
FINE_TUNED_MODEL_ID = "ft:gpt-3.5-turbo-0125:personal::9Uw4ZfRB"

# OpenAI Whisper - Convert audio to text
def convert_audio_to_text(audio_file):
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        message_text = transcript["text"]
        return message_text
    except Exception as e:
        print(f"Error in convert_audio_to_text: {e}")
        return ""

# OpenAI ChatGPT - Get chat response
def get_chat_response(message_input):
    messages = get_recent_messages()
    user_message = {
        "role": "user",
        "content": message_input + " Remember to time the response based on whether the question requires calculation (30 seconds) or not (15 seconds)"
    }
    messages.append(user_message)
    print(messages)
    
    try:
        response = openai.ChatCompletion.create(
            model=FINE_TUNED_MODEL_ID,
            messages=messages
        )
        message_text = response["choices"][0]["message"]["content"]
        return message_text
    except Exception as e:
        print(f"Error in get_chat_response: {e}")
        return ""
