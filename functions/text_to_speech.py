import requests
from decouple import config
import os


ELEVEN_LABS_API_KEY = "98ea649722fe058e3dca3c3ab897d84a"

 # Define data (Body)
def convert_text_to_speech(message):
    body = {
        "text": message,
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

# Define Voice

    Mwika_Kayange = "Yru1AaCztNSYkMNCbM1k"
    
    # Defining end points and Headers
    
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json",
        "accept": "audio/mpeg"
    }
    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{Mwika_Kayange}"

# Send request

    try:
        response = requests.post(endpoint, json=body, headers=headers)
    except Exception as e:
        print(f"Error in convert_text_to_speech: {e}")
        return
    
    # Handle Response
    if response.status_code == 200:
        return response.content
    else:
        return


