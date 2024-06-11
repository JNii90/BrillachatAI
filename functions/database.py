import os
import json
import random

# Get recent messages
def get_recent_messages():
    file_name = "stored_data.json"
    learn_instruction = {
        "role": "system",
        "content": "You are a dedicated science and maths teacher for final year students preparing for a national science and maths quiz. Your task is to ask the students questions in science and maths and time their responses. If the question requires calculation, students will have 30 seconds to answer; otherwise, they will have 15 seconds. For correct answers, award 3 points. For partially correct answers, award 1 point. Incorrect answers receive 0 points. For incorrect or partially correct answers, provide an explanation of the correct answer."
    }

    messages = []

    x = random.uniform(0, 1)
    if x < 0.2:
        learn_instruction["content"] += " Your response will include a useful study tip."
    elif x < 0.5:
        learn_instruction["content"] += " Your response will share an interesting science or maths trivia fact."
    else:
        learn_instruction["content"] += " Your response will recommend a practice exercise or problem."

    messages.append(learn_instruction)

    try:
        with open(file_name) as user_file:
            data = json.load(user_file)
            if data:
                if len(data) < 5:
                    for item in data:
                        messages.append(item)
                else:
                    for item in data[-5:]:
                        messages.append(item)
    except:
        pass

    return messages

# Store messages for retrieval later
def store_messages(request_message, response_message):
    file_name = "stored_data.json"
    messages = get_recent_messages()[1:]
    user_message = {"role": "user", "content": request_message}
    assistant_message = {"role": "assistant", "content": response_message}
    messages.append(user_message)
    messages.append(assistant_message)
    with open(file_name, "w") as f:
        json.dump(messages, f)

# Reset messages
def reset_messages():
    
    # Overwrite current file with nothing
    open("stored_data.json", "w").close()

