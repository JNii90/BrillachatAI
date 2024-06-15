# AFRICAIed HACKATHON PROJECT

## Introduction
Welcome to AFRICAIed HACKATHON PROJECT REPO. This is where you will find the source code for the backend to the Hackathon Project. 

## Installation and Setup
### 1. Clone the Repository
```bash
git clone [repo-name](https://github.com/JNii90/BrillachatAI.git)
cd BrillaChatAI
```

### 2. Create a Virtual Environment:
```bash
cd BrillaChatAI
```

On macOS and Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

On windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependancies
```bash
pip install -r requirements.txt
```

## Running the Application
### 1. Run the Server Notebook:
Open the [Release] Riddle Prep App Server.ipynb file, preferably in Colab if you don't have a GPU on your, follow the instructions to start the server, and copy the ngrok URL.

### 2. Update app.py:
Paste the ngrok URL into the BASE_URL variable at the top of the app.py file.

### 3. Run app.py
python app.py or using uvicorn, you can run the app: uvicorn -g main:app --reload




