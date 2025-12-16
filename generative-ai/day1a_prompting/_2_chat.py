import os, requests
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file

from google import genai
from google.genai import types

### Set up API key in Google AI Studio
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    GOOGLE_API_KEY = None

# Ollama runs locally, no API key needed
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2") # or "llama3"


### Start a chat
# a chat is stateful, it remembers what you said
# generate is stateless


### Google

client = genai.Client(api_key=GOOGLE_API_KEY)


chat = client.chats.create(
    model='gemini-2.0-flash', 
    history=[]
)

response = chat.send_message('Hello! My name is Zlork!')
print("Gemini response:")
print(response.text)

response = chat.send_message('Can you tell me something interesting about dinosaurs?')
print(response.text)

# while you have the chat object alive, the conversation state persists
response = chat.send_message('Do you remember what my name is?')
print(response.text)


### Ollama
# note /api/chat, not /api/generate

# biggest difference -- Ollama doesn't store the message history for you, you have to do it yourself

print("")
print("Gemma response:")
print("")

messages = []

## common roles: 
# "system" - high level instruction / policy ("you are.." constraints)
# "user" - user messages
# "assistant" - model's messages

def send_message(content):
    messages.append({
        "role": "user",
        "content": content
    })

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False, 
            # wait until full response is ready before sending tokens
        },
        timeout=120, # give up after 120s
    )
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]

    message = {
        "role": "assistant",
        "content": content
    }

    messages.append(message)

    return message


response = send_message('Hello! My name is Zlork!')
print(response)

response = send_message('Can you tell me something interesting about dinosaurs?')
print(response)

response = send_message('Do you remember what my name is?')
print(response)
