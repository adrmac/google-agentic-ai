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


### Run your first prompt
prompt = "Explain AI to me like I'm a kid."

### Google

client = genai.Client(api_key=GOOGLE_API_KEY)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

print("Gemini response:")
print(response.text)
print("")


### Ollama
# raw http request, not a bundled generate_content method

response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False, # wait until full response is ready before sending tokens
    },
    timeout=120, # give up after 120s
)
response.raise_for_status()

data = response.json()
print("Gemma2 response:")
print(data["response"])


response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    },
    timeout=120,
)
response.raise_for_status()

data = response.json()
print("Gemma2 response:")
print(data["response"])