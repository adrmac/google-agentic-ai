import os, requests
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file


### Google: Install the Google GenAI SDK:
# pip install -U -q "google-genai==1.7.0"
# this requires a different venv from the agentic AI course which uses a much more recent version

from google import genai
from google.genai import types


from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


### Set up API key in Google AI Studio
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    GOOGLE_API_KEY = None


### Ollama: Install Ollama and download models
# pip install ollama
# ollama pull gemma2
# ollama pull llama3
# ollama pull gemma2:2b
# ollama pull gemma2:9b
# ollama pull llama3.1.8b


# Runs locally, no API key needed
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2") # or "llama3"


