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

GOOGLE_MODEL=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# Ollama runs locally, no API key needed
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b") # or "llama3"



### Thinking mode
# You can ignore this for Gemini
# There was once an experimental "Thinking" model trained to use its own thinking process as context in its response, so you don't need to expose specialized prompting like in 5_prompting.py - ReAct.
# This was deprecated and rolled into the full production model. So now all Gemini models already do hidden CoT. 
# The industry at large has backed away from exposed CoT: it leaks training artifacts, encourages prompt hacking, and increases hallucination when users overfit to reasoning text

client = genai.Client(api_key=GOOGLE_API_KEY)

import io
from IPython.display import Markdown, clear_output


response = client.models.generate_content_stream(
    # model='gemini-2.0-flash-thinking-exp', # this no longer exists
    model='gemini-2.0-flash',
    contents='Who was the youngest author listed on the transformers NLP paper?',
)

buf = io.StringIO()
for chunk in response:
    buf.write(chunk.text)
    # Display the response as it is streamed
    print(chunk.text, end='')

# And then render the finished response as formatted markdown.
clear_output()
Markdown(buf.getvalue())
