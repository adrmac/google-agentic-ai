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

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


### Explore generation parameters

## Output length
# More compute, affects cost and performance
# Specify max_output_tokens with Gemini API
# Does not make it more succinct, just stops at limit

## Google

client = genai.Client(api_key=GOOGLE_API_KEY)

short_config = types.GenerateContentConfig(max_output_tokens=200)

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    config=short_config,
    contents="Write a 1000 word essay on the importance of olives in modern society."
)
print(response.text) # cuts off after 163 words


response = client.models.generate_content(
    model=GOOGLE_MODEL,
    config=short_config,
    contents='Write a short poem on the importance of olives in modern society.')
print(response.text) # this fits


## Temperature
# range 0.0 to 2.0, default 1.0
# degree of randomness in token selection
# higher means more candidate tokens from which the next output token is selected, and maybe more diverse results
# lower means selecting the most probable token for each step

high_temp_config = types.GenerateContentConfig(temperature=2.0) # prints a variety of colors


for _ in range(5):
    response = client.models.generate_content(
        model=GOOGLE_MODEL,
        config=high_temp_config,
        contents="Pick a random color... (respond in a single word)"
    )

if response.text:
    print(response.text, '-' * 25)

low_temp_config = types.GenerateContentConfig(temperature=0.0) # prints one color over and over


for _ in range(5):
    response = client.models.generate_content(
        model=GOOGLE_MODEL,
        config=high_temp_config,
        contents="Pick a random color... (respond in a single word)"
    )

if response.text:
    print(response.text, '-' * 25)


## Top-P - Probability Threshold
# similar to temperature, controls diversity of output
# Once tokens cumulatively exceed this threshold, they stop being selected as candidates. 0 = 'greedy decoding' / 1 = every token in vocabulary / default is 0.95

# Older: top-K, not configurable in Gemini 2, defines the most probable tokens from which to select the output token, 1 = select one token, greedy decoding

model_config = types.GenerateContentConfig(
    # these are the default values for gemini-2.0-flash
    temperature=1.0,
    top_p=0.95
)

story_prompt = "You are a creative writer. Write a 3-paragraph short story about a cat who goes on an adventure."
response = client.models.generate_content(
    model=GOOGLE_MODEL,
    config=model_config,
    contents=story_prompt
)

print("Gemini version:")
print(response.text) # change temp and top_p to see differences


## Ollama

options = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 200, # Google calls it max_output_tokens
}

response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": story_prompt,
        "stream": False,
        "options": options,
    }
)

data = response.json()
print("Gemma version:")
print(data["response"])


### Langchain with Ollama

model = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    num_predict=200,
    seed=42
    )

# OR (less common)

model = ChatOllama(
    model=OLLAMA_MODEL,
    options=options
    )


response = model.invoke("Give me a list of five creative startup ideas involving sensor data and building occupancy.")
print("Langchain Ollama version:")
print(response.content)