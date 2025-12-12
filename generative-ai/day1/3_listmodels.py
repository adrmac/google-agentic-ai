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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b") # or "llama3"


### Choose a model


### Google

client = genai.Client(api_key=GOOGLE_API_KEY)

for model in client.models.list():
    print(model.name)

from pprint import pprint

for model in client.models.list():
    if model.name == 'models/gemini-2.0-flash':
        pprint(model.to_json_dict())
        break


### Ollama

# list all the models
response = requests.get(
    f"{OLLAMA_HOST}/api/tags",
    timeout=120, # give up after 120s
)
response.raise_for_status()

data = response.json()
models = data["models"]
print("")
print("Ollama list:")

for model in models:
    print(model["name"])

# show summary status for one model
response = requests.post(
    f"{OLLAMA_HOST}/api/show",
    json={
        "name": "gemma2:2b"
    },
    timeout=120, # give up after 120s
)
response.raise_for_status()
info = response.json()
# pprint(info, depth=1)

## the default stats Ollama shows for a model are much different from what Google shows
# Google stats are "product / API metadata" for a hosted API product
# Ollama stats are "local runtime + weights metadata" for a local model + runtime

# You can drill down on the /api/show response to get more detail

def ollama_model_summary(model_name: str) -> dict:
    """
    Produce a Google-style summary dict for an Ollama model.
    """
    resp = requests.post(
        f"{OLLAMA_HOST}/api/show",
        json={"name": model_name},
        timeout=60,
    )
    resp.raise_for_status()
    info = resp.json()

    model = info.get("model", {})
    details = info.get("details", {})
    params = info.get("parameters", {})
    license_info = info.get("license", "")

    # Build a Google-like summary
    summary = {
        # Identification
        "name": f"ollama/{model_name}",
        "display_name": model_name,
        "description": details.get("description", "Local Ollama model"),

        # Capacity / limits (best-effort analogs)
        "input_token_limit": details.get("context_length"),
        "output_token_limit": None,  # not enforced globally in Ollama

        # Architecture info
        "architecture": model.get("architecture"),
        "parameters": model.get("parameters"),
        "quantization": model.get("quantization"),
        "embedding_length": model.get("embedding_length"),

        # Capabilities (derived, not intrinsic)
        "supported_actions": [
            "generate",
            "chat",
            "embeddings" if model.get("embedding_length") else None,
        ],

        # Defaults (not model-bound; request-time in Ollama)
        "temperature": None,
        "top_p": None,
        "top_k": None,

        # Metadata
        # "license": license_info.strip() if isinstance(license_info, str) else license_info,
        # "provider": "ollama (local)",
    }

    # Clean up None values
    summary["supported_actions"] = [
        a for a in summary["supported_actions"] if a is not None
    ]

    return summary



for model in models:
    name = model["name"]
    if name == OLLAMA_MODEL:
        pprint(ollama_model_summary(name))

