import os, requests
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file


### Google: Install the Google GenAI SDK:
# pip install -U -q "google-genai==1.7.0"
# this requires a different venv from the agentic AI course which uses a much more recent version

from google import genai
from google.genai import types


### Set up API key in Google AI Studio
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    GOOGLE_API_KEY = None

# Runs locally, no API key needed
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2") # or "llama3"

# LangChain Ollama integration
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# Google - list available models with embedding capability

client = genai.Client(api_key=GOOGLE_API_KEY)

for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)


# Ollama - no supported_actions metadata, so just check the model name

import ollama
models = ollama.list()

for model in models["models"]:
    if "embed" in model["model"]:
        print(model["model"])


# Some sample text for RAG
DOCUMENT1 = "Operating the Climate Control System  Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."
DOCUMENT2 = 'Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the "Navigation" icon to get directions to your destination or touch the "Music" icon to play your favorite songs.'
DOCUMENT3 = "Shifting Gears Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]


# Create an embedding database with ChromaDB

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry


# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

# create a custom function to generate embeddings using Gemini
class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document" # this is for implementing a retrieval system
        else:
            embedding_task = "retrieval_query" # this is for embedding user queries

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task, 
            ),
        )
        return [e.values for e in response.embeddings]
    
DB_NAME = "googlecardb"

embed_fn = GeminiEmbeddingFunction()

# Use document mode when adding documents.
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(
    name=DB_NAME, 
    embedding_function=embed_fn
    )

### Add documents to the DB ###
db.add(
    documents=documents, 
    ids=[str(i) for i in range(len(documents))]
    )

# confirm the data was added
print(db.count()) 


### Query the DB ###
# Switch to query mode when generating embeddings.
embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "How do you use the touchscreen to play music?"

result = db.query(
    query_texts=[query], 
    n_results=1 # only a single passage is retrieved -- in practice, you will want more than one and let the model determine which are relevant
    )
[all_passages] = result["documents"]

print(all_passages[0])


### Generate an answer using the retrieved documents ###

query_oneline = query.replace("\n", " ")

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and conversational tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
"""

# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"

print(prompt)

answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt)

print(answer.text)

