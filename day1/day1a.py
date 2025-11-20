import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    GOOGLE_API_KEY = None

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("✅ ADK components imported successfully.")

### 1.5: Configure Retry Options
# When working with LLMs, you may encounter transient errors like rate limits or temporary service unavailability. Retry options automatically handle these failures by retrying the request with exponential backoff.

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


### 2.2 Define your agent
# Now, let's build our agent. We'll configure an Agent by setting its key properties, which tell it what to do and how to operate.

root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for correct info or if unsure.",
    tools=[google_search],
)

print("✅ Root Agent defined.")

### 2.3 Run your agent
# Now it's time to bring your agent to life and send it a query. To do this, you need a [`Runner`](https://google.github.io/adk-docs/runtime/), which is the central component within ADK that acts as the orchestrator. It manages the conversation, sends our messages to the agent, and handles its responses.

# create an 'InMemoryRunner' to run the agent
runner = InMemoryRunner(root_agent)

print("✅ Runner created.")

# Now you can call the `.run_debug()` method to send our prompt and get an answer.

async def run_debug():
    response = await runner.run_debug(
        "What is asyncio? Please provide a brief explanation."
    )

import asyncio
asyncio.run(run_debug())