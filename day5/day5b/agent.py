import os
from dotenv import load_dotenv
import asyncio
import pydantic

print(pydantic.__version__)

load_dotenv()  # take environment variables from .env file

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    GOOGLE_API_KEY = None

import os
import random
import time
import vertexai
from vertexai import agent_engines

print("✅ Imports completed successfully")

PROJECT_ID = "bold-syntax-479023-g4"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

print(f"✅ Project ID set to: {PROJECT_ID}")

# 1.5: Enable Google Cloud APIs
# For this tutorial, you'll need to enable the following APIs in the Google Cloud Console.

# Vertex AI API
# Cloud Storage API
# Cloud Logging API
# Cloud Monitoring API
# Cloud Trace API
# Telemetry API
# You can use this link to open the Google Cloud Console and follow the steps there to enable these APIs.

from google.adk.agents import Agent
import vertexai
import os

vertexai.init(
    project="bold-syntax-479023-g4",
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

def get_weather(city: str) -> dict:
    """
    Returns weather information for a given city.

    This is a TOOL that the agent can call when users ask about weather.
    In production, this would call a real weather API (e.g., OpenWeatherMap).
    For this demo, we use mock data.

    Args:
        city: Name of the city (e.g., "Tokyo", "New York")

    Returns:
        dict: Dictionary with status and weather report or error message
    """
    # Mock weather database with structured responses
    weather_data = {
        "san francisco": {"status": "success", "report": "The weather in San Francisco is sunny with a temperature of 72°F (22°C)."},
        "new york": {"status": "success", "report": "The weather in New York is cloudy with a temperature of 65°F (18°C)."},
        "london": {"status": "success", "report": "The weather in London is rainy with a temperature of 58°F (14°C)."},
        "tokyo": {"status": "success", "report": "The weather in Tokyo is clear with a temperature of 70°F (21°C)."},
        "paris": {"status": "success", "report": "The weather in Paris is partly cloudy with a temperature of 68°F (20°C)."}
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        return weather_data[city_lower]
    else:
        available_cities = ", ".join([c.title() for c in weather_data.keys()])
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available. Try: {available_cities}"
        }

root_agent = Agent(
    name="weather_assistant",
    model="gemini-2.5-flash-lite",  # Fast, cost-effective Gemini model
    description="A helpful weather assistant that provides weather information for cities.",
    instruction="""
    You are a friendly weather assistant. When users ask about the weather:

    1. Identify the city name from their question
    2. Use the get_weather tool to fetch current weather information
    3. Respond in a friendly, conversational tone
    4. If the city isn't available, suggest one of the available cities

    Be helpful and concise in your responses.
    """,
    tools=[get_weather]
)

# Next we will deploy to: 
## Vertex AI Agent Engine -- monthly free tier. This agent will stay free if cleaned up promptly but can incur costs if left running.

# Other options:
## Cloud Run -- serverless, good for demos and small/medium workloads
## Google Kubernetes Engine (GKE) -- for large scale production deployments

regions_list = ["europe-west1", "europe-west4", "us-east4", "us-west1"]
deployed_region = random.choice(regions_list)

print(f"✅ Selected deployment region: {deployed_region}")

import subprocess
subprocess.run([
    "adk", "deploy", "agent_engine",
    "--project", PROJECT_ID,
    "--region", deployed_region,
    "day5b", "--agent_engine_config_file", "day5/day5b/.agent_engine_config.json"
])

# Deploy from CLI:
# adk deploy agent_engine --project=$PROJECT_ID --region=$deployed_region day5b --agent_engine_config_file=day5b/.agent_engine_config.json

# This doesn't work! 
# Deploy failed: Failed to create Agent Engine: {'code': 3, 'message': 'Reasoning Engine resource [projects/217151334676/locations/europe-west4/reasoningEngines/75681584363077632] failed to start and cannot serve traffic. Please refer to our documentation (https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/troubleshooting/deploy) for checking logs and other troubleshooting tips.'}
# Extensive troubleshooting with Gemini looking at the logs and troubleshooting advice gives no clues about this error.
# It does work locally using 'adk run'


# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=deployed_region)

# Get the most recently deployed agent
agents_list = list(agent_engines.list())
if agents_list:
    remote_agent = agents_list[0]  # Get the first (most recent) agent
    client = agent_engines
    print(f"✅ Connected to deployed agent: {remote_agent.resource_name}")
else:
    print("❌ No agents found. Please deploy first.")

async def run_query_test():
    async for item in remote_agent.async_stream_query(
    message="What is the weather in Tokyo?",
    user_id="user_42",
    ):
        print(item)

