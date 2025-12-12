import os
from dotenv import load_dotenv
import asyncio

load_dotenv()  # take environment variables from .env file

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    GOOGLE_API_KEY = None

import uuid
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("✅ ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


## Model Context Protocol

# McpToolset is ued to integrate an ADK Agent with an MCP Server.
## This runs npx -y @modelcontextprotocol/server-everything 
## filters to only use the getTinyImage tool.
## Everything MCP Server returns a test image 16x16 px, Base64 encoded.
## More servers: https://modelcontextprotocol.io/examples 

# MCP integration with Everything Server
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)

print("✅ MCP Tool created")



# Create image agent with MCP integration
root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction="Use the MCP Tool to generate images for user queries",
    tools=[mcp_image_server],
)

from google.adk.runners import InMemoryRunner

from IPython.display import display, Image as IPImage
import base64

async def run_image_agent():
    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug(
        "Generate a tiny image.",
        verbose=True,
    )
    # for event in response:
    #     if event.content and event.content.parts:
    #         for part in event.content.parts:
    #             if hasattr(part, "function_response") and part.function_response:
    #                 for item in part.function_response.response.get("content", []):
    #                     if item.get("type") == "image":
    #                         display(IPImage(data=base64.b64decode(item["data"])))

if __name__ == "__main__":
    asyncio.run(run_image_agent())


