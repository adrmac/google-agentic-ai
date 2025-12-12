import os
from dotenv import load_dotenv
import asyncio

load_dotenv()  # take environment variables from .env file

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    GOOGLE_API_KEY = None

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.genai import types

print("✅ ADK components imported successfully.")

# Helper function to run a session with queries
async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")


print("✅ Helper functions defined.")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Memory Management

# Session = Short-term memory (single conversation)
# Memory = Long-term knowledge (across multiple conversations)

# Three-step memory integration process:
## 1. Initialize → Create a MemoryService and provide it to your agent via the Runner
## 2. Ingest → Transfer session data to memory using add_session_to_memory()
## 3. Retrieve → Search stored memories using search_memory()

## 1. Initialize Memory Service
### InMemoryMemoryService is built in for prototyping and testing.
### VertexAiMemoryBankService is a managed cloud service for production.
### Custom -- build your own using databases, though managed services are recommended.
### The same workflow applies to all memory services.

memory_service = InMemoryMemoryService()

# Define constants used throughout the notebook
APP_NAME = "MemoryDemoApp"
USER_ID = "demo_user"

# Create agent
user_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="MemoryDemoAgent",
    instruction="Answer user questions in simple words.",
)

print("✅ Agent created")


# Create Session Service
session_service = InMemorySessionService()  # Handles conversations

# Create runner with BOTH services
runner = Runner(
    agent=user_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
)

print("✅ Agent and Runner created with memory support!")


# The memory is available but needs to be explicitly added/retrieved by the agent.


## 2. Ingest Session Data into Memory
### 

async def add_session_to_memory_example():
    # User tells agent about their favorite color
    await run_session(
        runner,
        "My favorite color is blue-green. Can you write a Haiku about it?",
        "conversation-01",  # Session ID
    )

    session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id="conversation-01"
    )

    # Let's see what's in the session
    print("📝 Session contains:")
    for event in session.events:
        text = (
            event.content.parts[0].text[:60]
            if event.content and event.content.parts
            else "(empty)"
        )
        print(f"  {event.content.role}: {text}...")

    # This is the key method!
    await memory_service.add_session_to_memory(session)

    print("✅ Session added to memory!")

# asyncio.run(add_session_to_memory_example())

# The session data is transferred to memory, but the agent needs tools to search it.

## 3. Enable Memory Retrieval

# Two built in retrieval tools:
#   1. load_memory() - reactive, agent decides when to use memory, more efficient
#   2. preload_memory() - proactive, memory is always provided, more guaranteed

# Create agent
user_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="MemoryDemoAgent",
    instruction="Answer user questions in simple words. Use load_memory tool if you need to recall past conversations.",
    tools=[
        load_memory
    ],  # Agent now has access to Memory and can search it whenever it decides to!
)

print("✅ Agent with load_memory tool created.")

# Create a new runner with the updated agent
runner = Runner(
    agent=user_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

async def use_memory_agent():
    await run_session(runner, "What is my favorite color?", "color-test")

# this remembers the favorite color from the haiku query in add_session_to_memory_example()


# Complete Manual Workflow Test

async def manual_workflow_test():
    await add_session_to_memory_example()
    await run_session(runner, "When is my birthday?", "birthday-session-00")
    await run_session(runner, "My birthday is on March 15th.", "birthday-session-01")
    birthday_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="birthday-session-01"
    )
    await memory_service.add_session_to_memory(birthday_session)
    print("✅ Birthday session saved to memory!")

    # Test retrieval in a NEW session
    await run_session(
        runner, "What is my favorite color?", "birthday-session-02"  # Different session ID
    )

#    await run_session(runner, "Tell me a joke.", "birthday-session-03")
#    try changing between load_memory and preload_memory -- with preload the agent searches memory even if it doesn't have to

# asyncio.run(manual_workflow_test())

# Manual Memory Search
# useful for debugging, building analytics dashboards, or custom memory management UIs

async def manual_memory_search():
    await manual_workflow_test()
    # Search for color preferences
    search_response = await memory_service.search_memory(
        app_name=APP_NAME, user_id=USER_ID, query="What is the user's favorite color?"
    )

    print("🔍 Search Results:")
    print(f"  Found {len(search_response.memories)} relevant memories")
    print()

    for memory in search_response.memories:
        if memory.content and memory.content.parts:
            text = memory.content.parts[0].text[:80]
            print(f"  [{memory.author}]: {text}...")


# asyncio.run(manual_memory_search())


# Automating Memory Storage
## Production systems need add_session_to_memory() to happen automatically

# Callbacks = hook into key execution points, like event listeners
## before_agent_callback / after_agent_callback
## before_tool_callback / after_tool_callback
## before_model_callback / after_model_callback
## on_model_error_callback


# When defining a callback, the ADK automatically passes a special parameter called callback_context
# This provides access to the Memory Service and other runtime components
# This context does not need to be created manually
async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

print("✅ Callback created.")

# Agent with automatic memory saving
auto_memory_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="AutoMemoryAgent",
    instruction="Answer user questions.",
    tools=[preload_memory],
    after_agent_callback=auto_save_to_memory,  # Saves after each turn!
)

print("✅ Agent created with automatic memory saving!")

# Create a runner for the auto-save agent
# This connects our automated agent to the session and memory services
auto_runner = Runner(
    agent=auto_memory_agent,  # Use the agent with callback + preload_memory
    app_name=APP_NAME,
    session_service=session_service,  # Same services from Section 3
    memory_service=memory_service,
)

print("✅ Runner created.")

async def test_auto_memory_agent():
    # Test 1: Tell the agent about a gift (first conversation)
    # The callback will automatically save this to memory when the turn completes
    await run_session(
        auto_runner,
        "I gifted a new toy to my nephew on his 1st birthday!",
        "auto-save-test",
    )

    # Test 2: Ask about the gift in a NEW session (second conversation)
    # The agent should retrieve the memory using preload_memory and answer correctly
    await run_session(
        auto_runner,
        "What did I gift my nephew?",
        "auto-save-test-2",  # Different session ID - proves memory works across sessions!
    )

# asyncio.run(test_auto_memory_agent())


# Memory Consolidation
## Storying every message, response, and tool call adds up
## 50 messages = 10,000 tokens
## We need to extract only important facts while discarding noise

# If you use VertexAiMemoryBankService, the LLM automatically summarizes and condenses memories into salient facts.
# InMemoryMemoryService stores raw events.