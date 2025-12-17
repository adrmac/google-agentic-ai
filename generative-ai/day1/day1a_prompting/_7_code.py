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



### Code prompting
# Use Gemini to generate code, configs, and scripts.
# Use for learning or rapidly generating a first draft.

client = genai.Client(api_key=GOOGLE_API_KEY)

## Generating code
print("Generating code: ", "")

# The Gemini models love to talk, so it helps to specify they stick to the code if that is all that you want.
code_prompt = """
Write a Python function to calculate the factorial of a number. No explanation, provide only the code.
"""

response = client.models.generate_content(
    model='gemini-2.0-flash',
    config=types.GenerateContentConfig(
        temperature=1,
        top_p=1,
        max_output_tokens=1024,
    ),
    contents=code_prompt)

print(response.text)
# prints: 
# def factorial(n): ...


## Code execution
print("Code execution: ", "")


from pprint import pprint

config = types.GenerateContentConfig(
    tools=[types.Tool(code_execution=types.ToolCodeExecution())]
)

code_exec_prompt = """
Generate the first 14 odd prime numbers, then calculate their sum.
"""

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    config=config,
    contents=code_exec_prompt
    )

for part in response.candidates[0].content.parts:
  pprint(part.to_json_dict())
  print("-----")

  # The code execution tool gives a response with multiple parts, including opening text, executable_code, code_execution_result, and closing text.

## Explaining code
print("Explaining code:")
print("")

file_contents = "curl https://raw.githubusercontent.com/magicmonty/bash-git-prompt/refs/heads/master/gitprompt.sh"

explain_prompt = f"""
Please explain what this file does at a very high level. What is it, and why would I use it?

```
{file_contents}
```
"""

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    contents=explain_prompt
    )

print(response.text)