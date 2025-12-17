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


### Prompting

## Zero-shot
# Prompts that describe the request for the model directly.
print("Zero-shot:")
print("-" * 10)
print("Gemini response:")

client = genai.Client(api_key=GOOGLE_API_KEY)

model_config = types.GenerateContentConfig(
    temperature=0.1,
    top_p=1,
    max_output_tokens=5,
)

prompt = """Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE.
Review: "Her" is a disturbing study revealing the direction
humanity is headed if AI is allowed to keep evolving,
unchecked. I wish there were more movies like this masterpiece.
Sentiment: """

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    config=model_config,
    contents=prompt,
)

print(response.text) # It just prints POSITIVE
# but depending on settings it might print more...

print("")
print("Ollama / Gemma response:")

options = {
    "temperature": 0.1,
    "top_p": 1,
    "num_predict": 5, # Google calls it max_output_tokens
}

response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
)

data = response.json()
print(data["response"]) 
# In Ollama / Gemma it isn't constrained to a one-word classification - the instruction wasn't clear enough, it prints:
# "The sentiment in the review" (5 tokens limit)


### Ollama via LangChain
model = ChatOllama(
    model=OLLAMA_MODEL,
    options=options,
    )

messages = [
    HumanMessage(content=prompt)
    ]

result = model.invoke(messages)
print("LangChain Ollama response:")
print(result.content)


## Enum mode
# constrain the output to a fixed set of values
print("")
print("Enum mode:")
print("-" * 10)

## Gemini SDK
print("Gemini response:")

import enum

class Sentiment(enum.Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

enum_config = types.GenerateContentConfig(
    response_mime_type="text/x.enum",
    response_schema=Sentiment
)

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    config=enum_config,
    contents=prompt
)

print(response.text) # prints 'positive'

# You can also have the SDK convert the response to a Python object
enum_response = response.parsed
print(enum_response) # prints Sentiment.POSITIVE
print(type(enum_response)) # prints <enum 'Sentiment'>


## Ollama

print("")
print("Ollama / Gemma response:")

prompt = prompt + "Return ONLY one of: positive, neutral, negative." # be explicit here to make up for lack of enum mode

options = {
    "temperature": 0.0,
    "top_p": 1,
    "num_predict": 3,
    "stop": ["\n", " "] # optional - stops generating at a space or new line, forcing a 1-token limit - equivalent to text/x.enum 
}

response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
)

data = response.json()["response"]
print(data)

# this works, but you can also keep retrying with stricter phrasing
# Pattern: "prompt + stop + validate + retry" is the open-model equivalent of enum/schema constraints

allowed = [
    "positive",
    "neutral",
    "negative"
]

if data.strip().lower() not in allowed:
    response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt + "\nOutput must be exactly one word from the list.",
        "stream": False,
        "options": options,
        }
    )
    data = response.json()["response"]
    print(data)


### Ollama via LangChain
# Ollama doesn't have enum mode, but LangChain can provide an output parser to validate
from langchain_core.output_parsers import EnumOutputParser

parser = EnumOutputParser(enum=Sentiment) # using the Sentiment enum defined for Gemini above

model = ChatOllama(
    model=OLLAMA_MODEL,
    options=options,
    )

response = model.invoke(prompt)
print("LangChain Ollama response:")
print(response.content)

sentiment = parser.parse(response.content)
print("Parsed sentiment:", sentiment)




## One-shot and few-shot
# Provide multiple examples of the expected response
print("One-shot and few-shot:")
print("-" * 10)

few_shot_prompt = """Parse a customer's pizza order into valid JSON:

EXAMPLE:
I want a small pizza with cheese, tomato sauce, and pepperoni.
JSON Response:
```
{
"size": "small",
"type": "normal",
"ingredients": ["cheese", "tomato sauce", "pepperoni"]
}
```

EXAMPLE:
Can I get a large pizza with tomato sauce, basil and mozzarella
JSON Response:
```
{
"size": "large",
"type": "normal",
"ingredients": ["tomato sauce", "basil", "mozzarella"]
}
```

ORDER:
"""

customer_order = "Give me a large with cheese & pineapple"

# Google SDK
print("")
print("Gemini response:")


model_config = types.GenerateContentConfig(
    temperature=0.1,
    top_p=1,
    max_output_tokens=250,
)

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    config=model_config,
    contents=[few_shot_prompt, customer_order]
)

print(response.text)
# prints
# ```json
# {
# "size": "large",
# "type": "normal",
# "ingredients": ["cheese", "pineapple"]
# }
# ```

# Ollama
print("")
print("Ollama / Gemma response:")


options = {
    "temperature": 0.1,
    "top_p": 1,
    "num_predict": 250,
}

response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": few_shot_prompt + customer_order,
        "stream": False,
        "options": options,
    }
)

data = response.json()["response"]
print(data)


## JSON mode
# Force the model to constrain decoding so token selection is guided by the schema
print("")
print("JSON mode:")
print("-" * 10)

# Google SDK
print("")
print("Gemini response:")

import typing_extensions as typing

class PizzaOrder(typing.TypedDict):
    size: str
    ingredients: list[str]
    type: str

model_config = types.GenerateContentConfig(
    temperature=0.1,
    response_mime_type="application/json",
    response_schema=PizzaOrder,
)

prompt="Can I have a large dessert pizza with apple and chocolate"

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    config=model_config,
    contents=prompt
)

print(response.text)
# prints
# {
#   "size": "large",
#   "ingredients": ["apple", "chocolate"],
#   "type": "dessert"
# }

# Ollama
print("")
print("Ollama response:")

# Gemini has true schema, Ollama uses "JSON-only" + parsing

ollama_prompt = f"""Return ONLY valid JSON (no code fences, no prose).
Schema: {{
  "size": string, 
  "type": string, 
  "ingredients": string[]
  }}
Order: {prompt}
"""

options = {
    "temperature": 0.1,
}

response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": ollama_prompt,
        "stream": False,
        "options": options,
    }
)

data = response.json()["response"]
print(data)

# this works but if you want robust extraction in case the model adds stray text you can do this:
import re, json
match = re.search(r"\{.*\}", data, re.DOTALL)
json_text = match.group(0) if match else data

obj = json.loads(json_text)  # will raise if invalid; catch + retry if needed
print("JSON schema confirmed: ", obj)


## Chain of Thought (CoT)
# Direct zero/multi shot prompting by itself can be prone to hallucination.
# Add CoT prompting to instruct the model to output intermediate reasoning steps for typically better results.
# Costs more tokens, not 100% hallucination free
# Don't ship the exposed reasoning by default, prefer "reason internally and return only the final answer"

print("")
print("Chain of Thought (CoT):")
print("-" * 10)

prompt = """When I was 4 years old, my partner was 3 times my age. Now, I am 20 years old. How old is my partner? Return the answer directly."""

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    contents=prompt
    )

print(response.text)
# prints 52 which is wrong

# Indicate that the model should think step by step.
cot_prompt = prompt + " Let's think step by step."

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    contents=cot_prompt
    )

print(response.text)
# prints:
# Here's how to solve the problem step-by-step:
# Find the age difference: When you were 4, your partner was 3 times your age, meaning they were 4 * 3 = 12 years old.
# Calculate the age difference: The age difference between you and your partner is 12 - 4 = 8 years.
# Determine the partner's current age: Since the age difference remains constant, your partner is currently 20 + 8 = 28 years old.
# Therefore, your partner is now 28 years old.

### UPDATE -- we no longer like to print reasoning steps (see deprecated mode in 6_thinking.py)
# However, prompting the model to simply not display the steps doesn't work. 
print("")
print("Attempting to hide reasoning steps by asking it to hide them doesn't work!")

better_prompt = cot_prompt + " Don't display your reasoning. Return the answer only as an integer."

response = client.models.generate_content(
    model=GOOGLE_MODEL,
    contents=better_prompt
    )

print("Gemini response to better_prompt", response.text)


options = {
    "temperature": 0.0,
    "num_predict": 10,
}

response = requests.post(
    f"{OLLAMA_HOST}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": better_prompt,
        "stream": False,
        # "options": options,
    }
)

data = response.json()["response"]
print("Ollama / Gemma response to better_prompt", data)
print("-" * 10)
print("consider other ways of constructing the prompt to walk through steps without displaying them...")


## ReAct: Reason and act
# Here we will perform the searching steps manually
# There are frameworks that wrap the prompt into easier-to-use APIs, e.g. LangChain

print("")
print("Reason and Act (ReAct):")
print("-" * 10)

model_instructions = """
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, Observation is understanding relevant information from an Action's output, and Action can be one of three types:
 (1) <search>entity</search>, which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it
     will return some similar entities to search and you can try to search the information from those topics.
 (2) <lookup>keyword</lookup>, which returns the next sentence containing keyword in the current context. This only does exact matches, so keep your searches short.
 (3) <finish>answer</finish>, which returns the answer and finishes the task.
"""

example1 = """Question
Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?

Thought 1
The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.

Action 1
<search>Milhouse</search>

Observation 1
Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Thought 2
The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".

Action 2
<lookup>named after</lookup>

Observation 2
Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.

Thought 3
Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.

Action 3
<finish>Richard Nixon</finish>
"""

example2 = """Question
What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Thought 1
I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.

Action 1
<search>Colorado orogeny</search>

Observation 1
The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.

Thought 2
It does not mention the eastern sector. So I need to look up eastern sector.

Action 2
<lookup>eastern sector</lookup>

Observation 2
The eastern sector extends into the High Plains and is called the Central Plains orogeny.

Thought 3
The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.

Action 3
<search>High Plains</search>

Observation 3
High Plains refers to one of two distinct land regions

Thought 4
I need to instead search High Plains (United States).

Action 4
<search>High Plains (United States)</search>

Observation 4
The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130m).

Thought 5
High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.

Action 5
<finish>1,800 to 7,000 ft</finish>
"""

# Come up with more examples yourself, or take a look through https://github.com/ysymyth/ReAct/

question = """Question
Who was the youngest author listed on the transformers NLP paper?
"""

# We will capture a single step at a time and manually do the actions and provide observations to avoid hallucinations
# stop_sequences ends the generation process after Thought and Action are completed

# Google
print("")
print("Gemini response:")

react_config = types.GenerateContentConfig(
    stop_sequences=["\nObservation"],
    system_instruction=model_instructions + example1 + example2,
)

react_chat = client.chats.create(
    model=GOOGLE_MODEL,
    config=react_config
)

response = react_chat.send_message(question)
print(response.text)

# prints:
# Thought 1
# I need to find the transformers NLP paper, then find the youngest author listed on it.
# Action 1
# <search>transformers NLP paper</search>


# Ollama
print("")
print("Ollama / Gemma response:")

messages = [
    {"role": "system", "content": model_instructions},
    {"role": "user", "content": f"Question\n{question}\n"}
]

options = {
    # "temperature": 0.2,
    # "num_predict": 300,
    "stop": ["\nObservation"]
}

response = requests.post(
    f"{OLLAMA_HOST}/api/chat",
    json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }
)
data = response.json()["message"]["content"]
print(data)
messages.append({"role": "assistant", "content": data})

# Now perform the search yourself, and supply back to the model as the Observation.

observation = """Observation 1
[1706.03762] Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
"""

# Google
print("")
print("Gemini response to new observation:")
response = react_chat.send_message(observation)
print(response.text)

# prints:
# Thought 2
# The paper is "Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. I need to find the youngest author among these. This is hard to determine without knowing their birthdates. I will search for each author individually and see if I can find their age or birthdate.

# Action 2
# <search>Ashish Vaswani</search>


# Ollama
print("")
print("Ollama / Gemma response to new observation:")
messages.append({"role": "user", "content": observation})

response = requests.post(
    f"{OLLAMA_HOST}/api/chat",
    json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }
)
data = response.json()["message"]["content"]
print(data)
messages.append({"role": "assistant", "content": data})


# repeat until the <finish> action is reached
# See complete automated example here: https://github.com/google-gemini/cookbook/blob/main/examples/Search_Wikipedia_using_ReAct.ipynb


