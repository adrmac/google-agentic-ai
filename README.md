This repo duplicates the contents of the 5-day Kaggle intensive on Google Agentic AI. 

To create a new agent template, `cd` to the parent directory and run:
`adk create sample-project --api-key $GOOGLE_API_KEY`

All of the tutorial examples use the genini-2.5-flash default, but other models can be filled later.

Files can be run using one of two CLI methods:
1. `adk run [containing folder]` -- e.g. from /day3, use `adk run day3a`
This runs the root_agent interactively in the terminal. 
2. python agent.py -- run as python to see the scripted prompts from the tutorial

Exceptions:
1. Day 4b-2 -- I was not able to make the 'User Simulation' test work using the Google Colab in the ADK documentation. In both the Colab and my LDE I get: `TypeError: 'NoneType' object is not iterable`. Looks like a bug in ADK?
2. Day 5b -- I was not able to deploy the agent to Vertex AI, despite consulting the troubleshooting page, checking logs, and double-checking prerequisites. The persistent error is a catch-all that doesn't surface any other errors: 
*"Deploy failed: Failed to create Agent Engine: {'code': 3, 'message': 'Reasoning Engine resource [projects/217151334676/locations/us-east4/reasoningEngines/9119472586076454912] failed to start and cannot serve traffic. Please refer to our documentation (https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/troubleshooting/deploy) for checking logs and other troubleshooting tips.'}"*

