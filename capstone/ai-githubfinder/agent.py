import os
import sys
import subprocess
import time
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor


from dotenv import load_dotenv
import asyncio

load_dotenv()  # take environment variables from .env file

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    GOOGLE_API_KEY = None


# ==========================================
# 1. ENVIRONMENT SETUP & DEPENDENCY INSTALL
# ==========================================
def install_dependencies():
    print("--- 📦 Installing Dependencies ---")
    packages = [
        "google-generativeai",
        "langchain",
        "langchain-community",
        "beautifulsoup4"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("--- ✅ Dependencies Installed ---")

try:
    import google.generativeai as genai
except ImportError:
    install_dependencies()
    import google.generativeai as genai

# from kaggle_secrets import UserSecretsClient

# # ==========================================
# # 2. CONFIGURATION
# # ==========================================
# # Try to retrieve secrets, otherwise rely on env vars or placeholder
# try:
#     user_secrets = UserSecretsClient()
#     GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
# except Exception:
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY_HERE")

# GitHub configuration
GITHUB_TOPIC = "docker"
GITHUB_SORT = "updated" # Get recently updated tools
OLLAMA_MODEL = "gemma:2b" # Google's lightweight model fits well in Kaggle memory

# ==========================================
# 3. OLLAMA INFRASTRUCTURE (Local LLM)
# ==========================================
class OllamaManager:
    def __init__(self):
        self.process = None

    def install_ollama(self):
        print("--- 🦙 Installing Ollama ---")
        # Kaggle runs as root in the docker container, so standard install works
        install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
        os.system(install_cmd)
        print("--- ✅ Ollama Installed ---")

    def start_server(self):
        print("--- 🚀 Starting Ollama Server ---")
        # Run in background
        self.process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Give it a moment to spin up
        time.sleep(5) 
        
    def pull_model(self, model_name):
        print(f"--- 📥 Pulling Model: {model_name} ---")
        # This might take a minute or two
        result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"--- ✅ Model {model_name} Ready ---")
        else:
            print(f"--- ❌ Failed to pull model: {result.stderr} ---")

    def generate(self, prompt):
        url = "http://localhost:11434/api/generate"
        data = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=data)
            return response.json().get('response', 'No response')
        except Exception as e:
            return f"Ollama Error: {str(e)}"

# ==========================================
# 4. GITHUB SCRAPER AGENT
# ==========================================
class GitHubAgent:
    def fetch_trending_tools(self, limit=3):
        print("--- 🕵️ Scrapping GitHub for New Tools ---")
        # Using GitHub API
        url = f"https://api.github.com/search/repositories?q=topic:{GITHUB_TOPIC}&sort={GITHUB_SORT}&order=desc"
        headers = {'Accept': 'application/vnd.github.v3+json'}
        
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"GitHub API Error: {resp.status_code}")
                return []
            
            items = resp.json().get('items', [])
            tools = []
            
            for item in items[:limit]:
                tools.append({
                    "name": item['name'],
                    "url": item['html_url'],
                    "description": item['description'],
                    "stars": item['stargazers_count'],
                    "language": item['language']
                })
            return tools
        except Exception as e:
            print(f"Scraping Error: {e}")
            return []

# ==========================================
# 5. GEMINI ANALYST AGENT (Cloud LLM)
# ==========================================
class GeminiAnalyst:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def summarize_and_validate(self, tool_data):
        """
        Gemini's job is to read the raw metadata and summarize what the tool actually does
        and ensure it's software that can be dockerized.
        """
        prompt = f"""
        Analyze this GitHub repository data:
        Name: {tool_data['name']}
        Description: {tool_data['description']}
        Language: {tool_data['language']}
        URL: {tool_data['url']}

        1. Summarize what this tool does in 2 sentences.
        2. Confirm if this looks like a deployable application (Yes/No).
        
        Output format:
        SUMMARY: [Your summary]
        DEPLOYABLE: [Yes/No]
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {e}"

# ==========================================
# 6. ORCHESTRATION (MAIN SCRIPT)
# ==========================================
def main():
    print("==========================================")
    print("🤖 AI AUTO-DEPLOYMENT RESEARCHER STARTED")
    print("==========================================")

    # 1. Initialize Infrastructure
    ollama_ops = OllamaManager()
    ollama_ops.install_ollama()
    ollama_ops.start_server()
    ollama_ops.pull_model(OLLAMA_MODEL)

    # 2. Initialize Agents
    scraper = GitHubAgent()
    if "YOUR_GEMINI_KEY" in GOOGLE_API_KEY:
        print("❌ Error: Please set GOOGLE_API_KEY in Kaggle Secrets.")
        return
    
    gemini_analyst = GeminiAnalyst(GOOGLE_API_KEY)

    # 3. Fetch Data
    tools = scraper.fetch_trending_tools(limit=3) # Limit to 3 for demo speed

    # 4. Processing Loop
    for tool in tools:
        print(f"\n\n🔹 PROCESSING: {tool['name']} ({tool['url']})")
        
        # --- Step A: Gemini summarizes the tool ---
        print("   ... Asking Gemini to analyze relevance ...")
        analysis = gemini_analyst.summarize_and_validate(tool)
        print(f"   📋 ANALYSIS:\n{analysis}")

        # Extract summary for Ollama
        summary_text = analysis.split("SUMMARY:")[-1].split("DEPLOYABLE:")[0].strip()

        # --- Step B: Ollama generates Docker instructions ---
        print(f"   ... Asking Local Ollama ({OLLAMA_MODEL}) for Install Guide ...")
        
        devops_prompt = f"""
        You are a DevOps Engineer. 
        Tool Name: {tool['name']}
        Tool Summary: {summary_text}
        
        Task: Create a valid 'docker run' command and a brief explanation on how to install 
        and run this system on a Linux Docker OS. 
        Keep it concise. If you don't know the specific image, assume a standard naming convention or suggest building from source.
        """
        
        install_guide = ollama_ops.generate(devops_prompt)
        
        print(f"   🛠️  LOCAL OLLAMA INSTRUCTIONS:")
        print("-" * 40)
        print(install_guide)
        print("-" * 40)

    print("\n✅ Workflow Complete.")

if __name__ == "__main__":
    main()