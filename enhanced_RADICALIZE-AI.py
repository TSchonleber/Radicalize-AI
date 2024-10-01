# Imports
import os
import threading
import subprocess
import logging
import time
import re
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# External Libraries
import openai
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
from colorama import init, Fore, Back, Style
import requests
import tiktoken  # For accurate token counting

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY').strip()
if not OPENAI_API_KEY:
    logging.error("OpenAI API key not found in environment variable 'OPENAI_API_KEY'. Please set it and rerun the script.")
    exit(1)

# Initialize OpenAI client
OPENAI_API_BASE = 'https://api.openai.com/v1'
client = OpenAI(api_key=OPENAI_API_KEY)

# Validate API key format
def validate_api_key(api_key):
    if not api_key:
        return False
    return bool(re.match(r'^sk-[a-zA-Z0-9]{32,}$', api_key))

if not validate_api_key(OPENAI_API_KEY):
    print(Fore.RED + f"Invalid API key format. Please check your API key.")
    exit(1)

# Test OpenAI API connection
def test_openai_connection():
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        if response:
            print(Fore.GREEN + "Successfully connected to OpenAI API.")
            return True
        else:
            print(Fore.RED + "Failed to connect to OpenAI API.")
            return False
    except Exception as e:
        print(Fore.RED + f"Error connecting to OpenAI API: {e}")
        return False

if not test_openai_connection():
    print(Fore.RED + "OpenAI API connection failed. Please check your setup and try again.")
    exit(1)

# Initialize MongoDB client
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    logging.warning("Environment variable 'MONGODB_URI' is not set. MongoDB functionality will be disabled.")
    collection = None
else:
    try:
        client_mongodb = MongoClient(MONGODB_URI)
        client_mongodb.list_database_names()
        print(Fore.GREEN + "MongoDB connection successful.")
        MONGODB_DB = os.getenv('MONGODB_DB') or 'default_db'
        MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION') or 'default_collection'
        db = client_mongodb[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]
    except Exception as e:
        print(Fore.RED + f"Failed to connect to MongoDB: {e}")
        collection = None

# Initialize Llama model (using Ollama directly)
def initialize_llama_model():
    def llama_generate(prompt):
        try:
            result = subprocess.run(
                ['ollama', 'run', 'llama3.1:8b', prompt],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(Fore.RED + f"Error from Llama model: {e}")
            return f"Error from Llama model: {str(e)}"
    return llama_generate

llama_model = initialize_llama_model()

# Constants for reasoning
MAX_TOTAL_TOKENS = 2048
RETRY_LIMIT = 3
RETRY_BACKOFF_FACTOR = 2
MAX_REFINEMENT_ATTEMPTS = 3

# Define colors for each agent
AGENT_COLORS = {
    'Jane Austen (INFJ)': Fore.MAGENTA,
    'George Orwell (INTJ)': Fore.CYAN,
    'Virginia Woolf (INFP)': Fore.YELLOW,
    'Ernest Hemingway (ESTP)': Fore.GREEN,
    'Agatha Christie (ISTJ)': Fore.RED,
    'Oscar Wilde (ENFP)': Fore.BLUE,
    'Sylvia Plath (INFJ)': Fore.LIGHTMAGENTA_EX,
    'Stephen King (INFP)': Fore.LIGHTRED_EX,
    'Ada Lovelace (INTP)': Fore.LIGHTCYAN_EX,
    'Alan Turing (ENTJ)': Fore.LIGHTGREEN_EX
}

# Agent class with advanced reasoning capabilities
class Agent:
    ACTION_DESCRIPTIONS = {
        'discuss': "formulating a response",
        'verify': "verifying data",
        'refine': "refining the response",
        'critique': "critiquing the other agent's response"
    }

    def __init__(self, name, color, model_name):
        self.name = name
        self.color = color
        self.model_name = model_name
        self.messages = []

    def _add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        # Token counting and truncation to maintain within limits
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            total_tokens = sum(len(encoding.encode(msg['content'])) for msg in self.messages)
            if total_tokens > MAX_TOTAL_TOKENS:
                while total_tokens > MAX_TOTAL_TOKENS and len(self.messages) > 1:
                    self.messages.pop(0)
                    total_tokens = sum(len(encoding.encode(msg['content'])) for msg in self.messages)
        except Exception as e:
            logging.error(f"Token encoding error: {e}")

    def _handle_chat_response(self, prompt):
        self._add_message("user", prompt)
        start_time = time.time()
        retries = 0
        backoff = 1

        while retries < RETRY_LIMIT:
            try:
                print(self.color + f"{self.name} is thinking..." + Style.RESET_ALL)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages
                )
                assistant_reply = response.choices[0].message.content.strip()
                self._add_message("assistant", assistant_reply)
                duration = time.time() - start_time
                return assistant_reply, duration
            except Exception as e:
                error_type = type(e).__name__
                logging.error(f"Error in agent '{self.name}': {error_type}: {e}")
                retries += 1
                if retries >= RETRY_LIMIT:
                    logging.error(f"Agent '{self.name}' reached maximum retry limit.")
                    break
                backoff_time = backoff * (RETRY_BACKOFF_FACTOR ** (retries - 1))
                logging.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
        return "An error occurred while generating a response.", time.time() - start_time

    def discuss(self, prompt):
        return self._handle_chat_response(prompt)

    def verify(self, data):
        verification_prompt = f"Verify the accuracy of the following information:\n\n{data}"
        return self._handle_chat_response(verification_prompt)

    def refine(self, data, more_time=False, iterations=2):
        refinement_prompt = f"Please refine the following response to improve its accuracy and completeness:\n\n{data}"
        if more_time:
            refinement_prompt += "\nTake additional time to improve the response thoroughly."
        total_duration = 0
        refined_response = data
        for i in range(iterations):
            refined_response, duration = self._handle_chat_response(refinement_prompt)
            total_duration += duration
            refinement_prompt = f"Please further refine the following response:\n\n{refined_response}"
        return refined_response, total_duration

    def critique(self, other_agent_response):
        critique_prompt = f"Critique the following response for accuracy and completeness:\n\n{other_agent_response}"
        return self._handle_chat_response(critique_prompt)

# Utility functions
def print_styled(text, color=Fore.WHITE, style=Style.NORMAL, end='\n'):
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)

def print_divider(char="═", length=100, color=Fore.YELLOW):
    print(color + char * length + Style.RESET_ALL)

def print_header(title, color=Fore.YELLOW):
    border = "═" * 58
    print(color + f"╔{border}╗")
    print(color + f"║{title.center(58)}║")
    print(color + f"╚{border}╝" + Style.RESET_ALL)

def blend_responses(agent_responses, user_prompt):
    combined_prompt = (
        f"Combine the following responses into a single, optimal answer to the question: '{user_prompt}'.\n\n"
        + "\n\n".join(f"Response {idx+1}:\n{response}" for idx, response in enumerate(agent_responses))
        + "\n\nProvide a concise and accurate combined response."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": combined_prompt}]
        )
        blended_reply = response.choices[0].message.content.strip()
        return blended_reply
    except Exception as e:
        logging.error(f"Error in blending responses: {e}")
        return "An error occurred while attempting to blend responses."

def process_agent_action(agent, action, *args, **kwargs):
    action_method = getattr(agent, action)
    action_description = agent.ACTION_DESCRIPTIONS.get(action, "performing an action")
    print_divider()
    print(agent.color + f"{agent.name} is {action_description}..." + Style.RESET_ALL)
    try:
        result, duration = action_method(*args, **kwargs)
        print(agent.color + f"{agent.name}'s action completed in {duration:.2f} seconds." + Style.RESET_ALL)
        return result, duration
    except Exception as e:
        logging.error(f"Error during {action} action for {agent.name}: {e}")
        return "An error occurred.", 0

def legion_mode(prompt):
    agents_info = {
        'Jane Austen (INFJ)': {'model': 'gpt-3.5-turbo', 'currency': 'Regency Era Pounds'},
        'George Orwell (INTJ)': {'model': 'gpt-3.5-turbo', 'currency': 'Dystopian Credits'},
        'Virginia Woolf (INFP)': {'model': 'gpt-3.5-turbo', 'currency': 'Stream of Consciousness Tokens'},
        'Ernest Hemingway (ESTP)': {'model': 'gpt-3.5-turbo', 'currency': 'Bullfighting Pesetas'},
        'Agatha Christie (ISTJ)': {'model': 'gpt-3.5-turbo', 'currency': 'Mystery Solving Guineas'},
        'Oscar Wilde (ENFP)': {'model': 'gpt-3.5-turbo', 'currency': 'Witty Epigram Coins'},
        'Sylvia Plath (INFJ)': {'model': 'gpt-3.5-turbo', 'currency': 'Poetic Introspection Points'},
        'Stephen King (INFP)': {'model': 'gpt-3.5-turbo', 'currency': 'Nightmare Fuel Dollars'},
        'Ada Lovelace (INTP)': {'model': 'gpt-3.5-turbo', 'currency': 'Analytical Engine Tokens'},
        'Alan Turing (ENTJ)': {'model': 'gpt-3.5-turbo', 'currency': 'Cryptographic Keys'}
    }

    agents = [Agent(name, AGENT_COLORS.get(name, Fore.WHITE), info['model']) for name, info in agents_info.items()]
    agent_responses = {}
    total_durations = {}

    print_styled("Legion Mode Activated with Advanced Reasoning", Fore.GREEN, Style.BRIGHT)
    print_divider('=')
    print_styled(f"Original Prompt: {prompt}", Fore.WHITE, Style.BRIGHT)
    print_divider()

    # Reasoning Step 1: Agents discuss the prompt
    print_header("Reasoning Step 1: Discussing the Prompt")
    for agent in agents:
        opinion, duration = process_agent_action(agent, 'discuss', prompt)
        agent_responses[agent.name] = opinion
        total_durations[agent.name] = duration

    # Reasoning Step 2: Agents verify their responses
    print_header("Reasoning Step 2: Verifying Responses")
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_agent_action, agent, 'verify', agent_responses[agent.name]): agent for agent in agents}
        for future in futures:
            agent = futures[future]
            verified_opinion, duration = future.result()
            agent_responses[agent.name] = verified_opinion
            total_durations[agent.name] += duration

    # Reasoning Step 3: Agents critique each other's responses
    print_header("Reasoning Step 3: Critiquing Responses")
    for agent in agents:
        other_agent = random.choice([a for a in agents if a != agent])
        critique, duration = process_agent_action(agent, 'critique', agent_responses[other_agent.name])
        agent_responses[agent.name] += f"\n\nCritique of {other_agent.name}:\n{critique}"
        total_durations[agent.name] += duration

    # Reasoning Step 4: Agents refine their responses
    print_header("Reasoning Step 4: Refining Responses")
    for agent in agents:
        refined_opinion, duration = process_agent_action(agent, 'refine', agent_responses[agent.name])
        agent_responses[agent.name] = refined_opinion
        total_durations[agent.name] += duration

    # Blending responses
    print_header("Blending Responses")
    responses_list = [agent_responses[agent.name] for agent in agents]
    blended_response = blend_responses(responses_list, prompt)
    print_styled("Final Blended Response:", Fore.GREEN, Style.BRIGHT)
    print_styled(blended_response, Fore.GREEN)
    print_divider('=')

    # Log the interaction to MongoDB if available
    if collection:
        log = {
            'prompt': prompt,
            'agent_responses': agent_responses,
            'blended_response': blended_response,
            'durations': total_durations
        }
        try:
            collection.insert_one(log)
            print_styled("Interaction logged to MongoDB.", Fore.GREEN, Style.DIM)
        except Exception as e:
            print_styled(f"Failed to log interaction to MongoDB: {e}", Fore.RED, Style.DIM)
    else:
        print_styled("Interaction not logged (MongoDB not available).", Fore.YELLOW, Style.DIM)

    return blended_response

# Main execution
if __name__ == "__main__":
    print_styled("Welcome to the Enhanced RADICALIZE-AI Assistant!", Fore.MAGENTA, Style.BRIGHT)
    print_styled("Type 'exit' to quit the program.", Fore.MAGENTA)
    print_styled("Type 'activate legion mode' to enable legion mode.", Fore.MAGENTA)
    print_styled("Type 'deactivate legion mode' to disable legion mode.", Fore.MAGENTA)
    print_divider('=')

    legion_active = False

    while True:
        prompt = input(Fore.BLUE + Style.BRIGHT + "\nYou: " + Style.RESET_ALL)
        if prompt.strip().lower() == 'exit':
            print_styled("Goodbye!", Fore.MAGENTA, Style.BRIGHT)
            break
        elif 'activate legion mode' in prompt.lower():
            legion_active = True
            print_styled("Legion mode activated.", Fore.GREEN, Style.BRIGHT)
            continue
        elif 'deactivate legion mode' in prompt.lower():
            legion_active = False
            print_styled("Legion mode deactivated.", Fore.RED, Style.BRIGHT)
            continue

        try:
            if legion_active:
                print_styled("Processing with legion mode...", Fore.YELLOW, Style.DIM)
                legion_response = legion_mode(prompt)
                print_styled("Assistant Response:", Fore.GREEN, Style.BRIGHT)
                print_styled(legion_response, Fore.GREEN)
            else:
                print_styled("Assistant is generating a response...", Fore.GREEN, Style.DIM)
                response = llama_model(prompt)
                print_styled("Assistant:", Fore.GREEN, Style.BRIGHT)
                print_styled(response, Fore.GREEN)
        except Exception as e:
            print_styled(f"An error occurred: {str(e)}", Fore.RED, Style.BRIGHT)
            print_styled("Please check your setup and try again.", Fore.YELLOW)
        print_divider('=')