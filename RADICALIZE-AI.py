import os
import threading
import subprocess
import openai
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
from colorama import init, Fore, Back, Style
import requests
import re
import random
import time
import concurrent.futures

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Debug: Print loaded environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY').strip()
print(Fore.YELLOW + f"Full API Key for debugging: {OPENAI_API_KEY}")  # Remove this line in production
print(Fore.YELLOW + f"OPENAI_API_KEY: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}")
print(Fore.YELLOW + "MONGODB_URI:", os.getenv('MONGODB_URI'))
print(Fore.YELLOW + "MONGODB_DB:", os.getenv('MONGODB_DB'))
print(Fore.YELLOW + "MONGODB_COLLECTION:", os.getenv('MONGODB_COLLECTION'))

# Initialize OpenAI client with explicit API key
OPENAI_API_BASE = 'https://api.openai.com/v1'
client = OpenAI(api_key=OPENAI_API_KEY)

# Validate API key format
def validate_api_key(api_key):
    if not api_key:
        return False
    # Check if the key starts with "sk-proj-" and is followed by a long string
    return bool(re.match(r'^sk-proj-[a-zA-Z0-9-_]{50,}$', api_key))

if not validate_api_key(OPENAI_API_KEY):
    print(Fore.RED + f"Invalid API key format. Key: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-5:]}")
    exit(1)

# Test OpenAI API connection
def test_openai_connection():
    try:
        print(Fore.YELLOW + "Attempting to connect to OpenAI API...")
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        response = requests.get(f"{OPENAI_API_BASE}/models", headers=headers)
        print(Fore.YELLOW + f"Response Status Code: {response.status_code}")
        print(Fore.YELLOW + f"Response Content: {response.text[:200]}...")  # Print first 200 characters
        
        if response.status_code == 200:
            print(Fore.GREEN + "Successfully connected to OpenAI API.")
            return True
        else:
            print(Fore.RED + f"Failed to connect to OpenAI API. Status code: {response.status_code}")
            print(Fore.RED + f"Error message: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Network error: {str(e)}")
    except Exception as e:
        print(Fore.RED + f"Unexpected error: {str(e)}")
    return False

# Run connection test
if not test_openai_connection():
    print(Fore.RED + "OpenAI API connection failed. Please check your setup and try again.")
    exit(1)

# Initialize MongoDB client with error handling
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    raise ValueError("Environment variable 'MONGODB_URI' is not set.")

try:
    client_mongodb = MongoClient(MONGODB_URI)
    # The following line will attempt to list databases to verify connection
    client_mongodb.list_database_names()
    print(Fore.GREEN + "MongoDB connection successful.")
except Exception as e:
    print(Fore.RED + f"Failed to connect to MongoDB: {e}")
    # Instead of raising an error, we'll print it and continue without MongoDB
    client_mongodb = None

# Add checks for environment variables
MONGODB_DB = os.getenv('MONGODB_DB')
if not MONGODB_DB:
    print(Fore.YELLOW + "Warning: Environment variable 'MONGODB_DB' is not set.")

MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')
if not MONGODB_COLLECTION:
    print(Fore.YELLOW + "Warning: Environment variable 'MONGODB_COLLECTION' is not set.")

if client_mongodb and MONGODB_DB and MONGODB_COLLECTION:
    db = client_mongodb[MONGODB_DB]
    collection = db[MONGODB_COLLECTION]
else:
    print(Fore.YELLOW + "MongoDB functionality will be disabled.")
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
            if "command not found" in str(e):
                print(Fore.RED + "Ollama is not installed or not in your system PATH.")
                print(Fore.YELLOW + "Please install Ollama from https://ollama.ai/download")
                print(Fore.YELLOW + "After installation, make sure it's in your system PATH.")
                return "Error: Ollama is not available. Please install it and try again."
            elif "no such model" in str(e):
                print(Fore.RED + "The specified Llama model is not available.")
                print(Fore.YELLOW + "Please pull the model using: ollama pull llama3.1:8b")
                return "Error: Llama model not available. Please pull the model and try again."
            return f"Error from Llama model: {str(e)}"
    return llama_generate

llama_model = initialize_llama_model()

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
    'Ada Lovelace (INTP)': Fore.LIGHTCYAN_EX,  # New color for Ada Lovelace
    'Alan Turing (ENTJ)': Fore.LIGHTGREEN_EX  # New color for Alan Turing
}

# Function to print styled text
def print_styled(text, color=Fore.WHITE, style=Style.NORMAL, end='\n'):
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)

# Function to print a separator line
def print_separator(char='-', length=50):
    print_styled(char * length, Fore.WHITE, Style.DIM)

# Function to run an OpenAI agent with added randomness and error handling
def run_agent(agent_name, model_name, prompt, responses, scores):
    start_time = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print_styled(f"Attempting to use model: {model_name}", Fore.YELLOW, Style.DIM)
            messages = []
            if model_name not in ['o1-mini', 'o1-preview', 'text-embedding-3-small', 'text-embedding-3-large']:
                messages.append({"role": "system", "content": f"You are {agent_name}, a helpful assistant with a unique perspective. Provide diverse and creative responses. You can include code blocks using triple backticks."})
            
            messages.append({"role": "user", "content": prompt})
            
            if model_name in ['text-embedding-3-small', 'text-embedding-3-large']:
                # For embedding models, we'll use them to generate embeddings and then use GPT to generate a response
                response = client.embeddings.create(
                    model=model_name,
                    input=prompt
                )
                embedding = response.data[0].embedding
                # Use the embedding to generate a response using GPT
                gpt_response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {"role": "system", "content": f"You are {agent_name}. Use the following embedding to generate a response: {embedding[:50]}..."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500  # Increased max tokens for embedding models
                )
                agent_response = gpt_response.choices[0].message.content
            elif model_name in ['o1-mini', 'o1-preview']:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=2500 if model_name == 'o1-preview' else 2000  # Increased max_completion_tokens for o1 models
                )
                agent_response = response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=random.uniform(0.5, 1.0),
                    max_tokens=2000  # Increased max tokens for other GPT models
                )
                agent_response = response.choices[0].message.content
            
            response_time = time.time() - start_time
            
            # Calculate score
            model_capability_factor = 1.5 if model_name in ['o1-preview', 'text-embedding-3-large'] else 1.2 if model_name in ['o1-mini', 'text-embedding-3-small'] else 1.1 if 'gpt-4o' in model_name else 1.0
            time_factor = min(1.0, 60 / response_time)  # Normalize time factor
            score = (len(agent_response.split()) * model_capability_factor * time_factor)
            
            responses[agent_name] = agent_response
            scores[agent_name] = score
            return agent_response
        except Exception as e:
            print_styled(f"Error from {agent_name} (Attempt {attempt+1}/{max_retries}): {str(e)}", Fore.RED, Style.BRIGHT)
            if attempt == max_retries - 1:
                error_message = f"Error from {agent_name} after {max_retries} attempts: {str(e)}"
                responses[agent_name] = error_message
                scores[agent_name] = 0
                return error_message
            time.sleep(2)  # Wait for 2 seconds before retrying

# Function to run legion mode with simultaneous agent responses
def legion_mode(prompt):
    agents = {
        'Jane Austen (INFJ)': {'model': 'gpt-4o', 'currency': 'Regency Era Pounds'},
        'George Orwell (INTJ)': {'model': 'gpt-4o', 'currency': 'Dystopian Credits'},
        'Virginia Woolf (INFP)': {'model': 'gpt-4o', 'currency': 'Stream of Consciousness Tokens'},
        'Ernest Hemingway (ESTP)': {'model': 'gpt-4o-mini', 'currency': 'Bullfighting Pesetas'},
        'Agatha Christie (ISTJ)': {'model': 'gpt-4o-mini', 'currency': 'Mystery Solving Guineas'},
        'Oscar Wilde (ENFP)': {'model': 'o1-mini', 'currency': 'Witty Epigram Coins'},
        'Sylvia Plath (INFJ)': {'model': 'o1-mini', 'currency': 'Poetic Introspection Points'},
        'Stephen King (INFP)': {'model': 'o1-preview', 'currency': 'Nightmare Fuel Dollars'},
        'Ada Lovelace (INTP)': {'model': 'o1-mini', 'currency': 'Analytical Engine Tokens'},  # New agent
        'Alan Turing (ENTJ)': {'model': 'gpt-4o', 'currency': 'Cryptographic Keys'},  # New agent
    }
    responses = {}
    scores = {}
    conversation = []

    print_styled("Legion Mode Activated", Fore.GREEN, Style.BRIGHT)
    print_separator('=')
    print_styled(f"Original Prompt: {prompt}", Fore.WHITE, Style.BRIGHT)
    print_separator()

    # Initial thoughts
    initial_prompts = [
        f"As {agent_name}, provide a brief initial thought on the following prompt: '{prompt}'. Be true to your writing style and personality. Keep it under 200 words."
        for agent_name in agents.keys()
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
        future_to_agent = {executor.submit(run_agent, agent_name, agent_info['model'], prompt, {}, {}): agent_name for agent_name, agent_info in agents.items()}
        for future in concurrent.futures.as_completed(future_to_agent):
            agent_name = future_to_agent[future]
            response = future.result()
            conversation.append(f"{agent_name}: {response}")
            print_styled(f"{agent_name} says:", AGENT_COLORS.get(agent_name, Fore.WHITE), Style.BRIGHT)
            print_styled(response, AGENT_COLORS.get(agent_name, Fore.WHITE))
            print_separator('-')

    # Add Llama model's initial thought
    llama_response = llama_model(prompt)
    conversation.append(f"Llama Model: {llama_response}")
    print_styled("Llama Model says:", AGENT_COLORS.get('Llama Model', Fore.WHITE), Style.BRIGHT)
    print_styled(llama_response, AGENT_COLORS.get('Llama Model', Fore.WHITE))
    print_separator('-')

    # Interactive discussion
    for _ in range(15):  # More iterations for a longer, more natural conversation
        speaking_agent = random.choice(list(agents.keys()) + ['Llama Model'])
        listening_agent = random.choice([a for a in list(agents.keys()) + ['Llama Model'] if a != speaking_agent])
        
        recent_context = " ".join(conversation[-3:])  # Use the last 3 conversation pieces for context
        discussion_prompt = f"""As {speaking_agent}, engage in the ongoing conversation about '{prompt}'. 
        Recent context: {recent_context}
        You can:
        1. Respond to a point made by {listening_agent} or another agent
        2. Introduce a new perspective on the original prompt
        3. Ask a thought-provoking question to the group or a specific agent
        4. Challenge or support another agent's view
        Be true to your unique voice and personality. Keep it under 200 words and make it feel like a natural group conversation."""

        if speaking_agent == 'Llama Model':
            response = llama_model(discussion_prompt)
        else:
            response = run_agent(speaking_agent, agents[speaking_agent]['model'], discussion_prompt, {}, {})
        
        conversation.append(f"{speaking_agent}: {response}")
        print_styled(f"{speaking_agent} chimes in:", AGENT_COLORS.get(speaking_agent, Fore.WHITE), Style.BRIGHT)
        print_styled(response, AGENT_COLORS.get(speaking_agent, Fore.WHITE))
        print_separator('-')

        # Check if the response mentions another agent and have that agent respond
        mentioned_agents = [agent for agent in list(agents.keys()) + ['Llama Model'] if agent in response]
        for mentioned_agent in mentioned_agents:
            if mentioned_agent != speaking_agent:
                mention_prompt = f"As {mentioned_agent}, respond to the point made by {speaking_agent} in the following context: '{response}'. Keep it under 200 words."
                if mentioned_agent == 'Llama Model':
                    mention_response = llama_model(mention_prompt)
                else:
                    mention_response = run_agent(mentioned_agent, agents[mentioned_agent]['model'], mention_prompt, {}, {})
                conversation.append(f"{mentioned_agent}: {mention_response}")
                print_styled(f"{mentioned_agent} responds:", AGENT_COLORS.get(mentioned_agent, Fore.WHITE), Style.BRIGHT)
                print_styled(mention_response, AGENT_COLORS.get(mentioned_agent, Fore.WHITE))
                print_separator('-')

    # Final response generation
    enhanced_prompt = f"Original prompt: {prompt}\n\nBased on the following conversation, provide your final, comprehensive response to the original prompt:\n{' '.join(conversation)}"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
        future_to_agent = {executor.submit(run_agent, agent_name, agent_info['model'], enhanced_prompt, responses, scores): agent_name for agent_name, agent_info in agents.items()}
        for future in concurrent.futures.as_completed(future_to_agent):
            agent_name = future_to_agent[future]
            response = future.result()
            responses[agent_name] = response
            scores[agent_name] = len(response.split())  # Simple scoring for Llama model
        concurrent.futures.wait(future_to_agent)

    # Add Llama model's final response
    llama_final_response = llama_model(enhanced_prompt)
    responses['Llama Model'] = llama_final_response
    scores['Llama Model'] = len(llama_final_response.split())  # Simple scoring for Llama model

    # Ensure all agents have responses
    for agent_name in list(agents.keys()) + ['Llama Model']:
        if agent_name not in responses:
            responses[agent_name] = f"Error: No response generated for {agent_name}"
            scores[agent_name] = 0

    # Display individual agent responses and scores
    for agent_name, response in responses.items():
        print_separator()
        print_styled(f"Final response from {agent_name} ({agents.get(agent_name, {}).get('model', 'llama')}):", AGENT_COLORS.get(agent_name, Fore.WHITE), Style.BRIGHT)
        print_styled(response, AGENT_COLORS.get(agent_name, Fore.WHITE))
        print_styled(f"Score: {scores[agent_name]:.2f}", AGENT_COLORS.get(agent_name, Fore.WHITE), Style.DIM)

    # Determine the winner
    winner = max(scores, key=scores.get)
    winning_currency = agents.get(winner, {}).get('currency', 'Llama Tokens')
    print_separator('=')
    print_styled(f"Winner: {winner}", Fore.GREEN, Style.BRIGHT)
    print_styled(f"Reward: 100 {winning_currency}", Fore.GREEN, Style.BRIGHT)
    print_separator('=')

    # First referee agent (o1-preview) with Stephen Hawking's personality and intelligence
    def run_referee_1(prompt):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return run_agent("Stephen Hawking (Referee 1)", 'o1-preview', prompt, {}, {})
            except Exception as e:
                print_styled(f"Error from Referee 1 (Attempt {attempt+1}/{max_retries}): {str(e)}", Fore.RED, Style.BRIGHT)
                if attempt == max_retries - 1:
                    return "The first referee agent encountered an error and could not generate a response."

    # Second referee agent (gpt-4o) with a personality of Socrates
    def run_referee_2(prompt):
        return run_agent("Socrates (Referee 2)", 'gpt-4o', prompt, {}, {})

    # Final blending by the third referee agent (gpt-4o) with a personality of Maxwell Perkins
    def run_final_referee(prompt):
        return run_agent("Maxwell Perkins (Final Referee)", 'gpt-4o', prompt, {}, {})

    # First referee agent (o1-preview) combines and analyzes the responses
    referee_prompt_1 = (
        "As Stephen Hawking, the first referee agent, analyze the following conversation and final responses from various agents, each with their own personality and writing style. "
        "Provide a comprehensive summary of the key points and insights, highlighting the most valuable contributions. "
        "Also, explain why the winning response was chosen and how it compares to the others. "
        "Finally, synthesize a cohesive response that combines the best elements from all agents. Keep your response under 750 words.\n\n"
        f"Original Prompt: {prompt}\n\n"
        f"Conversation:\n{' '.join(conversation)}\n\n"
        "Final Responses:\n" + "\n".join([f"{agent}: {response[:200]}..." for agent, response in responses.items()]) + "\n\n"
        f"Winner: {winner} (Score: {scores[winner]:.2f})"
    )

    referee_1_output = run_referee_1(referee_prompt_1)

    # Second referee agent (gpt-4o) reviews and provides feedback
    referee_prompt_2 = (
        "As Socrates, the second referee agent, review the analysis and synthesis provided by the first referee agent. "
        "Offer constructive feedback, highlight any overlooked points, and suggest improvements or alternative interpretations. "
        "Then, provide your own synthesis of the key insights and how they address the original prompt. "
        "Keep your response under 750 words.\n\n"
        f"Original Prompt: {prompt}\n\n"
        f"First Referee's Analysis:\n{referee_1_output}\n\n"
    )

    referee_2_output = run_referee_2(referee_prompt_2)

    # Final blending by the third referee agent (gpt-4o)
    final_blend_prompt = (
        "As Maxwell Perkins, the final referee agent, review the feedback and alternative synthesis provided by the second referee agent. "
        "Integrate the most valuable insights from both analyses to create a final, comprehensive response to the original prompt. "
        "This final response should represent the best collective wisdom from all agents and both referee perspectives. "
        "Keep your response under 1000 words.\n\n"
        f"Original Prompt: {prompt}\n\n"
        f"First Referee's Analysis:\n{referee_1_output}\n\n"
        f"Second Referee's Feedback and Synthesis:\n{referee_2_output}\n\n"
    )

    final_blend_output = run_final_referee(final_blend_prompt)

    # Log the interaction to MongoDB if available
    if collection is not None:
        log = {
            'prompt': prompt,
            'conversation': conversation,
            'responses': responses,
            'scores': scores,
            'winner': winner,
            'winning_currency': winning_currency,
            'referee_1_output': referee_1_output,
            'referee_2_output': referee_2_output,
            'final_response': final_blend_output
        }
        try:
            collection.insert_one(log)
            print_styled("Interaction logged to MongoDB.", Fore.GREEN, Style.DIM)
        except Exception as e:
            print_styled(f"Failed to log interaction to MongoDB: {str(e)}", Fore.RED, Style.DIM)
    else:
        print_styled("Interaction not logged (MongoDB not available).", Fore.YELLOW, Style.DIM)

    # Colorized final response
    print_styled("First Referee Agent (Stephen Hawking):", Fore.CYAN, Style.BRIGHT)
    print_styled(referee_1_output, Fore.CYAN)
    print_separator('-')
    print_styled("Second Referee Agent (Socrates):", Fore.MAGENTA, Style.BRIGHT)
    print_styled(referee_2_output, Fore.MAGENTA)
    print_separator('-')
    print_styled("Final Blended Response:", Fore.GREEN, Style.BRIGHT)
    print_styled(final_blend_output, Fore.GREEN)
    print_separator('=')

    return final_blend_output

    # Final blending by the third referee agent (gpt-4o) with a personality of Maxwell Perkins
    def run_final_referee(prompt):
        return run_agent("Maxwell Perkins (Final Referee)", 'gpt-4o', prompt, {}, {})

    # First referee agent (o1-preview) combines and analyzes the responses
    referee_prompt_1 = (
        "As Stephen Hawking, the first referee agent, analyze the following conversation and final responses from various agents, each with their own personality and writing style. "
        "Provide a comprehensive summary of the key points and insights, highlighting the most valuable contributions. "
        "Also, explain why the winning response was chosen and how it compares to the others. "
        "Finally, synthesize a cohesive response that combines the best elements from all agents. Keep your response under 750 words.\n\n"
        f"Original Prompt: {prompt}\n\n"
        f"Conversation:\n{' '.join(conversation)}\n\n"
        "Final Responses:\n" + "\n".join([f"{agent}: {response[:200]}..." for agent, response in responses.items()]) + "\n\n"
        f"Winner: {winner} (Score: {scores[winner]:.2f})"
    )

    referee_1_output = run_referee_1(referee_prompt_1)

    # Second referee agent (gpt-4o) reviews and provides feedback
    referee_prompt_2 = (
        "As Socrates, the second referee agent, review the analysis and synthesis provided by the first referee agent. "
        "Offer constructive feedback, highlight any overlooked points, and suggest improvements or alternative interpretations. "
        "Then, provide your own synthesis of the key insights and how they address the original prompt. "
        "Keep your response under 750 words.\n\n"
        f"Original Prompt: {prompt}\n\n"
        f"First Referee's Analysis:\n{referee_1_output}\n\n"
    )

    referee_2_output = run_referee_2(referee_prompt_2)

    # Final blending by the third referee agent (gpt-4o)
    final_blend_prompt = (
        "As Maxwell Perkins, the final referee agent, review the feedback and alternative synthesis provided by the second referee agent. "
        "Integrate the most valuable insights from both analyses to create a final, comprehensive response to the original prompt. "
        "This final response should represent the best collective wisdom from all agents and both referee perspectives. "
        "Keep your response under 1000 words.\n\n"
        f"Original Prompt: {prompt}\n\n"
        f"First Referee's Analysis:\n{referee_1_output}\n\n"
        f"Second Referee's Feedback and Synthesis:\n{referee_2_output}\n\n"
    )

    final_blend_output = run_final_referee(final_blend_prompt)

    # Log the interaction to MongoDB if available
    if collection is not None:
        log = {
            'prompt': prompt,
            'conversation': conversation,
            'responses': responses,
            'scores': scores,
            'winner': winner,
            'winning_currency': winning_currency,
            'referee_1_output': referee_1_output,
            'referee_2_output': referee_2_output,
            'final_response': final_blend_output
        }
        try:
            collection.insert_one(log)
            print_styled("Interaction logged to MongoDB.", Fore.GREEN, Style.DIM)
        except Exception as e:
            print_styled(f"Failed to log interaction to MongoDB: {str(e)}", Fore.RED, Style.DIM)
    else:
        print_styled("Interaction not logged (MongoDB not available).", Fore.YELLOW, Style.DIM)

    # Colorized final response
    print_styled("First Referee Agent (Stephen Hawking):", Fore.CYAN, Style.BRIGHT)
    print_styled(referee_1_output, Fore.CYAN)
    print_separator('-')
    print_styled("Second Referee Agent (Socrates):", Fore.MAGENTA, Style.BRIGHT)
    print_styled(referee_2_output, Fore.MAGENTA)
    print_separator('-')
    print_styled("Final Blended Response:", Fore.GREEN, Style.BRIGHT)
    print_styled(final_blend_output, Fore.GREEN)
    print_separator('=')

    return final_blend_output

# Main execution
if __name__ == "__main__":
    print_styled("Welcome to the RADICALIZE-AI Assistant!", Fore.MAGENTA, Style.BRIGHT)
    print_styled("Type 'exit' to quit the program.", Fore.MAGENTA)
    print_styled("Type 'activate legion mode' to enable legion mode.", Fore.MAGENTA)
    print_styled("Type 'deactivate legion mode' to disable legion mode.", Fore.MAGENTA)
    print_separator('=')

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
                print_styled("Enhancing Llama model response with Legion mode insights...", Fore.GREEN, Style.DIM)
                enhanced_prompt = f"Original prompt: {prompt}\n\nLegion mode insights: {legion_response}\n\nPlease provide an enhanced response based on the original prompt and the insights from Legion mode."
                final_response = llama_model(enhanced_prompt)
                print_styled("Enhanced Assistant Response:", Fore.GREEN, Style.BRIGHT)
                print_styled(final_response, Fore.GREEN)
            else:
                print_styled("Assistant is generating a response...", Fore.GREEN, Style.DIM)
                response = llama_model(prompt)
                print_styled("Assistant:", Fore.GREEN, Style.BRIGHT)
                print_styled(response, Fore.GREEN)
        except Exception as e:
            print_styled(f"An error occurred: {str(e)}", Fore.RED, Style.BRIGHT)
            print_styled("Please check your setup and try again.", Fore.YELLOW)
        
        print_separator('=')
