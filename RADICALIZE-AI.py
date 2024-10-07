import os
import asyncio
import logging
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import aiohttp

import threading
import subprocess
import time
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv
from colorama import init, Fore, Back, Style
import requests
import tiktoken
import anthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI, AsyncOpenAI
from openai import OpenAIError

import numpy as np
import hashlib

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Load API Keys (Make sure to set these in your .env file)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '').strip()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '').strip()
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', '').strip()
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', '').strip()

if not OPENAI_API_KEY or not CLAUDE_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX_NAME or not PINECONE_ENVIRONMENT:
    logging.error("API keys or Pinecone configuration not found. Please check your .env file.")
    exit(1)

# Initialize API clients
openai.api_key = OPENAI_API_KEY
claude_client = anthropic.AsyncAnthropic(api_key=CLAUDE_API_KEY)

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # List available indexes
    index_list = pc.list_indexes()
    
    if PINECONE_INDEX_NAME not in index_list.names():
        logging.warning(f"Index {PINECONE_INDEX_NAME} not found. Available indexes: {index_list.names()}")
        
        # Create the index if it doesn't exist
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI's embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="azure",
                region=PINECONE_ENVIRONMENT
            )
        )
        logging.info(f"Created new index: {PINECONE_INDEX_NAME}")
    
    index = pc.Index(PINECONE_INDEX_NAME)
    logging.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    logging.error(f"Error with Pinecone: {e}")
    exit(1)

# Initialize Llama model
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
            logging.error(f"Error from Llama model: {e}")
            return f"Error from Llama model: {str(e)}"
    return llama_generate

llama_model = initialize_llama_model()

# Constants
MAX_TOTAL_TOKENS = 4096
RETRY_LIMIT = 3
RETRY_BACKOFF_FACTOR = 2
MAX_REFINEMENT_ATTEMPTS = 3

# Load agents configuration from agents.json
def load_agents_config():
    try:
        with open('agents.json', 'r', encoding='utf-8') as f:
            agents_data = json.load(f)
        logging.info("Agents configuration loaded successfully.")
        return agents_data.get('agents', [])
    except Exception as e:
        logging.error(f"Error loading agents configuration: {e}")
        exit(1)

agents_config = load_agents_config()

# Agent colors
AGENT_COLORS = [
    Fore.MAGENTA, Fore.CYAN, Fore.YELLOW, Fore.GREEN, Fore.RED,
    Fore.BLUE, Fore.LIGHTMAGENTA_EX, Fore.LIGHTRED_EX, Fore.LIGHTCYAN_EX,
    Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX
]

class Agent:
    ACTION_DESCRIPTIONS = {
        'discuss': "formulating a response",
        'verify': "verifying data",
        'refine': "refining the response",
        'critique': "critiquing another agent's response",
        'reason': "applying advanced reasoning",
        'collaborate': "collaborating with other agents",
        'debate': "engaging in a debate",
        'synthesize': "synthesizing information",
    }

    def __init__(self, agent_config, color):
        self.name = agent_config.get('name', 'Unnamed Agent')
        self.color = color
        self.specialty = agent_config.get('personality', {}).get('personality_traits', [])
        self.system_purpose = agent_config.get('system_purpose', '')
        self.custom_instructions = self._build_system_message(agent_config)
        self.messages = []
        self.score = 0
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        self.memory = index  # Use the connected Pinecone index
        self.namespace = f"agent_{self.name.lower().replace(' ', '_')}"
        self.graph = NetworkxEntityGraph()
        self.initialize_model(agent_config)

    def initialize_model(self, agent_config):
        # Assign specific models based on agent name or other attributes
        model_assignments = {
            'Jane Austen': 'gpt-4o',
            'George Orwell': 'gpt-4o',
            'Virginia Woolf': 'gpt-4o',
            'Ernest Hemingway': 'gpt-4o-mini',
            'Agatha Christie': 'gpt-4o-mini',
            'Oscar Wilde': 'o1-mini',
            'Sylvia Plath': 'o1-mini',
            'Stephen King': 'o1-preview',
            'Ada Lovelace': 'o1-mini',
            'Alan Turing': 'gpt-4o',
            'Riot the Husky': 'claude-3-5-sonnet-20240620',
            'Zen Master': 'claude-3-5-sonnet-20240620',
            'Maestro': 'claude-3-5-sonnet-20240620',
            # Add other agents as needed
        }
        self.model_name = model_assignments.get(self.name, 'gpt-4o')

    def _build_system_message(self, agent_config):
        personality_description = f"Your name is {self.name}. {self.system_purpose}"

        # Include interaction style
        interaction_style = agent_config.get('interaction_style', {})
        if interaction_style:
            interaction_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in interaction_style.items())
            personality_description += f"\n\nInteraction Style:\n{interaction_details}"

        # Include personality traits
        personality_traits = agent_config.get('personality', {})
        if personality_traits:
            personality_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in personality_traits.items())
            personality_description += f"\n\nPersonality Traits:\n{personality_details}"

        # Include other aspects if needed...

        return personality_description

    async def _add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._truncate_messages()
        await self._update_memory(content)
        self._update_graph(content)

    def _truncate_messages(self):
        encoding = tiktoken.get_encoding("cl100k_base")
        while sum(len(encoding.encode(msg['content'])) for msg in self.messages) > MAX_TOTAL_TOKENS:
            self.messages.pop(0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, OpenAIError)))
    async def _generate_embedding(self, text):
        try:
            return await self.embedding_function.aembed_query(text)
        except Exception as e:
            logging.debug(f"Error generating embedding: {str(e)}. Using fallback method.")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text):
        # Simple hash-based embedding (not as good as real embeddings, but works offline)
        hash_object = hashlib.md5(text.encode())
        hash_value = int(hash_object.hexdigest(), 16)
        np.random.seed(hash_value % (2**32 - 1))
        return np.random.rand(1536).tolist()  # OpenAI embeddings are 1536-dimensional

    async def _update_memory(self, content):
        try:
            text_splitter = CharacterTextSplitter(
                chunk_size=4000,  # Increased from 2000
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_text(content)
            for i, text in enumerate(texts):
                try:
                    vector = await self._generate_embedding(text)
                    self.memory.upsert(
                        vectors=[(f"{self.name}-{i}", vector, {"text": text})],
                        namespace=self.namespace
                    )
                    # Also update the hive mind
                    self.memory.upsert(
                        vectors=[(f"hive-{self.name}-{i}", vector, {"text": text, "agent": self.name})],
                        namespace="hive_mind"
                    )
                except Exception as e:
                    logging.error(f"Failed to update memory for chunk {i} of {self.name}: {str(e)}")
                await asyncio.sleep(0.1)  # Add a small delay between requests
        except Exception as e:
            logging.error(f"Failed to process content for {self.name}: {str(e)}")

    def _update_graph(self, content):
        # Remove or comment out any code using GraphQAChain
        # If you need graph-based QA functionality, you may need to implement it differently
        # or use an alternative approach
        pass

    async def exponential_backoff(self, attempt):
        delay = min(2 ** attempt + random.uniform(0, 1), 60)
        await asyncio.sleep(delay)

    async def _handle_chat_response(self, prompt):
        try:
            await self._add_message("user", prompt)
            messages = [{"role": "system", "content": self.custom_instructions}] + self.messages

            max_retries = 5
            base_delay = 1

            for attempt in range(max_retries):
                try:
                    if self.model_name.startswith('claude'):
                        response = await self._claude_chat(prompt)
                    elif self.model_name.startswith('gpt-') or self.model_name.startswith('o1-'):
                        response = await self._openai_chat(messages)
                    else:
                        response = await self._ollama_chat(prompt)
                    reply = response.strip()
                    await self._add_message("assistant", reply)
                    return reply
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Error in agent '{self.name}': {type(e).__name__}: {e}")
                        return f"An error occurred while generating a response: {str(e)}"
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)

            return "Failed to generate a response after multiple attempts."
        except Exception as e:
            logging.error(f"Unexpected error in _handle_chat_response for {self.name}: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

    async def _openai_chat(self, messages):
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        try:
            # Remove 'system' message for models that don't support it
            if self.model_name in ['o1-mini', 'o1-preview']:
                messages = [msg for msg in messages if msg['role'] != 'system']
                if not messages:
                    # If messages is empty after removing 'system', add a default user message
                    messages.append({"role": "user", "content": "Hello"})
                elif messages[0]['role'] == 'user':
                    messages[0]['content'] = f"{self.custom_instructions}\n\n{messages[0]['content']}"
            
            if not messages:
                # If messages is still empty, add a default user message
                messages.append({"role": "user", "content": "Hello"})
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in OpenAI chat for agent '{self.name}': {e}")
            raise e

    async def _claude_chat(self, prompt):
        try:
            response = await claude_client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                system=self.custom_instructions,  # Use 'system' parameter instead of including it in messages
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Error in Claude model for agent '{self.name}': {e}")
            raise e

    async def _ollama_chat(self, prompt):
        # For agents using Llama model via Ollama
        try:
            result = subprocess.run(
                ['ollama', 'run', 'llama3.1:8b', prompt],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Error from Ollama model for agent '{self.name}': {e}")
            return f"Error from Ollama model: {str(e)}"

    async def discuss(self, prompt):
        return await self._handle_chat_response(f"{self.name}, {self.system_purpose}\n\nUser Prompt: {prompt}")

    async def verify(self, data):
        return await self._handle_chat_response(f"Please verify the following response for accuracy and completeness:\n\n{data}")

    async def refine(self, data, more_time=False):
        refinement_prompt = f"Please refine the following response:\n\n{data}"
        if more_time:
            refinement_prompt += "\nTake additional time to thoroughly improve the response."
        return await self._handle_chat_response(refinement_prompt)

    async def critique(self, data):
        return await self._handle_chat_response(f"Please critique the following response for accuracy and completeness:\n\n{data}")

    async def reason(self, prompt, context):
        reasoning_prompt = f"As {self.name}, apply advanced reasoning to address the following prompt:\n\nContext: {context}\n\nPrompt: {prompt}\n\nYour response should demonstrate critical thinking, logical consistency, and depth of analysis."
        return await self._handle_chat_response(reasoning_prompt)

    async def collaborate(self, other_agent, prompt):
        collaboration_prompt = f"As {self.name}, collaborate with {other_agent.name} to address the following prompt:\n\nPrompt: {prompt}\n\nCombine your unique perspectives and expertise to provide a comprehensive response."
        return await self._handle_chat_response(collaboration_prompt)

    async def debate(self, other_agent, topic):
        debate_prompt = f"As {self.name}, engage in a debate with {other_agent.name} on the following topic:\n\nTopic: {topic}\n\nPresent your arguments, counter-arguments, and reach a conclusion based on the debate."
        return await self._handle_chat_response(debate_prompt)

    async def synthesize(self, information):
        synthesis_prompt = f"As {self.name}, synthesize the following information into a coherent and insightful summary:\n\nInformation: {information}\n\nIdentify key themes, draw connections, and provide a comprehensive overview."
        return await self._handle_chat_response(synthesis_prompt)

def print_styled(text, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def print_divider(char="═", length=80, color=Fore.YELLOW):
    print(color + char * length + Style.RESET_ALL)

async def blend_responses(responses, prompt):
    blending_prompt = f"Combine these responses into a single, coherent answer to '{prompt}':\n\n" + "\n\n".join(responses)
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": blending_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in blending responses: {e}")
        return "An error occurred while blending responses."

def calculate_score(response, duration, complexity):
    word_count = len(response.split())
    return (word_count * complexity) * (1 / max(duration, 0.1)) * 10

async def process_agent_action(agent, action, *args):
    try:
        result = await getattr(agent, action)(*args)
        return result
    except Exception as e:
        logging.error(f"Error in {action} for {agent.name}: {str(e)}")
        return f"Error: {str(e)}"

async def legion_mode(prompt):
    agents = [Agent(agent_config, AGENT_COLORS[i % len(AGENT_COLORS)]) for i, agent_config in enumerate(agents_config)]

    print_styled("Legion Mode Activated", Fore.GREEN, Style.BRIGHT)
    print_divider()
    print_styled(f"Original Prompt: {prompt}", Fore.WHITE, Style.BRIGHT)
    print_divider()

    responses = {}
    scores = {}
    conversation = []

    try:
        # Step 1: Initial Responses
        print_styled("Step 1: Generating Initial Responses", Fore.YELLOW, Style.BRIGHT)
        tasks = [process_agent_action(agent, 'discuss', prompt) for agent in agents]
        initial_responses = await asyncio.gather(*tasks, return_exceptions=True)

        for agent, response in zip(agents, initial_responses):
            if isinstance(response, Exception):
                print_styled(f"Error for {agent.name}: {str(response)}", agent.color)
                responses[agent.name] = f"Error: {str(response)}"
                scores[agent.name] = 0
            else:
                responses[agent.name] = response
                scores[agent.name] = calculate_score(response, 1, 1)
                print_styled(f"{agent.name}: {responses[agent.name]}", agent.color)
                conversation.append(f"{agent.name}: {responses[agent.name]}")

        # Step 2: Verification and Fact-Checking
        print_styled("\nStep 2: Verifying and Fact-Checking Responses", Fore.YELLOW, Style.BRIGHT)
        verification_tasks = [process_agent_action(agent, 'verify', responses[agent.name]) for agent in agents]
        verification_responses = await asyncio.gather(*verification_tasks)

        for agent, response in zip(agents, verification_responses):
            if not response.startswith("Error:"):
                scores[agent.name] += calculate_score(response, 1, 1.2)
                print_styled(f"{agent.name} verification: {response}", agent.color)
                conversation.append(f"{agent.name} verification: {response}")

        # Step 3: Critique and Debate
        print_styled("\nStep 3: Critiquing and Debating Responses", Fore.YELLOW, Style.BRIGHT)
        critique_tasks = []
        for i, agent in enumerate(agents):
            other_agent = agents[(i + 1) % len(agents)]
            critique_tasks.append(process_agent_action(agent, 'critique', responses[other_agent.name]))
        critique_responses = await asyncio.gather(*critique_tasks)

        for agent, response in zip(agents, critique_responses):
            if not response.startswith("Error:"):
                scores[agent.name] += calculate_score(response, 1, 1.5)
                print_styled(f"{agent.name} critique: {response}", agent.color)
                conversation.append(f"{agent.name} critique: {response}")

        # Step 4: Collaborative Reasoning
        print_styled("\nStep 4: Collaborative Reasoning", Fore.YELLOW, Style.BRIGHT)
        collaboration_tasks = []
        for i in range(0, len(agents), 2):
            agent1 = agents[i]
            agent2 = agents[(i + 1) % len(agents)]
            collaboration_tasks.append(process_agent_action(agent1, 'collaborate', agent2, prompt))
        collaboration_responses = await asyncio.gather(*collaboration_tasks)

        for idx in range(len(collaboration_responses)):
            agent1 = agents[2*idx % len(agents)]
            agent2 = agents[(2*idx + 1) % len(agents)]
            response = collaboration_responses[idx]
            if not response.startswith("Error:"):
                scores[agent1.name] += calculate_score(response, 1, 1.8)
                scores[agent2.name] += calculate_score(response, 1, 1.8)
                print_styled(f"{agent1.name} and {agent2.name} collaboration: {response}", Fore.WHITE)
                conversation.append(f"{agent1.name} and {agent2.name} collaboration: {response}")

        # Step 5: Advanced Reasoning and Synthesis
        print_styled("\nStep 5: Advanced Reasoning and Synthesis", Fore.YELLOW, Style.BRIGHT)
        context = "\n".join(conversation)
        reasoning_tasks = [process_agent_action(agent, 'reason', prompt, context) for agent in agents]
        reasoning_responses = await asyncio.gather(*reasoning_tasks)

        for agent, response in zip(agents, reasoning_responses):
            if not response.startswith("Error:"):
                responses[agent.name] = response
                scores[agent.name] += calculate_score(response, 1, 2.0)
                print_styled(f"{agent.name} reasoning: {responses[agent.name]}", agent.color)
                conversation.append(f"{agent.name} reasoning: {responses[agent.name]}")

        # Step 6: Final Refinement
        print_styled("\nStep 6: Final Refinement", Fore.YELLOW, Style.BRIGHT)
        refinement_tasks = [process_agent_action(agent, 'refine', responses[agent.name], True) for agent in agents]
        refinement_responses = await asyncio.gather(*refinement_tasks)

        for agent, response in zip(agents, refinement_responses):
            if not response.startswith("Error:"):
                responses[agent.name] = response
                scores[agent.name] += calculate_score(response, 1, 1.5)
                print_styled(f"{agent.name} refinement: {responses[agent.name]}", agent.color)
                conversation.append(f"{agent.name} refinement: {responses[agent.name]}")

        # Step 7: Information Synthesis
        print_styled("\nStep 7: Information Synthesis", Fore.YELLOW, Style.BRIGHT)
        all_responses = list(responses.values())
        synthesis_tasks = [process_agent_action(agent, 'synthesize', "\n".join(all_responses)) for agent in agents]
        synthesis_responses = await asyncio.gather(*synthesis_tasks)
        synthesis_result = ""
        for agent, response in zip(agents, synthesis_responses):
            if not response.startswith("Error:"):
                synthesis_result = response
                print_styled(f"{agent.name} synthesized the information.", agent.color)
                break
        if not synthesis_result:
            synthesis_result = "No synthesis was successful."
        print_styled("Synthesized Information:", Fore.GREEN, Style.BRIGHT)
        print_styled(synthesis_result, Fore.GREEN)
        conversation.append(f"Synthesized Information: {synthesis_result}")

        # Step 8: Final Blending and Enhancement
        print_styled("\nStep 8: Final Blending and Enhancement", Fore.YELLOW, Style.BRIGHT)

        agents_responses = ''.join([f"{agent.name}: {responses[agent.name]}\n" for agent in agents])

        enhanced_prompt = (
            f"Original prompt: {prompt}\n\n"
            f"Agents' responses and synthesis:\n{agents_responses}\n\n"
            "Provide an enhanced final response that incorporates the best elements of "
            "all previous responses while maintaining coherence and addressing the original prompt comprehensively."
        )
        final_response = llama_model(enhanced_prompt)
        print_styled("Final Enhanced Response:", Fore.GREEN, Style.BRIGHT)
        print_styled(final_response, Fore.GREEN)

        # Determine winner and runner-ups
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_scores[0][0]
        runner_ups = [agent for agent, _ in sorted_scores[1:3]]

        print_styled(f"\nWinner: {winner}", Fore.GREEN, Style.BRIGHT)
        print_styled("Congratulations! 🎉", Fore.GREEN, Style.BRIGHT)
        print_styled(f"Runner-ups: {runner_ups[0]} and {runner_ups[1]}", Fore.YELLOW, Style.BRIGHT)

        # After all agents have processed, we can query the hive mind if needed
        hive_mind_query = index.query(
            namespace="hive_mind",
            vector=await agents[0]._generate_embedding(prompt),  # Use the first agent to generate the query vector
            top_k=5,
            include_values=True,
            include_metadata=True
        )
        
        print_styled("\nHive Mind Insights:", Fore.CYAN, Style.BRIGHT)
        for match in hive_mind_query['matches']:
            print_styled(f"Agent {match['metadata']['agent']}: {match['metadata']['text']}", Fore.CYAN)

        # Log interaction if desired
        # ... (same as previous code)

        return final_response

    except asyncio.CancelledError:
        print_styled("\nOperation cancelled. Shutting down gracefully...", Fore.YELLOW, Style.BRIGHT)
        return "Operation cancelled."
    except Exception as e:
        logging.error(f"An error occurred in legion mode: {str(e)}")
        return f"An error occurred: {str(e)}"

async def handle_feedback(response):
    print_styled("\nWas this response helpful? (yes/no)", Fore.YELLOW)
    feedback = await asyncio.get_event_loop().run_in_executor(None, input)
    feedback = feedback.strip().lower()
    if feedback == 'no':
        print_styled("We're sorry the response wasn't helpful. How can we improve it?", Fore.YELLOW)
        improvement = await asyncio.get_event_loop().run_in_executor(None, input)
        improvement = improvement.strip()
        return f"The previous response was not helpful. User suggests: {improvement}"
    return None

async def save_conversation(conversation_history):
    with open("conversation_history.txt", "w") as f:
        for entry in conversation_history:
            f.write(f"{entry}\n")

async def load_conversation():
    if os.path.exists("conversation_history.txt"):
        with open("conversation_history.txt", "r") as f:
            return f.read().splitlines()
    return []

async def individual_agent_chat(agent):
    print_styled(f"Chatting with {agent.name}", agent.color, Style.BRIGHT)
    while True:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, Fore.BLUE + Style.BRIGHT + f"\nYou to {agent.name}: " + Style.RESET_ALL)
        if user_input.lower() == 'exit':
            break
        response = await agent._handle_chat_response(user_input)
        print_styled(f"{agent.name}: {response}", agent.color)

async def display_options_menu(agents):
    while True:
        print_styled("\nOptions Menu:", Fore.YELLOW, Style.BRIGHT)
        print_styled("1. Activate Legion Mode", Fore.YELLOW)
        print_styled("2. Chat with Individual Agent", Fore.YELLOW)
        print_styled("3. Save Conversation", Fore.YELLOW)
        print_styled("4. Exit", Fore.YELLOW)
        
        choice = await asyncio.get_event_loop().run_in_executor(None, input, Fore.BLUE + Style.BRIGHT + "\nEnter your choice (1-4): " + Style.RESET_ALL)
        
        if choice == '1':
            return 'legion'
        elif choice == '2':
            print_styled("\nChoose an agent to chat with:", Fore.YELLOW)
            for i, agent in enumerate(agents, 1):
                print_styled(f"{i}. {agent.name}", agent.color)
            agent_choice = await asyncio.get_event_loop().run_in_executor(None, input, Fore.BLUE + Style.BRIGHT + "\nEnter agent number: " + Style.RESET_ALL)
            try:
                chosen_agent = agents[int(agent_choice) - 1]
                await individual_agent_chat(chosen_agent)
            except (ValueError, IndexError):
                print_styled("Invalid choice. Please try again.", Fore.RED)
        elif choice == '3':
            return 'save'
        elif choice == '4':
            return 'exit'
        else:
            print_styled("Invalid choice. Please try again.", Fore.RED)

async def main():
    print_styled("Welcome to RADICALIZE-AI!", Fore.MAGENTA, Style.BRIGHT)
    
    agents = [Agent(agent_config, AGENT_COLORS[i % len(AGENT_COLORS)]) for i, agent_config in enumerate(agents_config)]
    conversation_history = await load_conversation()

    while True:
        choice = await display_options_menu(agents)
        
        if choice == 'legion':
            print_styled("Legion Mode activated.", Fore.GREEN, Style.BRIGHT)
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, Fore.BLUE + Style.BRIGHT + "\nLegion Mode Prompt: " + Style.RESET_ALL)
            final_response = await legion_mode(user_input)
            print_styled("Final Response:", Fore.GREEN, Style.BRIGHT)
            print_styled(final_response, Fore.GREEN)
            conversation_history.append(f"User (Legion Mode): {user_input}")
            conversation_history.append(f"Legion Response: {final_response}")
        elif choice == 'save':
            await save_conversation(conversation_history)
            print_styled("Conversation saved.", Fore.GREEN, Style.BRIGHT)
        elif choice == 'exit':
            await save_conversation(conversation_history)
            print_styled("Conversation saved. Goodbye!", Fore.MAGENTA, Style.BRIGHT)
            break

    print_styled("Thank you for using RADICALIZE-AI!", Fore.MAGENTA, Style.BRIGHT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_styled("\nProgram terminated by user.", Fore.YELLOW, Style.BRIGHT)
