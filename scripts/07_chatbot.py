#!/usr/bin/env python3
"""
Phase 7: Interactive Chatbot using Fine-Tuned Model.

Inputs: data/output/finetuned_model.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging to hide HTTP requests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_client():
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def load_model_id(config):
    base_path = Path(__file__).parent.parent
    model_path = base_path / config['paths']['finetuned_model']
    
    if not model_path.exists():
        print("Error: Fine-tuned model not found. Have you run 06_finetune.py?")
        return None
        
    with open(model_path) as f:
        data = json.load(f)
        return data.get('model_id')


def main():
    parser = argparse.ArgumentParser(description='Chat with Fine-Tuned Model')
    parser.add_argument('--system', default='default', help='System prompt key to use')
    args = parser.parse_args()
    
    config = load_config()
    client = setup_client()
    
    model_id = load_model_id(config)
    if not model_id:
        sys.exit(1)
        
    # Get system prompt
    system_prompt = config['system_prompts'].get(args.system, config['system_prompts']['default'])
    
    print(f"\n{'='*50}")
    print(f"Chatting with: {model_id}")
    print(f"System: {args.system}")
    print(f"{'='*50}")
    print("Commands: /exit, /clear, /system <key>")
    print(f"{'='*50}\n")
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['/exit', '/quit']:
                break
                
            if user_input.lower() == '/clear':
                messages = [{"role": "system", "content": system_prompt}]
                print("Conversation history cleared.")
                continue
                
            if user_input.startswith('/system '):
                key = user_input.split(' ', 1)[1]
                new_prompt = config['system_prompts'].get(key)
                if new_prompt:
                    system_prompt = new_prompt
                    messages = [{"role": "system", "content": system_prompt}]
                    print(f"Switched to system prompt: {key}")
                else:
                    print(f"System prompt '{key}' not found.")
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get response
            print("Assistant: ", end="", flush=True)
            
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True
            )
            
            collected_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    collected_response += content
            
            print() # Newline
            
            # Add assistant message to history
            messages.append({"role": "assistant", "content": collected_response})
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == '__main__':
    main()
