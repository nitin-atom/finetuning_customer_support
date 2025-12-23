#!/usr/bin/env python3
"""
Web-based Chat Interface for Fine-Tuned Model.

FastAPI backend with SSE streaming for real-time responses.
"""

import os
import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Add parent to path for config access
sys.path.insert(0, str(Path(__file__).parent.parent))

# Global state
config = None
client = None
model_id = None
sessions: Dict[str, dict] = {}


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_client() -> OpenAI:
    """Initialize OpenAI client with API key from .env."""
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def load_model_id(cfg: dict) -> Optional[str]:
    """Load fine-tuned model ID from saved file."""
    base_path = Path(__file__).parent.parent
    model_path = base_path / cfg['paths']['finetuned_model']

    if not model_path.exists():
        return None

    with open(model_path) as f:
        data = json.load(f)
        return data.get('model_id')


def get_or_create_session(session_id: str) -> dict:
    """Get existing session or create new one with default system prompt."""
    if session_id not in sessions:
        system_prompt = config['system_prompts']['default']
        sessions[session_id] = {
            "messages": [{"role": "system", "content": system_prompt}],
            "system_prompt_key": "default"
        }
    return sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global config, client, model_id
    config = load_config()
    client = setup_client()
    model_id = load_model_id(config)

    if not model_id:
        # Fallback to hardcoded model ID
        model_id = "ft:gpt-4.1-2025-04-14:personal:atom-support:CpnziPun"

    print(f"Loaded model: {model_id}")
    print(f"Available system prompts: {list(config['system_prompts'].keys())}")
    yield


app = FastAPI(
    title="Atom Support Chatbot",
    description="Web interface for testing the fine-tuned customer support model",
    lifespan=lifespan
)


# Request models
class ChatRequest(BaseModel):
    session_id: str
    message: str


class SystemPromptRequest(BaseModel):
    session_id: str
    prompt_key: str


class ClearRequest(BaseModel):
    session_id: str


# Routes
@app.get("/")
async def serve_frontend():
    """Serve the HTML frontend."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/prompts")
async def get_prompts():
    """Return available system prompt keys."""
    return {"prompts": list(config['system_prompts'].keys())}


@app.get("/api/model")
async def get_model():
    """Return current model ID."""
    return {"model_id": model_id}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream chat response using Server-Sent Events."""
    session = get_or_create_session(request.session_id)

    # Add user message to history
    session['messages'].append({"role": "user", "content": request.message})

    async def generate():
        collected_response = ""
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=session['messages'],
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_response += content
                    yield {"event": "message", "data": json.dumps({"content": content})}

            # Add complete response to history
            session['messages'].append({"role": "assistant", "content": collected_response})
            yield {"event": "done", "data": json.dumps({"complete": True})}

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(generate())


@app.post("/api/clear")
async def clear_history(request: ClearRequest):
    """Clear conversation history, keeping system prompt."""
    session = get_or_create_session(request.session_id)
    system_prompt = config['system_prompts'].get(
        session['system_prompt_key'],
        config['system_prompts']['default']
    )
    session['messages'] = [{"role": "system", "content": system_prompt}]
    return {"status": "cleared"}


@app.post("/api/system")
async def switch_system_prompt(request: SystemPromptRequest):
    """Switch system prompt and clear history."""
    if request.prompt_key not in config['system_prompts']:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt key: {request.prompt_key}"
        )

    session = get_or_create_session(request.session_id)
    session['system_prompt_key'] = request.prompt_key
    system_prompt = config['system_prompts'][request.prompt_key]
    session['messages'] = [{"role": "system", "content": system_prompt}]

    return {"status": "switched", "prompt_key": request.prompt_key}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state (for debugging)."""
    session = sessions.get(session_id)
    if not session:
        return {"exists": False}
    return {
        "exists": True,
        "system_prompt_key": session['system_prompt_key'],
        "message_count": len(session['messages']) - 1  # Exclude system message
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
