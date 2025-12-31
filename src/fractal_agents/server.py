import logging
import os
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fractal_agents.core import FractalNode
from fractal_agents.knowledge import FractalKnowledgeGraph, QdrantFractalStore
from fractal_agents.llm_interface import LiteLLM
from fractal_agents.memory import FractalMemory

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fractal-agent-server")

app = FastAPI(title="Fractal Agent API")

# Setup Infrastructure
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://litellm.ai.svc.cluster.local:8000/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-litellm-local")
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.50.176:6333")

# Initialize Shared Components
memory = FractalMemory(redis_url=REDIS_URL)
llm = LiteLLM(api_base=LITELLM_API_BASE, api_key=LITELLM_API_KEY)

# Initialize Knowledge Graph
try:
    knowledge_store = QdrantFractalStore(url=QDRANT_URL)
    knowledge_graph = FractalKnowledgeGraph(store=knowledge_store, llm_client=llm)
    logger.info("Fractal Knowledge Graph Initialized")
except Exception as e:
    logger.warning(f"Could not initialize Knowledge Graph: {e}")
    knowledge_graph = None

# --- OpenAI Compatible Schemas ---


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]


# --- Endpoints ---


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    Standard OpenAI Chat Completion endpoint.
    It treats the last user message as the 'Goal' for a new Fractal Tree.
    """
    logger.info(f"Received request for model: {request.model}")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Extract Goal from the last user message
    last_message = request.messages[-1]
    if last_message.role != "user":
        # Fallback: search backwards for last user message
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_message = msg
                break

    goal = last_message.content
    context = ""

    # Optional: Compile previous messages into 'Context'
    if len(request.messages) > 1:
        context = "Conversation History:\n" + "\n".join(
            [f"{m.role}: {m.content}" for m in request.messages[:-1]]
        )

    logger.info(f"Starting Fractal Task: {goal[:100]}...")

    # Initialize Root Node
    root = FractalNode(
        goal=goal,
        context=context,
        llm=llm,
        memory=memory,
        knowledge=knowledge_graph,
        max_depth=2,  # Keep depth shallow for synchronous response
        task_type="general",
    )

    try:
        # Run the agent synchronously
        result = root.run()
    except Exception as e:
        logger.error(f"Fractal Agent Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Construct OpenAI Response
    return ChatCompletionResponse(
        id=root.id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=Message(role="assistant", content=result), finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },  # Usage tracking TODO
    )


if __name__ == "__main__":
    import uvicorn

    # Quick fix for the time call above if needed, but imported inside main
    uvicorn.run(app, host="0.0.0.0", port=8000)
