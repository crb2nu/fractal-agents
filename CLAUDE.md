# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fractal Agents is a recursive, self-similar agentic framework where every agent is a **FractalNode** that can split complex tasks into parallel subtasks (mitosis), with inactive branches serialized to Redis ("Frozen Memory") to optimize VRAM usage on consumer GPUs.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (uses direnv)
cp .env.example .env
direnv allow

# Run the API server (OpenAI-compatible endpoint)
PYTHONPATH=$PYTHONPATH:$(pwd)/src python src/fractal_agents/server.py

# Run the real-time visualizer
PYTHONPATH=$PYTHONPATH:$(pwd)/src uvicorn fractal_agents.visualizer.app:app --host 0.0.0.0 --port 8080

# Run an example
PYTHONPATH=$PYTHONPATH:$(pwd)/src python examples/codebase_agent.py

# Deploy to K8s
kubectl apply -k k8s/
```

## Architecture

### Core Components

**FractalNode** (`src/fractal_agents/core.py`)
- The fundamental recursive unit. Receives a goal, decides complexity, and either solves directly or splits via mitosis
- Statuses: `PENDING` → `IN_PROGRESS` → `SPLIT` or `COMPLETED`
- Child nodes execute in parallel via `asyncio.gather()`
- VRAM points are tracked per-node for resource estimation

**LLMInterface** (`src/fractal_agents/llm_interface.py`)
- Abstract interface with `LiteLLM` implementation using OpenAI-compatible async client
- Model routing via `model_map`: general, reasoning, vision, speculative, fast, summary
- Speculative solving: fast model drafts, reasoning model refines if needed
- Subgoal generation returns structured JSON for task decomposition

**FractalMemory** (`src/fractal_agents/memory.py`)
- Redis-backed state persistence for the fractal tree
- Key prefix: `fractal:node:{id}` for full state, `fractal:summaries` hash for quick lookups
- Enables "context zooming" where only active branch is in memory

**FractalKnowledgeGraph** (`src/fractal_agents/knowledge.py`)
- Hierarchical vector retrieval using Qdrant
- Recursive search: finds relevant nodes, then searches their children
- `KnowledgeNode` stores content with parent references for tree structure

**LangGraph Bridge** (`src/fractal_agents/langgraph_bridge.py`)
- Integrates FractalNode as a reasoning node within LangGraph workflows
- `AgentState` TypedDict flows through: input → fractal_result → final_output

### External Dependencies

| Service | Default URL | Purpose |
|---------|-------------|---------|
| LiteLLM | `http://litellm.ai.svc.cluster.local:8000/v1` | LLM routing |
| Redis | `redis://localhost:6379/0` | Frozen memory storage |
| Qdrant | `http://192.168.50.176:6333` | Vector knowledge store |

### Key Environment Variables

```bash
LITELLM_API_BASE    # LiteLLM endpoint
LITELLM_API_KEY     # Auth key
REDIS_URL           # Redis connection string
QDRANT_URL          # Qdrant vector DB
MAX_RECURSION_DEPTH # Default: 3
```

## Execution Flow

1. Root `FractalNode` receives goal and context
2. If `depth < max_depth`, node enters SPLIT state:
   - LLM generates subgoals (JSON list)
   - Child nodes spawned with incremented depth
   - `asyncio.gather()` runs all children concurrently
   - Results synthesized and summarized
3. Leaf nodes (at max_depth) use speculative solving:
   - Fast model generates draft
   - If too short, reasoning model refines
4. All state persisted to Redis after each status change

## Server API

The FastAPI server (`src/fractal_agents/server.py`) exposes an OpenAI-compatible chat completion endpoint:

```
POST /v1/chat/completions
```

Last user message becomes the fractal goal; prior messages become context. The fractal tree executes with `max_depth=2` for synchronous response.
