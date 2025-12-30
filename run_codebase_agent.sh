#!/bin/bash
export QDRANT_API_KEY="90FcWdIeQDR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export LITELLM_API_BASE="http://litellm.ai.svc.cluster.local:8000/v1"
export LITELLM_API_KEY="sk-litellm-local"
export REDIS_URL="redis://:changeme-redis@langgraph-redis-master.ai.svc.cluster.local:6379/0"
./.venv/bin/python3 examples/codebase_agent.py
