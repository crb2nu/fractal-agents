import os
import redis
import pytest
import asyncio
from typing import List, Dict, Any
from fractal_agents.memory import FractalMemory
from fractal_agents.core import FractalNode
from fractal_agents.llm_interface import LLMInterface, LiteLLM

# Integration test requires Redis.
# In CI, REDIS_URL should be set (e.g., redis://redis:6379/0)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def redis_client():
    client = redis.Redis.from_url(REDIS_URL)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not available for integration tests")

    # Clean up test data
    keys = client.keys("fractal:test:*")
    if keys:
        client.delete(*keys)
    yield client

    # Cleanup after
    keys = client.keys("fractal:test:*")
    if keys:
        client.delete(*keys)


@pytest.fixture
def integration_memory(redis_client):
    memory = FractalMemory(redis_url=REDIS_URL)
    memory.prefix = "fractal:test:"
    return memory


class RobustMockLLM(LLMInterface):
    """A more realistic mock that handles state and JSON better for integration testing."""

    def __init__(self):
        self.call_log = []

    async def generate_response(
        self, prompt: str, context: str = "", model_hint: str = "general"
    ) -> str:
        self.call_log.append(("generate", prompt))
        return f"Response to {prompt[:20]}"

    async def triage_task(self, goal: str, context: str = "") -> dict:
        self.call_log.append(("triage", goal))
        # Simple rule: if goal contains 'complex', SPLIT it
        if "complex" in goal.lower():
            return {"action": "SPLIT", "reason": "Task is complex", "num_subgoals": 2}
        return {"action": "SOLVE", "reason": "Task is simple", "num_subgoals": 0}

    async def generate_subgoals(self, goal: str, num_subgoals: int = 3) -> List[str]:
        self.call_log.append(("subgoals", goal))
        return [f"Sub-task {i + 1} for {goal[:10]}" for i in range(num_subgoals)]

    async def synthesize_results(self, goal: str, subtasks: List[dict], context: str = "") -> str:
        self.call_log.append(("synthesize", goal))
        results = [s.get("result", "") for s in subtasks]
        return f"Synthesized: {' | '.join(results)}"

    async def summarize(self, text: str) -> str:
        return text[:50]


@pytest.mark.asyncio
async def test_memory_integration_with_real_redis(integration_memory):
    """Test memory persistence, compression, and retrieval with a real Redis instance."""
    node_id = "test-node-int-1"
    state: Dict[str, Any] = {
        "id": node_id,
        "goal": "Integration test goal",
        "parent_id": None,
        "children_ids": [],
        "status": "IN_PROGRESS",
        "result": "",
        "summary": "",
        "depth": 0,
        "task_type": "reasoning",
        "vram_points": 100,
    }

    # Save
    integration_memory.save_node_state(node_id, state)

    # Retrieve
    retrieved = integration_memory.get_node_state(node_id)
    assert retrieved == state

    # Summary
    integration_memory.store_summary(node_id, "Short summary")
    assert integration_memory.get_summary(node_id) == "Short summary"


@pytest.mark.asyncio
async def test_node_mitosis_loop_integration(integration_memory):
    """Test the full FractalNode loop (Triage -> Split -> Execute -> Synthesize) with real storage."""
    llm = RobustMockLLM()

    # Use a goal that triggers a split
    root_node = FractalNode(
        goal="Perform a complex analysis of fractal patterns",
        llm=llm,
        memory=integration_memory,
        max_depth=2,
    )

    # Execute
    result = await root_node.run()

    # Verify logical flow
    assert "Synthesized:" in result

    # Verify state persistence in Redis
    state = integration_memory.get_node_state(root_node.id)
    assert state["status"] == "COMPLETED"
    assert len(state["children_ids"]) == 2

    # Check children states
    for child_id in state["children_ids"]:
        child_state = integration_memory.get_node_state(child_id)
        assert child_state["status"] == "COMPLETED"
        assert child_state["parent_id"] == root_node.id
