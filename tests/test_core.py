"""Tests for the FractalNode core class."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockLLMInterface:
    """Mock LLM interface for testing."""

    async def generate_response(
        self, prompt: str, context: str = "", model_hint: str = "general"
    ) -> str:
        return f"Response to: {prompt[:50]}"

    async def generate_subgoals(self, goal: str, num_subgoals: int = 3) -> list[str]:
        return [f"Subgoal {i + 1}" for i in range(num_subgoals)]

    async def summarize(self, text: str) -> str:
        return f"Summary of: {text[:30]}"


class MockMemory:
    """Mock memory for testing without Redis."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.summaries: dict[str, str] = {}

    def save_node_state(self, node_id: str, state: dict[str, Any]) -> None:
        self.nodes[node_id] = state

    def get_node_state(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes.get(node_id)

    def store_summary(self, node_id: str, summary: str) -> None:
        self.summaries[node_id] = summary

    def get_summary(self, node_id: str) -> str:
        return self.summaries.get(node_id, "")


class TestFractalNode:
    """Tests for the FractalNode class."""

    @pytest.fixture
    def mock_llm(self) -> MockLLMInterface:
        """Create a mock LLM interface."""
        return MockLLMInterface()

    @pytest.fixture
    def mock_memory(self) -> MockMemory:
        """Create a mock memory."""
        return MockMemory()

    def test_node_creation(
        self, mock_llm: MockLLMInterface, mock_memory: MockMemory
    ) -> None:
        """Test basic node creation."""
        from fractal_agents.core import FractalNode

        node = FractalNode(
            goal="Test goal",
            llm=mock_llm,
            memory=mock_memory,
            max_depth=2,
        )

        assert node.goal == "Test goal"
        assert node.status == "PENDING"
        assert node.depth == 0
        assert node.max_depth == 2
        assert len(node.id) == 36  # UUID length

    def test_node_persists_on_creation(
        self, mock_llm: MockLLMInterface, mock_memory: MockMemory
    ) -> None:
        """Test that node persists state on creation."""
        from fractal_agents.core import FractalNode

        node = FractalNode(
            goal="Persisted goal",
            llm=mock_llm,
            memory=mock_memory,
        )

        assert node.id in mock_memory.nodes
        state = mock_memory.nodes[node.id]
        assert state["goal"] == "Persisted goal"
        assert state["status"] == "PENDING"

    def test_is_complex_at_depth_zero(
        self, mock_llm: MockLLMInterface, mock_memory: MockMemory
    ) -> None:
        """Test complexity detection at root level."""
        from fractal_agents.core import FractalNode

        node = FractalNode(
            goal="Complex goal",
            llm=mock_llm,
            memory=mock_memory,
            depth=0,
            max_depth=3,
        )

        assert node.is_complex() is True

    def test_is_not_complex_at_max_depth(
        self, mock_llm: MockLLMInterface, mock_memory: MockMemory
    ) -> None:
        """Test complexity detection at max depth."""
        from fractal_agents.core import FractalNode

        node = FractalNode(
            goal="Leaf goal",
            llm=mock_llm,
            memory=mock_memory,
            depth=3,
            max_depth=3,
        )

        assert node.is_complex() is False

    @pytest.mark.asyncio
    async def test_leaf_node_execution(
        self, mock_llm: MockLLMInterface, mock_memory: MockMemory
    ) -> None:
        """Test execution of a leaf node (at max depth)."""
        from fractal_agents.core import FractalNode

        node = FractalNode(
            goal="Simple task",
            llm=mock_llm,
            memory=mock_memory,
            depth=3,
            max_depth=3,
        )

        result = await node.run()

        assert "Response to:" in result
        assert node.status == "COMPLETED"
        assert node.id in mock_memory.summaries

    @pytest.mark.asyncio
    async def test_complex_node_splits(
        self, mock_llm: MockLLMInterface, mock_memory: MockMemory
    ) -> None:
        """Test that complex nodes split into subgoals."""
        from fractal_agents.core import FractalNode

        node = FractalNode(
            goal="Complex task requiring breakdown",
            llm=mock_llm,
            memory=mock_memory,
            depth=0,
            max_depth=1,  # Will split once then execute leaves
        )

        result = await node.run()

        assert node.status == "COMPLETED"
        assert len(node.children_ids) == 3  # Default subgoals
        assert "Synthesized:" in result

    def test_parent_child_relationship(
        self, mock_llm: MockLLMInterface, mock_memory: MockMemory
    ) -> None:
        """Test parent-child ID tracking."""
        from fractal_agents.core import FractalNode

        parent = FractalNode(
            goal="Parent goal",
            llm=mock_llm,
            memory=mock_memory,
        )

        child = FractalNode(
            goal="Child goal",
            parent_id=parent.id,
            llm=mock_llm,
            memory=mock_memory,
            depth=1,
        )

        assert child.parent_id == parent.id
        assert child.depth == 1

    def test_vram_points_calculation(
        self, mock_llm: MockLLMInterface, mock_memory: MockMemory
    ) -> None:
        """Test VRAM points estimation in persisted state."""
        from fractal_agents.core import FractalNode

        node = FractalNode(
            goal="Reasoning task",
            llm=mock_llm,
            memory=mock_memory,
            task_type="reasoning",
        )

        # Initially PENDING, so 0 VRAM
        state = mock_memory.nodes[node.id]
        assert state["vram_points"] == 0

        # Manually set IN_PROGRESS and re-persist
        node.status = "IN_PROGRESS"
        node._persist()

        state = mock_memory.nodes[node.id]
        assert state["vram_points"] == 100  # Reasoning task
