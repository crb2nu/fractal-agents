"""Tests for the FractalMemory class."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestFractalMemory:
    """Tests for FractalMemory Redis-backed storage."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Create a mock Redis client."""
        with patch("redis.Redis.from_url") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def memory(self, mock_redis: MagicMock) -> "FractalMemory":
        """Create a FractalMemory instance with mocked Redis."""
        from fractal_agents.memory import FractalMemory

        return FractalMemory(redis_url="redis://test:6379/0")

    def test_save_node_state(self, memory: "FractalMemory", mock_redis: MagicMock) -> None:
        """Test saving node state to Redis with compression."""
        import lz4.frame

        state = {
            "id": "test-node-1",
            "goal": "Test goal",
            "status": "PENDING",
            "depth": 0,
        }

        memory.save_node_state("test-node-1", state)

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args

        # Verify Key
        assert call_args[0][0] == "fractal:node:test-node-1"

        # Verify Content (Decompress first)
        compressed_data = call_args[0][1]
        json_str = lz4.frame.decompress(compressed_data).decode("utf-8")
        assert json.loads(json_str) == state

    def test_get_node_state(self, memory: "FractalMemory", mock_redis: MagicMock) -> None:
        """Test retrieving node state from Redis."""
        import lz4.frame

        expected_state = {"id": "test-1", "goal": "Goal"}

        # Mock Redis returning compressed data
        json_str = json.dumps(expected_state)
        compressed_data = lz4.frame.compress(json_str.encode("utf-8"))
        mock_redis.get.return_value = compressed_data

        result = memory.get_node_state("test-1")

        assert result == expected_state

    def test_get_node_state_uncompressed_fallback(
        self, memory: "FractalMemory", mock_redis: MagicMock
    ) -> None:
        """Test retrieval falls back to raw JSON if decompression fails."""
        expected_state = {"id": "test-2", "goal": "Legacy"}
        # Return raw bytes (not compressed)
        mock_redis.get.return_value = json.dumps(expected_state).encode("utf-8")

        result = memory.get_node_state("test-2")
        assert result == expected_state

    def test_store_summary(self, memory: "FractalMemory", mock_redis: MagicMock) -> None:
        """Test storing summaries in Redis hash."""
        memory.store_summary("node-1", "This is a summary")

        mock_redis.hset.assert_called_once_with("fractal:summaries", "node-1", "This is a summary")

    def test_get_summary(self, memory: "FractalMemory", mock_redis: MagicMock) -> None:
        """Test retrieving summaries from Redis hash."""
        mock_redis.hget.return_value = "Stored summary"

        result = memory.get_summary("node-1")

        assert result == "Stored summary"
        mock_redis.hget.assert_called_once_with("fractal:summaries", "node-1")

    def test_get_summary_not_found(self, memory: "FractalMemory", mock_redis: MagicMock) -> None:
        """Test retrieving non-existent summary returns empty string."""
        mock_redis.hget.return_value = None

        result = memory.get_summary("nonexistent")

        assert result == ""

    def test_key_generation(self, memory: "FractalMemory") -> None:
        """Test internal key generation."""
        key = memory._get_key("my-node-id")
        assert key == "fractal:node:my-node-id"
