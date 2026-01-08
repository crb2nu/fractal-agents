"""Tests for observability integration."""

import time

from fractal_agents.observability import (
    FractalMetrics,
    TreeEvent,
)


class TestTreeEvent:
    """Tests for TreeEvent dataclass."""

    def test_event_creation(self):
        """Test creating a tree event."""
        event = TreeEvent(
            event_type="node_created",
            node_id="test-123",
            parent_id=None,
            depth=0,
            goal="Test goal",
            status="PENDING",
        )
        assert event.event_type == "node_created"
        assert event.node_id == "test-123"
        assert event.parent_id is None
        assert event.depth == 0
        assert event.goal == "Test goal"
        assert event.timestamp > 0

    def test_event_with_metadata(self):
        """Test event with metadata."""
        event = TreeEvent(
            event_type="node_started",
            node_id="test-123",
            parent_id="parent-456",
            depth=1,
            goal="Sub goal",
            status="IN_PROGRESS",
            metadata={"task_type": "reasoning"},
        )
        assert event.metadata["task_type"] == "reasoning"


class TestFractalMetrics:
    """Tests for FractalMetrics."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = FractalMetrics(job_name="test-agent")
        # Should work even without observability installed
        assert metrics is not None

    def test_event_callback_registration(self):
        """Test registering event callbacks."""
        metrics = FractalMetrics()
        events = []

        metrics.on_tree_event(lambda e: events.append(e))

        # Create a mock state
        state = {
            "id": "node-123",
            "parent_id": None,
            "depth": 0,
            "goal": "Test",
            "status": "PENDING",
            "task_type": "general",
            "vram_points": 0,
        }

        metrics.emit_tree_event(state, "node_created")

        assert len(events) == 1
        assert events[0].event_type == "node_created"
        assert events[0].node_id == "node-123"

    def test_multiple_callbacks(self):
        """Test multiple event callbacks."""
        metrics = FractalMetrics()
        events1 = []
        events2 = []

        metrics.on_tree_event(lambda e: events1.append(e))
        metrics.on_tree_event(lambda e: events2.append(e))

        state = {
            "id": "node-123",
            "depth": 0,
            "goal": "Test",
            "status": "PENDING",
        }

        metrics.emit_tree_event(state, "node_started")

        assert len(events1) == 1
        assert len(events2) == 1

    def test_callback_exception_handling(self):
        """Test that exceptions in callbacks don't break metrics."""
        metrics = FractalMetrics()
        good_events = []

        def bad_callback(e):
            raise ValueError("Callback error")

        metrics.on_tree_event(bad_callback)
        metrics.on_tree_event(lambda e: good_events.append(e))

        state = {"id": "node-123", "depth": 0, "goal": "Test", "status": "PENDING"}

        # Should not raise, should continue to good callback
        metrics.emit_tree_event(state, "node_created")

        assert len(good_events) == 1


class TestLLMCallTracker:
    """Tests for LLMCallTracker context manager."""

    def test_tracker_records_duration(self):
        """Test tracker records call duration."""
        metrics = FractalMetrics()
        recorded_calls = []

        def mock_record(**kwargs):
            recorded_calls.append(kwargs)

        metrics.record_llm_call = mock_record

        with metrics.track_llm_call("gpt-4", "openai") as tracker:
            time.sleep(0.01)  # Small delay
            tracker.set_tokens(100, 50)

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["model"] == "gpt-4"
        assert recorded_calls[0]["provider"] == "openai"
        assert recorded_calls[0]["prompt_tokens"] == 100
        assert recorded_calls[0]["completion_tokens"] == 50
        assert recorded_calls[0]["duration_seconds"] > 0
        assert recorded_calls[0]["success"] is True

    def test_tracker_records_failure(self):
        """Test tracker records failure on exception."""
        metrics = FractalMetrics()
        recorded_calls = []

        def mock_record(**kwargs):
            recorded_calls.append(kwargs)

        metrics.record_llm_call = mock_record

        try:
            with metrics.track_llm_call("gpt-4", "openai"):
                raise ValueError("LLM error")
        except ValueError:
            pass

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["success"] is False
