"""Observability integration for fractal agents.

Integrates with py-observability to track LLM usage metrics and emit
tree state events for real-time visualization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .state import NodeState

logger = logging.getLogger(__name__)

# Optional observability import
try:
    from observability import AIMetrics

    _HAS_OBSERVABILITY = True
except ImportError:
    AIMetrics = None  # type: ignore
    _HAS_OBSERVABILITY = False


@dataclass
class TreeEvent:
    """Event representing a change in the fractal tree."""

    event_type: str  # "node_created", "node_started", "node_completed", "node_failed"
    node_id: str
    parent_id: str | None
    depth: int
    goal: str
    status: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


TreeEventCallback = Callable[[TreeEvent], None]


class FractalMetrics:
    """Observability wrapper for fractal agent execution.

    Tracks LLM token usage and emits tree events for visualization.

    Example:
        metrics = FractalMetrics(job_name="my-agent")
        metrics.on_tree_event(lambda e: print(f"Event: {e.event_type}"))

        # Track an LLM call
        with metrics.track_llm_call("gpt-4", "openai") as tracker:
            result = await llm.generate(...)
            tracker.set_tokens(result.usage.prompt_tokens, result.usage.completion_tokens)
    """

    def __init__(self, job_name: str = "fractal-agents") -> None:
        self._ai_metrics: Any = None
        self._event_callbacks: list[TreeEventCallback] = []

        if _HAS_OBSERVABILITY and AIMetrics is not None:
            try:
                self._ai_metrics = AIMetrics(job_name=job_name)
                logger.info("Observability integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize AIMetrics: {e}")

    @property
    def is_enabled(self) -> bool:
        """Check if observability is available."""
        return self._ai_metrics is not None

    def on_tree_event(self, callback: TreeEventCallback) -> None:
        """Register a callback for tree events.

        Args:
            callback: Function to call with TreeEvent on each state change.
        """
        self._event_callbacks.append(callback)

    def emit_tree_event(self, state: "NodeState", event_type: str) -> None:
        """Emit a tree event based on node state change.

        Args:
            state: Current node state.
            event_type: Type of event (node_created, node_started, etc.)
        """
        event = TreeEvent(
            event_type=event_type,
            node_id=state["id"],
            parent_id=state.get("parent_id"),
            depth=state.get("depth", 0),
            goal=state.get("goal", ""),
            status=state.get("status", "PENDING"),
            metadata={
                "task_type": state.get("task_type", "general"),
                "vram_points": state.get("vram_points", 0),
            },
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Tree event callback failed: {e}")

    def record_llm_call(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record metrics for an LLM call.

        Args:
            model: Model name.
            provider: Provider name.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            duration_seconds: Call duration.
            success: Whether the call succeeded.
        """
        if self._ai_metrics is not None:
            try:
                self._ai_metrics.record_request(
                    model=model,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration_seconds=duration_seconds,
                    success=success,
                )
            except Exception as e:
                logger.debug(f"Failed to record metrics: {e}")

    def push(self) -> None:
        """Push accumulated metrics to the gateway."""
        if self._ai_metrics is not None:
            try:
                self._ai_metrics.push()
            except Exception as e:
                logger.debug(f"Failed to push metrics: {e}")

    def track_llm_call(self, model: str, provider: str) -> "LLMCallTracker":
        """Context manager to track an LLM call.

        Args:
            model: Model name.
            provider: Provider name.

        Returns:
            LLMCallTracker context manager.
        """
        return LLMCallTracker(self, model, provider)


class LLMCallTracker:
    """Context manager for tracking LLM call metrics."""

    def __init__(self, metrics: FractalMetrics, model: str, provider: str) -> None:
        self._metrics = metrics
        self._model = model
        self._provider = provider
        self._start_time: float = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._success = True

    def __enter__(self) -> "LLMCallTracker":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.perf_counter() - self._start_time
        self._success = exc_type is None
        self._metrics.record_llm_call(
            model=self._model,
            provider=self._provider,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            duration_seconds=duration,
            success=self._success,
        )

    def set_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Set token counts from the LLM response.

        Args:
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
        """
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
