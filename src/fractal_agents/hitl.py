"""Human-in-the-Loop (HITL) interrupt system for fractal agents.

Provides mechanisms for pausing execution to request user feedback
and resuming after receiving input.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class InterruptType(Enum):
    """Types of HITL interrupts."""

    CONFIRMATION = "confirmation"  # Yes/No decision
    INPUT = "input"  # Free-form text input
    CHOICE = "choice"  # Select from options
    REVIEW = "review"  # Review and approve/reject content


@dataclass
class InterruptRequest:
    """A request for user input during execution."""

    node_id: str
    interrupt_type: InterruptType
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    options: list[str] | None = None  # For CHOICE type
    content_to_review: str | None = None  # For REVIEW type


@dataclass
class InterruptResponse:
    """User's response to an interrupt request."""

    approved: bool = True
    value: str | None = None
    feedback: str | None = None


class HITLInterrupt(Exception):
    """Raised when a node requires user input to continue."""

    def __init__(self, request: InterruptRequest):
        self.request = request
        super().__init__(f"HITL interrupt: {request.prompt}")


InterruptCallback = Callable[[InterruptRequest], Awaitable[InterruptResponse]]


class InterruptManager:
    """Manage HITL interrupt requests and callbacks.

    The InterruptManager allows nodes to pause execution and request
    user feedback. A callback must be registered to handle interrupt
    requests.

    Example:
        async def handle_interrupt(request: InterruptRequest) -> InterruptResponse:
            # Show prompt to user and get response
            user_input = await show_dialog(request.prompt)
            return InterruptResponse(approved=True, value=user_input)

        manager = InterruptManager()
        manager.register_callback(handle_interrupt)

        # In a FractalNode:
        response = await self.interrupt_manager.request_input(
            node_id=self.id,
            prompt="Should I continue with this approach?",
            interrupt_type=InterruptType.CONFIRMATION
        )
    """

    def __init__(self) -> None:
        self._callback: InterruptCallback | None = None
        self._pending_requests: dict[str, asyncio.Future[InterruptResponse]] = {}
        self._enabled = True

    def register_callback(self, callback: InterruptCallback) -> None:
        """Register a callback to handle interrupt requests.

        Args:
            callback: Async function that takes an InterruptRequest
                     and returns an InterruptResponse.
        """
        self._callback = callback

    def enable(self) -> None:
        """Enable HITL interrupts."""
        self._enabled = True

    def disable(self) -> None:
        """Disable HITL interrupts (auto-approve all requests)."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if HITL interrupts are enabled."""
        return self._enabled and self._callback is not None

    async def request_confirmation(
        self,
        node_id: str,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Request a yes/no confirmation from the user.

        Args:
            node_id: ID of the requesting node.
            prompt: Question to ask the user.
            context: Optional context for the request.

        Returns:
            True if user approved, False otherwise.
        """
        response = await self._request(
            InterruptRequest(
                node_id=node_id,
                interrupt_type=InterruptType.CONFIRMATION,
                prompt=prompt,
                context=context or {},
            )
        )
        return response.approved

    async def request_input(
        self,
        node_id: str,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Request free-form text input from the user.

        Args:
            node_id: ID of the requesting node.
            prompt: Prompt for the user.
            context: Optional context for the request.

        Returns:
            User's input string.
        """
        response = await self._request(
            InterruptRequest(
                node_id=node_id,
                interrupt_type=InterruptType.INPUT,
                prompt=prompt,
                context=context or {},
            )
        )
        return response.value or ""

    async def request_choice(
        self,
        node_id: str,
        prompt: str,
        options: list[str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Request user to select from a list of options.

        Args:
            node_id: ID of the requesting node.
            prompt: Prompt for the user.
            options: List of options to choose from.
            context: Optional context for the request.

        Returns:
            Selected option string.
        """
        response = await self._request(
            InterruptRequest(
                node_id=node_id,
                interrupt_type=InterruptType.CHOICE,
                prompt=prompt,
                options=options,
                context=context or {},
            )
        )
        return response.value or options[0]

    async def request_review(
        self,
        node_id: str,
        prompt: str,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> InterruptResponse:
        """Request user to review and approve/reject content.

        Args:
            node_id: ID of the requesting node.
            prompt: Prompt explaining what to review.
            content: Content to review.
            context: Optional context for the request.

        Returns:
            InterruptResponse with approval status and optional feedback.
        """
        return await self._request(
            InterruptRequest(
                node_id=node_id,
                interrupt_type=InterruptType.REVIEW,
                prompt=prompt,
                content_to_review=content,
                context=context or {},
            )
        )

    async def _request(self, request: InterruptRequest) -> InterruptResponse:
        """Internal method to handle interrupt requests."""
        if not self._enabled:
            logger.debug(f"HITL disabled, auto-approving: {request.prompt}")
            return InterruptResponse(approved=True)

        if self._callback is None:
            logger.warning(f"No HITL callback registered, auto-approving: {request.prompt}")
            return InterruptResponse(approved=True)

        logger.info(f"HITL interrupt requested by node {request.node_id[:8]}: {request.prompt}")

        try:
            response = await self._callback(request)
            logger.info(
                f"HITL response for node {request.node_id[:8]}: "
                f"approved={response.approved}, value={response.value}"
            )
            return response
        except Exception as e:
            logger.error(f"HITL callback failed: {e}")
            # On callback failure, auto-approve to prevent blocking
            return InterruptResponse(approved=True)
