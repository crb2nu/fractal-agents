"""Tests for HITL interrupt system."""

import pytest

from fractal_agents.hitl import (
    InterruptManager,
    InterruptRequest,
    InterruptResponse,
    InterruptType,
)


class TestInterruptTypes:
    """Tests for interrupt type definitions."""

    def test_interrupt_types_defined(self):
        """Verify all interrupt types are defined."""
        assert InterruptType.CONFIRMATION.value == "confirmation"
        assert InterruptType.INPUT.value == "input"
        assert InterruptType.CHOICE.value == "choice"
        assert InterruptType.REVIEW.value == "review"


class TestInterruptRequest:
    """Tests for InterruptRequest dataclass."""

    def test_request_creation(self):
        """Test creating an interrupt request."""
        request = InterruptRequest(
            node_id="test-node-123",
            interrupt_type=InterruptType.CONFIRMATION,
            prompt="Continue?",
        )
        assert request.node_id == "test-node-123"
        assert request.interrupt_type == InterruptType.CONFIRMATION
        assert request.prompt == "Continue?"
        assert request.options is None

    def test_request_with_options(self):
        """Test request with choice options."""
        request = InterruptRequest(
            node_id="test-node",
            interrupt_type=InterruptType.CHOICE,
            prompt="Select model",
            options=["gpt-4", "claude-3"],
        )
        assert request.options == ["gpt-4", "claude-3"]


class TestInterruptResponse:
    """Tests for InterruptResponse dataclass."""

    def test_default_response(self):
        """Test default response is approved."""
        response = InterruptResponse()
        assert response.approved is True
        assert response.value is None

    def test_response_with_value(self):
        """Test response with value."""
        response = InterruptResponse(approved=True, value="user input")
        assert response.value == "user input"


class TestInterruptManager:
    """Tests for InterruptManager."""

    def test_manager_initialization(self):
        """Test manager starts enabled but without callback."""
        manager = InterruptManager()
        assert manager._enabled is True
        assert manager._callback is None
        assert manager.is_enabled is False  # No callback registered

    def test_enable_disable(self):
        """Test enabling and disabling interrupts."""
        manager = InterruptManager()
        manager.disable()
        assert manager._enabled is False
        manager.enable()
        assert manager._enabled is True

    def test_register_callback(self):
        """Test registering a callback."""

        async def callback(request: InterruptRequest) -> InterruptResponse:
            return InterruptResponse(approved=True)

        manager = InterruptManager()
        manager.register_callback(callback)
        assert manager._callback == callback
        assert manager.is_enabled is True

    @pytest.mark.asyncio
    async def test_request_confirmation_with_callback(self):
        """Test confirmation request with callback."""

        async def callback(request: InterruptRequest) -> InterruptResponse:
            assert request.interrupt_type == InterruptType.CONFIRMATION
            return InterruptResponse(approved=True)

        manager = InterruptManager()
        manager.register_callback(callback)

        result = await manager.request_confirmation("node-1", "Continue?")
        assert result is True

    @pytest.mark.asyncio
    async def test_request_confirmation_denied(self):
        """Test denied confirmation."""

        async def callback(request: InterruptRequest) -> InterruptResponse:
            return InterruptResponse(approved=False)

        manager = InterruptManager()
        manager.register_callback(callback)

        result = await manager.request_confirmation("node-1", "Continue?")
        assert result is False

    @pytest.mark.asyncio
    async def test_request_input_with_callback(self):
        """Test input request returns user value."""

        async def callback(request: InterruptRequest) -> InterruptResponse:
            return InterruptResponse(approved=True, value="user response")

        manager = InterruptManager()
        manager.register_callback(callback)

        result = await manager.request_input("node-1", "Enter value:")
        assert result == "user response"

    @pytest.mark.asyncio
    async def test_request_choice_with_callback(self):
        """Test choice request returns selected option."""

        async def callback(request: InterruptRequest) -> InterruptResponse:
            assert request.options == ["a", "b", "c"]
            return InterruptResponse(approved=True, value="b")

        manager = InterruptManager()
        manager.register_callback(callback)

        result = await manager.request_choice("node-1", "Choose:", ["a", "b", "c"])
        assert result == "b"

    @pytest.mark.asyncio
    async def test_request_review_with_feedback(self):
        """Test review request with feedback."""

        async def callback(request: InterruptRequest) -> InterruptResponse:
            assert request.content_to_review == "Draft content"
            return InterruptResponse(approved=False, feedback="Needs more detail")

        manager = InterruptManager()
        manager.register_callback(callback)

        result = await manager.request_review("node-1", "Review:", "Draft content")
        assert result.approved is False
        assert result.feedback == "Needs more detail"

    @pytest.mark.asyncio
    async def test_auto_approve_when_disabled(self):
        """Test requests auto-approve when disabled."""
        manager = InterruptManager()
        manager.disable()

        result = await manager.request_confirmation("node-1", "Continue?")
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_approve_without_callback(self):
        """Test requests auto-approve without callback."""
        manager = InterruptManager()

        result = await manager.request_confirmation("node-1", "Continue?")
        assert result is True
