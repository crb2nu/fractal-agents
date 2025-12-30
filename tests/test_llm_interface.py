"""Tests for the LLMInterface and LiteLLM classes."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMInterface:
    """Tests for the abstract LLMInterface contract."""

    def test_interface_defines_required_methods(self) -> None:
        """Test that interface defines all required abstract methods."""
        from fractal_agents.llm_interface import LLMInterface
        import inspect

        methods = [
            "generate_response",
            "generate_subgoals",
            "summarize",
        ]

        for method_name in methods:
            assert hasattr(LLMInterface, method_name)
            method = getattr(LLMInterface, method_name)
            assert inspect.iscoroutinefunction(method) or hasattr(
                method, "__isabstractmethod__"
            )


class TestLiteLLM:
    """Tests for the LiteLLM implementation."""

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Create a mock AsyncOpenAI client."""
        with patch("fractal_agents.llm_interface.AsyncOpenAI") as mock_class:
            client = AsyncMock()
            mock_class.return_value = client
            yield client

    @pytest.fixture
    def litellm(self, mock_openai_client: MagicMock) -> "LiteLLM":
        """Create a LiteLLM instance with mocked client."""
        from fractal_agents.llm_interface import LiteLLM

        return LiteLLM(api_base="http://test:8000/v1", api_key="test-key")

    def test_initialization_with_env_vars(self) -> None:
        """Test LiteLLM reads from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "LITELLM_API_BASE": "http://custom:8000/v1",
                "LITELLM_API_KEY": "custom-key",
            },
        ):
            with patch("fractal_agents.llm_interface.AsyncOpenAI"):
                from fractal_agents.llm_interface import LiteLLM

                llm = LiteLLM()
                assert llm.api_base == "http://custom:8000/v1"
                assert llm.api_key == "custom-key"

    def test_model_mapping(self, litellm: "LiteLLM") -> None:
        """Test model hint to model name mapping."""
        assert litellm._get_model("general") == "qwen2.5-7b"
        assert litellm._get_model("reasoning") == "reasoning"
        assert litellm._get_model("vision") == "vision"
        assert litellm._get_model("speculative") == "qwen2.5-7b"
        assert litellm._get_model("unknown") == "qwen2.5-7b"  # Default fallback

    @pytest.mark.asyncio
    async def test_generate_response(
        self, litellm: "LiteLLM", mock_openai_client: MagicMock
    ) -> None:
        """Test response generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await litellm.generate_response(
            prompt="Test prompt", context="Test context", model_hint="general"
        )

        assert result == "Generated response"
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_subgoals_json_response(
        self, litellm: "LiteLLM", mock_openai_client: MagicMock
    ) -> None:
        """Test subgoal generation with JSON response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["Goal 1", "Goal 2", "Goal 3"]'
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await litellm.generate_subgoals("Complex goal")

        assert result == ["Goal 1", "Goal 2", "Goal 3"]

    @pytest.mark.asyncio
    async def test_generate_subgoals_dict_response(
        self, litellm: "LiteLLM", mock_openai_client: MagicMock
    ) -> None:
        """Test subgoal generation handles dict responses."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"subgoals": ["A", "B", "C"]}'
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await litellm.generate_subgoals("Complex goal")

        assert result == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_generate_subgoals_fallback_parsing(
        self, litellm: "LiteLLM", mock_openai_client: MagicMock
    ) -> None:
        """Test subgoal generation falls back to line parsing."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "- First goal\n- Second goal\n- Third goal"
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await litellm.generate_subgoals("Complex goal")

        assert len(result) == 3
        assert "First goal" in result[0]

    @pytest.mark.asyncio
    async def test_summarize(
        self, litellm: "LiteLLM", mock_openai_client: MagicMock
    ) -> None:
        """Test text summarization."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Brief summary"
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await litellm.summarize("Long text to summarize...")

        assert result == "Brief summary"
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 150

    @pytest.mark.asyncio
    async def test_speculative_solve_short_draft(
        self, litellm: "LiteLLM", mock_openai_client: MagicMock
    ) -> None:
        """Test speculative solve refines short drafts."""
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="Short"))]),
            MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content="Refined and expanded response")
                    )
                ]
            ),
        ]
        mock_openai_client.chat.completions.create = AsyncMock(side_effect=responses)

        result = await litellm.speculative_solve("Complex prompt")

        assert result == "Refined and expanded response"
        assert mock_openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_speculative_solve_long_draft(
        self, litellm: "LiteLLM", mock_openai_client: MagicMock
    ) -> None:
        """Test speculative solve returns long drafts without refinement."""
        long_response = "A" * 150  # More than 100 chars
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = long_response
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await litellm.speculative_solve("Simple prompt")

        assert result == long_response
        mock_openai_client.chat.completions.create.assert_called_once()
