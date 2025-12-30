from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generates a text response based on the prompt and context."""
        pass

    @abstractmethod
    def generate_subgoals(self, goal: str, num_subgoals: int = 3) -> List[str]:
        """Decomposes a complex goal into subgoals."""
        pass

    @abstractmethod
    def summarize(self, text: str) -> str:
        """Compresses text into a summary."""
        pass

class MockLLM(LLMInterface):
    """A Mock LLM for testing the fractal structure without an API."""
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        return f"[Mock Response] Solved: {prompt}"

    def generate_subgoals(self, goal: str, num_subgoals: int = 2) -> List[str]:
        return [f"Subtask 1 for {goal}", f"Subtask 2 for {goal}"]

    def summarize(self, text: str) -> str:
        return f"[Summary] {text[:50]}..."
