from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json
from openai import OpenAI

class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: str = "", model_hint: str = "general") -> str:
        """Generates a text response based on the prompt and context.
           model_hint: 'general', 'reasoning', 'vision', 'speculative', 'fast'
        """
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
    def generate_response(self, prompt: str, context: str = "", model_hint: str = "general") -> str:
        return f"[Mock {model_hint}] Solved: {prompt}"

    def generate_subgoals(self, goal: str, num_subgoals: int = 2) -> List[str]:
        return [f"Subtask 1 for {goal}", f"Subtask 2 for {goal}"]

    def summarize(self, text: str) -> str:
        return f"[Summary] {text[:50]}..."

class LiteLLM(LLMInterface):
    """
    Interface for local LiteLLM service with specialized model routing.
    
    Model Map:
    - reasoning -> nemotron3-nano (cot, agent-reasoning)
    - vision -> qwen3-vl-8b (vision, ocr)
    - speculative/fast -> qwen2.5-7b-spec (speculative, fast-text)
    - general -> qwen2.5-7b (textgen)
    """
    def __init__(self, api_base: str = None, api_key: str = None):
        self.api_base = api_base or os.getenv("LITELLM_API_BASE", "http://litellm.ai.svc.cluster.local:8000/v1")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-litellm-local")
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Map logical intents to specific model aliases found in the cluster
        self.model_map = {
            "general": "qwen2.5-7b",    # qwen2.5-7b
            "reasoning": "reasoning",   # nemotron3-nano
            "vision": "vision",         # qwen3-vl-8b
            "speculative": "qwen2.5-7b", 
            "fast": "qwen2.5-7b",
            "summary": "qwen2.5-7b"
        }

    def _get_model(self, hint: str) -> str:
        return self.model_map.get(hint, self.model_map["general"])

    def generate_response(self, prompt: str, context: str = "", model_hint: str = "general") -> str:
        model = self._get_model(model_hint)
        
        # Add Chain-of-Thought instruction if reasoning model is selected
        system_prompt = "You are a helpful recursive agent node. Solve the task given the context."
        if model_hint == "reasoning":
            system_prompt += " Think step-by-step. Show your reasoning."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nTask: {prompt}"}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def generate_subgoals(self, goal: str, num_subgoals: int = 3) -> List[str]:
        # Use reasoning model for decomposition as it requires logic
        model = self._get_model("reasoning")
        
        prompt = (
            f"Break down the following complex goal into {num_subgoals} distinct, sequential sub-goals. "
            f"Return ONLY a JSON list of strings, e.g. [\"subgoal 1\", \"subgoal 2\"].\n\nGoal: {goal}"
        )
        
        messages = [
            {"role": "system", "content": "You are a task decomposition engine. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            # Note: Not all local models support response_format={"type": "json_object"} perfectly,
            # but we try. If it fails, the fallback parsing logic handles it.
            extra_body={"response_format": {"type": "json_object"}}
        )
        content = response.choices[0].message.content.strip()
        
        # Robust Parsing
        try:
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end != -1:
                return json.loads(content[start:end])
            
            data = json.loads(content)
            if isinstance(data, list): return data
            if isinstance(data, dict): return list(data.values())[0] if data else []
            return [content]
        except:
            return [line.strip("- *") for line in content.split("\n") if line.strip()]

    def summarize(self, text: str) -> str:
        # Use fast/speculative model for summarization
        model = self._get_model("summary") # or "speculative"
        
        messages = [
            {"role": "system", "content": "Summarize the following text efficiently, capturing the key outcome."},
            {"role": "user", "content": text}
        ]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
