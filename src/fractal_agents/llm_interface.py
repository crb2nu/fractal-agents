import json
import os
from abc import ABC, abstractmethod
from typing import List

from openai import AsyncOpenAI


class LLMInterface(ABC):
    @abstractmethod
    async def generate_response(
        self, prompt: str, context: str = "", model_hint: str = "general"
    ) -> str:
        pass

    @abstractmethod
    async def triage_task(self, goal: str, context: str = "") -> dict:
        pass

    @abstractmethod
    async def synthesize_results(self, goal: str, subtasks: List[dict], context: str = "") -> str:
        pass

    @abstractmethod
    async def generate_subgoals(self, goal: str, num_subgoals: int = 3) -> List[str]:
        pass

    @abstractmethod
    async def summarize(self, text: str, model_hint: str = "summary") -> str:
        pass


class LiteLLM(LLMInterface):
    def __init__(self, api_base: str = None, api_key: str = None):
        self.api_base = api_base or os.getenv(
            "LITELLM_API_BASE", "http://litellm.ai.svc.cluster.local:8000/v1"
        )
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-litellm-local")
        # Use Async Client for parallelism
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)

        self.model_map = {
            "general": "qwen2.5-7b",
            "reasoning": "reasoning",
            "vision": "vision",
            "speculative": "qwen2.5-7b",  # Speculative alias
            "fast": "qwen2.5-7b",
            "summary": "qwen2.5-7b",
        }

    def _get_model(self, hint: str) -> str:
        return self.model_map.get(hint, self.model_map["general"])

    async def generate_response(
        self, prompt: str, context: str = "", model_hint: str = "general"
    ) -> str:
        model = self._get_model(model_hint)
        system_prompt = "You are a helpful recursive agent node."
        if model_hint == "reasoning":
            system_prompt += " Think step-by-step. Show your reasoning."

        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nTask: {prompt}"},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    async def speculative_solve(self, prompt: str, context: str = "") -> str:
        """
        Intelligent Speculative Decode:
        1. Draft with 'speculative' model (Fast).
        2. Review with 'reasoning' model (Slow/Deep) ONLY if complexity is high.
        """
        # Step 1: Rapid Draft
        draft = await self.generate_response(prompt, context, model_hint="speculative")

        # Step 2: Complexity Check (Is the draft sufficient?)
        # For simplicity, we assume if it's < 100 chars, it might need more thought.
        if len(draft) < 100:
            return await self.generate_response(
                f"Refine this draft: {draft}", context, model_hint="reasoning"
            )

        return draft

    async def triage_task(self, goal: str, context: str = "") -> dict:
        """
        Decides whether to SOLVE or SPLIT.
        Returns: {"action": "SOLVE"|"SPLIT", "reason": str, "num_subgoals": int}
        """
        model = self._get_model("reasoning")
        prompt = (
            "Triage the following goal based on the provided context.\n"
            "Determine if the task can be solved directly or if it needs to be split "
            "into sub-tasks.\n"
            "If it needs splitting, specify the number of sub-tasks (2-5).\n"
            "Return ONLY a JSON object: "
            '{"action": "SOLVE"|"SPLIT", "reason": "str", "num_subgoals": int}\n\n'
            f"Goal: {goal}\n"
            f"Context: {context[:500]}"
        )
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content.strip())

    async def generate_subgoals(self, goal: str, num_subgoals: int = 3) -> List[str]:
        model = self._get_model("reasoning")
        prompt = (
            f"Break down the following complex goal into {num_subgoals} "
            f"distinct, sequential sub-goals. "
            f"Return ONLY a JSON list of strings.\n\nGoal: {goal}"
        )
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else list(data.values())[0]
        except Exception:
            # Fallback to line parsing
            return [line.strip("- *") for line in content.split("\n") if line.strip()][
                :num_subgoals
            ]

    async def synthesize_results(self, goal: str, subtasks: List[dict], context: str = "") -> str:
        """
        Merges results from multiple sub-tasks into a single cohesive response.
        """
        model = self._get_model("reasoning")
        results_summary = "\n".join([f"- {t['goal']}: {t['result']}" for t in subtasks])
        prompt = (
            f"You are synthesizing the results of sub-tasks for the following goal: {goal}\n"
            f"Context: {context[:500]}\n\n"
            f"Sub-task results:\n{results_summary}\n\n"
            f"Write a comprehensive and cohesive final response that solves the original goal."
        )
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    async def summarize(self, text: str, model_hint: str = "summary") -> str:
        model = self._get_model(model_hint)
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize efficiently."},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
