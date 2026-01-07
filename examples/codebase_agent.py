import asyncio
import json
import os

from fractal_agents.core import FractalNode
from fractal_agents.knowledge import FractalKnowledgeGraph, KnowledgeNode, QdrantFractalStore
from fractal_agents.llm_interface import LiteLLM
from fractal_agents.memory import FractalMemory


class CodebaseAgent:
    """
    A specialized agent for codebase exploration and implementation.
    Uses the Fractal Knowledge Graph to index file structures.
    """

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.llm = LiteLLM()
        self.memory = FractalMemory()

        # Init KG
        self.knowledge_store = QdrantFractalStore()
        self.kg = FractalKnowledgeGraph(store=self.knowledge_store, llm_client=self.llm)

    async def index_codebase(self):
        """Recursively scan files and add to FKG."""
        print(f"Indexing Codebase: {self.project_path}")
        for root, _, files in os.walk(self.project_path):
            if ".git" in root or "__pycache__" in root:
                continue

            for file in files:
                if file.endswith((".py", ".gd", ".yaml", ".json")):
                    path = os.path.join(root, file)
                    with open(path, "r") as f:
                        content = f.read()

                    # Store top-level file info
                    node = KnowledgeNode(
                        content=f"File: {path}\nContent Snippet: {content[:500]}",
                        metadata={"path": path, "type": "code"},
                    )
                    # Get embedding via KG utility
                    vector = self.kg._get_embedding(node.content)
                    self.knowledge_store.add_node(node, vector)
        print("Indexing Complete.")

    async def implement_feature(self, requirement: str):
        print(f"Implementing Feature: {requirement}")

        # Start Fractal Process with KG enabled
        root = FractalNode(
            goal=(
                f"Analyze the codebase at {self.project_path}. "
                "Return the final code changes in a JSON format: "
                "{'files': [{'path': str, 'content': str}]}"
            ),
            llm=self.llm,
            memory=self.memory,
            knowledge=self.kg,
            max_depth=2,
            task_type="reasoning",
        )

        result = await root.run()
        print("\n--- Final Implementation Plan ---")

        # 3. APPLY CHANGES Phase
        await self.apply_changes(result)

    async def apply_changes(self, result_text: str):
        print("Applying changes to codebase...")
        try:
            # Extract JSON from the synthesis result
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            data = json.loads(result_text[start:end])

            for file_change in data.get("files", []):
                path = file_change["path"]
                content = file_change["content"]

                # Safety: Ensure path is within project
                full_path = os.path.abspath(path)
                if not full_path.startswith(os.path.abspath(self.project_path)):
                    print(f"Skipping unsafe path: {path}")
                    continue

                print(f"Writing file: {path}")
                with open(full_path, "w") as f:
                    f.write(content)

            print("Successfully updated codebase.")
        except Exception as e:
            print(f"Failed to apply changes: {e}")
            print("Raw Result was:", result_text[:200], "...")


async def main():
    agent = CodebaseAgent(".")
    # 1. Index the fractal-agents codebase
    await agent.index_codebase()

    # 2. Parallel Fractal Reasoning to generate tests
    goal = (
        "Create a comprehensive unit test suite for the FractalNode class in 'tests/test_core.py'. "
        "The tests should cover initialization, mitosis (splitting), and result synthesis. "
        "Use pytest. Mock the LLM and Memory components to avoid real API calls during tests."
    )
    await agent.implement_feature(goal)


if __name__ == "__main__":
    asyncio.run(main())
