import os
import json
import asyncio
from typing import List, Dict, Any
from fractal_agents.core import FractalNode
from fractal_agents.llm_interface import LiteLLM
from fractal_agents.memory import FractalMemory
from fractal_agents.knowledge import FractalKnowledgeGraph, QdrantFractalStore, KnowledgeNode

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
        for root, dirs, files in os.walk(self.project_path):
            if ".git" in root or "__pycache__" in root: continue
            
            for file in files:
                if file.endswith((".py", ".gd", ".yaml", ".json")):
                    path = os.path.join(root, file)
                    with open(path, "r") as f:
                        content = f.read()
                    
                    # Store top-level file info
                    node = KnowledgeNode(
                        content=f"File: {path}\nContent Snippet: {content[:500]}",
                        metadata={"path": path, "type": "code"}
                    )
                    # Get embedding via KG utility
                    vector = self.kg._get_embedding(node.content)
                    self.knowledge_store.add_node(node, vector)
        print("Indexing Complete.")

    async def implement_feature(self, requirement: str):
        print(f"Implementing Feature: {requirement}")
        
        # Start Fractal Process with KG enabled
        root = FractalNode(
            goal=f"Implement feature: {requirement} in project {self.project_path}",
            llm=self.llm,
            memory=self.memory,
            knowledge=self.kg,
            max_depth=2,
            task_type="reasoning"
        )
        
        result = await root.run()
        print("\n--- Final Implementation Plan ---")
        print(result)

async def main():
    agent = CodebaseAgent(".")
    # 1. Index the fractal-agents codebase itself
    await agent.index_codebase()
    
    # 2. Parallel Fractal Reasoning
    await agent.implement_feature("Add a retry decorator to the LiteLLM class for robust API calls.")

if __name__ == "__main__":
    asyncio.run(main())
