import uuid
import asyncio
from typing import List, Optional, Dict, Any
from .llm_interface import LLMInterface, LiteLLM
from .memory import FractalMemory
from .knowledge import FractalKnowledgeGraph

class FractalNode:
    def __init__(
        self, 
        goal: str, 
        parent_id: Optional[str] = None, 
        context: str = "",
        llm: Optional[LLMInterface] = None,
        memory: Optional[FractalMemory] = None,
        knowledge: Optional[FractalKnowledgeGraph] = None,
        depth: int = 0,
        max_depth: int = 3,
        task_type: str = "general"
    ):
        self.id = str(uuid.uuid4())
        self.goal = goal
        self.parent_id = parent_id
        self.context = context 
        self.children_ids: List[str] = []
        self.status = "PENDING"
        self.result = ""
        self.summary = ""
        self.depth = depth
        self.max_depth = max_depth
        self.task_type = task_type

        self.llm = llm or LiteLLM()
        self.memory = memory or FractalMemory()
        self.knowledge = knowledge

        self._persist()

    def _persist(self):
        # VRAM Estimation logic
        vram_points = 0
        if self.status == "IN_PROGRESS":
            vram_points = 100 if self.task_type == "reasoning" else 30
        
        state = {
            "id": self.id,
            "goal": self.goal,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "status": self.status,
            "result": self.result,
            "summary": self.summary,
            "depth": self.depth,
            "task_type": self.task_type,
            "vram_points": vram_points
        }
        self.memory.save_node_state(self.id, state)

    def is_complex(self) -> bool:
        return self.depth < self.max_depth

    async def run(self) -> str:
        self.status = "IN_PROGRESS"
        self._persist()
        print(f"[{'  ' * self.depth}] Node {self.id[:4]} ({self.task_type}) START")

        # 1. Hierarchical Knowledge Retrieval
        if self.knowledge:
            retrieved = self.knowledge.query(self.goal)
            if retrieved:
                self.context += f"\n\nRef Knowledge:\n{retrieved}"

        if self.is_complex():
            self.status = "SPLIT"
            subgoals = await self.llm.generate_subgoals(self.goal)
            
            # --- PARALLEL EXECUTION ---
            # Create child nodes
            children = []
            for sg in subgoals:
                child = FractalNode(
                    goal=sg,
                    parent_id=self.id,
                    context=f"Intent: {self.goal}",
                    llm=self.llm,
                    memory=self.memory,
                    knowledge=self.knowledge,
                    depth=self.depth + 1,
                    max_depth=self.max_depth
                )
                self.children_ids.append(child.id)
                children.append(child)
            
            self._persist()

            # Execute all children concurrently
            print(f"[{'  ' * self.depth}] -> Mitosis: Spawning {len(children)} children...")
            child_results = await asyncio.gather(*[child.run() for child in children])
            
            # 2. Parallel Synthesis
            self.result = "Synthesized: " + " | ".join(child_results)
            self.summary = await self.llm.summarize(self.result)
            self.status = "COMPLETED"
            
        else:
            # --- SPECULATIVE SOLVE ---
            # Leaf nodes use the fast model first
            print(f"[{'  ' * self.depth}] -> Solving (Speculative)...")
            self.result = await self.llm.generate_response(self.goal, self.context, model_hint="speculative")
            self.summary = await self.llm.summarize(self.result)
            self.status = "COMPLETED"
            self.memory.store_summary(self.id, self.summary)

        self._persist()
        print(f"[{'  ' * self.depth}] Node {self.id[:4]} DONE.")
        return self.result