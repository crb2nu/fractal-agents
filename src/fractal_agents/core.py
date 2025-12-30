import uuid
from typing import List, Optional, Dict, Any
from .llm_interface import LLMInterface, MockLLM
from .memory import FractalMemory

class FractalNode:
    def __init__(
        self, 
        goal: str, 
        parent_id: Optional[str] = None, 
        context: str = "",
        llm: Optional[LLMInterface] = None,
        memory: Optional[FractalMemory] = None,
        depth: int = 0,
        max_depth: int = 3
    ):
        self.id = str(uuid.uuid4())
        self.goal = goal
        self.parent_id = parent_id
        self.context = context # Context from parent/siblings
        self.children_ids: List[str] = []
        self.status = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, SPLIT
        self.result = ""
        self.summary = ""
        self.depth = depth
        self.max_depth = max_depth

        self.llm = llm or MockLLM()
        self.memory = memory or FractalMemory()

        # Save initial state
        self._persist()

    def _persist(self):
        state = {
            "id": self.id,
            "goal": self.goal,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "status": self.status,
            "result": self.result,
            "summary": self.summary,
            "depth": self.depth
        }
        self.memory.save_node_state(self.id, state)

    def is_complex(self) -> bool:
        # Heuristic: If depth is low, assume complex. 
        # Real implementation would check token count or ask LLM.
        return self.depth < self.max_depth

    def run(self) -> str:
        print(f"[{'  ' * self.depth}] Node {self.id[:4]} processing: {self.goal}")
        self.status = "IN_PROGRESS"
        self._persist()

        if self.is_complex():
            print(f"[{'  ' * self.depth}] -> Goal is complex. Splitting (Mitosis)...")
            self.status = "SPLIT"
            subgoals = self.llm.generate_subgoals(self.goal)
            
            results = []
            previous_sibling_summary = ""

            for sg in subgoals:
                # Create child with condensed context:
                # Parent Intent + Summary of previous sibling's work
                child_context = f"Parent Goal: {self.goal}. Previous Progress: {previous_sibling_summary}"
                
                child = FractalNode(
                    goal=sg,
                    parent_id=self.id,
                    context=child_context,
                    llm=self.llm,
                    memory=self.memory,
                    depth=self.depth + 1,
                    max_depth=self.max_depth
                )
                self.children_ids.append(child.id)
                self._persist()
                
                # Execute child (Depth-First Traversal)
                child_result = child.run()
                results.append(child_result)
                
                # Update sibling context for the next loop
                previous_sibling_summary = child.summary

            # Synthesize results
            self.result = f"Synthesized result of {len(results)} subtasks."
            self.summary = self.llm.summarize(self.result)
            self.status = "COMPLETED"
            
        else:
            print(f"[{'  ' * self.depth}] -> Goal is simple. Solving directly.")
            # Solve directly
            self.result = self.llm.generate_response(self.goal, self.context)
            self.summary = self.llm.summarize(self.result)
            self.status = "COMPLETED"
            self.memory.store_summary(self.id, self.summary)

        self._persist()
        print(f"[{'  ' * self.depth}] Node {self.id[:4]} Finished.")
        return self.result
