import uuid
from typing import List, Optional, Dict, Any
from .llm_interface import LLMInterface, MockLLM, LiteLLM
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
        max_depth: int = 3,
        task_type: str = "general" # general, reasoning, vision, speculative
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

        self.llm = llm or MockLLM()
        self.memory = memory or FractalMemory()

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
            "depth": self.depth,
            "task_type": self.task_type
        }
        self.memory.save_node_state(self.id, state)

    def is_complex(self) -> bool:
        return self.depth < self.max_depth

    def run(self) -> str:
        print(f"[{'  ' * self.depth}] Node {self.id[:4]} ({self.task_type}) processing: {self.goal}")
        self.status = "IN_PROGRESS"
        self._persist()

        if self.is_complex():
            print(f"[{'  ' * self.depth}] -> Goal is complex. Splitting (Mitosis)...")
            self.status = "SPLIT"
            subgoals = self.llm.generate_subgoals(self.goal)
            
            results = []
            sibling_summaries = []

            for i, sg in enumerate(subgoals):
                progress_context = " ".join(sibling_summaries)
                child_context = f"Parent Goal: {self.goal}. Previous Steps Completed: {progress_context}"
                
                # Determine child task type (simple heuristic)
                child_type = "general"
                lower_sg = sg.lower()
                if "reason" in lower_sg or "plan" in lower_sg or "analy" in lower_sg:
                    child_type = "reasoning"
                elif "image" in lower_sg or "visual" in lower_sg or "look" in lower_sg:
                    child_type = "vision"
                elif "quick" in lower_sg or "fast" in lower_sg:
                    child_type = "speculative"

                child = FractalNode(
                    goal=sg,
                    parent_id=self.id,
                    context=child_context,
                    llm=self.llm,
                    memory=self.memory,
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    task_type=child_type
                )
                self.children_ids.append(child.id)
                self._persist()
                
                child_result = child.run()
                results.append(child_result)
                
                if child.summary:
                    sibling_summaries.append(f"Step {i+1} Summary: {child.summary}")

            self.result = f"Synthesized result of {len(results)} subtasks: " + " ".join(results)
            self.summary = self.llm.summarize(self.result)
            self.status = "COMPLETED"
            
        else:
            print(f"[{'  ' * self.depth}] -> Goal is simple. Solving directly with {self.task_type} model.")
            self.result = self.llm.generate_response(self.goal, self.context, model_hint=self.task_type)
            self.summary = self.llm.summarize(self.result)
            self.status = "COMPLETED"
            self.memory.store_summary(self.id, self.summary)

        self._persist()
        print(f"[{'  ' * self.depth}] Node {self.id[:4]} Finished.")
        return self.result
