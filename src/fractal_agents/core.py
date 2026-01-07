import asyncio
import uuid
import logging
from typing import List, Optional

from .knowledge import FractalKnowledgeGraph
from .llm_interface import LiteLLM, LLMInterface
from .memory import FractalMemory
from .state import NodeState, reduce_node_state

# Global safety limit to prevent infinite recursion bugs
GLOBAL_MAX_DEPTH = 50

logger = logging.getLogger(__name__)

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
        task_type: str = "general",
        timeout: int = 300,  # Default 5 minutes for recursion chain
    ):
        self.id = str(uuid.uuid4())
        self.goal = goal
        self.parent_id = parent_id
        self.context = context
        
        # Initialize State using TypedDict
        self.state: NodeState = {
            "id": self.id,
            "goal": self.goal,
            "parent_id": self.parent_id,
            "children_ids": [],
            "status": "PENDING",
            "result": "",
            "summary": "",
            "depth": depth,
            "task_type": task_type,
            "vram_points": 0
        }
        
        self.depth = depth
        self.max_depth = min(max_depth, GLOBAL_MAX_DEPTH)
        self.task_type = task_type
        self.timeout = timeout

        self.llm = llm or LiteLLM()
        self.memory = memory or FractalMemory()
        self.knowledge = knowledge

        self._persist()

    def _dispatch(self, action_type: str, payload: dict = None):
        """Apply a state change via the reducer."""
        if payload is None:
            payload = {}
        self.state = reduce_node_state(self.state, {"type": action_type, "payload": payload})
        self._persist()

    def _persist(self):
        # VRAM Estimation logic (computed during persist or reducer? Reducer is purer)
        # We'll update VRAM points based on status in reducer or here.
        # Ideally, it should be in the reducer, but for now we keep simple logic.
        if self.state["status"] == "IN_PROGRESS":
            self.state["vram_points"] = 100 if self.task_type == "reasoning" else 30
        
        try:
            self.memory.save_node_state(self.id, self.state)
            self.memory.publish_event("fractal:updates", self.state)
        except Exception as e:
            logger.error(f"Failed to persist node {self.id}: {e}")

    def is_complex(self) -> bool:
        return self.depth < self.max_depth

    async def run(self) -> str:
        try:
            return await asyncio.wait_for(self._execute(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self._dispatch("FAIL", {"error": f"Execution timed out after {self.timeout}s"})
            logger.warning(f"[{'  ' * self.depth}] Node {self.id[:4]} TIMEOUT")
            return self.state["result"]
        except asyncio.CancelledError:
            self._dispatch("CANCEL")
            logger.warning(f"[{'  ' * self.depth}] Node {self.id[:4]} CANCELLED")
            raise
        except Exception as e:
            self._dispatch("FAIL", {"error": f"Execution failed: {str(e)}"})
            logger.error(f"[{'  ' * self.depth}] Node {self.id[:4]} ERROR: {e}")
            return self.state["result"]

    async def _execute(self) -> str:
        self._dispatch("START")
        print(f"[{'  ' * self.depth}] Node {self.id[:4]} ({self.task_type}) START")

        child_tasks = []

        try:
            # 1. Hierarchical Knowledge Retrieval
            if self.knowledge:
                try:
                    retrieved = self.knowledge.query(self.goal)
                    if retrieved:
                        self.context += f"\n\nRef Knowledge:\n{retrieved}"
                except Exception as e:
                    logger.warning(f"Knowledge retrieval failed: {e}")

            if self.is_complex():
                self._dispatch("SPLIT", {}) # Status update only for now
                subgoals = await self.llm.generate_subgoals(self.goal)

                # Summarize parent context for children if it's too long
                summarized_context = self.context
                if len(self.context) > 1000:
                    print(f"[{'  ' * self.depth}] Compressing context for children...")
                    summarized_context = await self.llm.summarize(self.context)

                # Create child nodes
                children = []
                children_ids = []
                for sg in subgoals:
                    child = FractalNode(
                        goal=sg,
                        parent_id=self.id,
                        context=f"Intent: {self.goal}\nContext Snippet: {summarized_context}",
                        llm=self.llm,
                        memory=self.memory,
                        knowledge=self.knowledge,
                        depth=self.depth + 1,
                        max_depth=self.max_depth,
                        timeout=max(10, self.timeout - 10),  # Adjust child timeout
                    )
                    children_ids.append(child.id)
                    children.append(child)

                self._dispatch("SPLIT", {"children": children_ids})

                # Execute all children concurrently
                print(f"[{'  ' * self.depth}] -> Mitosis: Spawning {len(children)} children...")

                # Store tasks to allow cancellation
                child_tasks = [asyncio.create_task(child.run()) for child in children]
                
                # Run children with exception handling
                child_results = await asyncio.gather(*child_tasks, return_exceptions=True)

                valid_results = []
                failures = []

                for i, res in enumerate(child_results):
                    if isinstance(res, Exception):
                        failures.append(f"Child {children[i].id[:4]} failed: {str(res)}")
                    else:
                        valid_results.append(res)

                # 2. Parallel Synthesis
                final_result = ""
                if failures:
                    # If some failed, we mention it in the result but try to synthesize what we have
                    print(f"[{'  ' * self.depth}] WARNING: {len(failures)} children failed.")
                    final_result = (
                        "Synthesized (Partial): "
                        + " | ".join(valid_results)
                        + f" | Failures: {'; '.join(failures)}"
                    )
                else:
                    final_result = "Synthesized: " + " | ".join(valid_results)

                # If no valid results, mark as failed
                if not valid_results and failures:
                    self._dispatch("FAIL", {"error": f"All subtasks failed: {'; '.join(failures)}"})
                else:
                    summary = await self.llm.summarize(final_result)
                    self._dispatch("COMPLETE", {"result": final_result, "summary": summary})

            else:
                # --- SPECULATIVE SOLVE ---
                # Leaf nodes use the fast model first
                print(f"[{'  ' * self.depth}] -> Solving (Speculative)...")
                result = await self.llm.generate_response(
                    self.goal, self.context, model_hint="speculative"
                )
                summary = await self.llm.summarize(result)
                self._dispatch("COMPLETE", {"result": result, "summary": summary})
                try:
                    self.memory.store_summary(self.id, summary)
                except Exception:
                    pass

            print(f"[{'  ' * self.depth}] Node {self.id[:4]} DONE.")
            return self.state["result"]

        except asyncio.CancelledError:
            # Propagate cancellation to children
            if child_tasks:
                print(f"[{'  ' * self.depth}] Cancelling {len(child_tasks)} children...")
                for task in child_tasks:
                    task.cancel()
                await asyncio.gather(*child_tasks, return_exceptions=True)
            raise
