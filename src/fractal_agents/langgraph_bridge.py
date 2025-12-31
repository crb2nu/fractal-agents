from typing import Any, Dict

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from fractal_agents.core import FractalNode
from fractal_agents.llm_interface import LiteLLM
from fractal_agents.memory import FractalMemory


class AgentState(TypedDict):
    input: str
    fractal_result: str
    final_output: str

async def fractal_reasoning_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node that delegates a complex sub-problem to Fractal Agents.
    """
    print("--- LangGraph delegating to Fractal Agents ---")
    
    # Initialize Fractal Root
    llm = LiteLLM()
    memory = FractalMemory()
    
    root = FractalNode(
        goal=state["input"],
        llm=llm,
        memory=memory,
        max_depth=2,
        task_type="reasoning"
    )
    
    result = await root.run()
    
    return {"fractal_result": result}

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Simple supervisor to finalize the fractal output."""
    return {"final_output": f"Fractal reasoning complete. Summary: {state['fractal_result'][:200]}"}

# --- Construct the Graph ---

def create_fractal_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("fractal_reasoner", fractal_reasoning_node)
    workflow.add_node("supervisor", supervisor_node)

    workflow.set_entry_point("fractal_reasoner")
    workflow.add_edge("fractal_reasoner", "supervisor")
    workflow.add_edge("supervisor", END)

    return workflow.compile()

if __name__ == "__main__":
    import asyncio
    
    graph = create_fractal_graph()
    input_state = {"input": "Design a fractal-based compression algorithm for LLM KV caches."}
    
    async def run():
        async for output in graph.astream(input_state):
            print(output)

    asyncio.run(run())
