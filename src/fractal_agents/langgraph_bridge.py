from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from fractal_agents.core import FractalNode
from fractal_agents.knowledge import FractalKnowledgeGraph, QdrantFractalStore
from fractal_agents.llm_interface import LiteLLM
from fractal_agents.memory import FractalMemory


class AgentState(TypedDict):
    input: str
    context: str
    triage_decision: str
    fractal_result: str
    final_output: str


async def triage_node(state: AgentState) -> Dict[str, Any]:
    """
    Uses LLM triage to decide routing: direct answer, fractal decomposition, or knowledge lookup.
    """
    llm = LiteLLM()
    triage = await llm.triage_task(state["input"], state.get("context", ""))

    action = triage.get("action", "SOLVE")
    reason = triage.get("reason", "")

    print(f"[Triage] Decision: {action} | Reason: {reason}")

    return {"triage_decision": action}


def route_by_triage(state: AgentState) -> Literal["direct_solve", "fractal_decompose"]:
    """Conditional edge function for routing based on triage decision."""
    if state["triage_decision"] == "SPLIT":
        return "fractal_decompose"
    return "direct_solve"


async def direct_solve_node(state: AgentState) -> Dict[str, Any]:
    """Solves simple tasks directly with a single LLM call."""
    llm = LiteLLM()
    result = await llm.generate_response(
        prompt=state["input"], context=state.get("context", ""), model_hint="general"
    )
    return {"fractal_result": result}


async def fractal_decompose_node(state: AgentState) -> Dict[str, Any]:
    """
    Delegates complex problems to the FractalNode recursive engine.
    """
    print("--- Fractal Decomposition ---")

    llm = LiteLLM()
    memory = FractalMemory()

    root = FractalNode(
        goal=state["input"],
        llm=llm,
        memory=memory,
        context=state.get("context", ""),
        max_depth=3,
        task_type="reasoning",
    )

    result = await root.run()
    return {"fractal_result": result}


async def knowledge_augment_node(state: AgentState) -> Dict[str, Any]:
    """
    Augments the input with relevant context from the Fractal Knowledge Graph.
    """
    try:
        llm = LiteLLM()
        store = QdrantFractalStore()
        kg = FractalKnowledgeGraph(store, llm)

        # Recursive retrieval from the knowledge graph
        knowledge_context = kg.query(state["input"], max_depth=2, threshold=0.6)

        if knowledge_context:
            augmented_context = f"{state.get('context', '')}\n\n[Knowledge]:\n{knowledge_context}"
            return {"context": augmented_context}
    except Exception as e:
        print(f"[Knowledge] Retrieval failed: {e}")

    return {"context": state.get("context", "")}


async def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """Uses LLM synthesis to create a polished final output."""
    llm = LiteLLM()

    final = await llm.synthesize_results(
        goal=state["input"],
        subtasks=[{"goal": "Analysis", "result": state["fractal_result"]}],
        context=state.get("context", ""),
    )

    return {"final_output": final}


def create_fractal_graph(use_knowledge: bool = True):
    """
    Creates the LangGraph workflow with optional knowledge augmentation.

    Flow:
    [knowledge_augment] -> [triage] -> [direct_solve | fractal_decompose] -> [synthesize] -> END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    if use_knowledge:
        workflow.add_node("knowledge_augment", knowledge_augment_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("direct_solve", direct_solve_node)
    workflow.add_node("fractal_decompose", fractal_decompose_node)
    workflow.add_node("synthesize", synthesize_node)

    # Define edges
    if use_knowledge:
        workflow.set_entry_point("knowledge_augment")
        workflow.add_edge("knowledge_augment", "triage")
    else:
        workflow.set_entry_point("triage")

    # Conditional routing based on triage
    workflow.add_conditional_edges(
        "triage",
        route_by_triage,
        {"direct_solve": "direct_solve", "fractal_decompose": "fractal_decompose"},
    )

    workflow.add_edge("direct_solve", "synthesize")
    workflow.add_edge("fractal_decompose", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


if __name__ == "__main__":
    import asyncio

    graph = create_fractal_graph(use_knowledge=False)
    input_state = {
        "input": "Design a fractal-based compression algorithm for LLM KV caches.",
        "context": "",
    }

    async def run():
        async for output in graph.astream(input_state):
            print(output)

    asyncio.run(run())
