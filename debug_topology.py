from pathlib import Path
from diagram_gen.analyzers.langgraph import LangGraphAnalyzer
from diagram_gen.models import NodeType

analyzer = LangGraphAnalyzer()
path = Path("src/fractal_agents/langgraph_bridge.py")
graph = analyzer.analyze(path)

print(f"Total Nodes: {len(graph.nodes)}")
print("\n--- Workflow Components ---")
for n in graph.nodes.values():
    if n.type == NodeType.COMPONENT:
        print(f"Workflow: {n.name}")
        
print("\n--- Agent/Graph Nodes ---")
for n in graph.nodes.values():
    if n.metadata.get("workflow"):
        print(f"Node: {n.name} (Part of {n.metadata['workflow'].split(':')[-1]})")

print("\n--- Other Nodes (Noise?)")
for n in graph.nodes.values():
    if n.type not in (NodeType.COMPONENT,) and not n.metadata.get("workflow"):
        print(f"{n.type.value}: {n.name}")
