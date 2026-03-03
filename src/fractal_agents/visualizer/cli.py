"""CLI for visualizing Fractal Agents graphs."""

import argparse
import sys
from pathlib import Path

from diagram_gen.analyzers.langgraph import LangGraphAnalyzer
from diagram_gen.models import DiagramType, NodeType
from diagram_gen.renderers.svg import SVGRenderer


def main():
    parser = argparse.ArgumentParser(description="Visualize Fractal Agents / LangGraph workflows")
    parser.add_argument("file", help="Python file containing the graph definition")
    parser.add_argument("--output", "-o", default="graph.svg", help="Output SVG path")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {file_path} not found", file=sys.stderr)
        sys.exit(1)
        
    print(f"Analyzing {file_path}...")
    
    analyzer = LangGraphAnalyzer()
    graph = analyzer.analyze(file_path)
    
    # Filter to only keep workflow components and related classes
    # We want: COMPONENT (Workflow), Nodes added to workflow, and INTERFACE (Start)
    def is_topology_node(node):
        if node.type in (NodeType.COMPONENT, NodeType.INTERFACE):
            return True
        if node.metadata.get("workflow"):
            return True
        # Also keep END node if we have one (usually inferred or Class type)
        if node.name == "END":
            return True
        return False

    filtered_nodes = {nid: n for nid, n in graph.nodes.items() if is_topology_node(n)}
    
    # Rebuild edges
    filtered_edges = [
        e for e in graph.edges 
        if e.source_id in filtered_nodes and e.target_id in filtered_nodes
    ]
    
    # Replace graph data
    graph.nodes = filtered_nodes
    graph.edges = filtered_edges
    
    print(f"Found {len(graph.nodes)} nodes and {len(graph.edges)} edges (Topology Only).")
    
    renderer = SVGRenderer()
    # Force diagram type to generic or module to avoid strict class diagram layout if preferred
    # But SVGRenderer uses hierarchical layout anyway.
    
    diagram = renderer._render_graph(graph, DiagramType.ARCHITECTURE)
    
    diagram.save(args.output)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
