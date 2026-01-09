"""CLI for visualizing Fractal Agents graphs."""

import argparse
import sys
from pathlib import Path

from diagram_gen.analyzers.langgraph import LangGraphAnalyzer
from diagram_gen.renderers.svg import SVGRenderer
from diagram_gen.models import DiagramType

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
    # Optional: graph = graph.filter(...)
    
    print(f"Found {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    
    renderer = SVGRenderer()
    # Force diagram type to generic or module to avoid strict class diagram layout if preferred
    # But SVGRenderer uses hierarchical layout anyway.
    
    diagram = renderer._render_graph(graph, DiagramType.ARCHITECTURE)
    
    diagram.save(args.output)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
