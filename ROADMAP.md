# Roadmap: fractal-agents

## Vision

To provide a recursive, self-similar agentic framework ("Fractal Nodes") that enables infinite context depth and self-optimizing task execution on constrained hardware through intelligent state compression and distributed memory.

## Current Status

- **Core**: `FractalNode` implementation with basic recursion.
- **Memory**: Redis-backed "Frozen Memory" for inactive branches.
- **Integration**: `LangGraph` bridge for workflow orchestration.
- **LLM**: LiteLLM integration for model-agnostic inference.

## Immediate Priorities (Q1 2026)

### Core Stability & Resilience

- [x] **Recursive Engine**: Hardening the execution engine with proper error boundaries, timeouts, and infinite loop detection.
- [x] **State Compression**: Implement LZ4 compression for "Frozen Memory" to minimize Redis footprint during deep recursion.
- [ ] **Context Summarization**: Automatic summarization of parent contexts using cheaper models before passing to child nodes.

### Advanced Patterns (Q2 2026)

- [ ] **Multi-Agent Handoffs**: Implement explicit handoff protocols between specialized agents (e.g., 'Researcher' -> 'Coder') using LangGraph-style state transitions.
- [x] **Explicit State Reducers**: Define strict TypedDict schemas for node state and use reducer functions to manage state updates, preventing race conditions in parallel branches.

### Observability

- [ ] **Real-time Visualization**: Export tree state events to `py-observability` to visualize the fractal tree growth in real-time.

## Future Milestones (Q2 2026+)

### Advanced Tooling

- [ ] **MCP Integration**: Native support for `fi-mcp-kit` to allow agents to dynamically discover and bind tools.
- [ ] **Human-in-the-Loop**: "Interrupt" signals to pause execution and request user feedback.

### Scale

- [ ] **Multi-GPU/Node**: Distribute sibling nodes across different GPU workers or Kubernetes pods.
- [ ] **Self-Modification**: Safe sandboxed environment for agents to write and execute ephemeral Python scripts.

## Maintenance

- [ ] Update `langgraph` and `litellm` dependencies.
- [ ] Refine `pydantic` models for state serialization.
