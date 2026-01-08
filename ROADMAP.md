# Roadmap: fractal-agents

## Vision

To provide a recursive, self-similar agentic framework ("Fractal Nodes") that enables infinite context depth and self-optimizing task execution on constrained hardware through intelligent state compression and distributed memory.

## Current Status (v0.2.0)

- **Core**: `FractalNode` implementation with dynamic triage and cohesive synthesis.
- **Memory**: Redis-backed "Frozen Memory" with LZ4 compression.
- **Integration**: `LangGraph` bridge for workflow orchestration.
- **LLM**: LiteLLM integration for model-agnostic inference.
- **HITL**: Human-in-the-Loop interrupt system for user feedback.
- **Observability**: Tree event emission and AIMetrics integration.

## Completed (Q1 2026)

### Core Stability & Resilience

- [x] **Recursive Engine**: Hardening the execution engine with proper error boundaries, timeouts, and infinite loop detection.
- [x] **State Compression**: Implement LZ4 compression for "Frozen Memory" to minimize Redis footprint during deep recursion.
- [x] **Context Summarization**: Automatic summarization of parent contexts using cheaper models before passing to child nodes.
- [x] **Explicit State Reducers**: Define strict TypedDict schemas for node state and use reducer functions.

### Observability

- [x] **Real-time Visualization**: Export tree state events via `FractalMetrics` for visualizing fractal tree growth.
- [x] **AIMetrics Integration**: Optional integration with `py-observability` for LLM token tracking.

### Human-in-the-Loop

- [x] **Interrupt Signals**: `InterruptManager` to pause execution and request user feedback.
- [x] **Callback Mechanism**: Register async callbacks for handling interrupt requests.

## Future Milestones (Q2 2026+)

### Advanced Patterns

- [ ] **Multi-Agent Handoffs**: Implement explicit handoff protocols between specialized agents (e.g., 'Researcher' -> 'Coder') using LangGraph-style state transitions.

### Advanced Tooling

- [ ] **MCP Integration**: Native support for `fi-mcp-kit` to allow agents to dynamically discover and bind tools.
- [ ] **Self-Modification**: Safe sandboxed environment for agents to write and execute ephemeral Python scripts.

### Scale

- [ ] **Multi-GPU/Node**: Distribute sibling nodes across different GPU workers or Kubernetes pods.

## Maintenance

- [x] Update `langgraph` and `litellm` dependencies.
- [x] Refine `pydantic` models for state serialization.
