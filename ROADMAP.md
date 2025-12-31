# Fractal Agents Roadmap

## Quarter 1 2026: Core Stability & Memory

- [ ] **Recursive Execution Engine**: Finalize the `FractalNode` implementation with proper error boundaries and timeouts.
- [ ] **Redis "Frozen Memory"**: Implement efficient serialization/deserialization of node contexts to/from Redis (lz4 compression).
- [ ] **Context Summarization**: Integrate `litellm` to auto-summarize parent contexts before passing them to child nodes.

## Quarter 2 2026: Tooling & Observation

- [ ] **Real-time Visualization**: Export tree state events to `py-observability` for real-time visualization in `flexdeck`.
- [ ] **Human-in-the-Loop Mode**: Allow an agent to pause and request user approval/clarification via a dedicated "Interrupt" signal.
- [ ] **Standard Tool Registry**: Integrate with `fi-mcp-kit` to allow agents to discover and bind Model Context Protocol tools dynamically.

## Future / Backlog

- [ ] **Multi-GPU Parallelism**: Distribute sibling nodes across different GPU workers.
- [ ] **Self-modifying Code**: Allow agents to write and execute throwaway Python scripts for calculation-heavy subtasks.
