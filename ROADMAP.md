# fractal-agents Roadmap

> Last Updated: 2026-07-02
> Tier: 2 (see workspace AGENTS.md "Portfolio Tiers")
> Tracking Issue: none — maintain-mode; backlog = [open issues](https://gitlab.flexinfer.ai/libs/fractal-agents/-/issues)

## Current Status

Recursive "fractal node" agent framework (v0.2.0): FractalNode engine with triage/synthesis, Redis-backed LZ4-compressed "frozen memory", LangGraph bridge, LiteLLM model-agnostic inference, human-in-the-loop interrupts, and tree-event/AIMetrics observability. Maintain-mode: last organic change 2026-01-07 (Harbor publish variable fixes); 2026-03 commits are CI fixes (docker publish tag-gating, ruff lint) and the 2026-03-17 workspace-wide tech-radar CI bulk change. Default branch is `master`.

- **Plan store**: plan-workspace-portfolio-refresh-2026-h2-roadmaps-quality-baselin-f3db23 (this refresh; no repo-specific active plan)
- **Deployed**: not deployed (library; docker publish is tag-gated only)
- **CI**: python template family (platform/gitops `ci/templates/python.yml` + tech-radar `radar.yml`)

## Now

- Maintenance only — dependency and CI fixes as needed; no active feature work.

## Next

- [ ] Pre-commit hooks + standard Makefile targets if flagged at claim time by portfolio-refresh slice 7 (quality gate wave B, pending)

## Later

Directional themes salvaged from the 2026-01 roadmap (unfunded; promote to issues if picked up):

- Multi-agent handoff protocols between specialized agents
- fi-mcp-kit integration for dynamic tool discovery
- Distributing sibling nodes across GPU workers/pods

## Backlog

Full backlog: [P1](https://gitlab.flexinfer.ai/libs/fractal-agents/-/issues/?label_name[]=P1) · [P2](https://gitlab.flexinfer.ai/libs/fractal-agents/-/issues/?label_name[]=P2) · [P3](https://gitlab.flexinfer.ai/libs/fractal-agents/-/issues/?label_name[]=P3) · [all open](https://gitlab.flexinfer.ai/libs/fractal-agents/-/issues)
