# Agent execution entrypoint

Use the [coordination guide](implementation/README.md), [shared v1 contract](contracts/v1.md), and the individual [frontend](implementation/frontend.md), [backend](implementation/backend.md), and [LLM backend](implementation/llm-backend.md) plans.

The lead owns decisions, contracts, root infrastructure, root lockfiles, integration, and final reporting. Assign one fresh builder per bounded task with write access limited to its service. Builders preserve others' work, report to the lead, and do not spawn children. Testers/reviewers remain read-only; fixes go to the responsible builder.

Follow the personal AGENTS.md visible-team rules when executing: keep the lead in the current tmux window, use main-vertical layout with children stacked on the right, and launch each child as a real Codex session with `codex --profile luna-max --sandbox danger-full-access --ask-for-approval never -C /Users/sarinj/Documents/coding/learning-RAG`. If no tmux session exists, start `codex-team /Users/sarinj/Documents/coding/learning-RAG`. Use `codex-team-message` for follow-ups. Close completed child sessions and start fresh ones for the next task.

First confirm requirements and review v1. Then run foundation, upload, baseline chat, contextualization, hybrid retrieval, and final verification in order. Work within each wave can proceed in parallel once its contracts are ready. The lead integrates a real user flow before advancing. Only the lead installs TS workspace dependencies or changes Git state in a shared checkout.

Track task ID, owner, allowed paths, dependencies, status, evidence, and blockers. Each child returns changed files, checks/results, contract version, integration notes, and remaining issues. A fixture-backed demo is not a passed integration gate.
