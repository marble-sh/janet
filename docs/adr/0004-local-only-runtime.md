# ADR 0004 — Local-only runtime; hybrid build harness

**Status:** accepted 2026-10-23

## Context

The user selected "hybrid" for inference mode but explicitly disallowed sending
task content or failure traces to remote APIs. Those two choices conflict if
"hybrid" is read as remote runtime inference.

## Decision

The **runtime** is local-only: `llm.mode = "local"`,
`allow_remote_task_data = false`, and the LLM client refuses to send task
content, candidate artifacts, or traces to a non-loopback endpoint. The
**build harness** is hybrid: this Zed session may use remote models to write
code, review designs, and plan. Remote models may see repository code and
architecture documents, but not task content or run traces.

## Consequences

- The loop's throughput is bounded by the local 7B model.
- The judge can be a different family (Llama 3.2) without remote calls.
- Any future remote runtime requires an explicit config change and a new ADR.
