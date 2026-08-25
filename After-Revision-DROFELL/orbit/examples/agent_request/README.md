# Agent/Request streaming self-test

This example verifies the two `RequestNode` modes without loading a model or
using a GPU:

- a remote Request creates a proxy control, starts it with prefill, and consumes
  one streamed token per `infer_step` call while the Agent runs autonomously;
- a Request with `agent_node=None` constructs an Agent on its own `BaseNode` and
  delegates through the former Client interface.

The remote Request uses another Orbit thread, so its protocol crosses the
serialized transport rather than the same-thread local link.

Run from the repository root with the project environment:

```bash
python -m orbit.examples.agent_request.main
```
