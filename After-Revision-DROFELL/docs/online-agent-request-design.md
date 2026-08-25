# Online Agent/Request design

## Motivation

The former online entry placed the root inference control in the temporary
benchmark Satellite. Every autoregressive round therefore crossed the process
boundary twice:

```text
Client -> Dealer -> Client -> Dealer ...
```

The root control role is now `AgentNode`. A persistent Agent can share a host,
process and thread with a Dealer. An external `RequestNode` starts a session
once; the Agent then feeds each sampled token directly into the next local
Dealer round and streams a copy to Request:

```text
Request -- prefill/session --> Agent -> Dealer -> Agent -> Dealer ...
             Request <------ token --+       <------ token --+
```

The Request/Agent transport remains on the result-stream path, but it is no
longer part of the autoregressive feedback dependency.

## Common interface

`RequestNode` accepts `agent_node: str | None` and exposes the root-control
interface:

- `inspect_fastest_path()`;
- `inspect_fastest_n_path()`;
- `create_infer_control()`;
- `remove_infer_control()`;
- `create_train_control()`;
- `remove_train_control()`;
- the returned controls provide the corresponding step method and `ctrl_id`.

When `agent_node` is `None`, Request constructs `AgentNode` on the supplied
`BaseNode` and directly delegates every operation. Request and Agent then have
one Orbit identity and the behavior is equivalent to the original root-control
path.

When `agent_node` is a node name, path inspection is forwarded to that Agent.
`create_infer_control()` remains synchronous: it returns a local proxy and
defers remote session creation until the proxy's first `infer_step()`.
`create_train_control()` has the same property and defers creation until the
first `train_step()` or `train_step_with_detail()`.

## Remote infer semantics

The first proxy `infer_step()` must be prefill. It sends the control
configuration and prefill input to Agent. Agent creates the real
`InferControl`, performs prefill, then repeatedly calls the local control with
the preceding output token until the sequence stops. Each result is sent to
Request as an ordered event.

Subsequent proxy `infer_step()` calls consume one event each. They retain the
old method signature so existing inference and benchmark loops do not branch on
deployment, but they do not trigger another remote Dealer step. For correctness,
they reject another prefill, a changed `detail_trace` setting, or an integer
input different from the preceding returned token.

Removing a live proxy sends a cancellation message. Agent also cancels all
sessions owned by a Request when that Request disconnects. Completed Agent
controls are released without waiting for Request cleanup.

The current stream uses an unbounded Request queue and has no credit or
backpressure protocol. This is intentional for the current benchmark scope;
slow consumers require a separate flow-control design.

Remote adapter inference is still unsupported. `StartInferRequest` retains the
old in-process `UserModelDataProvider` shape, which may contain a Python
callback and is not a valid cross-process PEFT protocol. The benchmark therefore
allows an external Agent only for decode without adapter data. Training uses the
serializable protocol below.

## Remote train semantics

Request sends a serializable training-control specification containing the
PEFT, optimizer, scheduler, initialization, path and checkpoint-period
configuration. Python callbacks are not part of this interface. Agent creates
and owns the real `TrainControl`; Dealer continues to own the live PEFT and
optimizer tensors, while Agent retains the checkpoint and recovery metadata.

Each minibatch carries an RPC ticket, so concurrently submitted minibatches can
finish out of order without mismatching results. Agent returns only the loss
and optional detail trace in an object compatible with the local
`result.task.loss` and `result.detail_trace` access pattern. Resource-location,
checkpoint and recovery state remain inside Agent.

Unlike autoregressive inference, every training minibatch originates at
Request and therefore crosses the Request/Agent boundary once in each
direction. Dealer outputs do not return to Request between forward and backward
stages: the real root control remains in Agent. Removing the proxy releases the
Agent session, and an Agent removes all sessions owned by a disconnected
Request.

## Colocated Orbit links

`OrbitNetCommander` compares the source and destination `Location`. Equal host,
process, and thread fields select a local link even when `NetModel` requested
TCP. The local writer passes the original packet to the destination context's
`put_packet` method, without creating frames or invoking pickle.

This is object transfer by reference. A sender must not mutate a payload after
`Context.send` while the receiver may still consume it. Nodes in different
processes or threads continue to use serialized transport.

## Benchmark deployment and timing

`launch_network.py --agent NAME DEALER` starts a persistent Agent at the same
process/thread placement as the named Dealer. An independent
`benchmark_infer.py --agent NAME` or `benchmark_train.py --agent NAME`
constructs a remote `RequestNode`. Without `--agent`, either entry constructs
the combined fallback.

The benchmark implementation is identical in both cases. Warmup sessions are
fully consumed and removed. Decode timing starts on Request after prefill and
the configured decode warmup rounds, and stops when Request consumes the last
measured token. No Agent-side clock or benchmark-specific Agent RPC is used.

The remote protocol classes live in `node.agent.protocols`, an importable stable
module, because cross-process Orbit links serialize them with pickle.
