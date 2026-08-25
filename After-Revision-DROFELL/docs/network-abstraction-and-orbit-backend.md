# Network Abstraction and the Default Orbit Backend

## 1. Layering principle

DROFELL job controls access network capabilities through `Context` / `BaseNode`. Orbit provides the default backend, but control, Binding, GPU-memory scheduling, and recovery logic do not depend directly on Orbit tracker's internals.

Keep three questions separate:

- Who decides the upper-level task graph, paths, and recovery?
- How does the default backend discover nodes and establish connections?
- Does a concrete deployment provide NAT handling, relaying, replication, or network-partition semantics?

## 2. Minimal communication contract

Upper layers rely on the ability to:

- register a node at an addressable location;
- resolve a logical name into a location;
- establish, cache, and close connections;
- serialize, send, receive, and deliver messages to a target node;
- report connection establishment, disconnects, and send failures;
- support colocated and remote placements through one interface;
- expose round-trip latency or bandwidth observations for path probing.

A replacement backend must preserve this behavior and its failure semantics.

## 3. Components of the default Orbit backend

The default deployment contains:

- a central Orbit service for deployment control, discovery, connection coordination, and logs;
- Satellite/Context instances that host nodes and join the central service;
- link extensions that establish direct data paths between discovered nodes;
- `Location`, which describes host/process/thread/coroutine placement.

The central service does not receive model tensors for inference or training and does not maintain job Shadow state, Bindings, or a Memory Digital Twin.

## 4. Aux versus the Orbit central service

They are different roles:

| Component | Responsibility |
|---|---|
| AuxNode | DROFELL resource metadata such as providers, TPS, and Backbones |
| Orbit service | node registration, location, deployment, connection setup, and logging |

Aux participates in Dealer resource discovery. Orbit provides infrastructure for the communication backend. Replacing Orbit does not automatically replace Aux, and vice versa.

## 5. Direct links and direction restrictions

Nodes normally discover one another through the central service and then establish direct links through a link extension. The default implementation assumes that the deployment allows TCP connections in the required direction.

The default backend does not contain a general NAT-traversal or relay protocol. An environment that only permits outbound connections should provide a relay, overlay, or NAT-aware transport that satisfies the Context contract.

## 6. Path and resource decisions

Orbit tracker does not decide:

- which decoder groups a Dealer loads;
- which providers or Backbones a job selects;
- which handles receive a multi-path task;
- how a resource-location map propagates with a task;
- how a handle is reconstructed after a disconnect;
- how operations are merged and GPU memory is governed inside a node.

These behaviors live in Aux/Dealer, job controls, Backbones, and the Brain.

## 7. Network simulation

Orbit's built-in simulation delays messages according to message size and configured latency and is useful for functional debugging. It does not fully model:

- shared-bottleneck queues;
- fair competition among multiple flows;
- qdisc scheduling;
- retransmission and complex jitter;
- host-network-stack and cross-host NIC contention.

Use Linux network namespaces, `tc qdisc`, or an independent network testbed for controlled network measurements. Benchmark entry points do not create these rules automatically.

## 8. Consistency and failure boundaries

The default implementation does not provide:

- terms, leases, or quorums for resource maps;
- fencing or split-brain arbitration under a network partition;
- replication, leader election, or failover for Orbit tracker or Aux;
- shared-link congestion control;
- a general migration signal for a persistently overloaded node.

A disconnect can trigger job-level recovery. A connected node with divergent state is a different failure mode from an ordinary disconnect.

## 9. Replacing the network backend

Validate at least:

1. stable node naming and `Location` resolution;
2. consistent message semantics for colocated delivery and remote sends;
3. send failures and disconnects are reported once and reach the control;
4. reconnection cannot deliver old-session messages to a new handle;
5. path-probe metric units and directions are consistent;
6. serialization and lifecycle of large tensors/carriers are controlled;
7. shutdown releases listeners, connections, and background tasks.

## 10. Source entry points

- `node/base_node.py`
- `node/aux_node/`
- `node/path_utils.py`
- `orbit/`
- `orbit/link/`
- `scripts/online/benchmark/launch_network.py`
