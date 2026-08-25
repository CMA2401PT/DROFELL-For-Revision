# Local link self-test

When two Orbit nodes have the same host, process, and thread placement,
`OrbitNetCommander` selects an in-process link automatically. The sender places
the original packet directly into the destination context instead of serializing
it into frames.

Run the self-test from the repository root:

```bash
python -m orbit.examples.local_link.main
```

The payload contains a lambda and the sender verifies object identity after a
round trip. The test therefore fails if either direction uses serialization.
