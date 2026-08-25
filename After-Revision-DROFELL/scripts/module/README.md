# Module-Level Experiments

These scripts test key modules before assembling a complete LLM:

- `qlinear/`: forward/backward performance of quantized-linear implementations;
- `decoder/`: speed and CUDA-memory lifecycles of decoder and user-model checkpoint/replay strategies.

Both experiment families load GPU extensions and model weights directly and are not CPU unit tests. Prepare qlinear data through `converts/`, then begin with one configuration and a short sequence.
