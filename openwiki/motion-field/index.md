# Files

- [Config, training and the staleness contract](config-and-training.md) - The end-to-end chain from clips to a trained value function, and the two flags that decide whether an artefact still matches the config that will run it.
- [The motion field stage](motion-field-stage.md) - Driving a character from a neural motion field evaluated in Python — the four policies, why a value function beats greedy, and how the goal is expressed.
- [The pose manifold and its visualizer](pose-manifold-embedding.md) - Projecting the motion field to 3D so a stalled policy can be watched rather than inferred — why the cloud is poses and the edges are velocity.
- [Reaching Python](python-interop.md) - Two mutually exclusive transports to the motion field — an embedded CPython interpreter and an out-of-process ZeroMQ socket — and the constraints each imposes.
