# Files

- [Motion field policies](motion-field-policies.md) - The similarity metric, the candidate action set, and the two policies plus two debug modes the runtime steps the field with.
- [The pose data model](pose-data-bridge.md) - The packed array layout the motion field is defined over, how it is read from the exported database, and the algebra defined on it.
- [The Unity call surface](unity-call-surface.md) - What crosses the C#/Python boundary in each direction every tick, the virtual root the packed layout needs, and the debugger attach that runs by default.
- [Value function training](value-function-training.md) - Fitted value iteration over the whole database, what the produced artefact does and does not record, and why the loader deliberately checks nothing.
