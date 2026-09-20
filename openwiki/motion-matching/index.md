# Files

- [Control inputs](control-inputs.md) - Turning intent into the trajectory the search matches against, the simulation-object model every input shares, and an honest account of which inputs actually work.
- [Debug visualizers](debug-visualizers.md) - Four tools for looking at what the database holds and what the pipeline produced — two of them ordinary stages, which makes them a worked example of the extension seam.
- [Feature vectors and query masking](feature-vectors.md) - How an authored feature configuration becomes a fixed-width float vector per frame, and why an inactive query channel is weight-masked rather than filled.
- [Inertialization](inertialization.md) - Smoothing pose jumps by decaying offsets rather than crossfading, and why the root has to be blended in frame space.
- [Learned motion matching](learned-motion-matching.md) - How MoSynth replaces the database search with learned networks, the staging that makes each replacement measurable, and the parked Barracuda experiment that was removed to make room for it.
- [The motion matching stage](matching-stage.md) - Synthesis by search — playing a database frame by frame while periodically asking whether a better frame exists.
- [Search backends](search-backends.md) - The strategy seam for finding the closest database frame, the bounding-volume hierarchy that accelerates it, and the tag mechanism that is designed but unwired.
- [Spline pose keypoints](spline-pose-keypoints.md) - A path that carries authored poses — how facing is stored as a yaw offset from the tangent, and how bone constraints switch on only near a keypoint.
