# Files

- [Animation sources and clip baking](animation-sources.md) - How clips and BVH files become poses, why a clip is accepted on one animated bone, and what baking does and does not capture.
- [Benchmarking synthesis methods](benchmarking.md) - The Editor-driven sweep that runs every method against every path, what it measures, and what those numbers may and may not be compared against.
- [The channel and layout system](channel-layout-system.md) - The typed float-buffer substrate every pose, feature vector and recording is built on, and the rules that keep buffers from being read with the wrong offsets.
- [Tagging clips](clip-tags.md) - Hierarchical tags over stretches of a clip, why a channel is stored as boolean keyframes rather than intervals, and how a query turns them back into segments.
- [Motion recording](motion-recording.md) - Capturing a run to a flat float file plus a self-describing manifest, and the channel model that decides what lands in it.
- [Neural synthesis readiness](neural-synthesis.md) - What a learned motion model needs from this repository before it can be trained or run, which of those pieces exist, and the failure modes the shared definitions are there to prevent.
- [On-disk formats](on-disk-formats.md) - The four artefacts MoSynth ships to StreamingAssets, the byte convention that makes them readable from numpy, and how staleness is detected without a version header.
- [Path following and its metrics](path-following-metrics.md) - Windowed spline projection and why global nearest-point is wrong, plus how trajectory, heading, speed and lap progress are measured.
- [Pose buffers and layouts](pose-buffers.md) - How a pose is laid out over a skeleton, why layouts are shared between the database and the runtime, and what blending a pose actually does.
- [The pose database](pose-database.md) - How clips become a searchable set of poses, and why the extraction contract was narrowed to an interface that a motion field can satisfy.
- [Retargeting BVH onto the shared target rig](retargeting-pipeline.md) - Why source motion is retargeted in Blender before it reaches Unity, what the two stages do, and which hidden inputs decide whether the result is right.
- [The simulation frame, facing and travel](simulation-frame.md) - How a character frame is derived from a pose rather than stored, and why facing and direction of travel are different quantities that must not be substituted for one another.
- [Skeletons and rig binding](skeletons-and-rig-binding.md) - Bone identity and ordering for the whole project, why a skeleton must point at an asset rig, and how an asset skeleton binds to a live scene rig.
- [The synthesis pipeline](synthesis-pipeline.md) - How MotionSynthesisComponent drives a character each tick, and the MoSynthStage contract every synthesis method plugs into.
