using System.Collections.Generic;
using AnimationTools;
using UnityEngine;

namespace Reproduction
{
/// <summary>
/// The benchmark sweeps a reproduction of the paper's results runs. The configs it builds and trains
/// are derived from those sweeps' method prefabs by <see cref="ReproductionPipeline"/>, never listed
/// here, so the two cannot disagree.
/// </summary>
[CreateAssetMenu(fileName = "Reproduction", menuName = "MoSynth/Reproduction Manifest")]
public class ReproductionManifest : ScriptableObject
{
    public List<SynthesisBenchmarkConfig> benchmarks = new();
}
}
