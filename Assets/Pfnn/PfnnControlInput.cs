using System.Linq;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace Pfnn
{
/// <summary>
/// Base for components that steer a <see cref="PfnnStage"/> by saying where the character should be
/// going over the next second.
/// </summary>
/// <remarks>
/// Not a <c>MotionMatchingControlInput</c>, which resolves its horizons from a
/// <c>MotionMatchingData</c> and so needs a motion matching stage. Only the future half of the
/// window comes from here; the stage keeps the past half from where the character actually went.
/// </remarks>
public abstract class PfnnControlInput : MotionSynthesisControlInput
{
    private PfnnStage _stage;
    private bool _warnedNoStage;

    /// <summary>World position of the character's simulation frame.</summary>
    protected Vector3 RootPosition => synthesizer.transform.position;

    /// <summary>The stage this input drives, or null until the synthesis component has woken.</summary>
    protected PfnnStage Stage
    {
        get
        {
            // Stages are populated in the synthesis component's Awake; resolve lazily so this
            // component works regardless of Awake ordering.
            if (_stage != null) return _stage;

            _stage = synthesizer.stages?.OfType<PfnnStage>().FirstOrDefault();
            if (_stage == null && !_warnedNoStage)
            {
                Debug.LogWarning($"[PFNN] {GetType().Name} on '{name}' found no PfnnStage on the " +
                                 "synthesis component.", this);
                _warnedNoStage = true;
            }

            return _stage;
        }
    }

    private void Update()
    {
        if (Stage == null) return;
        OnUpdate();
    }

    /// <summary>Refresh whatever <see cref="TryGetFutureSample"/> reads, once per frame.</summary>
    protected virtual void OnUpdate()
    {
    }

    /// <summary>
    /// Where the character should be, and which way it should face, this many synthesis frames from
    /// now.
    /// </summary>
    /// <remarks>
    /// World space on the ground plane; the stage converts to the character frame. Returning false
    /// leaves the sample at the character's current position and facing, i.e. standing still.
    /// </remarks>
    /// <param name="frameOffset">Synthesis frames ahead. Always positive.</param>
    public abstract bool TryGetFutureSample(int frameOffset, out float2 position, out float2 direction);

#if UNITY_EDITOR
    private static readonly Color PastColor = new(0.35f, 0.45f, 0.55f);
    private static readonly Color FutureColor = new(1f, 0.6f, 0.1f);

    /// <summary>
    /// Draws the trajectory window the stage last assembled: where the character has been behind it,
    /// where it is being asked to go ahead of it.
    /// </summary>
    /// <remarks>
    /// An input with a path of its own to show overrides this and calls back into it.
    /// </remarks>
    protected virtual void OnDrawGizmos()
    {
        if (!Application.isPlaying || synthesizer == null || Stage == null) return;

        var height = RootPosition.y + 0.05f;
        var previous = Vector3.zero;

        for (var i = 0; i < Stage.WindowSampleCount; i++)
        {
            Stage.GetWindowSample(i, out var framesAhead, out var position, out var direction);

            Gizmos.color = framesAhead <= 0 ? PastColor : FutureColor;
            var world = new Vector3(position.x, height, position.y);
            Gizmos.DrawSphere(world, 0.04f);
            Gizmos.DrawLine(world, world + new Vector3(direction.x, 0f, direction.y) * 0.2f);
            if (i > 0) Gizmos.DrawLine(previous, world);
            previous = world;
        }
    }
#endif
}
}
