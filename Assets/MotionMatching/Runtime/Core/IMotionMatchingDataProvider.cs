namespace MotionMatching
{
/// <summary>
/// A stage that steers from a <see cref="MotionMatchingData"/>, whether or not it searches one.
/// </summary>
/// <remarks>
/// Every <see cref="MotionMatchingControlInput"/> reaches the database's prediction horizons and
/// feature definitions through <see cref="MotionSynthesisComponentExtensions.GetMmData"/>, and
/// none of them touches the stage itself. Matching on this interface rather than on
/// <see cref="MotionMatchingStage"/> is therefore what lets a learned matcher reuse the existing
/// control inputs unchanged, instead of needing a parallel hierarchy of its own.
/// </remarks>
public interface IMotionMatchingDataProvider
{
    /// <summary>The database whose feature definitions a control input builds its query against.</summary>
    MotionMatchingData MmData { get; }
}
}
