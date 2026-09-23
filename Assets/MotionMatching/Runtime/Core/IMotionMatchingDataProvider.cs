namespace MotionMatching
{
/// <summary>
/// A stage that steers from a <see cref="MotionMatchingData"/>, whether or not it searches one.
/// </summary>
/// <remarks>
/// Control inputs find their database through this interface rather than through
/// <see cref="MotionMatchingStage"/>, so a learned matcher can reuse them unchanged.
/// </remarks>
public interface IMotionMatchingDataProvider
{
    /// <summary>The database whose feature definitions a control input builds its query against.</summary>
    MotionMatchingData MmData { get; }
}
}
