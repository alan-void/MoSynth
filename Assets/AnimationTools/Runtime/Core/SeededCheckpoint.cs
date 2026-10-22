namespace AnimationTools
{
/// <summary>
/// Names the file one training run of a learned method writes: <c>&lt;name&gt;.s&lt;seed&gt;&lt;extension&gt;</c>,
/// e.g. <c>Walk.s42.pfnn.npz</c>. Every method trains once per seed, so every checkpoint carries one.
/// </summary>
public static class SeededCheckpoint
{
    /// <param name="extension">The method's compound extension, leading dot included (".pfnn.npz").</param>
    public static string FileName(string name, int seed, string extension) => $"{name}.s{seed}{extension}";

    /// <summary><see cref="FileName"/> inside <paramref name="folder"/>.</summary>
    public static string Path(string folder, string name, int seed, string extension) =>
        System.IO.Path.Combine(folder, FileName(name, seed, extension));
}

/// <summary>
/// A stage that runs a model trained once per seed, so a benchmark can choose which seed's
/// checkpoint it loads.
/// </summary>
public interface ISeededStage
{
    /// <summary>
    /// The seed whose checkpoint <see cref="MoSynthStage.Init"/> loads; null loads the config's own
    /// <c>seed</c>. Not serialized, and only read at Init.
    /// </summary>
    int? CheckpointSeed { get; set; }

    /// <summary>Whether the stage, as configured, loads a checkpoint at all; a run without one has no seed to vary.</summary>
    bool LoadsCheckpoint { get; }
}
}
