using System.Collections.Generic;
using System.Text;

namespace AnimationTools
{
/// <summary>
/// A stage whose database fixes which bones carry contact flags. The
/// <see cref="MotionSynthesisComponent"/> adopts the list before building its pose layout, so a
/// stage's frames and the pipeline pose share one layout.
/// </summary>
public interface IContactBoneSource
{
    /// <summary>
    /// Contact bone names in slot order, read from serialized configuration so it answers before
    /// <see cref="MoSynthStage.Init"/>. Null when the stage has no opinion, e.g. no config assigned.
    /// </summary>
    IReadOnlyList<string> ContactBoneNames { get; }
}

/// <summary>Reconciles the contact-bone lists a pipeline's stages ask for.</summary>
public static class ContactBoneSources
{
    /// <summary>
    /// The one list every <see cref="IContactBoneSource"/> among <paramref name="candidates"/>
    /// agrees on; empty when none has an opinion. Candidates that are not sources are skipped.
    /// False, with a message naming the disagreeing candidates, when two lists differ in names or
    /// order.
    /// </summary>
    public static bool TryAgree(IReadOnlyList<object> candidates, out IReadOnlyList<string> names,
        out string error)
    {
        names = System.Array.Empty<string>();
        error = null;

        var agreedIndex = -1;
        for (var i = 0; i < candidates.Count; i++)
        {
            if (candidates[i] is not IContactBoneSource source) continue;

            var answer = source.ContactBoneNames;
            if (answer == null) continue;

            if (agreedIndex < 0)
            {
                names = answer;
                agreedIndex = i;
                continue;
            }

            if (SameNames(names, answer)) continue;

            error = $"stage {agreedIndex} ({candidates[agreedIndex].GetType().Name}) lists contact bones " +
                    $"{Describe(names)} but stage {i} ({candidates[i].GetType().Name}) lists " +
                    $"{Describe(answer)}. Every database on one character must list the same bones in " +
                    "the same order.";
            names = System.Array.Empty<string>();
            return false;
        }

        return true;
    }

    /// <summary>
    /// The names of a config's serialized contact-bone list, in order; an unset entry is null. An
    /// absent list reads as empty.
    /// </summary>
    public static IReadOnlyList<string> NamesOf(IReadOnlyList<SkeletonBone> bones)
    {
        if (bones == null || bones.Count == 0) return System.Array.Empty<string>();

        var names = new string[bones.Count];
        for (var i = 0; i < names.Length; i++)
        {
            names[i] = bones[i]?.Name;
        }

        return names;
    }

    /// <summary>
    /// Whether a trained model's contact slots fit the pipeline's. <paramref name="storedNames"/> is
    /// null for a checkpoint that predates recording them, which is checked by count alone.
    /// </summary>
    public static bool TryMatchCheckpoint(int storedCount, IReadOnlyList<string> storedNames,
        IReadOnlyList<string> expected, out string error)
    {
        error = null;
        if (storedCount != expected.Count)
        {
            error = $"it predicts {storedCount} contact flags but the character has {expected.Count} " +
                    $"contact bones {Describe(expected)}";
            return false;
        }

        if (storedNames == null || SameNames(storedNames, expected)) return true;

        error = $"it was trained on contact bones {Describe(storedNames)} but the character has " +
                $"{Describe(expected)}";
        return false;
    }

    /// <summary>Same names in the same order. Two nulls are equal; null and empty are not.</summary>
    public static bool SameNames(IReadOnlyList<string> a, IReadOnlyList<string> b)
    {
        if (a == null || b == null) return a == null && b == null;
        if (a.Count != b.Count) return false;

        for (var i = 0; i < a.Count; i++)
        {
            if (a[i] != b[i]) return false;
        }

        return true;
    }

    public static string Describe(IReadOnlyList<string> names)
    {
        if (names == null) return "(none)";

        var builder = new StringBuilder("[");
        for (var i = 0; i < names.Count; i++)
        {
            if (i > 0) builder.Append(", ");
            builder.Append(names[i] ?? "<unset>");
        }

        return builder.Append(']').ToString();
    }
}
}
