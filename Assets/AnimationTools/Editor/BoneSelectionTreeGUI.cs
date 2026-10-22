using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Draws a model's bone selection — a sparse list of excluded bone names — as the skeleton
/// hierarchy, one toggle per joint, with the default-selection and clean-up buttons under it.
/// </summary>
/// <remarks>
/// Keeps the selection closed under subtree, as <see cref="PredictedBoneSelection"/> requires:
/// excluding a bone excludes everything below it, and including one includes its ancestors.
/// One instance per inspector, because it caches the skeleton across repaints.
/// </remarks>
public sealed class BoneSelectionTreeGUI
{
    /// <summary>Reads the skeleton the selection is expressed over; false when there is none.</summary>
    public delegate bool SkeletonSource(out Skeleton skeleton);

    private readonly string _includedWord;
    private readonly string _includeAllLabel;

    private Skeleton _skeleton;
    private int[] _jointDepths;
    private bool _skeletonRead;

    /// <param name="includedWord">What a ticked bone is, for the footer: "predicted", "styled".</param>
    /// <param name="includeAllLabel">The button that clears the exclusions: "Predict All".</param>
    public BoneSelectionTreeGUI(string includedWord, string includeAllLabel)
    {
        _includedWord = includedWord;
        _includeAllLabel = includeAllLabel;
    }

    /// <summary>Forget the cached skeleton, so the next draw reads it again.</summary>
    public void Invalidate()
    {
        _skeleton = null;
        _jointDepths = null;
        _skeletonRead = false;
    }

    /// <summary>Read and cache the skeleton, with each joint's depth for indentation.</summary>
    public bool TryRead(SkeletonSource source)
    {
        if (_skeletonRead) return _skeleton != null;
        _skeletonRead = true;

        if (!source(out var skeleton) || skeleton == null || skeleton.BoneCount == 0) return false;

        _skeleton = skeleton;
        _jointDepths = new int[skeleton.BoneCount];

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            var depth = 0;
            var parent = skeleton.GetParentIndex(i);
            // Bounded by the joint count so a cyclic parent index cannot hang the inspector.
            while (parent >= 0 && parent < skeleton.BoneCount && depth < skeleton.BoneCount)
            {
                depth++;
                parent = skeleton.GetParentIndex(parent);
            }

            _jointDepths[i] = depth;
        }

        return true;
    }

    /// <summary>
    /// Draws the rows and the footer over the skeleton <see cref="TryRead"/> cached. Returns true
    /// when the selection changed; the undo has then been recorded against <paramref name="owner"/>
    /// and the caller decides what goes stale.
    /// </summary>
    public bool Draw(Object owner, List<string> excludedBones)
    {
        if (_skeleton == null) return false;

        var changed = DrawRows(owner, excludedBones);
        changed |= DrawFooter(owner, excludedBones);
        return changed;
    }

    private bool DrawRows(Object owner, List<string> excludedBones)
    {
        var changed = false;
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            var boneName = _skeleton.GetBone(i).Name;
            var included = !excludedBones.Contains(boneName);

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Space(_jointDepths[i] * 12f);

                var toggled = EditorGUILayout.ToggleLeft(
                    new GUIContent(boneName, $"joint {i}"), included,
                    included ? EditorStyles.label : EditorStyles.miniLabel);

                if (toggled == included) continue;

                Undo.RecordObject(owner, $"Edit {_includedWord} bones");
                SetSubtreeIncluded(excludedBones, i, toggled);
                changed = true;
            }
        }

        return changed;
    }

    private void SetSubtreeIncluded(List<string> excludedBones, int boneIndex, bool included)
    {
        foreach (var bone in PredictedBoneSelection.Subtree(_skeleton, boneIndex))
        {
            SetIncluded(excludedBones, _skeleton.GetBone(bone).Name, included);
        }

        if (!included) return;

        for (var parent = _skeleton.GetParentIndex(boneIndex);
             parent >= 0;
             parent = _skeleton.GetParentIndex(parent))
        {
            SetIncluded(excludedBones, _skeleton.GetBone(parent).Name, true);
        }
    }

    private static void SetIncluded(List<string> excludedBones, string boneName, bool included)
    {
        if (included) excludedBones.Remove(boneName);
        else if (!excludedBones.Contains(boneName)) excludedBones.Add(boneName);
    }

    private bool DrawFooter(Object owner, List<string> excludedBones)
    {
        var changed = false;

        var included = 0;
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            if (!excludedBones.Contains(_skeleton.GetBone(i).Name)) included++;
        }

        EditorGUILayout.LabelField(
            $"{included} of {_skeleton.BoneCount} bones {_includedWord}.", EditorStyles.miniLabel);

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Exclude Fingers And Leaves"))
            {
                Undo.RecordObject(owner, "Exclude fingers and leaves");
                excludedBones.Clear();
                foreach (var boneName in DefaultPredictedBones.Excluded(_skeleton))
                {
                    SetIncluded(excludedBones, boneName, false);
                }

                changed = true;
            }

            using (new EditorGUI.DisabledScope(excludedBones.Count == 0))
            {
                if (GUILayout.Button(_includeAllLabel))
                {
                    Undo.RecordObject(owner, _includeAllLabel);
                    excludedBones.Clear();
                    changed = true;
                }
            }

            if (GUILayout.Button("Reload Skeleton")) Invalidate();
        }

        // The skeleton may have been dropped by Reload above, mid-draw.
        if (_skeleton == null) return changed;

        // Python refuses names the skeleton does not have, so surface them before training does.
        var orphans = excludedBones
            .Where(boneName => !_skeleton.TryFindByName(boneName, out _))
            .ToList();
        if (orphans.Count == 0) return changed;

        EditorGUILayout.HelpBox(
            $"{orphans.Count} excluded bone(s) are not in this skeleton, which training refuses: " +
            string.Join(", ", orphans), MessageType.Warning);

        if (GUILayout.Button("Remove Stale Entries"))
        {
            Undo.RecordObject(owner, "Remove stale excluded bones");
            excludedBones.RemoveAll(boneName => !_skeleton.TryFindByName(boneName, out _));
            changed = true;
        }

        return changed;
    }
}
}
