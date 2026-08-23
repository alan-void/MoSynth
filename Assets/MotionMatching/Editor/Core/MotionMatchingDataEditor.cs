using UnityEngine;
using UnityEditor;
using AnimationTools;
using AnimationTools.Editor;

namespace MotionMatching
{
/// <summary>
/// Inspector for a <see cref="MotionMatchingData"/> asset: authoring the animation clips, the
/// skeleton and the feature configuration, and the button that bakes it all into the database files
/// the runtime loads.
/// </summary>
/// <remarks>
/// The asset is a recipe, not the data. Edits take effect only when <see cref="GenerateDatabases"/>
/// runs, and stale generated files are not detected, so regenerate after any change.
/// </remarks>
[CustomEditor(typeof(MotionMatchingData))]
public class MotionMatchingDataEditor : UnityEditor.Editor
{
    private bool TrajectoryFeaturesSelectorFoldout;
    private bool PoseFeaturesSelectorFoldout;

    private bool _generateButtonError;

    private SerializedProperty _animationClipsProperty;
    private SerializedProperty _skeletonProperty;
    private SerializedProperty _rootMotionBoneProperty;
    private SerializedProperty _contactVelocityThresholdProperty;
    private SerializedProperty _leftContactBoneProperty;
    private SerializedProperty _rightContactBoneProperty;
    private SerializedProperty _trajectoryFeaturesProperty;
    private SerializedProperty _poseFeaturesProperty;

    /// <summary>
    /// Bakes the asset's configuration into the two files the runtime loads, and the one the Python
    /// side reads.
    /// </summary>
    /// <remarks>
    /// The steps depend on each other, so the order is fixed: extract poses, serialize them
    /// (.mmpose — also what the Python side reads), compute joint forward axes, extract the features
    /// derived from those poses, serialize those too.
    /// </remarks>
    public void GenerateDatabases(MotionMatchingData mmData)
    {
        PROFILE.BEGIN_SAMPLE_PROFILING("Pose Extract");
        mmData.ImportPoseSet();
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Pose Extract");

        PROFILE.BEGIN_SAMPLE_PROFILING("Pose Serialize");
        PoseSerializer poseSerializer = new();
        poseSerializer.Serialize(mmData.GetOrImportPoseSet(), mmData.GetAssetPath(), mmData.name);
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Pose Serialize");

        mmData.ComputeJointsLocalForward();

        PROFILE.BEGIN_SAMPLE_PROFILING("Feature Extract");
        mmData.ImportFeatureSet();
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Feature Extract");

        PROFILE.BEGIN_SAMPLE_PROFILING("Feature Serialize");
        FeatureSerializer featureSerializer = new();
        featureSerializer.Serialize(mmData.FeatureSet, mmData, mmData.GetAssetPath(), mmData.name);
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Feature Serialize");

        AssetDatabase.Refresh();
    }

    private void OnEnable()
    {
        _animationClipsProperty = serializedObject.FindProperty("animationClips");
        _skeletonProperty = serializedObject.FindProperty("skeleton");
        _rootMotionBoneProperty = serializedObject.FindProperty("rootMotionBone");
        _contactVelocityThresholdProperty = serializedObject.FindProperty("contactVelocityThreshold");
        _leftContactBoneProperty = serializedObject.FindProperty("leftContactBone");
        _rightContactBoneProperty = serializedObject.FindProperty("rightContactBone");
        _trajectoryFeaturesProperty = serializedObject.FindProperty("trajectoryFeatures");
        _poseFeaturesProperty = serializedObject.FindProperty("poseFeatures");
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        var data = (MotionMatchingData)target;
        _generateButtonError = false;
        var rigRoot = PoseSetSourceGUI.GetRigRoot(data.Skeleton);

        EditorGUILayout.PropertyField(_skeletonProperty);

        DrawAnimations();
        _generateButtonError |= PoseSetSourceGUI.DrawSkeletonValidation(data);

        // SmoothSimulationBone
        //data.SmoothSimulationBone = EditorGUILayout.Toggle(new GUIContent("Smooth Simulation Bone", "Smooth the simulation bone (articial root added during pose extraction) using Savitzky-Golay filter"),
        //                                                   data.SmoothSimulationBone);

        DrawContactThreshold(rigRoot);
        DrawTrajectoryFeatures(rigRoot);
        DrawPoseFeatures(rigRoot);
        DrawGenerateButton(data);

        serializedObject.ApplyModifiedProperties();
    }

    private void DrawAnimations()
    {
        EditorGUILayout.LabelField("Animations", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(_animationClipsProperty);
    }

    private void DrawSkeleton()
    {
        EditorGUILayout.Separator();
        EditorGUILayout.LabelField("Skeleton", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(_skeletonProperty);
    }

    // ContactVelocityThreshold + Contact Bones
    private void DrawContactThreshold(Transform rigRoot)
    {
        EditorGUILayout.Separator();
        EditorGUILayout.LabelField("Contacts", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(_contactVelocityThresholdProperty,
            new GUIContent("Contact Velocity Threshold",
                "Minimum velocity of the foot to be considered in movement and not in contact with the ground"));

        SkeletonBoneDrawer.DrawLayout(
            new GUIContent("Left Contact Bone", "Bone whose velocity drives foot-contact detection; leave unset to pick by name (LeftToe/RightToe)."),
            _leftContactBoneProperty, rigRoot);
        SkeletonBoneDrawer.DrawLayout(
            new GUIContent("Right Contact Bone", "Bone whose velocity drives foot-contact detection; leave unset to pick by name (LeftToe/RightToe)."),
            _rightContactBoneProperty, rigRoot);
    }

    // Trajectory Features ------------------------------------------------------------------------------------
    private void DrawTrajectoryFeatures(Transform rigRoot)
    {
        TrajectoryFeaturesSelectorFoldout =
            EditorGUILayout.BeginFoldoutHeaderGroup(TrajectoryFeaturesSelectorFoldout, "Trajectory Features");
        if (TrajectoryFeaturesSelectorFoldout)
        {
            EditorGUI.indentLevel++;
            var hasAMainPositionFeature = false;
            // Deleting mid-loop would change the control count between the layout and repaint passes
            var removeIndex = -1;
            for (var i = 0; i < _trajectoryFeaturesProperty.arraySize; i++)
            {
                var trajectoryFeature = _trajectoryFeaturesProperty.GetArrayElementAtIndex(i);
                // Header
                EditorGUILayout.BeginVertical(GUI.skin.box);
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField((i + 1).ToString());
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("x"))
                {
                    removeIndex = i;
                }

                EditorGUILayout.EndHorizontal();
                // Name
                var nameProp = trajectoryFeature.FindPropertyRelative("name");
                nameProp.stringValue = EditorGUILayout.TextField("Name", nameProp.stringValue);
                // Feature Type
                var featureTypeProp = trajectoryFeature.FindPropertyRelative("featureType");
                featureTypeProp.intValue = (int)(TrajectoryFeatureChannel.Type)EditorGUILayout.EnumPopup("Type",
                    (TrajectoryFeatureChannel.Type)featureTypeProp.intValue);
                if ((TrajectoryFeatureChannel.Type)featureTypeProp.intValue == TrajectoryFeatureChannel.Type.Position)
                {
                    var isMainPositionFeatureProp = trajectoryFeature.FindPropertyRelative("isMainPositionFeature");
                    isMainPositionFeatureProp.boolValue =
                        EditorGUILayout.Toggle("Main Position Feature", isMainPositionFeatureProp.boolValue);
                    if (isMainPositionFeatureProp.boolValue)
                    {
                        if (hasAMainPositionFeature)
                        {
                            EditorGUILayout.HelpBox("Only one main position feature is allowed", MessageType.Error);
                            _generateButtonError = true;
                        }

                        hasAMainPositionFeature = true;
                    }
                }

                _generateButtonError = _generateButtonError || TrajectoryFramesLayout(trajectoryFeature);
                _generateButtonError = _generateButtonError || TrajectoryTypeOptionsLayout(trajectoryFeature, rigRoot);

                EditorGUILayout.EndVertical();
            }

            if (removeIndex >= 0)
            {
                _trajectoryFeaturesProperty.DeleteArrayElementAtIndex(removeIndex);
            }

            if (GUILayout.Button("Add Trajectory Feature"))
            {
                var insertIndex = _trajectoryFeaturesProperty.arraySize;
                _trajectoryFeaturesProperty.arraySize++;
                ResetTrajectoryFeature(_trajectoryFeaturesProperty.GetArrayElementAtIndex(insertIndex));
            }

            EditorGUI.indentLevel--;
        }

        EditorGUILayout.EndFoldoutHeaderGroup();
    }

    // Pose Features ------------------------------------------------------------------------------------
    private void DrawPoseFeatures(Transform rigRoot)
    {
        PoseFeaturesSelectorFoldout =
            EditorGUILayout.BeginFoldoutHeaderGroup(PoseFeaturesSelectorFoldout, "Pose Features");
        if (PoseFeaturesSelectorFoldout)
        {
            EditorGUI.indentLevel++;
            // Deleting mid-loop would change the control count between the layout and repaint passes
            var removeIndex = -1;
            for (var i = 0; i < _poseFeaturesProperty.arraySize; i++)
            {
                var poseFeature = _poseFeaturesProperty.GetArrayElementAtIndex(i);
                // Header
                EditorGUILayout.BeginVertical(GUI.skin.box);
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField((i + 1).ToString());
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("x"))
                {
                    removeIndex = i;
                }

                EditorGUILayout.EndHorizontal();
                //  Properties
                var nameProp = poseFeature.FindPropertyRelative("name");
                nameProp.stringValue = EditorGUILayout.TextField("Name", nameProp.stringValue);
                var featureTypeProp = poseFeature.FindPropertyRelative("featureType");
                featureTypeProp.intValue = (int)(PoseFeatureChannel.Type)EditorGUILayout.EnumPopup("Type",
                    (PoseFeatureChannel.Type)featureTypeProp.intValue);
                var boneProp = poseFeature.FindPropertyRelative("bone");
                SkeletonBoneDrawer.DrawLayout(new GUIContent("Bone"), boneProp, rigRoot);

                EditorGUILayout.EndVertical();
            }

            if (removeIndex >= 0)
            {
                _poseFeaturesProperty.DeleteArrayElementAtIndex(removeIndex);
            }

            if (GUILayout.Button("Add Pose Feature"))
            {
                var insertIndex = _poseFeaturesProperty.arraySize;
                _poseFeaturesProperty.arraySize++;
                ResetPoseFeature(_poseFeaturesProperty.GetArrayElementAtIndex(insertIndex));
            }

            EditorGUI.indentLevel--;
        }

        EditorGUILayout.EndFoldoutHeaderGroup();
    }

    // Generate Databases
    private void DrawGenerateButton(MotionMatchingData data)
    {
        EditorGUILayout.Separator();
        if (_generateButtonError)
        {
            EditorGUILayout.HelpBox(
                "Oops! Looks like there are errors in the feature definition. Please resolve them before generating the databases.",
                MessageType.Error);
        }

        GUI.enabled = !_generateButtonError && data.animationClips.Count > 0;
        if (GUILayout.Button("Generate Databases", GUILayout.Height(30)))
        {
            serializedObject.ApplyModifiedProperties();
            GenerateDatabases(data);
            EditorUtility.SetDirty(data);
            serializedObject.Update();
        }

        GUI.enabled = true;

        // Error Check
        if (data.JointsLocalForwardError)
        {
            EditorGUILayout.HelpBox("Internal error detected. Please regenerate databases.", MessageType.Error);
        }
    }

    private static void ResetTrajectoryFeature(SerializedProperty element)
    {
        element.FindPropertyRelative("name").stringValue = "";
        element.FindPropertyRelative("featureType").intValue = 0;
        element.FindPropertyRelative("simulationBone").boolValue = false;
        ClearBone(element.FindPropertyRelative("bone"));
        element.FindPropertyRelative("zeroX").boolValue = false;
        element.FindPropertyRelative("zeroY").boolValue = false;
        element.FindPropertyRelative("zeroZ").boolValue = false;
        element.FindPropertyRelative("isMainPositionFeature").boolValue = false;
        element.FindPropertyRelative("predictionFrames").arraySize = 0;
    }

    private static void ResetPoseFeature(SerializedProperty element)
    {
        element.FindPropertyRelative("name").stringValue = "";
        element.FindPropertyRelative("featureType").intValue = 0;
        ClearBone(element.FindPropertyRelative("bone"));
    }

    private static void ClearBone(SerializedProperty skeletonBoneProperty)
    {
        skeletonBoneProperty.FindPropertyRelative("bone").objectReferenceValue = null;
        skeletonBoneProperty.FindPropertyRelative("skeleton.root").objectReferenceValue = null;
    }

    private bool TrajectoryFramesLayout(SerializedProperty trajectoryFeature)
    {
        var generateButtonError = false;
        var predictionFrames = trajectoryFeature.FindPropertyRelative("predictionFrames");
        EditorGUILayout.LabelField("Frames Prediction");
        EditorGUILayout.BeginHorizontal();
        for (var j = 0; j < predictionFrames.arraySize; j++)
        {
            var element = predictionFrames.GetArrayElementAtIndex(j);
            element.intValue = EditorGUILayout.IntField(element.intValue);
        }

        if (GUILayout.Button("Add"))
        {
            var addIndex = predictionFrames.arraySize;
            predictionFrames.arraySize++;
            // Growing an array copies the previous element rather than default-initialising it
            predictionFrames.GetArrayElementAtIndex(addIndex).intValue = 0;
        }

        if (predictionFrames.arraySize > 0 && GUILayout.Button("Remove"))
        {
            predictionFrames.arraySize--;
        }

        EditorGUILayout.EndHorizontal();
        return generateButtonError;
    }

    private bool TrajectoryTypeOptionsLayout(SerializedProperty trajectoryFeature, Transform rigRoot)
    {
        var generateButtonError = false;
        var featureType = (TrajectoryFeatureChannel.Type)trajectoryFeature.FindPropertyRelative("featureType").intValue;
        if (featureType == TrajectoryFeatureChannel.Type.Position ||
            featureType == TrajectoryFeatureChannel.Type.Direction)
        {
            // Bone
            var simulationBoneProp = trajectoryFeature.FindPropertyRelative("simulationBone");
            var boneProp = trajectoryFeature.FindPropertyRelative("bone");
            var zeroXProp = trajectoryFeature.FindPropertyRelative("zeroX");
            var zeroYProp = trajectoryFeature.FindPropertyRelative("zeroY");
            var zeroZProp = trajectoryFeature.FindPropertyRelative("zeroZ");

            simulationBoneProp.boolValue = EditorGUILayout.Toggle("Simulation Bone", simulationBoneProp.boolValue);
            if (!simulationBoneProp.boolValue)
            {
                SkeletonBoneDrawer.DrawLayout(new GUIContent("Bone"), boneProp, rigRoot);
                GUI.enabled = !zeroYProp.boolValue || !zeroZProp.boolValue;
                zeroXProp.boolValue = EditorGUILayout.Toggle("Zero X", zeroXProp.boolValue);
                GUI.enabled = !zeroXProp.boolValue || !zeroZProp.boolValue;
                zeroYProp.boolValue = EditorGUILayout.Toggle("Zero Y", zeroYProp.boolValue);
                GUI.enabled = !zeroXProp.boolValue || !zeroYProp.boolValue;
                zeroZProp.boolValue = EditorGUILayout.Toggle("Zero Z", zeroZProp.boolValue);
                GUI.enabled = true;
            }
            else
            {
                zeroXProp.boolValue = false;
                zeroYProp.boolValue = true; // project simulation bone to the ground
                zeroZProp.boolValue = false;
            }
        }

        return generateButtonError;
    }
}
}
