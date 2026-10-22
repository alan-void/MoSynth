using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AnimationTools;
using GameplayTags.Editor;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Style.Editor
{
/// <summary>
/// Plays the style test scene headless and checks the stage end to end over motion matching: it runs,
/// changes the pose, and leaves the pipeline error-free. Writes <c>Logs/style_smoke.json</c> and, in
/// batchmode, exits with 0 on a pass.
/// </summary>
/// <remarks>
/// The character walks forward through four segments: styled, unstyled, styled with the root
/// restyled, and a style switch. Lean and speed are compared between the first three.
/// </remarks>
[InitializeOnLoad]
public static class StyleSmokeCheck
{
    private const string ScenePath = "Assets/Scenes/StyleTransferMM.unity";
    private const string ArmedFlagPath = "Temp/style_smoke_armed";
    private const string ReportPath = "Logs/style_smoke.json";
    private const string HipsName = "mixamorig:Hips";
    private const string NeckName = "mixamorig:Neck";
    private const string SwitchStyleName = "style.Elated";

    // Segment boundaries in play-mode seconds; each segment is measured after a settle period, so
    // the blends between segments are not averaged in.
    private const float StyledEnd = 8f;
    private const float UnstyledEnd = 14f;
    private const float StylizedEnd = 20f;
    private const float SwitchEnd = 24f;
    private const float SettleSeconds = 2f;

    private const float MinLeanChangeDegrees = 3f;
    private const float MinMeanGate = 0.3f;

    /// <summary>Command-line switch that turns the stage's gate off, to see the style at full strength.</summary>
    private const string UngatedArgument = "-ungated";

    [Serializable]
    private class Segment
    {
        public string name;
        public float meanLeanDegrees;
        public float meanSpeed;
        public float meanGate;
        public int samples;

        [NonSerialized] public double LeanSum, SpeedSum, GateSum;

        public void Finish()
        {
            if (samples == 0) return;
            meanLeanDegrees = (float)(LeanSum / samples);
            meanSpeed = (float)(SpeedSum / samples);
            meanGate = (float)(GateSum / samples);
        }
    }

    [Serializable]
    private class Report
    {
        public bool passed;
        public List<string> failures = new();
        public List<Segment> segments = new();
        public int errorsLogged;
        public List<string> firstErrors = new();
        public bool poseStayedFinite = true;
        public float phaseSpread;
        public bool gated = true;
    }

    private static bool _armed;
    private static Report _report;
    private static MotionSynthesisComponent _character;
    private static StyleTransferStage _stage;
    private static IMotionSynthesisDirectionControlInput _input;
    private static Transform _hips, _neck;
    private static Vector3 _lastPosition;
    private static float _lastTime;
    private static float _phaseMin = float.MaxValue, _phaseMax = float.MinValue;
    private static int _segmentIndex = -1;

    static StyleSmokeCheck()
    {
        // Re-arms on the far side of the play-mode domain reload.
        if (File.Exists(ArmedFlagPath)) Arm();
    }

    [MenuItem("MoSynth/Style/Pipeline/7 Smoke Check (Play Mode)")]
    public static void Run()
    {
        if (!File.Exists(Path.Combine(Directory.GetParent(Application.dataPath)!.FullName, ScenePath)))
        {
            Debug.LogError($"[Style Smoke] No {ScenePath}; run Create MM Character first.");
            if (Application.isBatchMode) EditorApplication.Exit(1);
            return;
        }

        EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        File.WriteAllText(ArmedFlagPath, "armed");
        Arm();
        EditorApplication.EnterPlaymode();
    }

    private static void Arm()
    {
        if (_armed) return;
        _armed = true;
        _report = new Report();
        Application.logMessageReceived += OnLog;
        EditorApplication.update += Tick;
    }

    private static void OnLog(string message, string stackTrace, LogType type)
    {
        if (type != LogType.Error && type != LogType.Exception && type != LogType.Assert) return;
        if (message.StartsWith("[Licensing")) return;

        _report.errorsLogged++;
        if (_report.firstErrors.Count < 5) _report.firstErrors.Add(message);
    }

    private static void Tick()
    {
        if (!EditorApplication.isPlaying) return;

        try
        {
            if (_character == null && !TryBind()) return;
            Sample(Time.time);
        }
        catch (Exception e)
        {
            _report.failures.Add("check threw: " + e.Message);
            Finish();
        }
    }

    private static bool TryBind()
    {
        _character = UnityEngine.Object.FindFirstObjectByType<MotionSynthesisComponent>();
        if (_character == null) return false;

        _stage = _character.stages.OfType<StyleTransferStage>().FirstOrDefault();
        _input = _character.ControlInput as IMotionSynthesisDirectionControlInput;
        var bones = _character.GetComponentsInChildren<Transform>();
        _hips = bones.FirstOrDefault(t => t.name == HipsName);
        _neck = bones.FirstOrDefault(t => t.name == NeckName);

        if (_stage == null) _report.failures.Add("the character has no StyleTransferStage");
        if (_input == null) _report.failures.Add("the character's control input takes no direction");
        if (_hips == null || _neck == null) _report.failures.Add($"no {HipsName}/{NeckName} under the character");
        if (_report.failures.Count > 0)
        {
            Finish();
            return false;
        }

        _report.gated = !Environment.GetCommandLineArgs().Contains(UngatedArgument);
        _stage.gateEnabled = _report.gated;

        _lastPosition = _character.transform.position;
        _lastTime = Time.time;
        return true;
    }

    private static void Sample(float time)
    {
        var segment = SegmentAt(time);
        if (segment < 0)
        {
            Finish();
            return;
        }

        if (segment != _segmentIndex) EnterSegment(segment);
        _input.SetMovementDirection(Vector2.up);

        var position = _character.transform.position;
        var dt = time - _lastTime;
        var speed = dt > 0f ? Vector3.ProjectOnPlane(position - _lastPosition, Vector3.up).magnitude / dt : 0f;
        _lastPosition = position;
        _lastTime = time;

        if (!IsFinite(_hips.position)) _report.poseStayedFinite = false;
        if (time - SegmentStart(segment) < SettleSeconds || dt <= 0f) return;

        var forward = _character.transform.forward;
        var spine = _neck.position - _hips.position;
        var lean = Mathf.Atan2(Vector3.Dot(spine, forward), Vector3.Dot(spine, Vector3.up)) * Mathf.Rad2Deg;

        var current = _report.segments[segment];
        current.samples++;
        current.LeanSum += lean;
        current.SpeedSum += speed;
        current.GateSum += _stage.isEnabled && _stage.gateEnabled ? _stage.Gate : 0f;

        if (_stage.isEnabled)
        {
            _phaseMin = Mathf.Min(_phaseMin, _stage.Phase);
            _phaseMax = Mathf.Max(_phaseMax, _stage.Phase);
        }
    }

    private static int SegmentAt(float time) =>
        time < StyledEnd ? 0 : time < UnstyledEnd ? 1 : time < StylizedEnd ? 2 : time < SwitchEnd ? 3 : -1;

    private static float SegmentStart(int segment) =>
        segment switch { 0 => 0f, 1 => StyledEnd, 2 => UnstyledEnd, _ => StylizedEnd };

    private static void EnterSegment(int segment)
    {
        _segmentIndex = segment;
        switch (segment)
        {
            case 0:
                _report.segments.Add(new Segment { name = "styled, root preserved" });
                _stage.isEnabled = true;
                _stage.rootMotion = StyleTransferStage.RootMotionMode.Preserve;
                break;
            case 1:
                _report.segments.Add(new Segment { name = "unstyled" });
                _stage.isEnabled = false;
                break;
            case 2:
                _report.segments.Add(new Segment { name = "styled, root stylized" });
                _stage.isEnabled = true;
                _stage.rootMotion = StyleTransferStage.RootMotionMode.Stylize;
                break;
            case 3:
                _report.segments.Add(new Segment { name = "switched to " + SwitchStyleName });
                _stage.rootMotion = StyleTransferStage.RootMotionMode.Preserve;
                _stage.SetStyle(GameplayTagConfig.instance.GetTag(SwitchStyleName), _stage.styleBlendHalfLife);
                break;
        }
    }

    private static bool IsFinite(Vector3 v) => float.IsFinite(v.x) && float.IsFinite(v.y) && float.IsFinite(v.z);

    private static void Finish()
    {
        EditorApplication.update -= Tick;
        Application.logMessageReceived -= OnLog;
        _armed = false;
        File.Delete(ArmedFlagPath);

        foreach (var segment in _report.segments) segment.Finish();
        _report.phaseSpread = _phaseMax > _phaseMin ? _phaseMax - _phaseMin : 0f;
        Judge(_report);

        Directory.CreateDirectory(Path.GetDirectoryName(ReportPath)!);
        File.WriteAllText(ReportPath, JsonUtility.ToJson(_report, true));
        Debug.Log($"[Style Smoke] {(_report.passed ? "PASSED" : "FAILED")}: {JsonUtility.ToJson(_report)}");

        if (Application.isBatchMode)
        {
            EditorApplication.Exit(_report.passed ? 0 : 1);
            return;
        }

        EditorApplication.ExitPlaymode();
    }

    private static void Judge(Report report)
    {
        if (report.errorsLogged > 0) report.failures.Add($"{report.errorsLogged} errors logged");
        if (!report.poseStayedFinite) report.failures.Add("the pose went non-finite");
        if (report.segments.Count < 4 || report.segments.Any(s => s.samples == 0))
        {
            report.failures.Add("not every segment was sampled");
        }
        else
        {
            var styled = report.segments[0];
            var unstyled = report.segments[1];
            if (report.gated && styled.meanGate < MinMeanGate) report.failures.Add($"mean gate {styled.meanGate:F2}");
            if (Mathf.Abs(styled.meanLeanDegrees - unstyled.meanLeanDegrees) < MinLeanChangeDegrees)
            {
                report.failures.Add("the style did not change the torso lean");
            }
        }

        if (report.phaseSpread < 1f) report.failures.Add("the gait phase did not advance");
        report.passed = report.failures.Count == 0;
    }
}
}
