using System.Globalization;
using System.Text;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;
using Object = UnityEngine.Object;

namespace AnimationTools.Editor
{
/// <summary>
/// Draws what a method consumed on its last tick, from its <see cref="ISynthesisInputSource"/>:
/// world-space points and arrows coloured by <see cref="SynthesisInputRole"/>, and a HUD panel with
/// the numbers, a search flash and a phase dial.
/// </summary>
/// <remarks>
/// Only reads the stage's record, and nothing it creates has a collider, so a run draws the same
/// motion with or without it.
/// </remarks>
internal sealed class BenchmarkInputOverlay
{
    private const int PointCapacity = 192;
    private const int ArrowCapacity = 192;

    /// <summary>Raised off what it marks, so a ground-level feature clears the floor.</summary>
    private const float Lift = 0.02f;

    private const float RequestRadius = 0.045f;
    private const float StateRadius = 0.035f;
    private const float AnswerRingRadius = 0.08f;
    private const float AnswerRingWidth = 0.022f;
    private const float ShaftWidth = 0.026f;
    private const float AnswerShaftWidth = 0.018f;
    private const float HeadWidth = 0.085f;
    private const float HeadLength = 0.09f;

    private const float SearchFlashSeconds = 0.25f;
    private const float ChangeFlashSeconds = 0.6f;

    private const float PanelWidth = 600f;
    private const float Padding = 20f;
    private const int TitleSize = 30;
    private const int LineSize = 26;
    private const float LineHeight = 33f;
    private const float DialSize = 104f;

    public static readonly Color RequestColor = new(0.10f, 0.80f, 0.35f);
    public static readonly Color AnswerColor = new(0.62f, 0.28f, 0.98f);
    public static readonly Color StateColor = new(0.10f, 0.72f, 0.92f);

    private static readonly Color PanelColor = new(0.05f, 0.06f, 0.08f, 0.88f);
    private static readonly Color DimColor = new(0.45f, 0.48f, 0.52f);
    private static readonly Color SearchColor = new(1f, 0.85f, 0.25f);
    private static readonly Color ChangeColor = new(1f, 0.35f, 0.55f);

    private readonly GameObject _root;
    private readonly Material[] _roleMaterials = new Material[3];
    private readonly Material _lineMaterial;
    private readonly Mesh _sphere;
    private readonly Mesh _ring;
    private readonly Texture2D _ringTexture;
    private readonly Sprite _ringSprite;

    private readonly MeshRenderer[] _points = new MeshRenderer[PointCapacity];
    private readonly MeshFilter[] _pointMeshes = new MeshFilter[PointCapacity];
    private readonly Arrow[] _arrows = new Arrow[ArrowCapacity];
    private int _activePoints;
    private int _activeArrows;

    private readonly RectTransform _panel;
    private readonly Text _title;
    private readonly Text _searchFlag;
    private readonly Text _changeFlag;
    private readonly Text _legend;
    private readonly Text _readouts;
    private readonly RectTransform _dial;
    private readonly Image _dialFill;
    private readonly RectTransform _dialNeedle;
    private readonly Text _dialLabel;
    private readonly StringBuilder _text = new(256);

    private float _sinceSearch = float.PositiveInfinity;
    private float _sinceChange = float.PositiveInfinity;

    public BenchmarkInputOverlay(Transform parent, RectTransform canvas, Font font)
    {
        _root = new GameObject("MethodInputs") { hideFlags = HideFlags.HideAndDontSave };
        _root.transform.SetParent(parent, false);

        var shader = Shader.Find("Sprites/Default");
        _roleMaterials[(int)SynthesisInputRole.Request] = NewMaterial(shader, "InputRequest", RequestColor);
        _roleMaterials[(int)SynthesisInputRole.Answer] = NewMaterial(shader, "InputAnswer", AnswerColor);
        _roleMaterials[(int)SynthesisInputRole.State] = NewMaterial(shader, "InputState", StateColor);
        _lineMaterial = NewMaterial(shader, "InputLine", Color.white);

        _sphere = Sphere(12, 8);
        _ring = Ring(32, 1f - AnswerRingWidth / AnswerRingRadius);
        _ringTexture = RingTexture(128, 0.74f);
        _ringSprite = Sprite.Create(_ringTexture, new Rect(0f, 0f, 128f, 128f), new Vector2(0.5f, 0.5f));
        _ringSprite.hideFlags = HideFlags.HideAndDontSave;

        for (var i = 0; i < PointCapacity; i++)
        {
            _points[i] = CreatePoint();
            _pointMeshes[i] = _points[i].GetComponent<MeshFilter>();
        }
        for (var i = 0; i < ArrowCapacity; i++)
        {
            _arrows[i] = new Arrow { Shaft = CreateLine("Shaft"), Head = CreateLine("Head") };
        }

        _panel = CreateRect("InputPanel", canvas);
        _panel.gameObject.AddComponent<Image>().color = PanelColor;
        _panel.anchorMin = _panel.anchorMax = _panel.pivot = new Vector2(1f, 1f);
        _panel.anchoredPosition = new Vector2(-32f, -32f);

        _title = AddText(_panel, font, TitleSize, FontStyle.Bold, Color.white, TextAnchor.UpperLeft);
        Place(_title.rectTransform, Padding, Padding * 0.6f, PanelWidth - 2f * Padding, TitleSize * 1.3f);

        _changeFlag = AddText(_panel, font, LineSize - 2, FontStyle.Bold, DimColor, TextAnchor.UpperRight);
        _changeFlag.text = "● new match";
        Place(_changeFlag.rectTransform, PanelWidth - Padding - 170f, Padding * 0.6f + 3f, 170f, LineSize * 1.3f);

        _searchFlag = AddText(_panel, font, LineSize - 2, FontStyle.Bold, DimColor, TextAnchor.UpperRight);
        _searchFlag.text = "● search";
        Place(_searchFlag.rectTransform, PanelWidth - Padding - 290f, Padding * 0.6f + 3f, 115f, LineSize * 1.3f);

        _legend = AddText(_panel, font, LineSize - 2, FontStyle.Normal, Color.white, TextAnchor.UpperLeft);
        _legend.supportRichText = true;
        Place(_legend.rectTransform, Padding, Padding * 0.6f + 44f, PanelWidth - 2f * Padding, LineSize * 1.3f);

        _readouts = AddText(_panel, font, LineSize, FontStyle.Normal, new Color(0.86f, 0.88f, 0.91f),
            TextAnchor.UpperLeft);
        _readouts.lineSpacing = LineHeight / (LineSize * 1.15f);

        _dial = CreateRect("PhaseDial", _panel);
        _dial.anchorMin = _dial.anchorMax = _dial.pivot = new Vector2(1f, 1f);
        _dial.sizeDelta = new Vector2(DialSize, DialSize);
        _dial.anchoredPosition = new Vector2(-Padding, -(Padding * 0.6f + 84f));

        var dialBack = CreateRect("Back", _dial);
        Stretch(dialBack);
        dialBack.gameObject.AddComponent<Image>().sprite = _ringSprite;
        dialBack.GetComponent<Image>().color = new Color(1f, 1f, 1f, 0.18f);

        var fill = CreateRect("Fill", _dial);
        Stretch(fill);
        _dialFill = fill.gameObject.AddComponent<Image>();
        _dialFill.sprite = _ringSprite;
        _dialFill.color = StateColor;
        _dialFill.type = Image.Type.Filled;
        _dialFill.fillMethod = Image.FillMethod.Radial360;
        _dialFill.fillOrigin = (int)Image.Origin360.Top;
        _dialFill.fillClockwise = true;

        _dialNeedle = CreateRect("Needle", _dial);
        _dialNeedle.anchorMin = _dialNeedle.anchorMax = new Vector2(0.5f, 0.5f);
        _dialNeedle.pivot = new Vector2(0.5f, 0f);
        _dialNeedle.sizeDelta = new Vector2(4f, DialSize * 0.5f);
        _dialNeedle.gameObject.AddComponent<Image>().color = Color.white;

        _dialLabel = AddText(_dial, font, LineSize - 4, FontStyle.Bold, Color.white, TextAnchor.MiddleCenter);
        Stretch(_dialLabel.rectTransform);
        _dialLabel.rectTransform.anchoredPosition = new Vector2(0f, -DialSize * 0.62f);

        Hide();
    }

    /// <summary>Starts a run with nothing drawn.</summary>
    public void BeginRun()
    {
        _sinceSearch = float.PositiveInfinity;
        _sinceChange = float.PositiveInfinity;
        Hide();
    }

    public void EndRun() => Hide();

    /// <summary>Redraws from the stage's record of the tick just applied. Null hides everything.</summary>
    public void Tick(SynthesisInputs inputs, float deltaTime)
    {
        if (inputs == null)
        {
            Hide();
            return;
        }

        DrawMarkers(inputs);
        UpdatePanel(inputs, deltaTime);
    }

    private void DrawMarkers(SynthesisInputs inputs)
    {
        var points = 0;
        var arrows = 0;
        var lift = new float3(0f, Lift, 0f);

        for (var i = 0; i < inputs.MarkerCount; i++)
        {
            ref readonly var marker = ref inputs.GetMarker(i);
            var position = marker.Position + lift;

            if (!marker.IsArrow)
            {
                if (points == PointCapacity) continue;
                PlacePoint(points++, marker.Role, position);
                continue;
            }

            var length = math.length(marker.Vector);
            if (length < 0.01f || arrows == ArrowCapacity) continue;
            PlaceArrow(_arrows[arrows++], marker.Role, position, marker.Vector / length, length);
        }

        for (var i = points; i < _activePoints; i++) _points[i].gameObject.SetActive(false);
        for (var i = arrows; i < _activeArrows; i++) _arrows[i].SetActive(false);
        _activePoints = points;
        _activeArrows = arrows;
    }

    private void PlacePoint(int index, SynthesisInputRole role, float3 position)
    {
        var point = _points[index];
        var isAnswer = role == SynthesisInputRole.Answer;
        _pointMeshes[index].sharedMesh = isAnswer ? _ring : _sphere;
        point.sharedMaterial = _roleMaterials[(int)role];

        var radius = isAnswer ? AnswerRingRadius : role == SynthesisInputRole.State ? StateRadius : RequestRadius;
        point.transform.SetPositionAndRotation(position, Quaternion.identity);
        point.transform.localScale = new Vector3(radius, radius, radius);
        point.gameObject.SetActive(true);
    }

    private static void PlaceArrow(Arrow arrow, SynthesisInputRole role, float3 tail, float3 direction,
        float length)
    {
        var color = RoleColor(role);
        var headLength = math.min(HeadLength, 0.45f * length);
        var tip = tail + direction * length;
        var headBase = tip - direction * headLength;

        arrow.Shaft.startColor = arrow.Shaft.endColor = color;
        arrow.Shaft.widthMultiplier = role == SynthesisInputRole.Answer ? AnswerShaftWidth : ShaftWidth;
        arrow.Shaft.SetPosition(0, tail);
        arrow.Shaft.SetPosition(1, headBase);

        // A two-point line tapering to zero width is a triangle: the arrowhead.
        arrow.Head.startColor = arrow.Head.endColor = color;
        arrow.Head.startWidth = HeadWidth;
        arrow.Head.endWidth = 0f;
        arrow.Head.SetPosition(0, headBase);
        arrow.Head.SetPosition(1, tip);

        arrow.SetActive(true);
    }

    private void UpdatePanel(SynthesisInputs inputs, float deltaTime)
    {
        _panel.gameObject.SetActive(true);
        _title.text = inputs.Title;

        _sinceSearch = inputs.Searched ? 0f : _sinceSearch + deltaTime;
        _sinceChange = inputs.AnswerChanged ? 0f : _sinceChange + deltaTime;
        _searchFlag.gameObject.SetActive(inputs.Searches);
        _changeFlag.gameObject.SetActive(inputs.Searches);
        _searchFlag.color = Flash(SearchColor, _sinceSearch, SearchFlashSeconds);
        _changeFlag.color = Flash(ChangeColor, _sinceChange, ChangeFlashSeconds);

        _legend.text = Legend(inputs);

        _text.Clear();
        for (var i = 0; i < inputs.ReadoutCount; i++)
        {
            ref readonly var readout = ref inputs.GetReadout(i);
            if (i > 0) _text.Append('\n');
            _text.Append(readout.Label).Append("   ");
            if (readout.Text != null) _text.Append(readout.Text);
            else if (float.IsNaN(readout.Value) || float.IsInfinity(readout.Value)) _text.Append('–');
            else _text.Append(readout.Value.ToString(readout.Format, CultureInfo.InvariantCulture));
            if (readout.Text == null && readout.Unit != null) _text.Append(readout.Unit);
        }
        _readouts.text = _text.ToString();

        var readoutTop = Padding * 0.6f + 84f;
        var readoutHeight = math.max(1, inputs.ReadoutCount) * LineHeight;
        var textWidth = PanelWidth - 2f * Padding - (inputs.HasPhase ? DialSize + Padding : 0f);
        Place(_readouts.rectTransform, Padding, readoutTop, textWidth, readoutHeight);

        var dialHeight = inputs.HasPhase ? DialSize + LineHeight : 0f;
        _panel.sizeDelta = new Vector2(PanelWidth, readoutTop + math.max(readoutHeight, dialHeight) + Padding);

        _dial.gameObject.SetActive(inputs.HasPhase);
        if (!inputs.HasPhase) return;

        var phase = inputs.Phase - math.floor(inputs.Phase);
        _dialFill.fillAmount = phase;
        _dialNeedle.localRotation = Quaternion.Euler(0f, 0f, -360f * phase);
        _dialLabel.text = "phase";
    }

    /// <summary>One coloured swatch per role the record actually holds, in the method's own words.</summary>
    private string Legend(SynthesisInputs inputs)
    {
        bool request = false, answer = false, state = false;
        for (var i = 0; i < inputs.MarkerCount; i++)
        {
            switch (inputs.GetMarker(i).Role)
            {
                case SynthesisInputRole.Request: request = true; break;
                case SynthesisInputRole.Answer: answer = true; break;
                case SynthesisInputRole.State: state = true; break;
            }
        }

        _text.Clear();
        if (request) AppendSwatch(RequestColor, inputs.RequestLabel);
        if (answer) AppendSwatch(AnswerColor, inputs.AnswerLabel);
        if (state) AppendSwatch(StateColor, inputs.StateLabel);
        return _text.ToString();
    }

    private void AppendSwatch(Color color, string label)
    {
        if (_text.Length > 0) _text.Append("     ");
        _text.Append("<color=#").Append(ColorUtility.ToHtmlStringRGB(color)).Append(">■</color> ")
            .Append(label);
    }

    private static Color Flash(Color lit, float since, float seconds)
    {
        var t = math.saturate(since / seconds);
        return Color.Lerp(lit, DimColor, t * t);
    }

    private static Color RoleColor(SynthesisInputRole role) => role switch
    {
        SynthesisInputRole.Answer => AnswerColor,
        SynthesisInputRole.State => StateColor,
        _ => RequestColor
    };

    private void Hide()
    {
        for (var i = 0; i < _activePoints; i++) _points[i].gameObject.SetActive(false);
        for (var i = 0; i < _activeArrows; i++) _arrows[i].SetActive(false);
        _activePoints = 0;
        _activeArrows = 0;
        _panel.gameObject.SetActive(false);
    }

    private MeshRenderer CreatePoint()
    {
        var go = new GameObject("Point") { hideFlags = HideFlags.HideAndDontSave };
        go.transform.SetParent(_root.transform, false);
        go.AddComponent<MeshFilter>().sharedMesh = _sphere;
        var renderer = go.AddComponent<MeshRenderer>();
        renderer.sharedMaterial = _roleMaterials[0];
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        go.SetActive(false);
        return renderer;
    }

    private LineRenderer CreateLine(string name)
    {
        var go = new GameObject(name) { hideFlags = HideFlags.HideAndDontSave };
        go.transform.SetParent(_root.transform, false);
        var line = go.AddComponent<LineRenderer>();
        line.sharedMaterial = _lineMaterial;
        line.useWorldSpace = true;
        line.alignment = LineAlignment.View;
        line.positionCount = 2;
        line.numCapVertices = 0;
        line.shadowCastingMode = ShadowCastingMode.Off;
        line.receiveShadows = false;
        go.SetActive(false);
        return line;
    }

    private static Material NewMaterial(Shader shader, string name, Color color)
    {
        return new Material(shader) { name = name, color = color, hideFlags = HideFlags.HideAndDontSave };
    }

    private static RectTransform CreateRect(string name, Transform parent)
    {
        var go = new GameObject(name, typeof(RectTransform)) { hideFlags = HideFlags.HideAndDontSave };
        go.transform.SetParent(parent, false);
        return (RectTransform)go.transform;
    }

    private static Text AddText(RectTransform parent, Font font, int size, FontStyle style, Color color,
        TextAnchor alignment)
    {
        var text = CreateRect("Text", parent).gameObject.AddComponent<Text>();
        text.font = font;
        text.fontSize = size;
        text.fontStyle = style;
        text.color = color;
        text.alignment = alignment;
        text.horizontalOverflow = HorizontalWrapMode.Overflow;
        text.verticalOverflow = VerticalWrapMode.Overflow;
        return text;
    }

    /// <summary>Puts a rect at a top-left offset inside its parent, in reference pixels.</summary>
    private static void Place(RectTransform rect, float left, float top, float width, float height)
    {
        rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = new Vector2(left, -top);
        rect.sizeDelta = new Vector2(width, height);
    }

    private static void Stretch(RectTransform rect)
    {
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = rect.offsetMax = Vector2.zero;
    }

    /// <summary>A unit-radius sphere; unlit, it reads as a disc from any view.</summary>
    private static Mesh Sphere(int segments, int rings)
    {
        var vertices = new Vector3[(segments + 1) * (rings + 1)];
        for (var r = 0; r <= rings; r++)
        {
            var polar = math.PI * r / rings;
            for (var s = 0; s <= segments; s++)
            {
                var azimuth = 2f * math.PI * s / segments;
                vertices[r * (segments + 1) + s] = new Vector3(math.sin(polar) * math.cos(azimuth), math.cos(polar),
                    math.sin(polar) * math.sin(azimuth));
            }
        }

        var triangles = new int[segments * rings * 6];
        var t = 0;
        for (var r = 0; r < rings; r++)
        for (var s = 0; s < segments; s++)
        {
            var a = r * (segments + 1) + s;
            var b = a + segments + 1;
            triangles[t++] = a;
            triangles[t++] = a + 1;
            triangles[t++] = b;
            triangles[t++] = a + 1;
            triangles[t++] = b + 1;
            triangles[t++] = b;
        }

        var mesh = new Mesh { name = "InputSphere", vertices = vertices, triangles = triangles,
            hideFlags = HideFlags.HideAndDontSave };
        mesh.RecalculateBounds();
        return mesh;
    }

    /// <summary>A flat unit-radius annulus in the XZ plane, inner radius <paramref name="inner"/>.</summary>
    private static Mesh Ring(int segments, float inner)
    {
        var vertices = new Vector3[segments * 2];
        var triangles = new int[segments * 6];
        for (var i = 0; i < segments; i++)
        {
            var angle = 2f * math.PI * i / segments;
            var direction = new Vector3(math.sin(angle), 0f, math.cos(angle));
            vertices[i * 2] = direction * inner;
            vertices[i * 2 + 1] = direction;

            var next = (i + 1) % segments;
            triangles[i * 6] = i * 2;
            triangles[i * 6 + 1] = i * 2 + 1;
            triangles[i * 6 + 2] = next * 2 + 1;
            triangles[i * 6 + 3] = i * 2;
            triangles[i * 6 + 4] = next * 2 + 1;
            triangles[i * 6 + 5] = next * 2;
        }

        var mesh = new Mesh { name = "InputRing", vertices = vertices, triangles = triangles,
            hideFlags = HideFlags.HideAndDontSave };
        mesh.RecalculateBounds();
        return mesh;
    }

    /// <summary>A white anti-aliased ring for the phase dial, inner radius <paramref name="inner"/> of the outer.</summary>
    private static Texture2D RingTexture(int size, float inner)
    {
        var texture = new Texture2D(size, size, TextureFormat.RGBA32, false)
        {
            hideFlags = HideFlags.HideAndDontSave,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };

        var pixels = new Color32[size * size];
        var half = size * 0.5f;
        for (var y = 0; y < size; y++)
        for (var x = 0; x < size; x++)
        {
            var radius = math.length(new float2(x + 0.5f - half, y + 0.5f - half)) / half;
            var edge = 1.5f / half;
            var alpha = math.saturate((1f - radius) / edge) * math.saturate((radius - inner) / edge);
            pixels[y * size + x] = new Color32(255, 255, 255, (byte)(alpha * 255f));
        }

        texture.SetPixels32(pixels);
        texture.Apply(false, true);
        return texture;
    }

    public void Dispose()
    {
        foreach (var material in _roleMaterials)
        {
            if (material != null) Object.DestroyImmediate(material);
        }

        if (_lineMaterial != null) Object.DestroyImmediate(_lineMaterial);
        if (_sphere != null) Object.DestroyImmediate(_sphere);
        if (_ring != null) Object.DestroyImmediate(_ring);
        if (_ringSprite != null) Object.DestroyImmediate(_ringSprite);
        if (_ringTexture != null) Object.DestroyImmediate(_ringTexture);
        if (_root != null) Object.DestroyImmediate(_root);
        if (_panel != null) Object.DestroyImmediate(_panel.gameObject);
    }

    private sealed class Arrow
    {
        public LineRenderer Shaft;
        public LineRenderer Head;

        public void SetActive(bool active)
        {
            Shaft.gameObject.SetActive(active);
            Head.gameObject.SetActive(active);
        }
    }
}
}
