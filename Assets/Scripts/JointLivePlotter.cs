using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif
using XCharts.Runtime;

public class JointLivePlotter : MonoBehaviour
{
    [Header("Target")]
    public Transform targetJoint;

    public enum PlottableType
    {
        PositionX,
        PositionY,
        PositionZ,
        VelocityX,
        VelocityY,
        VelocityZ,
        AccelerationX,
        AccelerationY,
        AccelerationZ,
        AngleX,
        AngleY,
        AngleZ,
        AngularVelocityX,
        AngularVelocityY,
        AngularVelocityZ,
    }

    [Header("Chart Settings")]
    public GameObject chartTemplate;

    public List<PlottableType> plots;
    [Tooltip("How many seconds of data to show on the graph")]
    public float timeWindow = 5f;
    [Tooltip("How often to sample the data (seconds)")]
    public float updateInterval = 0.05f;

    private readonly List<LineChart> _charts = new List<LineChart>();
    private float _timer;
    private Vector3 _lastPosition;
    private Vector3 _lastVelocity;
    private Vector3 _lastAngles;

    void Start()
    {
        RebuildCharts();

        _lastPosition = targetJoint != null ? targetJoint.position : Vector3.zero;
        _lastAngles = targetJoint != null ? targetJoint.eulerAngles : Vector3.zero;

        UpdateTimeWindow();
    }

    void Update()
    {
        if (targetJoint == null || _charts.Count == 0) return;

        _timer += Time.deltaTime;
        if (_timer >= updateInterval)
        {
            PlotData();
            _timer = 0f;
        }
    }

    private void PlotData()
    {
        // Finite differences over one sampling interval.
        var position = targetJoint.position;
        var velocity = (position - _lastPosition) / updateInterval;
        var acceleration = (velocity - _lastVelocity) / updateInterval;

        var angles = targetJoint.eulerAngles;
        var angularVelocity = new Vector3(
            Mathf.DeltaAngle(_lastAngles.x, angles.x),
            Mathf.DeltaAngle(_lastAngles.y, angles.y),
            Mathf.DeltaAngle(_lastAngles.z, angles.z)) / updateInterval;

        var timeLabel = Time.time.ToString("F1");

        var count = Mathf.Min(plots.Count, _charts.Count);
        for (var i = 0; i < count; i++)
        {
            var chart = _charts[i];
            if (chart == null) continue;

            var value = GetPlotValue(plots[i], position, velocity, acceleration, angles, angularVelocity);
            chart.AddXAxisData(timeLabel);
            chart.AddData(0, value);
        }

        _lastPosition = position;
        _lastVelocity = velocity;
        _lastAngles = angles;
    }

    private static float GetPlotValue(
        PlottableType plot,
        Vector3 position,
        Vector3 velocity,
        Vector3 acceleration,
        Vector3 angles,
        Vector3 angularVelocity)
    {
        switch (plot)
        {
            case PlottableType.PositionX: return position.x;
            case PlottableType.PositionY: return position.y;
            case PlottableType.PositionZ: return position.z;
            case PlottableType.VelocityX: return velocity.x;
            case PlottableType.VelocityY: return velocity.y;
            case PlottableType.VelocityZ: return velocity.z;
            case PlottableType.AccelerationX: return acceleration.x;
            case PlottableType.AccelerationY: return acceleration.y;
            case PlottableType.AccelerationZ: return acceleration.z;
            case PlottableType.AngleX: return angles.x;
            case PlottableType.AngleY: return angles.y;
            case PlottableType.AngleZ: return angles.z;
            case PlottableType.AngularVelocityX: return angularVelocity.x;
            case PlottableType.AngularVelocityY: return angularVelocity.y;
            case PlottableType.AngularVelocityZ: return angularVelocity.z;
            default: return 0f;
        }
    }

    /// <summary>Applies <see cref="timeWindow"/> to every chart; call after changing it at runtime.</summary>
    public void UpdateTimeWindow()
    {
        var maxDataPoints = Mathf.CeilToInt(timeWindow / updateInterval);

        foreach (var chart in _charts)
        {
            if (chart == null) continue;

            var xAxis = chart.GetChartComponent<XAxis>();
            if (xAxis != null) xAxis.maxCache = maxDataPoints;

            foreach (var serie in chart.series)
            {
                serie.maxCache = maxDataPoints;
            }
        }
    }

    public void RebuildCharts()
    {
        for (var i = transform.childCount - 1; i >= 0; i--)
        {
            var child = transform.GetChild(i);
            if (child == chartTemplate.transform) continue;
#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                // Outside play mode this is reached from inspector callbacks, where destroying a
                // child immediately is refused, so it has to wait for the next editor tick.
                EditorApplication.delayCall += () =>
                {
                    if (this == null) return;
                    if (child.gameObject != null)
                    {
                        DestroyImmediate(child.gameObject);
                    }
                };
                continue;
            }
#endif
            Destroy(child.gameObject);
        }

        _charts.Clear();

        if (plots == null) plots = new List<PlottableType>();

        foreach (var plot in plots)
        {
            var instance = Instantiate(chartTemplate, transform);
            instance.SetActive(true);
            instance.name = chartTemplate.name + "_" + plot;
            var lineChart = instance.GetComponent<LineChart>();

            lineChart.ClearData();
            _charts.Add(lineChart);

            var title = lineChart.GetChartComponent<Title>();
            title.text = plot.ToString();
        }
    }

    private void OnValidate()
    {
        RebuildCharts();
        UpdateTimeWindow();
    }
}
