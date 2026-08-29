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

    private readonly List<LineChart> charts = new List<LineChart>();
    private readonly List<GameObject> spawnedCharts = new List<GameObject>();
    private float timer = 0f;
    private Vector3 lastPosition;
    private Vector3 lastVelocity;
    private Vector3 lastAngles;

    private bool rebuildScheduled;

    void Start()
    {
        RebuildCharts();

        lastPosition = targetJoint != null ? targetJoint.position : Vector3.zero;
        lastAngles = targetJoint != null ? targetJoint.eulerAngles : Vector3.zero;

        UpdateTimeWindow();
    }

    void Update()
    {
        if (targetJoint == null || charts.Count == 0) return;

        timer += Time.deltaTime;
        
        // Only sample data at the specified interval for performance and readability
        if (timer >= updateInterval)
        {
            PlotData();
            timer = 0f;
        }
    }

    private void PlotData()
    {
        // 1. Calculate Kinematics (Finite Differences)
        Vector3 currentPosition = targetJoint.position;
        Vector3 currentVelocity = (currentPosition - lastPosition) / updateInterval;
        Vector3 currentAcceleration = (currentVelocity - lastVelocity) / updateInterval;

        Vector3 currentAngles = targetJoint.eulerAngles;
        Vector3 currentAngularVelocity = new Vector3(
            Mathf.DeltaAngle(lastAngles.x, currentAngles.x),
            Mathf.DeltaAngle(lastAngles.y, currentAngles.y),
            Mathf.DeltaAngle(lastAngles.z, currentAngles.z)) / updateInterval;

        // 2. Add Data to Charts
        string timeStr = Time.time.ToString("F1");

        int count = Mathf.Min(plots.Count, charts.Count);
        for (int i = 0; i < count; i++)
        {
            var chart = charts[i];
            if (chart == null) continue;

            float value = GetPlotValue(plots[i], currentPosition, currentVelocity, currentAcceleration, currentAngles, currentAngularVelocity);
            chart.AddXAxisData(timeStr);
            chart.AddData(0, value);
        }

        // 3. Cache states for next frame calculation
        lastPosition = currentPosition;
        lastVelocity = currentVelocity;
        lastAngles = currentAngles;
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

    // Call this if you change the timeWindow variable at runtime
    public void UpdateTimeWindow()
    {
        int maxDataPoints = Mathf.CeilToInt(timeWindow / updateInterval);

        for (int i = 0; i < charts.Count; i++)
        {
            var chart = charts[i];
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
        for (int i = transform.childCount - 1; i >= 0; i--)
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
        
        spawnedCharts.Clear();
        charts.Clear();

        if (plots == null) plots = new List<PlottableType>();


        for (int i = 0; i < plots.Count; i++)
        {
            var instance = Instantiate(chartTemplate, transform);
            instance.SetActive(true);
            instance.name = chartTemplate.name + "_" + plots[i];
            var lineChart = instance.GetComponent<LineChart>();

            lineChart.ClearData();
            lineChart.ClearData();
            charts.Add(lineChart);
            spawnedCharts.Add(instance);

            var title = lineChart.GetChartComponent<Title>();
            title.text = plots[i].ToString();
        }

    }

    private void OnValidate()
    {
        RebuildCharts();
        UpdateTimeWindow();
    }

    // private void OnDisable()
    // {
    //     if (!Application.isPlaying) return;
    //
    //     for (int i = 0; i < spawnedCharts.Count; i++)
    //     {
    //         if (spawnedCharts[i] != null) Destroy(spawnedCharts[i]);
    //     }
    //
    //     spawnedCharts.Clear();
    //     charts.Clear();
    // }
}
