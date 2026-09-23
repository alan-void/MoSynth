using System;
using System.Collections.Generic;
using UnityEngine;

namespace MotionMatching
{
    /// <summary>
    /// Scene-wide registry of <see cref="Obstacle"/>s, so crowd control inputs can find who to avoid
    /// without each searching the scene itself.
    /// </summary>
    /// <remarks>
    /// Runs early (order -1000) so obstacles have somewhere to register before control inputs look
    /// for them. Membership changes are announced once in LateUpdate, so a batch of spawns costs
    /// subscribers one rebuild instead of one each.
    /// </remarks>
    [DefaultExecutionOrder(-1000)]
    public class ObstacleManager : MonoBehaviour
    {
        public static ObstacleManager Instance { get; private set; }

        public event Action<List<Obstacle>> OnObstaclesUpdated;

        private readonly List<Obstacle> _obstacles = new();
        private bool _obstaclesUpdated;

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
            }
            else
            {
                Debug.LogError($"Multiple instances of {nameof(ObstacleManager)} detected.");
            }
        }

        public List<Obstacle> GetObstacles()
        {
            return _obstacles;
        }

        public void RegisterObstacle(Obstacle obstacle)
        {
            if (!_obstacles.Contains(obstacle))
            {
                _obstacles.Add(obstacle);
                _obstaclesUpdated = true;
            }
        }

        public void UnregisterObstacle(Obstacle obstacle)
        {
            if (_obstacles.Remove(obstacle))
            {
                _obstaclesUpdated = true;
            }
        }

        private void LateUpdate()
        {
            if (_obstaclesUpdated)
            {
                OnObstaclesUpdated?.Invoke(_obstacles);
                _obstaclesUpdated = false;
            }
        }
    }
}