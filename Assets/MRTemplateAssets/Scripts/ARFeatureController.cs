using UnityEngine.Events;
using UnityEngine.XR.ARFoundation;

namespace UnityEngine.XR.Templates.MR
{
    /// <summary>
    /// Utility class used to control various AR features like occlusion, AR bounding boxes, and AR planes.
    /// </summary>
    public class ARFeatureController : MonoBehaviour
    {
        [SerializeField, Tooltip("AR Occlusion Manager that is in charge of changing MR Occlusion Features.")]
        OcclusionManager m_OcclusionManager;

        /// <summary>
        /// AR Occlusion Manager that is in charge of changing MR Occlusion Features.
        /// </summary>
        public OcclusionManager occlusionManager
        {
            get => m_OcclusionManager;
            set => m_OcclusionManager = value;
        }

        [SerializeField, Tooltip("AR Plane Manager that is in charge of spawning new AR Plane prefabs into the scene.")]
        ARPlaneManager m_PlaneManager;

        /// <summary>
        /// AR Plane Manager that is in charge of spawning new AR Plane prefabs into the scene.
        /// </summary>
        public ARPlaneManager planeManager
        {
            get => m_PlaneManager;
            set => m_PlaneManager = value;
        }

        [SerializeField, Tooltip("Toggle that dictates whether AR Planes should be visualized at runtime.")]
        bool m_PlaneVisualsEnabled = true;

        /// <summary>
        /// Toggle that dictates whether AR Planes should be visualized at runtime.
        /// </summary>
        public bool PlaneVisualsEnabled => m_PlaneVisualsEnabled;

        [SerializeField, Tooltip("AR Bounding Box Manager that is in charge of spawning new AR Bounding Box prefabs into the scene")]
        ARBoundingBoxManager m_BoundingBoxManager;

        /// <summary>
        /// AR Bounding Box Manager that is in charge of spawning new AR Bounding Box prefabs into the scene.
        /// </summary>
        public ARBoundingBoxManager BoundingBoxManager
        {
            get => m_BoundingBoxManager;
            set => m_BoundingBoxManager = value;
        }

        [SerializeField, Tooltip("Toggle that dictates whether AR Bounding Boxes should be visualized at runtime.")]
        bool m_BoundingBoxVisualsEnabled = true;

        /// <summary>
        /// Toggle that dictates whether AR Bounding Boxes should be visualized at runtime.
        /// </summary>
        public bool BoundingBoxVisualsEnabled => m_BoundingBoxVisualsEnabled;

        [SerializeField, Tooltip("Toggle that dictates whether AR Bounding Box visualizations should show additional debug information.")]
        bool m_BoundingBoxDebugInfoEnabled = true;

        /// <summary>
        /// Toggle that dictates whether AR Bounding Box visualizations should show additional debug information.
        /// </summary>
        public bool boundingBoxDebugInfoEnabled => m_BoundingBoxDebugInfoEnabled;

        [Header("Feature Changed Events")]

        [SerializeField]
        UnityEvent<bool> m_OnARPlaneFeatureChanged = new UnityEvent<bool>();

        public UnityEvent<bool> onARPlaneFeatureChanged => m_OnARPlaneFeatureChanged;

        [SerializeField]
        UnityEvent<bool> m_OnARPlaneFeatureVisualizationChanged = new UnityEvent<bool>();

        public UnityEvent<bool> onARPlaneFeatureVisualizationChanged => m_OnARPlaneFeatureVisualizationChanged;

        [SerializeField]
        UnityEvent<bool> m_OnARBoundingBoxFeatureChanged = new UnityEvent<bool>();

        public UnityEvent<bool> onARBoundingBoxFeatureChanged => m_OnARBoundingBoxFeatureChanged;

        [SerializeField]
        UnityEvent<bool> m_OnARBoundingBoxFeatureVisualizationChanged = new UnityEvent<bool>();

        public UnityEvent<bool> onARBoundingBoxFeatureVisualizationChanged => m_OnARBoundingBoxFeatureVisualizationChanged;

        [SerializeField]
        UnityEvent<bool> m_OnARBoundingBoxFeatureDebugVisualizationChanged = new UnityEvent<bool>();

        public UnityEvent<bool> onARBoundingBoxFeatureDebugVisualizationChanged => m_OnARBoundingBoxFeatureDebugVisualizationChanged;

        /// <summary>
        /// Allows access to easily see if the AR Features are enabled and there is at least one bounding box
        /// </summary>
        /// <returns>Will return True if there is 1 or more AR Bounding Boxes found in the AR Scene.</returns>
        public bool HasBoundingBoxes() => m_BoundingBoxManager != null && m_BoundingBoxManager.trackables.count > 0;

        bool m_BoundingBoxManagerEnabled;
        bool m_PlaneManagerEnabled;

        /// <summary>
        /// Functionally turns AR Planes on and off in a scene.
        /// </summary>
        /// <param name="enabled">Whether to enable or disable the currently detected planes.</param>
        public void TogglePlanes(bool enabled)
        {
            if (m_PlaneManager == null)
                return;

            m_PlaneManagerEnabled = enabled;
            m_OnARPlaneFeatureChanged?.Invoke(m_PlaneManagerEnabled);

            if (m_PlaneManagerEnabled)
            {
                m_PlaneManager.enabled = m_PlaneManagerEnabled;
                m_PlaneManager.SetTrackablesActive(m_PlaneManagerEnabled);
            }
            else
            {
                m_PlaneManager.SetTrackablesActive(m_PlaneManagerEnabled);
                m_PlaneManager.enabled = m_PlaneManagerEnabled;
            }
        }

        /// <summary>
        /// Toggles the AR plane visualizations in a scene.
        /// </summary>
        /// <param name="enabled">If <see langword="true"/>, AR plane visualizations will be enabled. Otherwise AR plane visualizations be disabled.</param>
        public void TogglePlaneVisualization(bool enabled)
        {
            if (m_PlaneManager == null)
                return;

            m_PlaneVisualsEnabled = enabled;
            m_OnARPlaneFeatureVisualizationChanged?.Invoke(m_PlaneVisualsEnabled);

            var trackables = m_PlaneManager.trackables;
            foreach (var trackable in trackables)
            {
                if (trackable.TryGetComponent(out FadePlaneMaterial fader))
                {
                    fader.FadePlane(m_PlaneVisualsEnabled);
                }
                if (trackable.TryGetComponent(out ARPlaneMeshVisualizer visualizer))
                {
                    visualizer.enabled = m_PlaneVisualsEnabled;
                }
            }
        }

        /// <summary>
        /// Functionally turns AR Bounding Boxes on and off in a scene.
        /// </summary>
        /// <param name="enabled">Whether to enable or disable the currently detected bounding boxes.</param>
        public void ToggleBoundingBoxes(bool enabled)
        {
            if (m_BoundingBoxManager == null)
                return;

            m_BoundingBoxManagerEnabled = enabled;
            m_OnARBoundingBoxFeatureChanged?.Invoke(m_BoundingBoxManagerEnabled);

            if (m_BoundingBoxManagerEnabled)
            {
                m_BoundingBoxManager.enabled = m_BoundingBoxManagerEnabled;
                m_BoundingBoxManager.SetTrackablesActive(m_BoundingBoxManagerEnabled);
            }
            else
            {
                m_BoundingBoxManager.SetTrackablesActive(m_BoundingBoxManagerEnabled);
                m_BoundingBoxManager.enabled = m_BoundingBoxManagerEnabled;
            }
        }

        /// <summary>
        /// Toggles the AR Bounding Boxes visualizations in a scene.
        /// </summary>
        /// <param name="enabled">If <see langword="true"/>, AR Bounding Boxes visualizations will be enabled. Otherwise AR Bounding Boxes visualizations be disabled.</param>
        public void ToggleBoundingBoxVisualization(bool enabled)
        {
            if (m_BoundingBoxManager == null)
                return;

            m_BoundingBoxVisualsEnabled = enabled;
            m_OnARBoundingBoxFeatureVisualizationChanged?.Invoke(m_BoundingBoxVisualsEnabled);

            var trackables = m_BoundingBoxManager.trackables;
            foreach (var trackable in trackables)
            {
                if (trackable.TryGetComponent(out ARBoundingBoxDebugVisualizer visualizer))
                {
                    visualizer.enabled = m_BoundingBoxVisualsEnabled;
                    visualizer.ShowDebugInfoCanvas(m_BoundingBoxVisualsEnabled && m_BoundingBoxDebugInfoEnabled);
                }
            }
        }

        /// <summary>
        /// Toggles the visualization of the debug information for AR Bounding Boxes.
        /// </summary>
        /// <param name="enabled">If <see langword="true"/>, debug information will be shown for AR Bounding Boxes. Otherwise, debug information will not be shown.</param>
        public void ToggleDebugInfo(bool enabled)
        {
            if (m_BoundingBoxManager == null)
                return;

            m_BoundingBoxDebugInfoEnabled = enabled;
            m_OnARBoundingBoxFeatureDebugVisualizationChanged?.Invoke(m_BoundingBoxDebugInfoEnabled);

            // If general bounding box visuals are not enabled, do not enable the debug info.
            if (!m_BoundingBoxVisualsEnabled)
                return;

            var trackables = m_BoundingBoxManager.trackables;
            foreach (var trackable in trackables)
            {
                if (trackable.TryGetComponent(out ARBoundingBoxDebugVisualizer visualizer))
                {
                    visualizer.ShowDebugInfoCanvas(m_BoundingBoxDebugInfoEnabled);
                }
            }
        }
    }
}
