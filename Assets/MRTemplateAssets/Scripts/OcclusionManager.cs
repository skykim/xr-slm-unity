#if USING_META_OCCLUSION && (UNITY_ANDROID || UNITY_EDITOR)
#define META_OCCLUSION_AVAILABLE
using UnityEngine.XR.OpenXR.Features.Meta;
using UnityEngine.XR.OpenXR.NativeTypes;
#endif

using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace UnityEngine.XR.Templates.MR
{
    /// <summary>
    /// Utility class that provides configuration for occlusion support.
    /// </summary>
    public class OcclusionManager : MonoBehaviour
    {
        [SerializeField, Tooltip("Enable the occlusion manager on start.")]
        bool m_EnableManagerOnStart = false;

        /// <summary>
        /// Enable the occlusion manager on start.
        /// </summary>
        public bool enableManagerOnStart
        {
            get => m_EnableManagerOnStart;
            set => m_EnableManagerOnStart = value;
        }

        [SerializeField, Tooltip("The UI Toggle GameObject that will be disabled if occlusion is not supported.")]
        GameObject m_UIToggleObject;

        /// <summary>
        /// The UI Toggle GameObject that will be disabled if occlusion is not supported.
        /// </summary>
        public GameObject uiToggleObject
        {
            get => m_UIToggleObject;
            set => m_UIToggleObject = value;
        }

        [SerializeField, Tooltip("The GameObject containing the Quest-specific settings.")]
        GameObject m_QuestSettings;

        /// <summary>
        /// The GameObject containing the Quest-specific settings.
        /// </summary>
        public GameObject questSettings
        {
            get => m_QuestSettings;
            set => m_QuestSettings = value;
        }

        [SerializeField, Tooltip("The GameObject containing the Android XR-specific settings.")]
        GameObject m_AndroidXRSettings;

        /// <summary>
        /// The GameObject containing the Android XR-specific settings.
        /// </summary>
        public GameObject AndroidXRSettings
        {
            get => m_AndroidXRSettings;
            set => m_AndroidXRSettings = value;
        }

        AROcclusionManager m_AROcclusionManager;
        ARShaderOcclusion m_ARShaderOcclusion;

        void Start()
        {
            if (TrySetupOcclusion() && m_EnableManagerOnStart)
            {
                SetupManager();
            }
        }

        bool TrySetupOcclusion()
        {
            // Check if the platform supports occlusion
            if (LoaderUtility
                    .GetActiveLoader()?
                    .GetLoadedSubsystem<XROcclusionSubsystem>() != null)
            {
                // XROcclusionSubsystem was loaded. The platform supports occlusion.
                m_AROcclusionManager = FindAnyObjectByType<AROcclusionManager>();
                if (m_AROcclusionManager == null)
                {
                    Debug.LogWarning("No AROcclusionManager found, yet Use Occlusion is enabled. Disabling Object.", this);
                    m_UIToggleObject.SetActive(false);
                    return false;
                }
                else
                {
                    if (!m_AROcclusionManager.gameObject.TryGetComponent(out m_ARShaderOcclusion))
                    {
                        Debug.LogWarning($"No {nameof(ARShaderOcclusion)} component found. Adding one manually now.", this);
                        m_ARShaderOcclusion = m_AROcclusionManager.gameObject.AddComponent<ARShaderOcclusion>();
                    }
                }

                SetupDeviceSpecificSettings();

                return true;
            }
            // If the platform does not support occlusion, disable the object
            else
            {
                Debug.LogWarning("Occlusion is not supported on this platform. Disabling Object.", this);
                m_UIToggleObject.SetActive(false);
                return false;
            }
        }

        /// <summary>
        /// Sets up device-specific settings for the occlusion manager.
        /// This is called in the Start method to ensure that the correct settings are applied based on the current XR platform.
        /// This function is important for ensuring that the occlusion manager works correctly on different devices based on the platform specifics.
        /// </summary>
        void SetupDeviceSpecificSettings()
        {
            Debug.Log($"Current XR Platform: {XRPlatformUnderstanding.CurrentPlatform}");
            switch (XRPlatformUnderstanding.CurrentPlatform)
            {
                case XRPlatformType.Quest:
                    m_QuestSettings.SetActive(true);
                    m_AndroidXRSettings.SetActive(false);
                    break;
                case XRPlatformType.AndroidXR:
                    m_QuestSettings.SetActive(false);
                    m_AndroidXRSettings.SetActive(true);
                    m_AROcclusionManager.environmentDepthTemporalSmoothingRequested = true;
                    SetShaderMode(AROcclusionShaderMode.HardOcclusion);
                    break;
                case XRPlatformType.Other:
                    m_QuestSettings.SetActive(false);
                    m_AndroidXRSettings.SetActive(false);
                    break;
            }
        }

        /// <summary>
        /// Sets up the occlusion manager and enables it if necessary.
        /// This is called from the goal manager to ensure that the occlusion manager is set up correctly and enabled when needed.
        /// </summary>
        public void SetupManager()
        {
            if (m_AROcclusionManager != null && !m_AROcclusionManager.enabled)
                m_AROcclusionManager.enabled = true;

            if (m_ARShaderOcclusion != null && !m_ARShaderOcclusion.enabled)
                m_ARShaderOcclusion.enabled = true;
        }

        /// <summary>
        /// Sets the occlusion manager to be enabled or disabled.
        /// This is called from the UI toggle to enable or disable the occlusion manager based on user input.
        /// </summary>
        /// <param name="isOn">Enables or disables <see cref="AROcclusionManager"/> based on value.</param>
        public void SetOcclusionIsOn(bool isOn)
        {
            if (m_AROcclusionManager != null)
            {
                m_AROcclusionManager.enabled = isOn;
            }
        }

        /// <summary>
        /// Sets the shader mode for the occlusion manager.
        /// This is called from the UI dropdown to set the shader mode based on user input.
        /// </summary>
        /// <param name="shaderMode">The shader mode to set for the occlusion manager.</param>
        /// <remarks>
        /// The shader mode is an integer value that corresponds to the <see cref="AROcclusionShaderMode"/> enum.
        /// The value is offset by 1 because the first value in the enum is None, which is not used in this context.
        /// </remarks>
        public void SetShaderMode(int shaderMode)
        {
            // Offset by 1 because we don't use None
            SetShaderMode((AROcclusionShaderMode)(++shaderMode));
        }

        void SetShaderMode(AROcclusionShaderMode mode)
        {
            if (m_ARShaderOcclusion != null)
            {
                m_ARShaderOcclusion.occlusionShaderMode = mode;
                Debug.Log("Setting shader mode to " + mode);
            }
        }

        /// <summary>
        /// Sets the state of temporal smoothing for the <see cref="OcclusionManager"/>.
        /// </summary>
        /// <param name="isEnabled">If true, temporal smoothing will be applied to the environment depth image.</param>
        public void SetTemporalSmoothingEnabled(bool isEnabled)
        {
            if (m_AROcclusionManager != null)
            {
                m_AROcclusionManager.environmentDepthTemporalSmoothingRequested = isEnabled;
            }
        }

        /// <summary>
        /// Sets the hand removal feature for the Meta OpenXR occlusion subsystem.
        /// </summary>
        /// <param name="isEnabled">Sets the Hand Removal Feature value.</param>
        public void SetHandHandRemovalEnabled(bool isEnabled)
        {
#if META_OCCLUSION_AVAILABLE
            var subsystem = m_AROcclusionManager.subsystem as MetaOpenXROcclusionSubsystem;
            var result = subsystem.TrySetHandRemovalEnabled(isEnabled);
            if (result.IsError())
            {
                // Handle error
                Debug.LogWarning("Error setting hand removal enabled: " + result.ToString());
            }
#endif
        }
    }
}
