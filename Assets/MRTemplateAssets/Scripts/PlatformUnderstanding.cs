using UnityEngine.XR.ARSubsystems;
using UnityEngine.XR.ARFoundation;

namespace UnityEngine.XR.Templates.MR
{
    public enum XRPlatformType
    {
        Quest,
        AndroidXR,
        Other,
        None
    }

    public class XRPlatformUnderstanding
    {
        // Copied from ARF Samples repo. Commit: 4a7e7b6 - Class: MenuLoader.cs

        /// <summary>
        /// The current platform based on the active XRSessionSubsystem.
        /// </summary>
        public static XRPlatformType CurrentPlatform
        {
            get
            {
                if (!k_Initialized)
                {
                    k_CurrentPlatform = GetCurrentXRPlatform();
                    k_Initialized = true;
                }
                return k_CurrentPlatform;
            }
        }

        static XRPlatformType k_CurrentPlatform = XRPlatformType.None;

        static bool k_Initialized = false;

        /// <summary>
        /// Returns the current platform based on the active XRSessionSubsystem.
        /// </summary>
        /// <returns>The current platform based on the active XRSessionSubsystem.</returns>
        static XRPlatformType GetCurrentXRPlatform()
        {
            // If we have already initialized, just return the current platform
            if (k_Initialized)
                return k_CurrentPlatform;

            var loader = LoaderUtility.GetActiveLoader();
            var sessionSubsystem = loader != null ? loader.GetLoadedSubsystem<XRSessionSubsystem>() : null;

            if (sessionSubsystem == null)
            {
                Debug.Log("No active XRSessionSubsystem found. Defaulting to XRPlatformType.None.");
                k_Initialized = true;
                k_CurrentPlatform = XRPlatformType.None;
                return k_CurrentPlatform;
            }

            // We switch on Session Descriptor id because we can't guarantee with current preprocessor directives whether
            // a provider package (and its types) will be present. For example, UNITY_ANDROID could signal that either
            // ARCore or OpenXR loader is present. Because we don't know for sure, we are unable to switch on the loader
            // type without introducing a build-time error in case that package was stripped.
            switch (sessionSubsystem.subsystemDescriptor.id)
            {
                case "Meta-Session":
                    Debug.Log("Meta-Session detected.");
                    k_CurrentPlatform = XRPlatformType.Quest;
                    break;
                case "Android-Session":
                    Debug.Log("Android-Session detected.");
                    k_CurrentPlatform = XRPlatformType.AndroidXR;
                    break;
                default:
                    // Default case includes other third-party providers
                    Debug.Log($"Unknown platform detected, setting platform to Other. Subsystem id: {sessionSubsystem.subsystemDescriptor.id}");
                    k_CurrentPlatform = XRPlatformType.Other;
                    break;
            }

            return k_CurrentPlatform;
        }
    }
}
