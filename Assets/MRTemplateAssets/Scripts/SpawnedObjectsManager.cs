using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.XR.Interaction.Toolkit.Samples.StarterAssets;

namespace UnityEngine.XR.Templates.MR
{
    [RequireComponent(typeof(ObjectSpawner))]
    public class SpawnedObjectsManager : MonoBehaviour
    {
        [Tooltip("Spawn objects with a persistent anchor.")]
        [SerializeField]
        bool m_SpawnAsPersistentAnchor = true;

        /// <summary>
        /// Spawn objects with a persistent anchor.
        /// </summary>
        public bool spawnAsPersistentAnchor
        {
            get => m_SpawnAsPersistentAnchor;
            set => m_SpawnAsPersistentAnchor = value;
        }

        [SerializeField]
        [Tooltip("Load saved anchors on start.")]
        bool m_LoadSavedAnchorsOnStart = false;

        /// <summary>
        /// Load saved anchors on start.
        /// </summary>
        public bool loadSavedAnchorsOnStart
        {
            get => m_LoadSavedAnchorsOnStart;
            set => m_LoadSavedAnchorsOnStart = value;
        }

        /// <summary>
        /// UI drop down representing object spawn selection for the scene.
        /// </summary>
        [SerializeField]
        TMP_Dropdown m_ObjectSelectorDropdown;

        /// <summary>
        /// UGUI Button clicked to destroy all spawned objects by this spawner.
        /// </summary>
        [SerializeField]
        Button m_DestroyObjectsButton;

        [SerializeField]
        TMP_Text m_AnchorText;

        [SerializeField]
        ARAnchorManager m_AnchorManager;

        ObjectSpawner m_Spawner;

        readonly List<SpawnedObjectHelper> m_SpawnedObjects = new();

        SaveAndLoadAnchorDataToFile m_SaveAndLoadAnchorIdsToFile;

        async void Start()
        {
            m_SaveAndLoadAnchorIdsToFile = new SaveAndLoadAnchorDataToFile();

            // Wait for the SaveAndLoadAnchorIdsToFile to initialize before continuing.
            if (!m_SaveAndLoadAnchorIdsToFile.initialized)
                await m_SaveAndLoadAnchorIdsToFile.initializeAwaitable;

            if (m_LoadSavedAnchorsOnStart)
            {
                LoadAnchors();
            }
            else
            {
                m_AnchorText.text = "<b><u><align=center>- Currently Saved Objects -</b></u></align>\n\n";
                if (m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData.Count == 0)
                {
                    m_AnchorText.text += "<align=center>- No Anchors Saved -</align>";
                }
                else
                {
                    foreach (var kvp in m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData)
                    {
                        m_AnchorText.text += $"GUID: [{kvp.Key}] Object: {SpawnedObjectName(kvp.Value)}\n\n";
                    }
                }
            }
        }

        void OnEnable()
        {
            if (m_Spawner == null)
                m_Spawner = GetComponent<ObjectSpawner>();

            if (m_SpawnAsPersistentAnchor)
            {
                m_Spawner.onlySpawnInView = false;
                m_Spawner.spawnAsChildren = false;
            }

            if (m_AnchorManager == null)
                m_AnchorManager = FindAnyObjectByType<ARAnchorManager>();

            if (m_ObjectSelectorDropdown != null)
            {
                OnObjectSelectorDropdownValueChanged(m_ObjectSelectorDropdown.value);
                m_ObjectSelectorDropdown.onValueChanged.AddListener(OnObjectSelectorDropdownValueChanged);
            }

            if (m_DestroyObjectsButton != null)
                m_DestroyObjectsButton.onClick.AddListener(OnDestroyObjectsButtonClicked);

            m_Spawner.objectSpawned += ObjectSpawned;
        }

        void OnDisable()
        {
            if (m_ObjectSelectorDropdown != null)
                m_ObjectSelectorDropdown.onValueChanged.RemoveListener(OnObjectSelectorDropdownValueChanged);

            if (m_DestroyObjectsButton != null)
                m_DestroyObjectsButton.onClick.RemoveListener(OnDestroyObjectsButtonClicked);

            m_Spawner.objectSpawned -= ObjectSpawned;
        }

        /// <summary>
        /// Called from the UI to destroy all spawned objects.
        /// It will remove non persistent anchors and delete the objects.
        /// </summary>
        public void OnDestroyObjectsButtonClicked()
        {
            while (m_SpawnedObjects.Count > 0)
            {
                SpawnedObjectHelper spawnedObject = m_SpawnedObjects[0];
                m_SpawnedObjects.RemoveAt(0);

                if (spawnedObject.attachedAnchor != null)
                {
                    if (!m_AnchorManager.TryRemoveAnchor(spawnedObject.attachedAnchor))
                    {
                        Debug.LogWarning("Failed to remove anchor, manually destroying anchor object.", this);
                        Destroy(spawnedObject.attachedAnchor);
                        return;
                    }
                }

                Destroy(spawnedObject.gameObject);
            }
        }

        /// <summary>
        /// Deletes all anchors and clears the saved anchor data list.
        /// This method is called when the Delete Anchors button is clicked from the UI.
        /// </summary>
        public async void DeleteAnchors()
        {
            m_AnchorText.text = "<b><u><align=center>- Deleted Persistent Anchors -</b></u></align>\n";
            await EraseAnchorsAsync();

            if (m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData.Count != 0)
            {
                m_AnchorText.text += "\n\n<align=center>Failed to delete all anchors.</align>\n\n";
                foreach (var kvp in m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData)
                {
                    m_AnchorText.text += $"GUID failed to remove: [{kvp.Key}]\n\n";
                }
            }
            else
            {
                m_AnchorText.text += "\n\n<align=center>All anchors deleted.</align>";
            }
        }

        /// <summary>
        /// Loads all anchors from the saved anchor data list.
        /// This method is called when the Load Anchors button is clicked from the UI.
        /// </summary>
        public async void LoadAnchors()
        {
            m_AnchorText.text = "<b><u><align=center>- Loaded Persistent Anchors -</b></u></align>\n";
            await LoadAnchorsAsync();
        }

        /// <summary>
        /// Saves all anchors from the spawned objects list.
        /// This method is called when the Save Anchors button is clicked from the UI.
        /// </summary>
        public async void SaveAnchors()
        {
            if (!m_AnchorManager.descriptor.supportsSaveAnchor)
            {
                Debug.LogWarning("Save anchor is not supported on this device.", this);
                m_AnchorText.text = "Save anchor is not supported on this device.";
                return;
            }
            // Clear existing anchors
            await EraseAnchorsAsync();

            // Save the current spawned anchors
            m_AnchorText.text = "<b><u><align=center>- Saved Persistent Anchors -</b></u></align>\n";
            await SaveAchorsAsync();
        }

        // This method is called when the Object Selector Dropdown value changes.
        void OnObjectSelectorDropdownValueChanged(int value)
        {
            if (m_Spawner == null)
                return;

            if (value == 0)
            {
                m_Spawner.RandomizeSpawnOption();
                return;
            }

            m_Spawner.spawnOptionIndex = value - 1;
        }

        // This method is called when an object is spawned by the ObjectSpawner.
        void ObjectSpawned(GameObject spawnedObject)
        {
            SpawnedObjectHelper spawnedObjectHelper = new SpawnedObjectHelper
            {
                gameObject = spawnedObject,
                spawnObjectIdx = m_Spawner.spawnOptionIndex,
                isPersistent = m_SpawnAsPersistentAnchor,
                spawnWithAnchor = true
            };

            m_SpawnedObjects.Add(spawnedObjectHelper);

            CreateAndParentAnchorForObject();
        }

        // This method Adds the spawned object to an anchor.
        async void CreateAndParentAnchorForObject()
        {
            SpawnedObjectHelper spawnedObjectHelper = m_SpawnedObjects[^1];
            var result = await m_AnchorManager.TryAddAnchorAsync(new Pose(spawnedObjectHelper.gameObject.transform.position, spawnedObjectHelper.gameObject.transform.rotation));
            if (result.status.IsSuccess())
            {
                var anchor = result.value;
                spawnedObjectHelper.gameObject.transform.SetParent(anchor.transform);
                spawnedObjectHelper.attachedAnchor = anchor;
                m_SpawnedObjects[^1] = spawnedObjectHelper;
            }
        }

        // This method deletes all anchors from the saved anchor data list.
        async Awaitable EraseAnchorsAsync()
        {
            try
            {
                var deletionList = m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData.Keys.ToArray();
                foreach (var guid in deletionList)
                {
                    await EraseAnchorAsync(guid);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error deleting anchors: {e.Message}", this);
            }
        }

        // This method erases the anchor based on the GUID.
        async Awaitable EraseAnchorAsync(SerializableGuid guid)
        {
            var result = await m_AnchorManager.TryEraseAnchorAsync(guid);
            if (!result.IsError() && result.statusCode != XRResultStatus.StatusCode.UnqualifiedSuccess)
            {
                // handle error
                Debug.LogError($"Error erasing GUID: [{guid}].\n\nStatus Code:{result.statusCode}\n\nError: {result}", this);
                return;
            }

            // The anchor was successfully erased.
            await m_SaveAndLoadAnchorIdsToFile.EraseAnchorIdAsync(guid);
        }

        // Loads All anchors from the saved anchor data list.
        async Awaitable LoadAnchorsAsync(bool updateText = true)
        {
            foreach (var kvp in m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData)
            {
                await TryLoadAnchorAsync(kvp.Key);
            }
            if (m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData.Count == 0 && updateText)
            {
                m_AnchorText.text += "\n\n<align=center>- No Anchors To Load -</align>";
            }
        }

        // Lodas the individual anchor based on the GUID.
        async Awaitable TryLoadAnchorAsync(SerializableGuid guid)
        {
            // Don't respawn objects that are already loaded
            foreach (var spawnedObjectHelper in m_SpawnedObjects)
            {
                if (spawnedObjectHelper.persistentGuid == guid && spawnedObjectHelper.gameObject != null)
                {
                    m_AnchorText.text += $"Object {SpawnedObjectName(spawnedObjectHelper.spawnObjectIdx)} is already loaded.\nGUID {guid}.\n\n";
                    return;
                }
            }

            var result = await m_AnchorManager.TryLoadAnchorAsync(guid);
            if (result.status.IsError())
            {
                // handle error
                Debug.Log($"Error Loading Anchor - Status Code:{result.status.statusCode}\n\nError: {result.status} GUID: [{guid}].\n\n", this);
                m_AnchorText.text += $"Error loading GUID: [{guid}].\n{result.status.statusCode}\n";
                return;
            }

            var newAnchor = result.value;
            if (m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData.ContainsKey(guid))
            {
                spawnAsPersistentAnchor = true;
                int spawnId = m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData[guid];
                CreateObjectForLoadedAnchor(newAnchor, guid, spawnId);
                m_AnchorText.text += $"Loaded Object{SpawnedObjectName(spawnId)}\nGUID {guid}.\n\n";
            }
            else
            {
                m_AnchorText.text += $"Persistent Dictionary did not contain GUID: [{guid}].\n\n";
                m_AnchorManager.TryRemoveAnchor(newAnchor);
            }
        }

        // This method creates a new object for the loaded anchor reference.
        void CreateObjectForLoadedAnchor(ARAnchor newAnchor, SerializableGuid guid, int spawnId)
        {
            int nonZeroIndex = spawnId < 0 ? Random.Range(0, m_Spawner.objectPrefabs.Count) : spawnId;
            GameObject respawnedObject = Instantiate(m_Spawner.objectPrefabs[nonZeroIndex], newAnchor.transform);
            respawnedObject.transform.SetLocalPositionAndRotation(Vector3.zero, Quaternion.identity);

            // Create new SpawnedObjectHelper object to store the new anchor reference.
            SpawnedObjectHelper spawnedObjectHelper = new SpawnedObjectHelper
            {
                gameObject = respawnedObject,
                attachedAnchor = newAnchor,
                spawnObjectIdx = spawnId,
                isPersistent = true,
                spawnWithAnchor = true,
                persistentGuid = guid
            };

            bool anchorAlreadyExists = false;
            // Find and replace the SpawnedObjectHelper in the list with the new reference.
            for (int i = 0; i < m_SpawnedObjects.Count; i++)
            {
                if (m_SpawnedObjects[i].persistentGuid == guid)
                {
                    m_SpawnedObjects[i] = spawnedObjectHelper;
                    anchorAlreadyExists = true;
                    break;
                }
            }

            if (!anchorAlreadyExists)
                m_SpawnedObjects.Add(spawnedObjectHelper);
        }

        // Saves all anchors from the spawned objects list.
        async Awaitable SaveAchorsAsync(bool updateText = true)
        {
            for (int i = 0; i < m_SpawnedObjects.Count; i++)
            {
                if (m_SpawnedObjects[i].isPersistent && m_SpawnedObjects[i].spawnWithAnchor)
                    await TrySaveAnchorAsync(i);
            }

            if (m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData.Count == 0 && updateText)
            {
                m_AnchorText.text += "\n\n<align=center>- No Anchors To Save -</align>";
            }
        }

        // Loop through the spawned objects and save the anchors.
        async Awaitable TrySaveAnchorAsync(int idx)
        {
            if (m_SaveAndLoadAnchorIdsToFile.SavedAnchorsData.ContainsKey(m_SpawnedObjects[idx].persistentGuid))
            {
                m_AnchorText.text += $"Object {SpawnedObjectName(m_SpawnedObjects[idx].spawnObjectIdx)} already saved.\nGUID: [{m_SpawnedObjects[idx].persistentGuid}].\n\n";
                return;
            }

            var result = await m_AnchorManager.TrySaveAnchorAsync(m_SpawnedObjects[idx].attachedAnchor);
            if (result.status.IsError())
            {
                // handle error
                Debug.Log($"Error Saving Object: [{SpawnedObjectName(m_SpawnedObjects[idx].spawnObjectIdx)}].\n\nStatus Code:{result.status.statusCode}\n\nError: {result.status}", this);
                m_AnchorText.text += $"Failed to save {SpawnedObjectName(m_SpawnedObjects[idx].spawnObjectIdx)}\n\n";
                return;
            }

            // Save this value, then use it as an input parameter
            SerializableGuid guid = result.value;
            m_SaveAndLoadAnchorIdsToFile ??= new SaveAndLoadAnchorDataToFile();
            await m_SaveAndLoadAnchorIdsToFile.SaveAnchorIdAsync(guid, m_SpawnedObjects[idx].spawnObjectIdx);

            var spawnedObjectHelper = m_SpawnedObjects[idx];
            spawnedObjectHelper.persistentGuid = guid;
            m_SpawnedObjects[idx] = spawnedObjectHelper;

            m_AnchorText.text += $"Saved Object {SpawnedObjectName(m_SpawnedObjects[idx].spawnObjectIdx)}\nGUID: [{guid}].\n\n";
        }

        string SpawnedObjectName(int id)
        {
            return id < 0 ? "Random" : m_Spawner.objectPrefabs[id].name;
        }
    }

    /// <summary>
    /// Helper struct to store the spawned object, anchor, and spawn object index, persistence, and persistent GUID.
    /// </summary>
    struct SpawnedObjectHelper
    {
        public GameObject gameObject;
        public ARAnchor attachedAnchor;
        public int spawnObjectIdx;
        public bool isPersistent;
        public bool spawnWithAnchor;
        public SerializableGuid persistentGuid;
    }
}
