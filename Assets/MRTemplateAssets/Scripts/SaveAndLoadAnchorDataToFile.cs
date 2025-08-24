using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine.XR.ARSubsystems;

namespace UnityEngine.XR.Templates.MR
{
    // Copied from ARF Samples repo. Commit: 4a7e7b6
    public class SaveAndLoadAnchorDataToFile
    {
        readonly string m_FilePath = Path.Combine(Application.persistentDataPath, "SavedAnchorIds.json");

        public bool initialized => m_Initialized;
        bool m_Initialized;

        public Awaitable initializeAwaitable => m_InitializeAwaitable;
        Awaitable m_InitializeAwaitable;

        public Dictionary<SerializableGuid, int> SavedAnchorsData => m_SavedAnchorsData;
        Dictionary<SerializableGuid, int> m_SavedAnchorsData = new();

        public SaveAndLoadAnchorDataToFile()
        {
            m_InitializeAwaitable = PopulateSavedAnchorIdsFromFile();
        }

        /// <summary>
        /// Saves a `SerializableGuid` to a file asynchronously, appending to the list of ids already saved.
        /// If no file exists or the file is unreadable, a new file is created.
        /// </summary>
        /// <param name="savedAnchorId">The `SerializableGuid` to save.</param>
        public async Awaitable SaveAnchorIdAsync(SerializableGuid savedAnchorId, int prefabId)
        {
            try
            {
                if (!m_Initialized)
                    await m_InitializeAwaitable;

                if (!m_SavedAnchorsData.TryAdd(savedAnchorId, prefabId))
                    m_SavedAnchorsData[savedAnchorId] = prefabId;

                await WriteSavedAnchorIdsToFile();
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        public async Awaitable EraseAnchorIdAsync(SerializableGuid savedAnchorId)
        {
            try
            {
                if (!m_Initialized)
                    await m_InitializeAwaitable;

                m_SavedAnchorsData.Remove(savedAnchorId);
                await WriteSavedAnchorIdsToFile();
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        /// <summary>
        /// Returns the set of `SerializableGuid`s from the save file.
        /// </summary>
        /// <returns>The set of `SerializableGuid`s that were saved to the file.
        /// If no file exists or the file is unreadable, an an empty set is returned.</returns>
        public async Awaitable<Dictionary<SerializableGuid, int>> LoadSavedAnchorsDataAsync()
        {
            if (!m_Initialized)
                await m_InitializeAwaitable;

            return m_SavedAnchorsData;
        }

        async Awaitable PopulateSavedAnchorIdsFromFile()
        {
            try
            {
                m_SavedAnchorsData.Clear();
                if (!File.Exists(m_FilePath))
                    return;

                using var streamReader = File.OpenText(m_FilePath);
                using var jsonTextReader = new JsonTextReader(streamReader);

                var kvp = (JObject)await JToken.ReadFromAsync(jsonTextReader);
                foreach (var (idAsString, prefabId) in kvp)
                {
                    var tokens = idAsString.Split("-");
                    var low = Convert.ToUInt64(tokens[0], 16);
                    var high = Convert.ToUInt64(tokens[1], 16);
                    var serializableGuid = new SerializableGuid(low, high);
                    m_SavedAnchorsData.Add(serializableGuid, (int)prefabId);
                }
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
            finally
            {
                m_Initialized = true;
            }
        }

        async Awaitable WriteSavedAnchorIdsToFile()
        {
            var jsonString = JsonConvert.SerializeObject(m_SavedAnchorsData, Formatting.Indented);
            await File.WriteAllTextAsync(m_FilePath, jsonString);
        }
    }
}
