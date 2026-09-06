using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Finds the <see cref="AnimationClipComponentTrack"/> registered for a component type, falling
    /// back to <see cref="DefaultComponentTrack"/> so every component is always viewable.
    /// </summary>
    public static class ClipComponentTrackRegistry
    {
        // Built on first use rather than from a static initializer: TypeCache is rebuilt by the same
        // domain reload that clears this, and an [InitializeOnLoad] can run before every assembly
        // is ready.
        private static Dictionary<Type, Type> _trackTypeByComponentType;

        /// <summary>Whether a component type resolves to something other than the default track.</summary>
        public static bool HasTrack(Type componentType) => FindTrackType(componentType) != null;

        /// <summary>The registered track type for a component type, or null if there is none.</summary>
        public static Type FindTrackType(Type componentType)
        {
            if (componentType == null) return null;

            var registry = Registry;

            // Walk up to, but not including, the abstract base, so a component family can share one
            // track registered on its common ancestor.
            for (var type = componentType;
                 type != null && type != typeof(AnimationClipComponent);
                 type = type.BaseType)
            {
                if (registry.TryGetValue(type, out var trackType)) return trackType;
            }

            return null;
        }

        /// <summary>Creates the track for a component, never returning null.</summary>
        public static AnimationClipComponentTrack Create(AnimationClipComponent component)
        {
            var trackType = component == null ? null : FindTrackType(component.GetType());
            if (trackType == null) return new DefaultComponentTrack();

            try
            {
                return (AnimationClipComponentTrack)Activator.CreateInstance(trackType);
            }
            catch (Exception exception)
            {
                Debug.LogError($"[ClipEditor] Could not create track '{trackType.FullName}': " +
                               $"{exception.Message}. Falling back to the default track.");
                return new DefaultComponentTrack();
            }
        }

        private static Dictionary<Type, Type> Registry
        {
            get
            {
                _trackTypeByComponentType ??= Build();
                return _trackTypeByComponentType;
            }
        }

        private static Dictionary<Type, Type> Build()
        {
            var registry = new Dictionary<Type, Type>();

            foreach (var trackType in TypeCache.GetTypesWithAttribute<ClipComponentTrackAttribute>())
            {
                if (!IsUsable(trackType)) continue;

                foreach (ClipComponentTrackAttribute attribute in
                         trackType.GetCustomAttributes(typeof(ClipComponentTrackAttribute), false))
                {
                    var componentType = attribute.ComponentType;
                    if (componentType == null ||
                        !typeof(AnimationClipComponent).IsAssignableFrom(componentType))
                    {
                        Debug.LogWarning(
                            $"[ClipEditor] '{trackType.FullName}' is registered for " +
                            $"'{componentType?.FullName ?? "null"}', which is not an " +
                            $"{nameof(AnimationClipComponent)}. Ignoring it.");
                        continue;
                    }

                    if (registry.TryGetValue(componentType, out var existing))
                    {
                        // Order by name so the winner does not depend on assembly load order: an
                        // intermittently-applied track is far harder to diagnose than a consistent one.
                        var winner = string.CompareOrdinal(existing.FullName, trackType.FullName) <= 0
                            ? existing
                            : trackType;

                        Debug.LogWarning(
                            $"[ClipEditor] '{existing.FullName}' and '{trackType.FullName}' are both " +
                            $"registered for '{componentType.FullName}'. Using '{winner.FullName}'.");

                        registry[componentType] = winner;
                        continue;
                    }

                    registry[componentType] = trackType;
                }
            }

            return registry;
        }

        private static bool IsUsable(Type trackType)
        {
            if (!typeof(AnimationClipComponentTrack).IsAssignableFrom(trackType))
            {
                Debug.LogWarning(
                    $"[ClipEditor] '{trackType.FullName}' carries " +
                    $"[{nameof(ClipComponentTrackAttribute)}] but does not derive from " +
                    $"{nameof(AnimationClipComponentTrack)}. Ignoring it.");
                return false;
            }

            if (trackType.IsAbstract)
            {
                Debug.LogWarning($"[ClipEditor] Track '{trackType.FullName}' is abstract. Ignoring it.");
                return false;
            }

            if (trackType.GetConstructor(Type.EmptyTypes) == null)
            {
                Debug.LogWarning(
                    $"[ClipEditor] Track '{trackType.FullName}' has no public parameterless " +
                    "constructor. Ignoring it.");
                return false;
            }

            return true;
        }
    }
}
