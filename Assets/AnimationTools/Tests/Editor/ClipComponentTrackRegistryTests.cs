using System;
using AnimationTools.Editor;
using NUnit.Framework;

namespace AnimationTools.Tests
{
[Serializable]
public class RegistryTestComponentBase : AnimationClipComponent
{
    public override string Describe() => "registry test base";
}

[Serializable]
public class RegistryTestComponentDerived : RegistryTestComponentBase
{
}

[Serializable]
public class RegistryTestUnregisteredComponent : AnimationClipComponent
{
    public override string Describe() => "registry test unregistered";
}

[ClipComponentTrack(typeof(RegistryTestComponentBase))]
public class RegistryTestBaseTrack : AnimationClipComponentTrack
{
}

/// <summary>
/// Which track the clip editor picks for a component.
/// </summary>
/// <remarks>
/// The case worth guarding is the silent one: renaming or moving a component type leaves its track
/// registered against a type that no longer exists, and the window falls back to the default track
/// with no error, because nothing can tell that apart from a component that simply has no track.
/// </remarks>
public class ClipComponentTrackRegistryTests
{
    [Test]
    public void GaitPhaseResolvesToItsOwnTrack()
    {
        var track = ClipComponentTrackRegistry.Create(new GaitPhaseComponent());
        Assert.IsInstanceOf<GaitPhaseTrack>(track);
    }

    [Test]
    public void AComponentWithNoTrackFallsBackToTheDefault()
    {
        var track = ClipComponentTrackRegistry.Create(new RegistryTestUnregisteredComponent());
        Assert.IsInstanceOf<DefaultComponentTrack>(track);
    }

    [Test]
    public void ATrackRegisteredOnAnAncestorCoversItsSubclasses()
    {
        var track = ClipComponentTrackRegistry.Create(new RegistryTestComponentDerived());
        Assert.IsInstanceOf<RegistryTestBaseTrack>(track);
    }

    [Test]
    public void ANullComponentStillYieldsATrack()
    {
        Assert.IsInstanceOf<DefaultComponentTrack>(ClipComponentTrackRegistry.Create(null));
    }

    [Test]
    public void HasTrackAgreesWithWhatCreateReturns()
    {
        Assert.IsTrue(ClipComponentTrackRegistry.HasTrack(typeof(GaitPhaseComponent)));
        Assert.IsFalse(ClipComponentTrackRegistry.HasTrack(typeof(RegistryTestUnregisteredComponent)));
    }
}
}
