using System;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// The clip component list: lookup by type, and the null tolerance a <c>[SerializeReference]</c>
/// list forces on every consumer.
/// </summary>
public class AnimationClipComponentTests
{
    [Serializable]
    private sealed class ProbeComponent : AnimationClipComponent
    {
        public int validatedCount;

        public override string Describe() => "probe";

        public override void OnValidate(AnnotatedAnimationClip clip) => validatedCount++;
    }

    private AnnotatedAnimationClip _clip;

    [SetUp]
    public void SetUp()
    {
        _clip = ScriptableObject.CreateInstance<AnnotatedAnimationClip>();
    }

    [TearDown]
    public void TearDown()
    {
        UnityEngine.Object.DestroyImmediate(_clip);
    }

    [Test]
    public void AClipStartsWithAnEmptyComponentList()
    {
        // A field initializer rather than menu seeding, because [CreateAssetMenu] makes assets
        // without going through the creation menu.
        Assert.That(_clip.components, Is.Not.Null);
        Assert.That(_clip.components, Is.Empty);
    }

    [Test]
    public void GetComponentFindsByType()
    {
        var probe = new ProbeComponent();
        _clip.components.Add(new GaitPhaseComponent());
        _clip.components.Add(probe);

        Assert.That(_clip.GetComponent<ProbeComponent>(), Is.SameAs(probe));
        Assert.That(_clip.GetComponent<GaitPhaseComponent>(), Is.Not.Null);
    }

    [Test]
    public void GetComponentReturnsNullWhenAbsent()
    {
        Assert.That(_clip.GetComponent<ProbeComponent>(), Is.Null);
        Assert.That(_clip.TryGetComponent<ProbeComponent>(out _), Is.False);
    }

    [Test]
    public void TryGetComponentReportsWhatItFound()
    {
        var probe = new ProbeComponent();
        _clip.components.Add(probe);

        Assert.That(_clip.TryGetComponent<ProbeComponent>(out var found), Is.True);
        Assert.That(found, Is.SameAs(probe));
    }

    [Test]
    public void LookupToleratesANullEntry()
    {
        // An element whose type was renamed or deleted deserializes as null. Every consumer has to
        // survive that; this is the one that matters most.
        _clip.components.Add(null);
        _clip.components.Add(new ProbeComponent());

        Assert.That(_clip.GetComponent<ProbeComponent>(), Is.Not.Null);
    }

    [Test]
    public void GetComponentReturnsTheFirstMatch()
    {
        var first = new ProbeComponent();
        _clip.components.Add(first);
        _clip.components.Add(new ProbeComponent());

        Assert.That(_clip.GetComponent<ProbeComponent>(), Is.SameAs(first));
    }

    [Test]
    public void ADisabledComponentIsStillFound()
    {
        // isEnabled says what consumers should act on; it does not hide the data.
        var probe = new ProbeComponent { isEnabled = false };
        _clip.components.Add(probe);

        Assert.That(_clip.GetComponent<ProbeComponent>(), Is.SameAs(probe));
    }
}
}
