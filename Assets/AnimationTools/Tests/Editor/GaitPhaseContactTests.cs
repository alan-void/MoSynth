using NUnit.Framework;

namespace AnimationTools.Tests
{
/// <summary>
/// Turning foot contact flags into footfall anchors — the step between measuring a clip and having
/// a phase.
/// </summary>
/// <remarks>
/// The fixtures build contact patterns directly rather than driving a real clip, because detection
/// on a clip needs a persistent rig asset that edit-mode tests cannot cheaply make. What is worth
/// pinning here is the edge rule and the smoothing, which is where the anchors actually come from.
/// </remarks>
public class GaitPhaseContactTests
{
    /// <summary>
    /// Contacts for <paramref name="frameCount"/> frames, planted over the given half-open spans.
    /// </summary>
    private static bool[] Contacts(int frameCount, (int start, int end)[] left,
        (int start, int end)[] right)
    {
        var contacts = new bool[frameCount * 2];

        foreach (var (start, end) in left)
        {
            for (var frame = start; frame < end; frame++) contacts[frame * 2] = true;
        }

        foreach (var (start, end) in right)
        {
            for (var frame = start; frame < end; frame++) contacts[frame * 2 + 1] = true;
        }

        return contacts;
    }

    [Test]
    public void AFootfallIsARisingEdge()
    {
        var contacts = Contacts(40, new[] { (10, 20) }, new[] { (25, 35) });

        var footfalls = GaitPhase.FootfallsFromContacts(contacts, 40);

        Assert.That(footfalls.Count, Is.EqualTo(2));
        Assert.That(footfalls[0].frame, Is.EqualTo(10));
        Assert.That(footfalls[0].foot, Is.EqualTo(GaitPhase.Foot.Left));
        Assert.That(footfalls[1].frame, Is.EqualTo(25));
        Assert.That(footfalls[1].foot, Is.EqualTo(GaitPhase.Foot.Right));
    }

    [Test]
    public void AFootAlreadyPlantedOnFrameZeroIsNotAFootfall()
    {
        // The clip started mid-stance; nothing says when that foot went down, so anchoring the cycle
        // there would be a guess.
        var contacts = Contacts(40, new[] { (0, 10), (20, 30) }, new (int, int)[0]);

        var footfalls = GaitPhase.FootfallsFromContacts(contacts, 40);

        Assert.That(footfalls.Count, Is.EqualTo(1));
        Assert.That(footfalls[0].frame, Is.EqualTo(20));
    }

    [Test]
    public void FootfallsComeBackInAscendingFrameOrder()
    {
        var contacts = Contacts(60, new[] { (30, 40) }, new[] { (10, 20), (50, 55) });

        var footfalls = GaitPhase.FootfallsFromContacts(contacts, 60);

        Assert.That(footfalls.Count, Is.EqualTo(3));
        for (var i = 1; i < footfalls.Count; i++)
        {
            Assert.That(footfalls[i].frame, Is.GreaterThanOrEqualTo(footfalls[i - 1].frame));
        }
    }

    [Test]
    public void AlternatingFeetProduceNoMissedContacts()
    {
        var contacts = Contacts(80, new[] { (10, 25), (50, 65) }, new[] { (30, 45), (70, 78) });

        var footfalls = GaitPhase.FootfallsFromContacts(contacts, 80);

        Assert.That(footfalls.Count, Is.EqualTo(4));
        Assert.That(GaitPhase.CountRepeatedFeet(footfalls), Is.EqualTo(0));
    }

    [Test]
    public void AFootThatFallsTwiceRunningIsReportedAsAMissedContact()
    {
        // The other foot's stance was never detected. This is what the fast sections of the shipped
        // walk clips look like, and it is the count the inspector warns on.
        var contacts = Contacts(80, new[] { (10, 25), (40, 55) }, new (int, int)[0]);

        var footfalls = GaitPhase.FootfallsFromContacts(contacts, 80);

        Assert.That(GaitPhase.CountRepeatedFeet(footfalls), Is.EqualTo(1));
    }

    [Test]
    public void SmoothingRemovesSingleFrameChatter()
    {
        var contacts = Contacts(40, new[] { (10, 11), (20, 30) }, new (int, int)[0]);

        GaitPhase.SmoothContacts(contacts, 40, radius: 3);
        var footfalls = GaitPhase.FootfallsFromContacts(contacts, 40);

        Assert.That(footfalls.Count, Is.EqualTo(1), "the one-frame blip must not become a footfall");
        Assert.That(footfalls[0].frame, Is.EqualTo(20).Within(3));
    }

    [Test]
    public void SmoothingKeepsAStanceLongerThanHalfTheWindow()
    {
        var contacts = Contacts(40, new[] { (10, 25) }, new (int, int)[0]);

        GaitPhase.SmoothContacts(contacts, 40, radius: 3);
        var footfalls = GaitPhase.FootfallsFromContacts(contacts, 40);

        Assert.That(footfalls.Count, Is.EqualTo(1));
    }

    [Test]
    public void SmoothingWithNoRadiusLeavesTheFlagsAlone()
    {
        var contacts = Contacts(20, new[] { (5, 6) }, new (int, int)[0]);
        var before = (bool[])contacts.Clone();

        GaitPhase.SmoothContacts(contacts, 20, radius: 0);

        Assert.That(contacts, Is.EqualTo(before));
    }

    [Test]
    public void SmoothingClampsAtTheClipEdgesRatherThanReadingPastThem()
    {
        // A foot planted across frame 0 must survive smoothing, not be diluted by frames that do
        // not exist.
        var contacts = Contacts(40, new[] { (0, 20) }, new (int, int)[0]);

        GaitPhase.SmoothContacts(contacts, 40, radius: 6);

        Assert.That(contacts[0], Is.True);
        Assert.That(contacts[5 * 2], Is.True);
    }

    [Test]
    public void ContactsAndPhaseLineUpEndToEnd()
    {
        // The whole chain a clip goes through: flags, smoothing, anchors, phase.
        var contacts = Contacts(120,
            new[] { (10, 25), (50, 65), (90, 105) },
            new[] { (30, 45), (70, 85) });

        GaitPhase.SmoothContacts(contacts, 120, radius: 3);
        var footfalls = GaitPhase.FootfallsFromContacts(contacts, 120);

        var phase = new float[120];
        var rate = new float[120];
        GaitPhase.Evaluate(footfalls, 1f / 30f, phase, rate);

        Assert.That(footfalls.Count, Is.EqualTo(5));
        Assert.That(GaitPhase.CountRepeatedFeet(footfalls), Is.EqualTo(0));

        // Every frame between the outer anchors has a cycle; the ends do not.
        Assert.That(rate[0], Is.EqualTo(0f));
        Assert.That(rate[60], Is.Not.EqualTo(0f));
        Assert.That(rate[119], Is.EqualTo(0f));

        foreach (var value in phase)
        {
            Assert.That(value, Is.GreaterThanOrEqualTo(0f));
            Assert.That(value, Is.LessThan(GaitPhase.Tau));
        }
    }
}
}
