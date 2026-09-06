# Tagging clips

Some things about an animation cannot be read off its curves. That a stretch of a capture is a
*tired walk* rather than a brisk one, or that the middle six seconds are a turn, is knowledge a
person has and the clip does not. Tag channels are where that knowledge goes, and a tag query is how
you get it back out: *give me every part of the library that is a walk and is not tired.*

## Tags are hierarchical, and questions run downhill

Tags come from the [GameplayTags](https://github.com/alan-void/GameplayTags) package, embedded here
as a submodule under `Packages/com.alanvoid.gameplaytags`. A tag is an asset, and tags form a tree
written with dots: `action.walk` and `action.run` both sit under `action`.

The hierarchy exists so the vocabulary can be refined without re-annotating anything. A question
asked about `action` is answered by every clip annotated anywhere below it, so introducing
`action.walk.limping` next month does not invalidate a query written today.

That direction is one-way, and the asymmetry is the point:

- a segment tagged `action.walk` **does** answer a query for `action` — a walk is an action
- a segment tagged `action` **does not** answer a query for `action.walk` — it never claimed to be
  a walk, and treating it as one would make every filter quietly too permissive

A `GameplayTagQuery` has three clauses, all optional: **all** of these must be matched, **any** one
of these must be matched, **none** of these may be. An empty query matches everything, so a
serialized query field can be left blank to mean "no filter" with no separate on/off flag.

## A channel is a boolean signal, stored as the frames it flips

An `AnimationTagComponent` is a list of channels, and a channel is one tag plus the frames that tag
switches on and off at. The channel starts off; the first key turns it on, the second turns it off,
and so on by parity. A trailing key with no partner simply stays on to the end of the clip, so "on
until the end" needs no closing key and cannot be lost by trimming.

Storing the flips rather than the intervals is deliberate. Every keyframe is then a thing in its own
right — something the clip editor can select, drag and delete individually — where an interval's two
edges are second-class and can be dragged into an invalid state. And any sorted list of distinct
frames is a valid boolean signal, so there is no overlapping-or-inverted case to guard against: two
keys landing on the same frame describe a flip and an immediate flip back, which is the same signal
as neither, so they cancel. That cancellation *is* the collision rule, and it is what makes dragging
one key onto another do something sensible instead of producing a duplicate.

Frames are **clip-local** — numbered against the whole baked clip, not from `startFrame` — exactly
as gait phase footfalls are. That is what makes trimming safe: a trim changes which frames get
extracted and nothing else, so no annotation has to move and none has to be deleted.

They were slice-local once, justified as "the markers stay correct when a clip's start or end frame
moves". That was backwards twice over. Measuring from `startFrame` is precisely the encoding that
*cannot* survive the start moving — every marker silently slides across the motion, with no change
in the asset to notice. And to keep the numbers meaningful, `OnValidate` deleted every marker
outside the range on every validate, so pulling the slice in and pushing it back out destroyed
everything between. It cost one clip 44 hand-corrected footfalls before the encoding was changed.

## Three words that are not interchangeable

The code leans on a vocabulary, and confusing any two of these is the easiest mistake to make here:

| word | means |
|---|---|
| **slice** | a clip's own `[startFrame, endFrame)` window — `SliceFrameCount`, `SliceToClipFrame` |
| **span** | one channel's on-interval, derived by pairing its keys |
| **segment** | a run of frames some question selected — `AnimationClipSegment` |

An `AnimationClipSegment` carries the clip alongside the frame range, and nothing about it is
tag-specific: it is the shared answer type for "which parts of these clips", whatever the question
was. Both the single-clip and the whole-library search fill the same list, so results concatenate.

## Asking

```csharp
var segments = new List<AnimationClipSegment>();
AnimationTagComponent.FindSegments(clips, query, segments);
```

The search sweeps boundaries rather than frames, and the keys *are* the boundaries: the active set
of tags can only change where some channel flips, so the query is evaluated once per interval
between keys and adjacent matches are merged into one segment. Two channels that hand over at the
same frame therefore come back as one continuous run rather than two abutting ones.

## Authoring, and what it costs you

Channels are authored in the clip editor (`MoSynth/Animation/Clip Editor...`), which draws each one
as a dope-sheet row: the on-spans as bars, the keys as diamonds. The keymap is Blender's — click and
shift-click to select, drag from empty space or `B` to box select, `G` to move, `S` to scale, `X` to
delete, and a typed number for an exact offset. Click a lane first: that is what focuses the row and
hands it the keyboard. It is written up in [the clip editor](clip-editor.md); the hazards are in
[`agents/animation-tools/clip-editor-tracks.md`](../agents/animation-tools/clip-editor-tracks.md).

Channels are added with **Add Channel**, which opens the same searchable tag dropdown a serialized
tag field uses, with a *New Tag…* entry — so building the vocabulary does not mean leaving the
window for the Tags Browser.

Annotation outside the slice is drawn and edited exactly like annotation inside it — the timeline
already dims those regions, which says all that needs saying. Nothing is ever removed for being out
of range; an edit clamps only the frames it produces, so a key a trim left beyond the end is none of
an unrelated edit's business. To be rid of one, select it and press `X` like any other.

## What this does not do yet

Tags do not reach the pose database. MoSynth has carried a complete tag pipe for a long time —
`PoseSet.AddTag`, an `AnimationTag` runtime type, a tag block in `.mmpose` that `PoseSerializer`
writes and `Python/pose_set_importer.py` reads — with nothing able to author a tag, so the block has
always been written empty.

`AnimationTagComponent` is the author that pipe never had, but it is not wired to it. Nothing
downstream reads tags out of the database yet, so writing them there would be dead weight with a
full database regeneration attached. The dead code stays where it is until something wants to read
it; see [the pose database](pose-database.md).
