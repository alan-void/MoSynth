"""Replays a retarget setup that carries its own helper rig, without Rokoko.

    blender --background <setup>.blend --python Tools/Retargeting/direct_retarget.py -- \
        --bvh-dir Assets/LFS/Animation/Edinburgh/bvh --pattern "edin_test_0*.bvh" \
        --out Assets/LFS/Animation/LafanCorrected/edinburgh/edin_test.fbx

Same command line and the same FBX as batch_retarget.py, and it borrows that module for
everything except the retarget itself -- import, export, manifest, the `done` sentinel.

A setup built by make_edinburgh_setup.py --live already holds the rig Rokoko would build and
throw away: one helper bone per mapped bone, sitting at the model bone's rest but parented to the
cleaned bone, with the model constrained to follow it. So the whole chain is live --

    source action -> cleaned (constraints) -> helpers (parenting) -> model (constraints)

-- and one bake of the model replaces stage one's bake, the constraint-free proxy, and Rokoko's own
duplicate-and-bake. Measured against a real Rokoko bake of the same clip and rest pose, the two
agree to 0.016 degrees mean and 0.079 max.

Three things come for free with it. Rokoko's rest-pose drift has no mechanism here, so the guard
that watched it is unnecessary. Nothing depends on the addon being installed. And the blend used to
author a rest pose runs the same code as the batch, so a preview cannot disagree with the output.

batch_retarget.py still owns the two setups that have no helper rig; this one refuses them.
"""

import os
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_retarget as batch

from batch_retarget import SetupError, log, DONE, EXIT_OK, EXIT_FATAL, EXIT_SKIPPED

# The prefix make_edinburgh_setup.py gives the helper bones. Their presence is what tells a setup
# that carries its own rig from one that still needs Rokoko.
HELPER_PREFIX = "RT_"

# Without Rokoko nothing applies and un-applies the model's object transform, so the rest pose
# cannot move at all. Checked rather than assumed, because that is the claim this script rests on.
MAX_REST_DRIFT = 1e-9


def resolve_setup(scene):
    """Finds the three armatures by how they are wired, the way batch_retarget does, but for a
    setup whose model is constrained rather than baked into."""
    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    bridges = [o for o in armatures
               if any(b.name.startswith(HELPER_PREFIX) for b in o.data.bones)]
    if len(bridges) != 1:
        raise SetupError(
            "Expected exactly one armature carrying {}* helper bones; found {}. A setup without "
            "them has no live rig, so retarget it with batch_retarget.py instead."
            .format(HELPER_PREFIX, len(bridges)))
    bridge = bridges[0]

    models = [o for o in armatures if o is not bridge
              and any(c.target is bridge for pb in o.pose.bones for c in pb.constraints)]
    if len(models) != 1:
        raise SetupError('Expected exactly one armature constrained to "{}"; found {}.'
                         .format(bridge.name, len(models)))
    model = models[0]

    sources = {c.target for pb in bridge.pose.bones for c in pb.constraints
               if getattr(c, "target", None) is not None}
    sources.discard(bridge)
    if len(sources) != 1:
        raise SetupError('The constraints on "{}" must all point at one source armature; they '
                         "point at {}.".format(bridge.name, len(sources)))

    # Every bone the model actually follows, read off its constraints rather than a stored list:
    # the wiring is the mapping here, so the two cannot drift apart.
    mapped = sorted({pb.name for pb in model.pose.bones for c in pb.constraints
                     if c.type == "COPY_ROTATION" and c.target is bridge})
    if not mapped:
        raise SetupError('No bone on "{}" copies rotation from "{}".'.format(model.name,
                                                                             bridge.name))
    missing = [n for n in mapped if not bridge.pose.bones.get(HELPER_PREFIX + n)]
    if missing:
        raise SetupError("The model follows helpers that do not exist: " + ", ".join(missing))

    bone_map = [(name, name, "", False) for name in mapped]
    return batch.Setup(sources.pop(), bridge, model, bone_map)


def select_mapped(model, setup):
    """Selects exactly the bones the model follows, so a bake writes curves for those and leaves
    an unmapped bone -- the target's Spine2 -- at rest instead of on the previous clip's take."""
    batch.activate(model)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="DESELECT")
    for source_name, _, _, _ in setup.bone_map:
        try:
            model.data.bones[source_name].select = True          # Blender before 5.0
        except AttributeError:
            model.pose.bones[source_name].select = True


def silence_takes(model, muted):
    """Mutes or unmutes every take already stacked on the model.

    Constraints override an action underneath them, so the takes cannot disturb a *mapped* bone --
    but the target's Spine2 is unmapped and has no constraint, so it would be posed by whatever the
    previous clips left on the stack, and everything hanging off it moves with it. Left unmuted the
    error compounds clip by clip: take 1 matched Rokoko to 0.08 degrees, take 10 was out by 28 and
    16 cm.
    """
    for track in (model.animation_data.nla_tracks if model.animation_data else ()):
        track.mute = muted


def bake_model(setup, frame_start, frame_end):
    """One bake of the model, straight through the live chain.

    `clear_constraints` stays off: the rig has to survive for the next clip.
    """
    silence_takes(setup.model, True)
    select_mapped(setup.model, setup)
    bpy.ops.nla.bake(
        frame_start=frame_start, frame_end=frame_end, step=1,
        only_selected=True, visual_keying=True,
        clear_constraints=False, clear_parents=False,
        use_current_action=False, clean_curves=False,
        bake_types={"POSE"})
    action = setup.model.animation_data.action
    bpy.ops.object.mode_set(mode="OBJECT")
    return action


def release_model(setup):
    """Drops the live constraints once every clip is baked.

    The exporter bakes the NLA stack, and a live constraint evaluates on top of it -- left in
    place, every take would export as whatever the last clip left the source holding.
    """
    removed = 0
    for pose_bone in setup.model.pose.bones:
        for constraint in list(pose_bone.constraints):
            if constraint.target is setup.target:
                pose_bone.constraints.remove(constraint)
                removed += 1
    log("released {} constraints on {}".format(removed, setup.model.name))


def process_clip(scene, setup, path, keep_position):
    clip_name = os.path.splitext(os.path.basename(path))[0]
    started = time.time()

    raw = batch.import_bvh_action(path, setup, keep_position)
    batch.assign_action(setup.source, raw)
    frame_start, frame_end = batch.action_frame_range(raw)
    scene.frame_start, scene.frame_end = frame_start, frame_end

    retargeted = bake_model(setup, frame_start, frame_end)
    retargeted.name = clip_name
    root_start = batch.root_world_position(setup.model, scene, frame_start)
    batch.push_to_nla(setup.model, retargeted, clip_name)

    batch.assign_action(setup.source, None)
    batch.drop_action(raw)
    bpy.data.orphans_purge(do_recursive=True)

    seconds = time.time() - started
    log('  {} -> take "{}"  {} frames  {:.1f}s'.format(
        os.path.basename(path), clip_name, frame_end - frame_start + 1, seconds))
    return {"bvh": os.path.basename(path), "take": clip_name,
            "frames": frame_end - frame_start + 1,
            "frame_start": frame_start, "frame_end": frame_end,
            "seconds": round(seconds, 2), "root_start_xyz": root_start}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = batch.parse_args(argv)

    scene = bpy.context.scene
    setup = resolve_setup(scene)
    batch.make_operable(bpy.context.view_layer, [setup.source, setup.target, setup.model])
    log('setup: source "{}" -> bridge "{}" -> model "{}" ({} mapped bones, no Rokoko)'.format(
        setup.source.name, setup.target.name, setup.model.name, len(setup.bone_map)))

    # A setup doubling as a rest-pose authoring blend keeps clips parked on the source for
    # scrubbing, and an assigned action evaluates on top of the NLA stack rather than instead of
    # it -- so an unmuted take there would blend into every clip of the run.
    anim = setup.source.animation_data
    if anim and anim.use_tweak_mode:
        # Left open on a strip by whoever was last scrubbing clips in here. Tweak mode makes
        # AnimData.action read-only, so every clip would fail on assignment.
        log("leaving NLA tweak mode on " + setup.source.name)
        anim.use_tweak_mode = False
    parked = sum(1 for t in (anim.nla_tracks if anim else ()) if not t.mute)
    if parked:
        log("muting {} take(s) parked on {}".format(parked, setup.source.name))
        silence_takes(setup.source, True)

    paths = batch.collect_paths(args)
    if args.limit:
        paths = paths[:args.limit]
    log("{} clip(s) to process".format(len(paths)))

    if args.dry_run:
        for path in paths:
            log("  would process " + os.path.basename(path))
        log(DONE)
        return EXIT_OK

    snapshot = batch.rest_snapshot(setup.model)
    batch.clear_clip_tracks(setup.model)

    started = time.time()
    takes, skipped = [], []
    for index, path in enumerate(paths, start=1):
        log("[{}/{}]".format(index, len(paths)))
        try:
            takes.append(process_clip(scene, setup, path, args.keep_capture_position))
        except Exception as error:
            if not args.continue_on_error:
                raise
            log("  SKIPPED {}: {}".format(os.path.basename(path), error))
            skipped.append({"bvh": os.path.basename(path), "error": str(error)})

    drift = batch.rest_drift(setup.model, snapshot)
    if drift > MAX_REST_DRIFT:
        raise SetupError("The model's rest pose moved by {:.2e} m, which nothing here should be "
                         "able to do. The setup is not what it appears to be.".format(drift))
    log("retargeted {} clip(s) in {:.0f}s; rest-pose drift {:.2e}m".format(
        len(takes), time.time() - started, drift))

    silence_takes(setup.model, False)
    release_model(setup)
    batch.order_rest_take_first(setup)
    batch.export_fbx(setup, args.out, args.simplify)
    log("wrote {} ({:.1f} MB)".format(args.out, os.path.getsize(args.out) / 1e6))
    if args.manifest:
        batch.write_manifest(args, setup, takes, skipped, time.time() - started, drift)
    log(DONE)
    return EXIT_SKIPPED if skipped else EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SetupError as error:
        log("ERROR: " + str(error))
        sys.exit(EXIT_FATAL)
