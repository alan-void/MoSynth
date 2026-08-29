"""Replays a Blender retarget setup over a folder of BVH files and exports one FBX.

Run headless against a setup .blend (see README.md for what makes a blend a valid setup):

    blender --background <setup>.blend --python Tools/Retargeting/batch_retarget.py -- \
        --bvh-dir Assets/LFS/Animation/lafan1/bvh --pattern "*_subject1.bvh" \
        --out Assets/LFS/Animation/lafan1/retargeted/lafan1_subject1.fbx

Never pass --factory-startup: the Rokoko addon supplies the second retarget stage. The
.blend is only ever read; nothing here saves it.
"""

import argparse
import fnmatch
import math
import os
import sys
import time

import bpy
import mathutils

# The rest-pose track that belongs to the setup rather than to any one clip. Kept across a
# run so the exported FBX still carries a bind-pose take.
REST_TRACK_NAME = "T-Pose"

# Rokoko applies and un-applies the target armature's object transform on every retarget,
# which is only identity in exact arithmetic. Past this much drift in a rest bone position
# the exported rig is no longer the rig the setup was authored against.
MAX_REST_DRIFT = 1e-4

# Bone lengths are invariant under the rigid correction between a raw BVH import and a
# setup's source skeleton, so they are what identifies a BVH as belonging to this setup.
MAX_BONE_LENGTH_MISMATCH = 1e-3


class SetupError(Exception):
    """The .blend is not a usable retarget setup, or a clip does not belong to it."""


# Blender exits 0 even when the script raises, and even when it fails to parse at all, so
# the exit code alone cannot tell a caller that a batch actually ran. This is the sentinel the
# Unity launcher looks for.
DONE = "done"


def log(message):
    print("[retarget] " + message, flush=True)


# --- setup discovery ------------------------------------------------------------------


class Setup:
    """The three armatures a retarget setup is built from, plus the bone mapping, resolved
    out of the .blend."""

    def __init__(self, source, target, model, bone_map):
        self.source = source
        self.target = target
        self.model = model
        self.bone_map = bone_map


def resolve_setup(scene):
    """Finds the setup's armatures by role rather than by name, so a blend for a new dataset
    only has to be wired correctly, not named a particular way."""
    model = getattr(scene, "rsl_retargeting_armature_target", None)
    if model is None:
        raise SetupError(
            "No Rokoko target armature set. Open the blend, pick the target rig under "
            "Rokoko > Retargeting, build the bone list, and save.")

    constrained = [o for o in scene.objects
                   if o.type == "ARMATURE" and any(pb.constraints for pb in o.pose.bones)]
    if len(constrained) != 1:
        names = ", ".join(sorted(o.name for o in constrained)) or "none"
        raise SetupError(
            "Expected exactly one armature carrying bone constraints (the cleaned-rest-pose "
            "skeleton stage one bakes); found {}: {}.".format(len(constrained), names))
    target = constrained[0]

    sources = {c.target for pb in target.pose.bones for c in pb.constraints
               if getattr(c, "target", None) is not None}
    sources.discard(target)
    if len(sources) != 1:
        names = ", ".join(sorted(o.name for o in sources)) or "none"
        raise SetupError(
            'The constraints on "{}" must all point at one source armature; they point at '
            "{}: {}.".format(target.name, len(sources), names))

    if not scene.rsl_retargeting_bone_list:
        raise SetupError(
            "The Rokoko bone list is empty. Build and check it in the blend, then save - the "
            "list is a scene property, so it travels with the file.")

    # Both Rokoko armature pointers clear the bone list when assigned, and every clip has to
    # reassign the source. The mapping is by bone name and the names never change across a
    # run, so it is read once here and written back before each retarget.
    bone_map = [(i.bone_name_source, i.bone_name_target, i.bone_name_key, i.is_custom)
                for i in scene.rsl_retargeting_bone_list]

    return Setup(sources.pop(), target, model, bone_map)


def restore_bone_list(scene, setup):
    """Writes the setup's saved mapping back over whatever assigning the armature pointers
    left behind."""
    scene.rsl_retargeting_bone_list.clear()
    for source_name, target_name, key, is_custom in setup.bone_map:
        item = scene.rsl_retargeting_bone_list.add()
        item.bone_name_source = source_name
        item.bone_name_target = target_name
        item.bone_name_key = key
        item.is_custom = is_custom


def check_bone_list(setup):
    """Rokoko silently skips list entries naming bones that do not exist, which surfaces as a
    partly-posed character rather than as an error."""
    missing = []
    for source_name, target_name, _, _ in setup.bone_map:
        if source_name and not setup.target.pose.bones.get(source_name):
            missing.append('source "{}"'.format(source_name))
        if target_name and not setup.model.pose.bones.get(target_name):
            missing.append('target "{}"'.format(target_name))
    if missing:
        raise SetupError("Rokoko bone list names bones that do not exist: " + ", ".join(missing))


# --- object and action plumbing -------------------------------------------------------


def make_operable(view_layer, objects):
    """Un-excludes and unhides whatever holds these objects. A finished setup usually has its
    stage-one collection switched off in the view layer, and operators refuse to touch an
    object that is not in it. Only view-layer state, and the blend is never saved."""
    wanted = {o.name for o in objects}

    def visit(layer_collection):
        holds = any(name in wanted for name in layer_collection.collection.objects.keys())
        holds |= any(visit(child) for child in layer_collection.children)
        if holds:
            layer_collection.exclude = False
            layer_collection.hide_viewport = False
        return holds

    visit(view_layer.layer_collection)

    for obj in objects:
        obj.hide_viewport = False
        obj.hide_set(False)


def activate(obj):
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def assign_action(obj, action):
    """Assigns an action together with a slot to bind it through. Blender 4.4+ routes an
    action's channels through slots, and an action assigned without one animates nothing."""
    anim = obj.animation_data or obj.animation_data_create()
    anim.action = action
    if action is not None and getattr(anim, "action_slot", "unsupported") is None and action.slots:
        anim.action_slot = action.slots[0]


def action_frame_range(action):
    start, end = action.frame_range
    return int(math.floor(start)), int(math.ceil(end))


def drop_action(action):
    if action is not None and action.name in bpy.data.actions:
        bpy.data.actions.remove(action)


# --- pipeline stages ------------------------------------------------------------------


def import_bvh_action(path, setup):
    """Imports a BVH and returns its action, discarding the armature it arrived on.

    The setup's source skeleton already carries the rest pose the constraints were authored
    against, and a raw import differs from it by a rigid transform (for LAFAN, a half turn
    about Z plus the root's horizontal offset). Transplanting only the action keeps every
    clip in the frame the setup was tuned in.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(
        filepath=path, global_scale=0.01, rotate_mode="NATIVE",
        axis_forward="Z", axis_up="Y",
        use_fps_scale=False, update_scene_fps=False, update_scene_duration=False)

    imported = [o for o in bpy.data.objects if o not in before]
    if len(imported) != 1:
        raise SetupError("BVH import produced {} objects, expected 1: {}".format(
            len(imported), os.path.basename(path)))
    obj = imported[0]

    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        raise SetupError("BVH has no animation: " + os.path.basename(path))

    check_skeleton_matches(obj, setup.source, os.path.basename(path))

    # Removing the object would take its single-user action with it.
    action.use_fake_user = True
    armature = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.armatures.remove(armature)
    action.use_fake_user = False
    return action


def check_skeleton_matches(imported, source, label):
    """Guards against a BVH from another dataset or subject reaching this setup, which would
    otherwise retarget into plausible-looking nonsense."""
    missing = sorted({b.name for b in source.data.bones} - {b.name for b in imported.data.bones})
    if missing:
        raise SetupError("{} is missing bones the setup's source skeleton has: {}".format(
            label, ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")))

    worst_name, worst = None, 0.0
    for bone in source.data.bones:
        delta = abs(imported.data.bones[bone.name].length - bone.length)
        if delta > worst:
            worst_name, worst = bone.name, delta
    if worst > MAX_BONE_LENGTH_MISMATCH:
        raise SetupError(
            '{} has different bone proportions from the setup\'s source skeleton ("{}" differs '
            "by {:.4f}m). It is probably from another subject or dataset; give it its own setup "
            "blend.".format(label, worst_name, worst))


def bake_pose(obj, frame_start, frame_end):
    """Bakes the constrained result of stage one into a plain action, leaving the setup's
    constraints in place for the next clip."""
    activate(obj)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(
        frame_start=frame_start, frame_end=frame_end, step=1,
        only_selected=False, visual_keying=True,
        clear_constraints=False, clear_parents=False,
        use_current_action=False, clean_curves=False,
        bake_types={"POSE"})
    baked = obj.animation_data.action
    bpy.ops.object.mode_set(mode="OBJECT")
    return baked


def retarget_to_model(scene, setup, baked, clip_name):
    """Runs Rokoko's retarget from a constraint-free copy of the cleaned skeleton onto the
    model rig, and returns the action it produced."""
    # Duplicating carries the per-pose-bone rotation mode across, which building an object
    # over the armature data would not: a synthesised object defaults to quaternion bones,
    # and the stage-one bake's rotation_euler curves would then drive nothing.
    activate(setup.target)
    bpy.ops.object.duplicate(linked=True)
    proxy = bpy.context.object
    proxy.name = clip_name
    for pose_bone in proxy.pose.bones:
        for constraint in list(pose_bone.constraints):
            pose_bone.constraints.remove(constraint)
    assign_action(proxy, baked)

    scene.rsl_retargeting_armature_source = proxy
    scene.rsl_retargeting_armature_target = setup.model
    restore_bone_list(scene, setup)

    # Rokoko takes the source's rest pose by applying whatever pose the armature holds, which
    # would make the current frame an input; it unlinks the action first, so in practice the
    # result is frame-independent. Pinning the frame anyway costs nothing and keeps that from
    # quietly becoming untrue, since the bake above leaves the scene on the last frame.
    scene.frame_set(scene.frame_start)

    activate(proxy)
    result = bpy.ops.rsl.retarget_animation()
    if "FINISHED" not in result:
        raise SetupError("Rokoko retarget failed for {} ({}).".format(clip_name, ", ".join(result)))

    action = setup.model.animation_data.action
    if proxy.name in bpy.data.objects:
        bpy.data.objects.remove(proxy, do_unlink=True)
    return action


def push_to_nla(obj, action, track_name):
    anim = obj.animation_data or obj.animation_data_create()
    start, _ = action_frame_range(action)
    track = anim.nla_tracks.new()
    track.name = track_name
    strip = track.strips.new(track_name, start, action)
    strip.name = track_name
    assign_action(obj, None)


# --- rest-pose drift ------------------------------------------------------------------


def rest_snapshot(obj):
    return {b.name: (mathutils.Vector(b.head_local), mathutils.Vector(b.tail_local))
            for b in obj.data.bones}


def rest_drift(obj, snapshot):
    worst = 0.0
    for bone in obj.data.bones:
        head, tail = snapshot[bone.name]
        worst = max(worst, (bone.head_local - head).length, (bone.tail_local - tail).length)
    return worst


# --- run ------------------------------------------------------------------------------


def clear_clip_tracks(obj):
    """Starts every run from the same NLA state, so a rerun cannot stack a second take with
    the same name onto whatever the blend happened to be saved with."""
    anim = obj.animation_data
    if anim is None:
        return
    for track in [t for t in anim.nla_tracks if t.name != REST_TRACK_NAME]:
        anim.nla_tracks.remove(track)
    assign_action(obj, None)


def process_clip(scene, setup, path):
    clip_name = os.path.splitext(os.path.basename(path))[0]
    started = time.time()

    raw = import_bvh_action(path, setup)
    assign_action(setup.source, raw)
    frame_start, frame_end = action_frame_range(raw)
    scene.frame_start, scene.frame_end = frame_start, frame_end

    baked = bake_pose(setup.target, frame_start, frame_end)
    assign_action(setup.target, None)

    retargeted = retarget_to_model(scene, setup, baked, clip_name)
    retargeted.name = clip_name
    push_to_nla(setup.model, retargeted, clip_name)

    assign_action(setup.source, None)
    drop_action(baked)
    drop_action(raw)
    bpy.data.orphans_purge(do_recursive=True)

    log('  {} -> take "{}"  {} frames  {:.1f}s'.format(
        os.path.basename(path), clip_name, frame_end - frame_start + 1, time.time() - started))


def export_fbx(setup, out_path, simplify):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Selection-scoped, so the source and cleaned skeletons never reach the FBX no matter
    # what else the setup blend holds.
    activate(setup.model)
    for child in setup.model.children:
        child.select_set(True)

    bpy.ops.export_scene.fbx(
        filepath=out_path,
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=True,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_simplify_factor=simplify,
        apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Z",
        axis_up="Y",
        path_mode="COPY",
        embed_textures=False)


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="batch_retarget", description=__doc__)
    parser.add_argument("--bvh-dir", required=True)
    parser.add_argument("--pattern", default="*.bvh",
                        help="glob matched against file names in --bvh-dir")
    parser.add_argument("--out", required=True, help="destination .fbx")
    parser.add_argument("--limit", type=int, default=0, help="process at most N clips (smoke runs)")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="FBX keyframe reduction; 0 keeps every key. On a 7840-frame LAFAN "
                             "clip, 1.0 costs about 0.2 degrees and saves roughly two thirds of "
                             "the file size")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve the setup and list the clips, then stop")
    return parser.parse_args(argv)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)

    scene = bpy.context.scene
    setup = resolve_setup(scene)
    check_bone_list(setup)
    make_operable(bpy.context.view_layer, [setup.source, setup.target, setup.model])
    log('setup: source "{}" -> cleaned "{}" -> model "{}" ({} mapped bones)'.format(
        setup.source.name, setup.target.name, setup.model.name, len(setup.bone_map)))

    if not os.path.isdir(args.bvh_dir):
        raise SetupError("No such BVH folder: " + args.bvh_dir)
    paths = [os.path.join(args.bvh_dir, n) for n in sorted(os.listdir(args.bvh_dir))
             if fnmatch.fnmatch(n, args.pattern)]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SetupError('No files matching "{}" in {}'.format(args.pattern, args.bvh_dir))
    log('{} clip(s) matching "{}"'.format(len(paths), args.pattern))

    if args.dry_run:
        for path in paths:
            log("  would process " + os.path.basename(path))
        log(DONE)
        return

    snapshot = rest_snapshot(setup.model)
    clear_clip_tracks(setup.model)

    started = time.time()
    for index, path in enumerate(paths, 1):
        log("[{}/{}]".format(index, len(paths)))
        process_clip(scene, setup, path)

        drift = rest_drift(setup.model, snapshot)
        if drift > MAX_REST_DRIFT:
            raise SetupError(
                'The model rig\'s rest pose drifted by {:.6f}m after "{}". Rokoko applies and '
                "un-applies the target's object transform on every clip; split the batch into "
                "smaller runs.".format(drift, os.path.basename(path)))

    log("retargeted {} clip(s) in {:.0f}s; rest-pose drift {:.2e}m".format(
        len(paths), time.time() - started, rest_drift(setup.model, snapshot)))

    export_fbx(setup, args.out, args.simplify)
    log("wrote {} ({:.1f} MB)".format(args.out, os.path.getsize(args.out) / 1e6))
    log(DONE)


if __name__ == "__main__":
    # Blender exits 0 after an uncaught exception in --python, so the launcher would read a
    # failed batch as a success. Every failure path has to set the exit code itself.
    try:
        main()
    except SetupError as error:
        log("ERROR: {}".format(error))
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        log("ERROR: batch failed; see the traceback above.")
        sys.exit(1)
