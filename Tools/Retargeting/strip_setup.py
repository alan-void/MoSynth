"""Strips a retarget setup blend down to the calibration, so it can be published.

    blender --background Assets/LFS/Retargeting/lafan_bvh_to_lafan_corrected.blend \
        --python Tools/Retargeting/strip_setup.py -- \
        --out Tools/Retargeting/setups/lafan_bvh_to_lafan_corrected.blend

Never pass --factory-startup: a Rokoko setup's bone list and target are scene properties that only
load with the addon enabled, and a blend saved without them has lost its mapping.

What a setup holds that took judgement -- the source rest, the cleaned rest that is the
calibration pose, its constraints, the mapping, a live setup's helper bones and RT_ground -- is
armature data and properties. Everything else in a working setup is capture data or a character's
assets: clips parked for scrubbing, the actions left over from authoring, the corrected rig's
skinned mesh. This keeps the three armatures the retarget scripts resolve by role, plus anything
their constraints point at, and drops the rest. The corrected rig's "T-Pose" track survives only if
it really is the rest pose, so nothing captured can leave in it.

Parked clips are dropped, so whoever rebuilds a Mixamo setup from the result passes the same clips
back with make_mixamo_setup.py --clips. The input blend is only read.
"""

import argparse
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_retarget as batch
import direct_retarget as direct

from batch_retarget import SetupError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# What a key on each transform channel holds when a pose bone is at rest.
REST_VALUES = {
    "location": (0.0, 0.0, 0.0),
    "rotation_quaternion": (1.0, 0.0, 0.0, 0.0),
    "rotation_euler": (0.0, 0.0, 0.0),
    "rotation_axis_angle": (0.0, 0.0, 1.0, 0.0),
    "scale": (1.0, 1.0, 1.0),
}
REST_TOLERANCE = 1e-6

# Data that only a mesh, a material or a viewport reference can be using. Removed outright rather
# than left to the orphan purge, because a fake user would keep any of it alive.
DROPPED_DATA = ("meshes", "materials", "images", "textures", "curves", "cameras", "lights",
                "sounds", "movieclips", "fonts", "grease_pencils", "volumes", "pointclouds")


def log(message):
    print("[strip-setup] " + message, flush=True)


def resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def same_file(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Stripped blend to write.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --out.")
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def resolve_roles(scene):
    """The setup's three armatures, found exactly the way the retarget scripts find them, so a
    blend that strips cleanly is one they can still read."""
    live = any(o.type == "ARMATURE" and any(b.name.startswith(direct.HELPER_PREFIX)
                                            for b in o.data.bones)
               for o in scene.objects)
    setup = direct.resolve_setup(scene) if live else batch.resolve_setup(scene)
    return setup, live


def constraint_targets(obj):
    """Every object a constraint on this object or its bones points at.

    Read off every object pointer rather than `target` alone: a constraint evaluated in a custom
    space reads it from `space_object`, and LAFAN's cleaned Hips does exactly that.
    """
    constraints = list(obj.constraints)
    if obj.type == "ARMATURE":
        constraints += [c for pb in obj.pose.bones for c in pb.constraints]
    found = set()
    for constraint in constraints:
        for prop in constraint.bl_rna.properties:
            if prop.type != "POINTER":
                continue
            value = getattr(constraint, prop.identifier, None)
            if isinstance(value, bpy.types.Object):
                found.add(value)
    return found


def kept_objects(setup):
    """The setup's armatures, closed over whatever their constraints reference."""
    keep = {setup.source, setup.target, setup.model}
    pending = list(keep)
    while pending:
        for target in constraint_targets(pending.pop()):
            if target not in keep:
                keep.add(target)
                pending.append(target)
    for obj in keep:
        if obj.parent is not None and obj.parent not in keep:
            raise SetupError('"{}" is parented to "{}", which the setup does not need; unparent it '
                             "in the input blend first.".format(obj.name, obj.parent.name))
    return keep


def is_rest_action(action):
    """True if every key of every curve holds its channel's rest value, so the action carries a
    pose of the rig and nothing captured."""
    curves = list(batch.action_fcurves(action))
    if not curves:
        return False
    for curve in curves:
        channel = curve.data_path.rsplit(".", 1)[-1]
        rest = REST_VALUES.get(channel)
        if rest is None or curve.array_index >= len(rest):
            return False
        want = rest[curve.array_index]
        if any(abs(key.co[1] - want) > REST_TOLERANCE for key in curve.keyframe_points):
            return False
    return True


def strip_animation(keep, model):
    """Clears every kept object's animation except the model's rest-pose track, and returns that
    track's action, or None if the model has no such track."""
    rest_action = None
    for obj in keep:
        anim = obj.animation_data
        if anim is None:
            continue
        if anim.use_tweak_mode:
            anim.use_tweak_mode = False
        batch.assign_action(obj, None)
        for track in list(anim.nla_tracks):
            if obj is model and track.name == batch.REST_TRACK_NAME and len(track.strips) == 1 \
                    and track.strips[0].action is not None:
                action = track.strips[0].action
                if not is_rest_action(action):
                    raise SetupError(
                        'The "{}" track on "{}" holds keys away from the rest pose, so publishing '
                        "it could publish captured motion. Re-key it at rest in the input blend."
                        .format(track.name, obj.name))
                rest_action = action
                continue
            log('dropping NLA track "{}" on {}'.format(track.name, obj.name))
            anim.nla_tracks.remove(track)
    return rest_action


def strip(scene):
    setup, live = resolve_roles(scene)
    bone_list = len(getattr(scene, "rsl_retargeting_bone_list", ()))
    log('{} setup: source "{}" -> cleaned "{}" -> model "{}", {} mapped bones'.format(
        "live" if live else "Rokoko", setup.source.name, setup.target.name, setup.model.name,
        len(setup.bone_map)))

    keep = kept_objects(setup)
    rest_action = strip_animation(keep, setup.model)
    if rest_action is None:
        log('note: "{}" has no "{}" track, so the result cannot be batched onto it directly'
            .format(setup.model.name, batch.REST_TRACK_NAME))

    for obj in [o for o in bpy.data.objects if o not in keep]:
        log("dropping {} {}".format(obj.type.lower(), obj.name))
        bpy.data.objects.remove(obj, do_unlink=True)

    for action in [a for a in bpy.data.actions if a is not rest_action]:
        action.use_fake_user = False
        bpy.data.actions.remove(action)
    for name in DROPPED_DATA:
        collection = getattr(bpy.data, name, None)
        for block in list(collection or ()):
            collection.remove(block)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    # Removing an object a Rokoko pointer referenced must not have cost the mapping with it.
    if bone_list != len(getattr(scene, "rsl_retargeting_bone_list", ())):
        raise SetupError("Stripping emptied the Rokoko bone list; the result would have no "
                         "mapping.")
    return setup


def check_published(path):
    """Re-opens the saved blend and confirms it still resolves and holds nothing captured."""
    bpy.ops.wm.open_mainfile(filepath=path)
    scene = bpy.context.scene
    setup, live = resolve_roles(scene)

    problems = []
    if bpy.data.meshes:
        problems.append("{} mesh(es) left".format(len(bpy.data.meshes)))
    for action in bpy.data.actions:
        if not is_rest_action(action):
            problems.append('action "{}" is not a rest pose'.format(action.name))
    for obj in (setup.source, setup.target):
        if obj.animation_data and (obj.animation_data.nla_tracks or obj.animation_data.action):
            problems.append('"{}" still carries animation'.format(obj.name))
    if problems:
        raise SetupError("The stripped blend is not publishable: " + "; ".join(problems))

    log('checked: {} setup "{}" -> "{}" -> "{}", {} mapped bones, {} action(s), {} object(s)'
        .format("live" if live else "Rokoko", setup.source.name, setup.target.name,
                setup.model.name, len(setup.bone_map), len(bpy.data.actions),
                len(bpy.data.objects)))


def main():
    args = parse_args(sys.argv)
    source_blend = bpy.data.filepath
    if not source_blend:
        raise SetupError("No blend open. Pass the setup as blender's positional argument.")
    out = resolve(args.out)
    if same_file(out, source_blend):
        raise SetupError("--out is the input blend. This script never writes over its input.")
    if os.path.exists(out) and not args.force:
        raise SetupError("{} already exists. Pass --force to replace it.".format(out))

    log("from " + source_blend)
    strip(bpy.context.scene)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Saving over a file leaves the old one beside it as a .blend1, a second copy for nobody.
    if os.path.exists(out):
        os.remove(out)
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True, copy=True)
    log("wrote {} ({:.0f} KB)".format(out, os.path.getsize(out) / 1024.0))

    check_published(out)
    log("done")


if __name__ == "__main__":
    try:
        main()
    except SetupError as error:
        log("ERROR: " + str(error))
        sys.exit(1)
