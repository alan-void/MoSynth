"""Composes the LAFAN-to-corrected-rig retarget setup from the two blends that already exist.

    blender --background Assets/LFS/Retargeting/retargeting.blend \
        --python Tools/Retargeting/make_lafan_corrected_setup.py -- \
        --out Assets/LFS/Retargeting/lafan_bvh_to_lafan_corrected.blend

Everything in a retarget setup that needs judgement -- the cleaned skeleton whose rest pose is
the calibration pose stage two retargets from -- is already authored in retargeting.blend, and
the corrected rig is already a proven export target in the Bandai setup. All that separates the
two is which rig stage two aims at, so composing them is mechanical and belongs in a script
rather than in a hand-edited binary nobody can re-derive. Re-run it if either input moves.

The input blend is opened read-only and saved out under a new name; writing back over it is
refused outright.
"""

import argparse
import os
import sys

import bpy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DEFAULT_RIG_BLEND = ("Assets/LFS/Retargeting/bandai-namco/"
                     "bandai_namco_retarget_to_corrected_lafan_claude.blend")
DEFAULT_OUT = "Assets/LFS/Retargeting/lafan_bvh_to_lafan_corrected.blend"

# The corrected rig and its skinned mesh, taken from the Bandai setup rather than from
# lafan_rig_correction.blend. Both hold the same 22-bone armature at the same transform, but only
# this copy carries the "T-Pose" NLA track batch_retarget.order_rest_take_first requires; the
# correction blend has no animation data at all.
MODEL_NAME = "Lafan_corrected"
MODEL_MESH_NAME = "Mesh"

# The old model rig's bones are namespaced; the corrected rig's are not. Stripping the namespace
# is the whole difference between the two bone lists, and every stripped name is checked against
# the corrected rig before the blend is saved.
TARGET_PREFIX = "Model:"


class SetupError(Exception):
    """The inputs are not what this composition expects."""


def log(message):
    print("[setup] " + message, flush=True)


def resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-blend", default=DEFAULT_RIG_BLEND,
                        help="Blend to take the corrected rig and its T-Pose track from.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Setup blend to write.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --out.")
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def find_cleaned_and_source(scene):
    """The cleaned skeleton and the source it is constrained to, found the way resolve_setup in
    batch_retarget.py finds them, so a blend this script writes resolves there without surprises."""
    constrained = [o for o in scene.objects
                   if o.type == "ARMATURE" and any(pb.constraints for pb in o.pose.bones)]
    if len(constrained) != 1:
        names = ", ".join(sorted(o.name for o in constrained)) or "none"
        raise SetupError("Expected exactly one armature carrying bone constraints; found {}: {}."
                         .format(len(constrained), names))
    cleaned = constrained[0]

    sources = {c.target for pb in cleaned.pose.bones for c in pb.constraints
               if getattr(c, "target", None) is not None}
    sources.discard(cleaned)
    if len(sources) != 1:
        names = ", ".join(sorted(o.name for o in sources)) or "none"
        raise SetupError('The constraints on "{}" must all point at one source armature; they '
                         "point at {}: {}.".format(cleaned.name, len(sources), names))
    return cleaned, sources.pop()


def capture_bone_list(scene):
    """Reads the mapping out before anything touches the armature pointers, which clear it."""
    rows = getattr(scene, "rsl_retargeting_bone_list", None)
    if rows is None:
        raise SetupError("No Rokoko bone list on the scene. The Rokoko addon is not loaded; do "
                         "not pass --factory-startup.")
    if not rows:
        raise SetupError("The input blend's Rokoko bone list is empty, so there is no mapping to "
                         "carry over.")
    return [(i.bone_name_source, i.bone_name_target, i.bone_name_key, i.is_custom) for i in rows]


def drop_old_model(scene, keep):
    """Removes the old model rig, its mesh, and any stale per-clip proxy armature.

    A second constrained armature or a leftover retarget proxy would make role resolution
    ambiguous, and the old rig is most of the file's size for a run that loads it once per shard.
    """
    doomed = []
    for obj in list(scene.objects):
        if obj in keep or obj.type not in {"ARMATURE", "MESH"}:
            continue
        if obj.parent in keep:
            continue
        doomed.append(obj)

    for obj in doomed:
        log("dropping " + obj.name)
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.orphans_purge(do_recursive=True)


def append_model(rig_blend):
    """Brings the corrected rig and its mesh across in one call, so parenting, the armature
    modifier and the NLA tracks arrive intact."""
    for name in (MODEL_NAME, MODEL_MESH_NAME):
        if name in bpy.data.objects:
            raise SetupError('An object named "{}" is already in the blend, so the appended one '
                             "would be renamed and could not be found.".format(name))

    directory = rig_blend + "/Object/"
    bpy.ops.wm.append(directory=directory,
                      files=[{"name": MODEL_NAME}, {"name": MODEL_MESH_NAME}])

    model = bpy.data.objects.get(MODEL_NAME)
    if model is None:
        raise SetupError('No object named "{}" in {}.'.format(MODEL_NAME, rig_blend))

    anim = model.animation_data
    rest = next((t for t in anim.nla_tracks if t.name == "T-Pose"), None) if anim else None
    if rest is None or not rest.strips or rest.strips[0].action is None:
        raise SetupError('The appended "{}" has no "T-Pose" NLA track holding an action, so an '
                         "export would lead with a locomotion take.".format(MODEL_NAME))

    # push_to_nla appends a track per clip, and an empty leftover track would be carried into
    # every shard's stack for no reason.
    for track in [t for t in anim.nla_tracks if t is not rest and not t.strips]:
        log("dropping empty NLA track " + track.name)
        anim.nla_tracks.remove(track)

    # The append also brings whatever action the rig was last posed with, which on the Bandai
    # setup is one of its retarget results. The rest pose and the T-Pose track are all this setup
    # takes from it.
    if anim.action:
        log('clearing active action "{}" on {}'.format(anim.action.name, model.name))
        anim.action = None
        bpy.data.orphans_purge(do_recursive=True)

    return model


def drop_existing_actions():
    """Empties the blend of actions before the model rig is appended.

    Every action in a setup blend belongs to the retired rig or to the author's reference clips,
    and several are held by a fake user, so a purge alone leaves them behind. They are also why an
    appended rest-pose action would otherwise arrive renamed "T-Pose.001" and export a take under
    that name. batch_retarget brings each clip's action in fresh, so the setup needs none of them;
    the originals stay in the source blend, which this script never writes to.
    """
    for action in list(bpy.data.actions):
        log("dropping action " + action.name)
        action.use_fake_user = False
        bpy.data.actions.remove(action)
    bpy.data.orphans_purge(do_recursive=True)


def write_bone_list(scene, bone_map, cleaned, model):
    """Re-points the mapping at the corrected rig, and refuses any name that is not on it.

    Rokoko silently skips a row naming a bone that does not exist, which shows up as a partly
    posed character rather than as an error, so the names are checked here instead.
    """
    missing = []
    rewritten = []
    for source_name, target_name, key, is_custom in bone_map:
        stripped = target_name[len(TARGET_PREFIX):] if target_name.startswith(TARGET_PREFIX) \
            else target_name
        if source_name and not cleaned.pose.bones.get(source_name):
            missing.append('source "{}"'.format(source_name))
        if stripped and not model.pose.bones.get(stripped):
            missing.append('target "{}"'.format(stripped))
        rewritten.append((source_name, stripped, key, is_custom))

    if missing:
        raise SetupError("The mapping names bones that are not on the rigs: " + ", ".join(missing))

    scene.rsl_retargeting_bone_list.clear()
    for source_name, target_name, key, is_custom in rewritten:
        item = scene.rsl_retargeting_bone_list.add()
        item.bone_name_source = source_name
        item.bone_name_target = target_name
        item.bone_name_key = key
        item.is_custom = is_custom
    return rewritten


def main():
    args = parse_args(sys.argv)
    scene = bpy.context.scene

    source_blend = bpy.data.filepath
    if not source_blend:
        raise SetupError("No blend open. Pass the input setup as blender's positional argument.")

    out = resolve(args.out)
    if os.path.abspath(out) == os.path.abspath(source_blend):
        raise SetupError("--out is the input blend. This script never writes back over its input; "
                         "give it a new file name.")
    if os.path.exists(out) and not args.force:
        raise SetupError("{} already exists. Pass --force to replace it.".format(out))

    rig_blend = resolve(args.rig_blend)
    if not os.path.isfile(rig_blend):
        raise SetupError("No rig blend at " + rig_blend)

    log("from " + source_blend)
    log("rig  " + rig_blend)

    bone_map = capture_bone_list(scene)
    cleaned, source = find_cleaned_and_source(scene)
    log('cleaned "{}" constrained to source "{}", {} mapped bones'
        .format(cleaned.name, source.name, len(bone_map)))

    drop_old_model(scene, keep={cleaned, source})
    drop_existing_actions()
    model = append_model(rig_blend)
    log('model "{}" appended, {} bones'.format(model.name, len(model.data.bones)))

    # Both armature pointers clear the bone list when assigned, so the mapping is written last.
    scene.rsl_retargeting_armature_source = cleaned
    scene.rsl_retargeting_armature_target = model
    # LAFAN's source skeleton and the corrected rig are the same skeleton, so auto-scaling has
    # nothing to correct and would only introduce a factor. Matches the Bandai setup.
    scene.rsl_retargeting_auto_scaling = False
    scene.rsl_retargeting_use_pose = "REST"

    rows = write_bone_list(scene, bone_map, cleaned, model)
    for source_name, target_name, _, _ in rows:
        log("  {:<14} -> {}".format(source_name, target_name))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    log("wrote " + out)
    log("done")


if __name__ == "__main__":
    try:
        main()
    except SetupError as error:
        log("ERROR: " + str(error))
        sys.exit(1)
