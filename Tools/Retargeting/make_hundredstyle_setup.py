"""Composes the 100STYLE-to-corrected-rig retarget setup from one BVH and the shared rig.

    blender --background --python Tools/Retargeting/make_hundredstyle_setup.py -- \
        --bvh External/LFS/100STYLE/bvh/Neutral/Neutral_FW.bvh \
        --out Assets/LFS/Retargeting/hundredstyle_bvh_to_lafan_corrected.blend

Never pass --factory-startup: the Rokoko addon supplies the scene properties this writes.

100STYLE is real joint rotations on one actor, and its BVH rest is already a clean T-pose, so like
the Edinburgh setup the cleaned skeleton is a plain duplicate of the source wired straight through
with COPY_ROTATION -- no IK and no spine damping. Two things differ from Edinburgh: the bone names
are the dataset's own, so the Rokoko map is not an identity, and the raw import faces the opposite
way from the corrected rig, so the source rest is turned to face it.

Every one of the 810 BVH declares the same skeleton, so one blend serves the whole dataset.
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_retarget
from make_edinburgh_setup import (BVH_IMPORT, DEFAULT_RIG_BLEND, SetupError, activate,
                                  append_model, drop_existing_actions, log, read_bvh_frame_time,
                                  resolve)

DEFAULT_BVH = "External/LFS/100STYLE/bvh/Neutral/Neutral_FW.bvh"
DEFAULT_OUT = "Assets/LFS/Retargeting/hundredstyle_bvh_to_lafan_corrected.blend"

SOURCE_NAME = "hundredstyle_source"
CLEANED_NAME = "hundredstyle_cleaned"
ROOT_BONE = "Hips"

# 100STYLE is captured at 60 Hz. The importer keys one sample per frame regardless of the file's
# Frame Time, so this scene rate is the only thing that turns frames into seconds -- and the
# exported FBX takes its rate from it.
SCENE_FPS = 60

# The raw import faces +Y and the corrected rig faces -Y. Retargeting copies the root's position
# but only the rotation *delta* from rest, so with the two rests a half turn apart the character
# would face one way and travel the other. Turning the source's rest about the vertical through
# its root turns every clip transplanted onto it with it, because a bone's channels are read in its
# own rest frame.
SOURCE_REST_TURN = Matrix.Rotation(math.pi, 4, "Z")

# (source bone, corrected-rig bone, Rokoko's auto-detection key). The keys are Rokoko's own: it
# files every spine and chest bone under "spine".
BONE_MAP = (
    ("Hips", "Hips", "hip"),
    ("Chest", "Spine", "spine"), ("Chest2", "Spine1", "spine"), ("Chest4", "Spine2", "spine"),
    ("Neck", "Neck", "neck"), ("Head", "Head", "head"),
    ("LeftCollar", "LeftShoulder", "leftShoulder"), ("LeftShoulder", "LeftArm", "leftUpperArm"),
    ("LeftElbow", "LeftForeArm", "leftLowerArm"), ("LeftWrist", "LeftHand", "leftHand"),
    ("RightCollar", "RightShoulder", "rightShoulder"),
    ("RightShoulder", "RightArm", "rightUpperArm"),
    ("RightElbow", "RightForeArm", "rightLowerArm"), ("RightWrist", "RightHand", "rightHand"),
    ("LeftHip", "LeftUpLeg", "leftUpLeg"), ("LeftKnee", "LeftLeg", "leftLeg"),
    ("LeftAnkle", "LeftFoot", "leftFoot"), ("LeftToe", "LeftToe", "leftToe"),
    ("RightHip", "RightUpLeg", "rightUpLeg"), ("RightKnee", "RightLeg", "rightLeg"),
    ("RightAnkle", "RightFoot", "rightFoot"), ("RightToe", "RightToe", "rightToe"),
)

# The source bends its back in four segments where the target has three. Chest3's bend is not lost
# by leaving it out: Spine2 copies Chest4's world rotation, which already carries it.
UNMAPPED_SOURCE_BONES = ("Chest3",)

EXPECTED_SOURCE_BONES = frozenset([s for s, _, _ in BONE_MAP] + list(UNMAPPED_SOURCE_BONES))
MODEL_BONE_COUNT = 22

# Limb bones whose rest direction a T-pose fixes, each with the joint it points at, in
# corrected-rig names; the same set make_mixamo_setup calibrates a character's limbs on. Any rest
# difference here is carried into every retargeted frame, so it is measured and reported.
LIMB_CHILDREN = {
    "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand",
    "RightArm": "RightForeArm", "RightForeArm": "RightHand",
    "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": "LeftToe",
    "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": "RightToe",
}
# Past this, a limb row is mapping the wrong bone or the wrong side rather than a slightly
# different T-pose.
MAX_LIMB_REST_DEGREES = 15.0
MAX_FACING_DEGREES = 2.0

# The --bvh clip is a forward walk, so on average the actor must travel the way it faces. A setup
# whose source faced backwards would read close to -1 here.
MIN_FORWARD_TRAVEL = 0.5


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", default=DEFAULT_BVH,
                        help="One forward-walking 100STYLE BVH; its skeleton becomes the setup's "
                             "source, and its motion checks the setup's facing.")
    parser.add_argument("--rig-blend", default=DEFAULT_RIG_BLEND,
                        help="Blend to take the corrected rig and its T-Pose track from.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Setup blend to write.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --out.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check the blend already at --out and write nothing.")
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def configure_scene(scene, frame_time):
    expected = 1.0 / SCENE_FPS
    if abs(frame_time - expected) > 1e-4:
        raise SetupError("The BVH is {:.1f} Hz but this setup runs at {} fps, so every clip would "
                         "play at {:.2f}x speed.".format(1.0 / frame_time, SCENE_FPS,
                                                        expected / frame_time))
    scene.render.fps = SCENE_FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1


def import_source(path):
    """Materialises the source skeleton by importing one clip exactly as the batch will, then
    turns its rest to face the corrected rig."""
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=path, **BVH_IMPORT)
    imported = [o for o in bpy.data.objects if o not in before]
    if len(imported) != 1:
        raise SetupError("BVH import produced {} objects, expected 1.".format(len(imported)))

    source = imported[0]
    source.name = source.data.name = SOURCE_NAME
    # Cleared before anything duplicates it, so a copied action cannot survive into the setup.
    source.animation_data_clear()

    found = {bone.name for bone in source.data.bones}
    if found != EXPECTED_SOURCE_BONES:
        raise SetupError("The BVH's skeleton is not 100STYLE's. Missing {}; unexpected {}."
                         .format(sorted(EXPECTED_SOURCE_BONES - found) or "none",
                                 sorted(found - EXPECTED_SOURCE_BONES) or "none"))
    roots = [bone.name for bone in source.data.bones if bone.parent is None]
    if roots != [ROOT_BONE]:
        raise SetupError("Expected {} to be the only root bone; found {}.".format(ROOT_BONE, roots))
    modes = {pb.rotation_mode for pb in source.pose.bones}
    if "QUATERNION" in modes:
        raise SetupError("Imported pose bones are quaternion, so rotate_mode was not NATIVE.")

    activate(source)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone in source.data.edit_bones:
        bone.transform(SOURCE_REST_TURN)
    bpy.ops.object.mode_set(mode="OBJECT")
    log("source imported: {} bones, euler order {}, rest turned a half turn about Z"
        .format(len(source.data.bones), "/".join(sorted(modes))))
    return source


def duplicate_cleaned(source):
    """Stage one's skeleton: the same rest pose, wired straight through.

    Duplicated rather than built over the armature data, because only a duplicate keeps the
    per-pose-bone euler order the stage-one bake's rotation_euler curves need.
    """
    activate(source)
    bpy.ops.object.duplicate(linked=False)
    cleaned = bpy.context.object
    cleaned.name = CLEANED_NAME
    if cleaned.data is source.data:
        cleaned.data = source.data.copy()
    cleaned.data.name = CLEANED_NAME
    cleaned.animation_data_clear()

    if any(pb.rotation_mode != source.pose.bones[pb.name].rotation_mode
           for pb in cleaned.pose.bones):
        raise SetupError("The duplicate's pose bones lost their euler order.")
    return cleaned


def constrain(cleaned, source):
    """Copies the source's world rotations onto every cleaned bone, and its root's location.

    The two rests are identical, so rotations alone reproduce the source exactly.
    """
    for pose_bone in cleaned.pose.bones:
        if pose_bone.name == ROOT_BONE:
            location = pose_bone.constraints.new("COPY_LOCATION")
            location.target, location.subtarget = source, pose_bone.name
        rotation = pose_bone.constraints.new("COPY_ROTATION")
        rotation.target, rotation.subtarget = source, pose_bone.name
    log("constrained {} bones to {}, plain pass-through".format(len(cleaned.pose.bones),
                                                                 source.name))


def write_bone_list(scene, cleaned, model):
    """Writes the mapping, refusing a row naming a bone that is not on its rig.

    Rokoko silently skips such a row, which shows up as a partly posed character rather than as an
    error.
    """
    problems = []
    for source_name, target_name, _ in BONE_MAP:
        if not cleaned.pose.bones.get(source_name):
            problems.append('source "{}"'.format(source_name))
        if not model.pose.bones.get(target_name):
            problems.append('target "{}"'.format(target_name))
    if problems:
        raise SetupError("The mapping names bones that are not on the rigs: " + ", ".join(problems))
    if len(model.data.bones) != MODEL_BONE_COUNT:
        raise SetupError("The target rig has {} bones, expected {}."
                         .format(len(model.data.bones), MODEL_BONE_COUNT))

    scene.rsl_retargeting_bone_list.clear()
    for source_name, target_name, key in BONE_MAP:
        item = scene.rsl_retargeting_bone_list.add()
        item.bone_name_source = source_name
        item.bone_name_target = target_name
        item.bone_name_key = key
        item.is_custom = False


def compose(args):
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene

    bvh = resolve(args.bvh)
    rig_blend = resolve(args.rig_blend)
    for label, path in (("BVH", bvh), ("rig blend", rig_blend)):
        if not os.path.isfile(path):
            raise SetupError("No {} at {}".format(label, path))
    log("bvh " + bvh)
    log("rig " + rig_blend)

    configure_scene(scene, read_bvh_frame_time(bvh))
    source = import_source(bvh)
    cleaned = duplicate_cleaned(source)
    constrain(cleaned, source)
    drop_existing_actions()
    model = append_model(rig_blend)
    log('model "{}" appended, {} bones'.format(model.name, len(model.data.bones)))

    # Both armature pointers clear the bone list when assigned, so the mapping is written last.
    scene.rsl_retargeting_armature_source = cleaned
    scene.rsl_retargeting_armature_target = model
    # The actor keeps his size and the rig its own; auto-scaling defaults to True and would only
    # float the feet, so this line is load-bearing.
    scene.rsl_retargeting_auto_scaling = False
    scene.rsl_retargeting_use_pose = "REST"
    write_bone_list(scene, cleaned, model)
    log("{} rows; source {} left unmapped on purpose"
        .format(len(BONE_MAP), ", ".join(UNMAPPED_SOURCE_BONES)))

    out = resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    log("wrote " + out)
    return out


# --- verification -------------------------------------------------------------------------------


def degrees_between(a, b):
    return math.degrees(a.angle(b))


def ground_plane(vector):
    return Vector((vector.x, vector.y, 0.0))


def rest_facing(armature, left_foot, right_foot, left_toe, right_toe):
    """The direction the rest pose faces, from its feet: heel-to-toe, in the ground plane."""
    heads = {b.name: armature.matrix_world @ b.head_local for b in armature.data.bones}
    return ground_plane(heads[left_toe] + heads[right_toe] - heads[left_foot] - heads[right_foot])


def check_rest_geometry(setup):
    """Compares the source rest with the corrected rig's in world space, never bone frame against
    bone frame: the two rigs' local axes need not agree, so only directions mean anything.

    Returns the worst limb disagreement after logging every one.
    """
    source_of = {target: source for source, target, _ in BONE_MAP}
    source_heads = {b.name: setup.source.matrix_world @ b.head_local
                    for b in setup.source.data.bones}
    model_heads = {b.name: setup.model.matrix_world @ b.head_local for b in setup.model.data.bones}

    facing = degrees_between(
        rest_facing(setup.source, *(source_of[n] for n in
                                    ("LeftFoot", "RightFoot", "LeftToe", "RightToe"))),
        rest_facing(setup.model, "LeftFoot", "RightFoot", "LeftToe", "RightToe"))
    lateral = degrees_between(source_heads[source_of["LeftUpLeg"]]
                              - source_heads[source_of["RightUpLeg"]],
                              model_heads["LeftUpLeg"] - model_heads["RightUpLeg"])
    log("rest facing differs by {:.2f} deg, left-to-right by {:.2f} deg".format(facing, lateral))
    if facing > MAX_FACING_DEGREES or lateral > MAX_FACING_DEGREES:
        raise SetupError("The source rest does not face the way the corrected rig does, or has its "
                         "sides swapped; every clip would travel against its facing.")

    worst = 0.0
    for target, child in LIMB_CHILDREN.items():
        have = source_heads[source_of[child]] - source_heads[source_of[target]]
        want = model_heads[child] - model_heads[target]
        angle = degrees_between(have, want)
        worst = max(worst, angle)
        log("  {:<14} <- {:<14} rest direction differs by {:5.2f} deg"
            .format(target, source_of[target], angle))
    if worst > MAX_LIMB_REST_DEGREES:
        raise SetupError("A limb's rest differs from the corrected rig's by {:.1f} deg, which is a "
                         "wrong row rather than a different T-pose.".format(worst))
    return worst


def check_forward_travel(scene, setup, bvh):
    """Plays the --bvh clip through the setup and checks the actor travels the way he faces.

    The rest comparison cannot catch a turn that also flips the motion, so this measures it on
    real motion: the cleaned root's ground-plane velocity against its facing, weighted by speed.
    Also confirms the cleaned skeleton lands on the source, since that is the constraints' job.
    """
    action = batch_retarget.import_bvh_action(bvh, setup, keep_position=False)
    batch_retarget.assign_action(setup.source, action)
    start, end = batch_retarget.action_frame_range(action)

    root_rest = setup.target.data.bones[ROOT_BONE].matrix_local.to_3x3()
    rest_forward = rest_facing(setup.source, "LeftAnkle", "RightAnkle", "LeftToe",
                               "RightToe").normalized()
    step = SCENE_FPS // 2
    previous, along, total, worst_joint = None, 0.0, 0.0, 0.0
    for frame in range(start, end + 1, step):
        scene.frame_set(frame)
        root = setup.target.pose.bones[ROOT_BONE].matrix
        position = setup.target.matrix_world @ root.to_translation()
        facing = ground_plane((root.to_3x3() @ root_rest.inverted()) @ rest_forward)
        if previous is not None and facing.length > 1e-6:
            travel = ground_plane(position - previous)
            if travel.length > 1e-6:
                along += travel.dot(facing.normalized())
                total += travel.length
        previous = position
        for bone in setup.target.pose.bones:
            here = setup.target.matrix_world @ bone.head
            there = setup.source.matrix_world @ setup.source.pose.bones[bone.name].head
            worst_joint = max(worst_joint, (here - there).length)

    batch_retarget.assign_action(setup.source, None)
    batch_retarget.drop_action(action)

    forward = along / total if total > 0.0 else 0.0
    log("{}: travels {:.2f} m, {:.2f} of it along its facing; cleaned joints off the source by "
        "{:.1e} m at worst".format(os.path.basename(bvh), total, forward, worst_joint))
    if forward < MIN_FORWARD_TRAVEL:
        raise SetupError("A forward walk travels {:.2f} along its facing; the source faces the "
                         "wrong way.".format(forward))
    if worst_joint > 1e-4:
        raise SetupError("The cleaned skeleton does not follow the source ({:.1e} m)."
                         .format(worst_joint))


def verify(path, bvh):
    """Re-opens the saved file and resolves it with batch_retarget's own code.

    Checking the blend in memory would prove nothing about what reached the disk, and reproducing
    the resolver here would let the two drift; this is the code the batch actually runs.
    """
    bpy.ops.wm.open_mainfile(filepath=path)
    scene = bpy.context.scene
    setup = batch_retarget.resolve_setup(scene)
    batch_retarget.check_bone_list(setup)
    batch_retarget.order_rest_take_first(setup)
    log('resolved: source "{}" -> cleaned "{}" -> model "{}" ({} mapped bones)'
        .format(setup.source.name, setup.target.name, setup.model.name, len(setup.bone_map)))

    problems = []
    if (scene.render.fps, scene.render.fps_base) != (SCENE_FPS, 1.0):
        problems.append("scene runs at {}/{}".format(scene.render.fps, scene.render.fps_base))
    if {a.name for a in bpy.data.actions} != {batch_retarget.REST_TRACK_NAME}:
        problems.append("actions are " + ", ".join(sorted(a.name for a in bpy.data.actions)))
    if scene.rsl_retargeting_auto_scaling:
        problems.append("Rokoko auto-scaling is on")
    if scene.rsl_retargeting_use_pose != "REST":
        problems.append("Rokoko pose mode is " + scene.rsl_retargeting_use_pose)

    rows = [(source, target, key) for source, target, key, _ in setup.bone_map]
    if rows != list(BONE_MAP):
        problems.append("the saved mapping is not BONE_MAP")
    targets = [target for _, target, _ in rows]
    if sorted(targets) != sorted(b.name for b in setup.model.data.bones):
        problems.append("the corrected rig's bones are not each mapped exactly once")
    sources = {source for source, _, _ in rows}
    if any(name in sources or name not in setup.source.data.bones
           for name in UNMAPPED_SOURCE_BONES):
        problems.append("an unmapped source bone is mapped after all, or missing")

    armatures = [o.name for o in scene.objects if o.type == "ARMATURE"]
    if len(armatures) != 3:
        problems.append("armatures are " + ", ".join(sorted(armatures)))
    if "QUATERNION" in {pb.rotation_mode for pb in setup.target.pose.bones}:
        problems.append("the cleaned skeleton has quaternion pose bones")
    for bone in setup.target.data.bones:
        original = setup.source.data.bones.get(bone.name)
        if original is None or (bone.head_local - original.head_local).length > 1e-9 or \
                (bone.tail_local - original.tail_local).length > 1e-9:
            problems.append("the cleaned rest differs from the source at " + bone.name)
            break
    if problems:
        raise SetupError("The saved blend is not usable: " + "; ".join(problems))

    worst_limb = check_rest_geometry(setup)
    check_forward_travel(scene, setup, bvh)
    log("verified: {} fps, {} armatures, {} rows, auto-scaling off, T-Pose track first, limbs "
        "within {:.1f} deg of the corrected rest".format(scene.render.fps, len(armatures),
                                                         len(rows), worst_limb))


def main():
    args = parse_args(sys.argv)
    out = resolve(args.out)

    if args.verify_only:
        if not os.path.isfile(out):
            raise SetupError("No blend at " + out)
    else:
        for label, other in (("--rig-blend", args.rig_blend), ("--bvh", args.bvh)):
            if os.path.abspath(resolve(other)) == os.path.abspath(out):
                raise SetupError("--out is the same file as {}.".format(label))
        if os.path.exists(out) and not args.force:
            raise SetupError("{} already exists. Pass --force to replace it.".format(out))
        compose(args)

    verify(out, resolve(args.bvh))
    log("done")


if __name__ == "__main__":
    try:
        main()
    # make_edinburgh_setup's helpers raise its own SetupError, the batch's code raises the batch's.
    except (SetupError, batch_retarget.SetupError) as error:
        log("ERROR: " + str(error))
        sys.exit(1)
