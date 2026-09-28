"""Rebuilds an existing retarget setup onto a Mixamo character, as a live setup.

    blender --background --python Tools/Retargeting/make_mixamo_setup.py -- \
        --setup Assets/LFS/Retargeting/lafan_bvh_to_lafan_corrected.blend \
        --model-fbx "Assets/LFS/Characters/Y Bot.fbx" \
        --clips Assets/LFS/Animation/lafan1/bvh/walk1_subject1.bvh \
        --out Assets/LFS/Retargeting/ybot/lafan_bvh_to_ybot.blend

Never pass --factory-startup: the setups this reads carry Rokoko scene properties, which only load
with the addon enabled. The input blend is only read.

Every existing setup retargets onto the shared corrected rig, and the judgement it holds -- a
cleaned skeleton whose rest pose is the calibration pose -- was authored against that rig's rest.
This keeps the source, the cleaned skeleton and its constraints, swaps the corrected rig for the
Mixamo character, and carries the calibration across: the character is posed so that it looks
the way the corrected rig did when the cleaned skeleton stands at rest.

The result is a live setup (see make_edinburgh_setup.py --live): open it, press play, and the
character follows whichever source clip is unmuted. direct_retarget.py batches from it unchanged.
"""

import argparse
import os
import sys

import bpy
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_retarget as batch
import direct_retarget as direct

from batch_retarget import SetupError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

HELPER_PREFIX = direct.HELPER_PREFIX
HELPER_COLLECTION = "Retarget"
GROUND_BONE = HELPER_PREFIX + "ground"

# Every existing setup targets the corrected rig, so its bone names are what the setups' mappings
# are written in. Mixamo names the same joints with a namespace, and calls the toe joint ToeBase.
CORRECTED_TO_MIXAMO = {
    "Hips": "mixamorig:Hips",
    "Spine": "mixamorig:Spine", "Spine1": "mixamorig:Spine1", "Spine2": "mixamorig:Spine2",
    "Neck": "mixamorig:Neck", "Head": "mixamorig:Head",
    "LeftShoulder": "mixamorig:LeftShoulder", "LeftArm": "mixamorig:LeftArm",
    "LeftForeArm": "mixamorig:LeftForeArm", "LeftHand": "mixamorig:LeftHand",
    "RightShoulder": "mixamorig:RightShoulder", "RightArm": "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm", "RightHand": "mixamorig:RightHand",
    "LeftUpLeg": "mixamorig:LeftUpLeg", "LeftLeg": "mixamorig:LeftLeg",
    "LeftFoot": "mixamorig:LeftFoot", "LeftToe": "mixamorig:LeftToeBase",
    "RightUpLeg": "mixamorig:RightUpLeg", "RightLeg": "mixamorig:RightLeg",
    "RightFoot": "mixamorig:RightFoot", "RightToe": "mixamorig:RightToeBase",
}
ROOT = "Hips"
TOES = ("LeftToe", "RightToe")

# The bones whose rest direction means the same thing on both rigs, each with the joint it points
# at, in corrected-rig names. A T-pose fixes where an arm or a leg points, so a difference there is
# a real difference in pose and is corrected. The torso, clavicles, head and hands are left alone:
# where a rig puts its spine and neck joints inside the body is a design choice, and two upright
# rests disagreeing by up to 15 degrees there are still both upright.
LIMB_CHILDREN = {
    "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand",
    "RightArm": "RightForeArm", "RightForeArm": "RightHand",
    "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": "LeftToe",
    "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": "RightToe",
}

# Scene properties the Rokoko addon stores. A live setup does not use them, and left pointing at
# the deleted rig they would only mislead whoever opens the Rokoko panel.
ROKOKO_POINTERS = ("rsl_retargeting_armature_source", "rsl_retargeting_armature_target")


def log(message):
    print("[mixamo-setup] " + message, flush=True)


def resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", required=True,
                        help="Existing setup blend retargeting onto the corrected rig.")
    parser.add_argument("--model-fbx", default="Assets/LFS/Characters/Y Bot.fbx",
                        help="Mixamo character to retarget onto.")
    parser.add_argument("--out", required=True, help="Setup blend to write.")
    parser.add_argument("--clips", nargs="*", default=[],
                        help="BVH to park on the source as NLA tracks, for previewing. Clips the "
                             "setup already holds are kept.")
    parser.add_argument("--keep-capture-position", action="store_true",
                        help="Import --clips the way a run with this flag would.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --out.")
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def world_heads(armature):
    return {b.name: armature.matrix_world @ b.head_local for b in armature.data.bones}


# --- reading the existing setup -----------------------------------------------------------------


def resolve_existing(scene):
    """The setup's source, cleaned skeleton and corrected rig, with the mapping between the last
    two -- read from its helper rig if it is live, else from its Rokoko bone list."""
    live = any(o.type == "ARMATURE" and any(b.name.startswith(HELPER_PREFIX) for b in o.data.bones)
               for o in scene.objects)
    setup = direct.resolve_setup(scene) if live else batch.resolve_setup(scene)
    rows = [(cleaned, corrected) for cleaned, corrected, _, _ in setup.bone_map
            if cleaned and corrected]
    unknown = sorted(c for _, c in rows if c not in CORRECTED_TO_MIXAMO)
    if unknown:
        raise SetupError("The setup maps onto bones this script has no Mixamo name for: "
                         + ", ".join(unknown))
    log('existing {} setup: source "{}" -> cleaned "{}" -> "{}", {} mapped bones'.format(
        "live" if live else "Rokoko", setup.source.name, setup.target.name, setup.model.name,
        len(rows)))
    return setup, rows


def strip_corrected_rig(scene, setup):
    """Removes the corrected rig and everything that only served it, keeping the source's parked
    clips."""
    keep = {setup.source, setup.target}
    for obj in [o for o in scene.objects if o.type == "ARMATURE" and o not in keep]:
        for child in obj.children_recursive:
            bpy.data.objects.remove(child, do_unlink=True)
        log("removing " + obj.name)
        bpy.data.objects.remove(obj, do_unlink=True)

    batch.activate(setup.target)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = setup.target.data.edit_bones
    helpers = [b.name for b in bones if b.name.startswith(HELPER_PREFIX)]
    for name in helpers:
        bones.remove(bones[name])
    bpy.ops.object.mode_set(mode="OBJECT")
    if helpers:
        log("removed {} helper bones from {}".format(len(helpers), setup.target.name))
    for collection in [c for c in setup.target.data.collections if c.name == HELPER_COLLECTION]:
        setup.target.data.collections.remove(collection)

    for obj in keep:
        if obj.animation_data:
            if obj.animation_data.use_tweak_mode:
                obj.animation_data.use_tweak_mode = False
            batch.assign_action(obj, None)
    parked = {s.action for t in (setup.source.animation_data.nla_tracks
                                 if setup.source.animation_data else ())
              for s in t.strips if s.action}
    for action in [a for a in bpy.data.actions if a not in parked]:
        action.use_fake_user = False
        bpy.data.actions.remove(action)
    bpy.data.orphans_purge(do_recursive=True)

    for pointer in ROKOKO_POINTERS:
        if hasattr(scene, pointer):
            setattr(scene, pointer, None)
    if hasattr(scene, "rsl_retargeting_bone_list"):
        scene.rsl_retargeting_bone_list.clear()


# --- the Mixamo character -----------------------------------------------------------------------


def import_model(path):
    """Imports the character at unit object scale, with no action of its own.

    The importer puts the file's centimetre units on the armature as a scale of 0.01. Exported like
    that, the scale lands on the FBX's armature node, which the Unity side rejects, so it is applied
    into the data here. The character's own take is dropped: its rest pose is re-keyed below.
    """
    before_objects, before_actions = set(bpy.data.objects), set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=path)
    imported = [o for o in bpy.data.objects if o not in before_objects]
    armatures = [o for o in imported if o.type == "ARMATURE"]
    if len(armatures) != 1:
        raise SetupError("{} holds {} armatures, expected 1.".format(os.path.basename(path),
                                                                     len(armatures)))
    model = armatures[0]
    model.name = os.path.splitext(os.path.basename(path))[0]

    meshes = [o for o in imported if o.type == "MESH"]
    bounds_before = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices[:1]]
    heads_before = world_heads(model)

    for action in [a for a in bpy.data.actions if a not in before_actions]:
        bpy.data.actions.remove(action)
    if model.animation_data:
        batch.assign_action(model, None)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = model
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()

    worst = max((a - b).length for a, b in zip(bounds_before, (o.matrix_world @ v.co for o in meshes
                                                               for v in o.data.vertices[:1])))
    worst = max([worst] + [(heads_before[n] - h).length for n, h in world_heads(model).items()])
    if worst > 1e-5:
        raise SetupError("Applying the import scale moved the character by {:.2e} m.".format(worst))
    if any(abs(s - 1.0) > 1e-6 for o in imported for s in o.scale):
        raise SetupError("An imported object still carries a scale after applying it.")

    # A Mixamo rig's "_End" bones are real bones; the export must not add leaves of its own.
    model[batch.ADD_LEAF_BONES_PROPERTY] = False
    log('model "{}": {} bones, {} meshes'.format(model.name, len(model.data.bones), len(meshes)))
    return model


def key_rest_track(model):
    """Keys the rest pose into the "T-Pose" track every export leads with."""
    batch.activate(model)
    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in model.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
        for path in ("location", "rotation_quaternion" if pose_bone.rotation_mode == "QUATERNION"
                     else "rotation_euler", "scale"):
            pose_bone.keyframe_insert(path, frame=1)
    bpy.ops.object.mode_set(mode="OBJECT")
    action = model.animation_data.action
    action.name = batch.REST_TRACK_NAME
    batch.push_to_nla(model, action, batch.REST_TRACK_NAME)


# --- calibration --------------------------------------------------------------------------------


def leg_height(heads, root, toes):
    """Root height above the lower toe joint: the length a stride scales with."""
    return heads[root].z - min(heads[t].z for t in toes)


def scale_actor(setup, rows, model):
    """Scales the source and cleaned skeletons so the actor's legs are as long as the character's.

    The character's root copies the actor's root position, so without this a longer-legged
    character covers the actor's stride on its own legs and slides its feet. A uniform scale of
    both objects about the origin is a similarity transform of everything upstream: rotations,
    constraints and IK all behave exactly as before.
    """
    identity = Matrix.Identity(4)
    for obj in (setup.source, setup.target):
        if any(abs(obj.matrix_world[i][j] - identity[i][j]) > 1e-6
               for i in range(4) for j in range(4)):
            raise SetupError('"{}" has an object transform; this setup expects it at the origin.'
                             .format(obj.name))

    cleaned_of = {corrected: cleaned for cleaned, corrected in rows}
    cleaned_heads = world_heads(setup.target)
    actor = leg_height(cleaned_heads, cleaned_of[ROOT], [cleaned_of[t] for t in TOES])
    character = leg_height(world_heads(model), CORRECTED_TO_MIXAMO[ROOT],
                           [CORRECTED_TO_MIXAMO[t] for t in TOES])
    factor = character / actor
    for obj in (setup.source, setup.target):
        obj.scale = (factor, factor, factor)
    bpy.context.view_layer.update()
    log("actor leg {:.3f} m, character leg {:.3f} m: actor scaled by {:.4f}".format(
        actor, character, factor))
    return factor


def calibrated_rest(model, corrected_heads):
    """Each mapped character bone's world rest orientation, posed to match the corrected rig.

    A limb is swung, by the smallest rotation, until it points where the corrected rig's did;
    everything else keeps the character's own rest. The swing leaves the limb's twist alone, so
    only direction is corrected, never roll -- the two rigs' local axes need not agree.
    """
    heads = world_heads(model)
    rest = {}
    for corrected, mixamo in CORRECTED_TO_MIXAMO.items():
        rotation = (model.matrix_world @ model.data.bones[mixamo].matrix_local).to_3x3().normalized()
        child = LIMB_CHILDREN.get(corrected)
        if child is not None:
            have = heads[CORRECTED_TO_MIXAMO[child]] - heads[mixamo]
            want = corrected_heads[child] - corrected_heads[corrected]
            swing = have.rotation_difference(want)
            rotation = swing.to_matrix() @ rotation
            log("  {:<14} swung {:5.2f} deg".format(corrected, swing.angle * 57.29578))
        rest[corrected] = rotation
    return rest


def build_live_rig(setup, rows, model, rest):
    """Helper bones on the cleaned skeleton, and the constraints that make the character follow
    them.

    Each helper sits at its character bone's calibrated rest but is parented to the cleaned bone it
    maps from, so Blender evaluates helper_world = cleaned_world @ cleaned_rest^-1 @ helper_rest --
    the delta Rokoko would bake, kept live.
    """
    cleaned = setup.target
    to_cleaned = cleaned.matrix_world.inverted()
    rotation_to_cleaned = cleaned.matrix_world.to_3x3().normalized().inverted()
    model_heads = world_heads(model)
    cleaned_of = {corrected: name for name, corrected in rows}

    batch.activate(cleaned)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = cleaned.data.edit_bones
    collection = cleaned.data.collections.new(HELPER_COLLECTION)

    root = bones[cleaned_of[ROOT]]
    ground = bones.new(GROUND_BONE)
    ground.head = root.head.copy()
    # Along +Y with zero roll, so its local axes are the world's and its Z reads as height.
    ground.tail = (root.head.x, root.head.y + 0.2, root.head.z)
    ground.roll = 0.0
    ground.parent = root
    ground.use_connect = False
    ground.use_deform = False
    # Not inheriting rotation keeps a height correction vertical instead of tipping with the pelvis.
    for attr in ("inherit_rotation", "use_inherit_rotation"):
        if hasattr(ground, attr):
            setattr(ground, attr, False)
    collection.assign(ground)

    for cleaned_name, corrected in rows:
        mixamo = CORRECTED_TO_MIXAMO[corrected]
        axis, roll = bpy.types.Bone.AxisRollFromMatrix(rotation_to_cleaned @ rest[corrected])
        head = to_cleaned @ model_heads[mixamo]
        length = model.data.bones[mixamo].length / cleaned.matrix_world.to_scale().x
        helper = bones.new(HELPER_PREFIX + mixamo)
        helper.head, helper.tail, helper.roll = head, head + axis * length, roll
        helper.parent = bones[cleaned_name]
        # Disconnected so an edit-mode drag of the cleaned bone does not carry its helper along.
        helper.use_connect = False
        helper.use_deform = False
        collection.assign(helper)
    bpy.ops.object.mode_set(mode="OBJECT")
    collection.is_visible = False

    for cleaned_name, corrected in rows:
        pose_bone = model.pose.bones[CORRECTED_TO_MIXAMO[corrected]]
        rotation = pose_bone.constraints.new("COPY_ROTATION")
        rotation.target, rotation.subtarget = cleaned, HELPER_PREFIX + pose_bone.name
        if corrected == ROOT:
            location = pose_bone.constraints.new("COPY_LOCATION")
            location.target, location.subtarget = cleaned, GROUND_BONE
    log("live rig: {} helpers plus {} on {}".format(len(rows), GROUND_BONE, cleaned.name))


# --- preview clips ------------------------------------------------------------------------------


def park_clips(setup, rows, paths, keep_position):
    """Loads clips onto the source as NLA tracks, through the batch's own import and checks."""
    checked = batch.Setup(setup.source, setup.target, None,
                          [(cleaned, corrected, "", False) for cleaned, corrected in rows])
    anim = setup.source.animation_data or setup.source.animation_data_create()
    for path in paths:
        action = batch.import_bvh_action(path, checked, keep_position)
        action.name = os.path.splitext(os.path.basename(path))[0]
        track = anim.nla_tracks.new()
        track.name = action.name
        track.strips.new(action.name, int(action.frame_range[0]), action)
        log("  clip " + action.name)

    tracks = list(anim.nla_tracks)
    if tracks and all(t.mute for t in tracks):
        tracks[0].mute = False
    for track in tracks:
        track.mute = track is not next(t for t in tracks if not t.mute)
    return tracks


def ground_on_clips(scene, setup, model, tracks, factor):
    """Sets the character's height from the parked clips, by the rule that a planted toe sits at
    its rest height.

    The actor's feet and the character's do not reach the floor the same way even with the legs
    scaled to match, so the height has to be measured in motion. In locomotion some foot is planted
    on most frames, so the median over every frame of the lower toe joint is the planted height.
    """
    if not tracks:
        log("no clips parked, so the height is left uncorrected")
        return
    toes = [CORRECTED_TO_MIXAMO[t] for t in TOES]
    rest = min((model.matrix_world @ model.data.bones[t].head_local).z for t in toes)
    muted = [t.mute for t in tracks]
    lows = []
    for solo in tracks:
        for track in tracks:
            track.mute = track is not solo
        strip = solo.strips[0]
        for frame in range(int(strip.frame_start), int(strip.frame_end) + 1, 2):
            scene.frame_set(frame)
            lows.append(min((model.matrix_world @ model.pose.bones[t].head).z for t in toes))
    for track, mute in zip(tracks, muted):
        track.mute = mute

    lows.sort()
    offset = rest - lows[len(lows) // 2]
    # The ground bone's axes are the cleaned armature's, which the actor scale also scales.
    setup.target.pose.bones[GROUND_BONE].location = (0.0, 0.0, offset / factor)
    log("planted toe measured {:.3f} m against a rest of {:.3f} m over {} frames: raised by "
        "{:+.3f} m".format(lows[len(lows) // 2], rest, len(lows), offset))


def frame_preview(scene, setup, tracks):
    """Opens on the unmuted clip, with the actor drawn over the character and the source hidden."""
    playing = next((t for t in tracks if not t.mute), None)
    if playing is not None and playing.strips:
        strip = playing.strips[0]
        scene.frame_start, scene.frame_end = int(strip.frame_start), int(strip.frame_end)
        scene.frame_set(scene.frame_start)
    setup.source.hide_set(True)
    setup.target.show_in_front = True


# --- run ----------------------------------------------------------------------------------------


def compose(args):
    setup_path = resolve(args.setup)
    bpy.ops.wm.open_mainfile(filepath=setup_path)
    scene = bpy.context.scene
    log("setup " + setup_path)

    setup, rows = resolve_existing(scene)
    corrected_heads = world_heads(setup.model)
    batch.make_operable(bpy.context.view_layer, [setup.source, setup.target, setup.model])
    strip_corrected_rig(scene, setup)

    model = import_model(resolve(args.model_fbx))
    missing = [CORRECTED_TO_MIXAMO[c] for _, c in rows
               if CORRECTED_TO_MIXAMO[c] not in model.data.bones]
    if missing:
        raise SetupError("The character lacks mapped bones: " + ", ".join(missing))
    key_rest_track(model)

    factor = scale_actor(setup, rows, model)
    build_live_rig(setup, rows, model, calibrated_rest(model, corrected_heads))

    tracks = park_clips(setup, rows, [resolve(c) for c in args.clips], args.keep_capture_position)
    ground_on_clips(scene, setup, model, tracks, factor)
    frame_preview(scene, setup, tracks)

    out = resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    log("wrote " + out)
    return out


def verify(path):
    """Re-opens the saved blend and resolves it with direct_retarget's own code."""
    bpy.ops.wm.open_mainfile(filepath=path)
    scene = bpy.context.scene
    setup = direct.resolve_setup(scene)
    batch.order_rest_take_first(setup)

    problems = []
    if setup.model.get(batch.ADD_LEAF_BONES_PROPERTY, True):
        problems.append("the model would be exported with extra leaf bones")
    if any(abs(s - 1.0) > 1e-6 for s in setup.model.scale):
        problems.append("the model carries an object scale")
    unmapped = [b for b in CORRECTED_TO_MIXAMO.values() if b not in
                {m for _, m, _, _ in setup.bone_map}]
    log('verified: source "{}" -> bridge "{}" -> model "{}", {} mapped bones; left at rest: {}'
        .format(setup.source.name, setup.target.name, setup.model.name, len(setup.bone_map),
                ", ".join(unmapped) or "none"))
    if problems:
        raise SetupError("The saved blend is not usable: " + "; ".join(problems))


def main():
    args = parse_args(sys.argv)
    out = resolve(args.out)
    if os.path.abspath(out) == os.path.abspath(resolve(args.setup)):
        raise SetupError("--out is the same file as --setup.")
    if os.path.exists(out) and not args.force:
        raise SetupError("{} already exists. Pass --force to replace it.".format(out))
    verify(compose(args))
    log("done")


if __name__ == "__main__":
    try:
        main()
    except SetupError as error:
        log("ERROR: " + str(error))
        sys.exit(1)
