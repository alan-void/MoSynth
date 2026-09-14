"""Composes the Edinburgh-to-corrected-rig retarget setup from one converted BVH and the shared rig.

    blender --background --python Tools/Retargeting/make_edinburgh_setup.py -- \
        --bvh Assets/LFS/Animation/Edinburgh/bvh/edin_locomotion_0000.bvh \
        --out Assets/LFS/Retargeting/edinburgh_bvh_to_lafan_corrected.blend

Never pass --factory-startup: the Rokoko addon supplies stage two and the scene properties this
writes. Unlike the other two setups there is no blend to start from, so this one begins empty.

The judgement a setup blend normally carries -- a cleaned skeleton whose rest pose is the
calibration pose stage two retargets from -- was spent upstream instead: edinburgh_npz_to_bvh.py
authors its BVH already in a T-pose, with the target rig's own bone names. That leaves stage one
with nothing to clean, so the cleaned skeleton here is a plain duplicate of the source wired
straight through, which the README sanctions: "where the cleaned skeleton is a rotation-only copy,
rotations alone already reproduce the source exactly and IK can only add solver noise."

Re-run it if the converter's skeleton changes; every BVH it writes must match this blend's source.
"""

import argparse
import os
import sys

import bpy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DEFAULT_RIG_BLEND = ("Assets/LFS/Retargeting/bandai-namco/"
                     "bandai_namco_retarget_to_corrected_lafan_claude.blend")
DEFAULT_BVH = "Assets/LFS/Animation/Edinburgh/bvh/edin_locomotion_0000.bvh"
DEFAULT_OUT = "Assets/LFS/Retargeting/edinburgh_bvh_to_lafan_corrected.blend"

# The corrected rig and its skinned mesh, taken from the Bandai setup because only that copy
# carries the "T-Pose" NLA track batch_retarget.order_rest_take_first requires.
MODEL_NAME = "Lafan_corrected"
MODEL_MESH_NAME = "Mesh"

# How much of the actor's spine bend to pass on. Edinburgh's markers read about 2.4 times his real
# lean -- measured against where his head sits over his ankles, the only indicator on this skeleton
# that does not involve a pelvis marker -- because this marker set samples the spine differently
# from the target rig. Copying the bend verbatim puts a 20 to 30 degree hunch on a character whose
# body only leaned 8 to 14, so it is scaled here rather than in the BVH, which stays exactly
# faithful to the point cloud.
#
# A scale and not an offset: subtracting a fixed angle flattens every clip to the same posture and
# tips the strongly-leaning ones backwards, while a scale keeps each clip's own lean in proportion.
# One factor covers the whole dataset; nothing here is per clip.
SPINE_LEAN_SCALE = 0.42
SPINE_LEAN_BONES = ("Spine", "Spine1", "Neck")

# Splitting the root's rotation per axis means Blender decomposes it to euler, and an euler order
# gimbal-locks on its middle axis. The BVH's bones are ZYX, whose middle axis is Y -- the very one
# carrying the character's heading -- so a clip that turns far enough drags the damped axes through
# a 94 degree snap as the yaw crosses 90. YXZ puts the middle on X, where a pelvis would have to
# pitch 90 degrees to reach it.
ROOT_ROTATION_MODE = "YXZ"

# The live-preview rig. Each helper sits at one model bone's rest but is parented to the cleaned
# bone of the same name, so Blender evaluates
#     helper_world = cleaned_world @ cleaned_rest^-1 @ helper_rest
# which is exactly the delta Rokoko bakes. Because the rest matrices are read fresh every
# evaluation, editing the cleaned skeleton's rest pose updates the character immediately.
HELPER_PREFIX = "RT_"
HELPER_COLLECTION = "Retarget"

# The one dial for where the character stands. The model's root copies its location from this bone
# rather than straight from the cleaned root, so the height can be corrected without touching the
# model rig at all. It rests along +Y with zero roll, which makes its local frame the world
# identity, and it does not inherit rotation -- so moving it is a plain world-space translation
# that does not tip with the pelvis. Needed because the two skeletons divide their leg length
# differently: the model's hip-to-toe chain is 4.3 cm longer overall yet reaches less vertically,
# so its feet float even with the hips placed exactly right.
GROUND_BONE = HELPER_PREFIX + "ground"

SOURCE_NAME = "edinburgh_source"
CLEANED_NAME = "edinburgh_cleaned"
ROOT_BONE = "Hips"

# One BVH sample is one Blender frame: the importer is called with use_fps_scale off, so it
# ignores the file's Frame Time entirely and the scene's rate is the only thing that turns frames
# into seconds. An empty blend starts at 24, which would silently run every clip a quarter slow.
SCENE_FPS = 30

# Must stay identical to batch_retarget.import_bvh_action's flags. A divergence here -- a missing
# global_scale above all -- puts this blend's rest pose a factor of 100 from every clip's, and
# check_skeleton_matches then rejects the entire dataset.
BVH_IMPORT = dict(global_scale=0.01, rotate_mode="NATIVE", axis_forward="Z", axis_up="Y",
                  use_fps_scale=False, update_scene_fps=False, update_scene_duration=False)

# (bone name on both rigs, Rokoko's auto-detection key). Every row is an identity row because the
# converter authors the BVH with the target's own names. The keys are Rokoko's, read off the rows
# it generated for the LAFAN setup rather than invented; they are recorded in the blend and shown
# in its UI, but the retarget itself matches on names.
BONE_MAP = (
    ("Hips", "hip"),
    ("LeftUpLeg", "leftUpLeg"), ("LeftLeg", "leftLeg"),
    ("LeftFoot", "leftFoot"), ("LeftToe", "leftToe"),
    ("RightUpLeg", "rightUpLeg"), ("RightLeg", "rightLeg"),
    ("RightFoot", "rightFoot"), ("RightToe", "rightToe"),
    ("Spine", "spine"), ("Spine1", "spine"),
    ("Neck", "neck"), ("Head", "head"),
    ("LeftShoulder", "leftShoulder"), ("LeftArm", "leftUpperArm"),
    ("LeftForeArm", "leftLowerArm"), ("LeftHand", "leftHand"),
    ("RightShoulder", "rightShoulder"), ("RightArm", "rightUpperArm"),
    ("RightForeArm", "rightLowerArm"), ("RightHand", "rightHand"),
)

# Deliberate omissions, checked rather than assumed. The source bends its back in two places where
# the target has three, so the target's Spine2 rides at rest -- the compromise the Bandai setup
# makes with Spine1. The source's two hip-joint segments are real rotating bones the target has no
# counterpart for, since it hangs its thighs off a rigid pelvis.
UNMAPPED_TARGET_BONES = ("Spine2",)
UNMAPPED_SOURCE_BONES = ("LeftHipJoint", "RightHipJoint")

EXPECTED_BONES = frozenset([name for name, _ in BONE_MAP] + list(UNMAPPED_SOURCE_BONES))
MODEL_BONE_COUNT = 22


class SetupError(Exception):
    """The inputs are not what this composition expects."""


def log(message):
    print("[setup] " + message, flush=True)


def resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", default=DEFAULT_BVH,
                        help="One converted BVH; its skeleton becomes the setup's source.")
    parser.add_argument("--rig-blend", default=DEFAULT_RIG_BLEND,
                        help="Blend to take the corrected rig and its T-Pose track from.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Setup blend to write.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --out.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check the blend already at --out and write nothing.")
    parser.add_argument("--clips", nargs="*", default=[],
                        help="BVH to load onto the source as NLA tracks, for authoring by hand.")
    parser.add_argument("--no-damping", action="store_true",
                        help="Plain pass-through constraints, leaving the rest pose the only knob.")
    parser.add_argument("--live", action="store_true",
                        help="Drive the model rig from the cleaned skeleton live, for authoring a "
                             "rest pose. Gives the model bone constraints, which batch_retarget's "
                             "resolve_setup refuses, so a live blend is for editing and not runs.")
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def read_bvh_frame_time(path):
    """Reads the header in plain Python, to catch a converter that changed rate before the scene
    rate silently rescales every clip."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Frame Time:"):
                return float(line.split(":", 1)[1])
    raise SetupError("No 'Frame Time:' in " + path)


def configure_scene(scene, frame_time):
    expected = 1.0 / SCENE_FPS
    if abs(frame_time - expected) > 1e-4:
        raise SetupError(
            "The BVH is {:.1f} Hz but this setup runs at {} fps, and the importer keys one sample "
            "per frame regardless, so every clip would play at {:.2f}x speed. Re-run the converter "
            "at {} Hz.".format(1.0 / frame_time, SCENE_FPS, expected / frame_time, SCENE_FPS))
    scene.render.fps = SCENE_FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1


def activate(obj):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def import_source(path):
    """Materialises the source skeleton by importing one clip, exactly as the batch will."""
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=path, **BVH_IMPORT)
    imported = [o for o in bpy.data.objects if o not in before]
    if len(imported) != 1:
        raise SetupError("BVH import produced {} objects, expected 1.".format(len(imported)))

    source = imported[0]
    source.name = source.data.name = SOURCE_NAME

    found = {bone.name for bone in source.data.bones}
    if found != EXPECTED_BONES:
        raise SetupError("The BVH's skeleton is not the converter's. Missing {}; unexpected {}."
                         .format(sorted(EXPECTED_BONES - found) or "none",
                                 sorted(found - EXPECTED_BONES) or "none"))
    roots = [bone.name for bone in source.data.bones if bone.parent is None]
    if roots != [ROOT_BONE]:
        raise SetupError("Expected {} to be the only root bone; found {}.".format(ROOT_BONE, roots))

    modes = {pb.rotation_mode for pb in source.pose.bones}
    if "QUATERNION" in modes:
        raise SetupError("Imported pose bones are quaternion, so rotate_mode was not NATIVE.")
    log("source imported: {} bones, euler order {}".format(len(source.data.bones),
                                                           "/".join(sorted(modes))))
    for bone in sorted(source.data.bones, key=lambda b: b.name):
        log("  {:<14} {:.5f} m".format(bone.name, bone.length))

    # Cleared before anything duplicates it, so a copied action cannot survive into the setup.
    source.animation_data_clear()
    return source


def duplicate_cleaned(source):
    """Stage one's skeleton: the same rest pose, wired straight through.

    Duplicating carries the per-pose-bone rotation mode across, which building an object over the
    armature data would not: a synthesised object defaults to quaternion bones, and the stage-one
    bake's rotation_euler curves would then drive nothing. batch_retarget makes stage two's proxy
    by duplicating this one, so the modes have to be right here too.
    """
    activate(source)
    bpy.ops.object.duplicate(linked=False)
    cleaned = bpy.context.object
    cleaned.name = CLEANED_NAME
    if cleaned.data is source.data:
        cleaned.data = source.data.copy()
    cleaned.data.name = CLEANED_NAME
    cleaned.animation_data_clear()

    modes = {pb.name: pb.rotation_mode for pb in cleaned.pose.bones}
    if modes != {pb.name: pb.rotation_mode for pb in source.pose.bones}:
        raise SetupError("The duplicate's pose bones lost their euler order.")
    for bone in cleaned.data.bones:
        original = source.data.bones[bone.name]
        if (bone.head_local - original.head_local).length > 1e-9 or \
                (bone.tail_local - original.tail_local).length > 1e-9:
            raise SetupError("The duplicate's rest pose differs at " + bone.name)
    return cleaned


def add_clips(source, paths):
    """Loads clips onto the source as NLA tracks, so a rest pose can be judged against real motion.

    The cleaned skeleton is constrained to the source, and those constraints are live, so it
    follows whichever track is unmuted while its rest pose is being edited.
    """
    for path in paths:
        before = set(bpy.data.objects)
        bpy.ops.import_anim.bvh(filepath=path, **BVH_IMPORT)
        imported = [o for o in bpy.data.objects if o not in before]
        action = imported[0].animation_data.action if imported[0].animation_data else None
        if action is None:
            raise SetupError("BVH has no animation: " + os.path.basename(path))

        action.name = os.path.splitext(os.path.basename(path))[0]
        action.use_fake_user = True
        armature = imported[0].data
        bpy.data.objects.remove(imported[0], do_unlink=True)
        bpy.data.armatures.remove(armature)
        action.use_fake_user = False

        if source.animation_data is None:
            source.animation_data_create()
        track = source.animation_data.nla_tracks.new()
        track.name = action.name
        track.strips.new(action.name, 1, action)
        track.mute = len(source.animation_data.nla_tracks) > 1
        log("  clip " + action.name)

    if paths:
        first = source.animation_data.nla_tracks[0].strips[0]
        bpy.context.scene.frame_end = int(first.frame_end)
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def live_retarget(cleaned, model):
    """Wires the model rig to follow the cleaned skeleton through helper bones, permanently.

    Rokoko builds this same rig, bakes through it and throws it away. Kept instead, it makes the
    retarget live -- which is the only way to judge a rest pose while editing it, since Rokoko
    cannot be re-run cheaply and mis-reads a constraint-driven rig anyway: it clears a bone's
    pose by zeroing rotation_quaternion, which a constraint simply re-applies, so it builds its
    helpers against whatever frame the playhead happens to be on.
    """
    # The model's rest bones, re-expressed in the cleaned armature's space. Both objects carry
    # their own transform -- the model rig sits at -90 degrees about X -- so neither can be ignored.
    into_cleaned = cleaned.matrix_world.inverted() @ model.matrix_world
    activate(model)
    bpy.ops.object.mode_set(mode="EDIT")
    transforms = {}
    for bone in model.data.edit_bones:
        _, roll = bpy.types.Bone.AxisRollFromMatrix(into_cleaned.to_3x3() @ bone.matrix.to_3x3())
        transforms[bone.name] = (into_cleaned @ bone.head, into_cleaned @ bone.tail, roll)
    bpy.ops.object.mode_set(mode="OBJECT")

    activate(cleaned)
    bpy.ops.object.mode_set(mode="EDIT")
    collection = cleaned.data.collections.new(HELPER_COLLECTION)

    root = cleaned.data.edit_bones[ROOT_BONE]
    ground = cleaned.data.edit_bones.new(GROUND_BONE)
    ground.head = root.head.copy()
    ground.tail = (root.head.x, root.head.y + 0.2, root.head.z)
    ground.roll = 0.0
    ground.parent = root
    ground.use_connect = False
    ground.use_deform = False
    for attr in ("inherit_rotation", "use_inherit_rotation"):
        if hasattr(ground, attr):
            setattr(ground, attr, False)
    collection.assign(ground)

    for name, _ in BONE_MAP:
        helper = cleaned.data.edit_bones.new(HELPER_PREFIX + name)
        helper.head, helper.tail, helper.roll = transforms[name]
        helper.parent = cleaned.data.edit_bones[name]
        # Disconnected on purpose: an edit-mode transform moves a connected child with its parent,
        # and a helper dragged along by the bone being edited would show no change at all.
        helper.use_connect = False
        helper.use_deform = False
        collection.assign(helper)
    bpy.ops.object.mode_set(mode="OBJECT")

    for name, _ in BONE_MAP:
        pose_bone = model.pose.bones[name]
        rotation = pose_bone.constraints.new("COPY_ROTATION")
        rotation.target, rotation.subtarget = cleaned, HELPER_PREFIX + name
        if name == ROOT_BONE:
            location = pose_bone.constraints.new("COPY_LOCATION")
            location.target, location.subtarget = cleaned, GROUND_BONE
    collection.is_visible = False
    log("live preview: {} helper bones plus {} on {}, {} driving {}"
        .format(len(BONE_MAP), GROUND_BONE, cleaned.name, HELPER_COLLECTION, model.name))


def constrain(cleaned, source, damping=True):
    """Copies the source's world rotations onto the cleaned skeleton, and its root's location.

    No IK and no custom spaces: the two rests are identical, so rotations alone reproduce the
    source exactly and a solver could only add noise. LAFAN's setup needs more only because its
    cleaning moved bones.
    """
    def copy_rotation(pose_bone, axes, influence):
        rotation = pose_bone.constraints.new("COPY_ROTATION")
        rotation.target, rotation.subtarget = source, pose_bone.name
        # Local space, so an influence below 1 scales this bone's bend against its parent and
        # leaves the heading it inherits alone. In world space it would drag the whole torso back
        # towards the rest facing every time the character turned.
        if influence < 1.0 or axes != "xyz":
            rotation.owner_space = rotation.target_space = "LOCAL"
        rotation.use_x, rotation.use_y, rotation.use_z = (a in axes for a in "xyz")
        rotation.influence = influence
        return rotation

    for pose_bone in cleaned.pose.bones:
        if pose_bone.name == ROOT_BONE and not damping:
            location = pose_bone.constraints.new("COPY_LOCATION")
            location.target, location.subtarget = source, pose_bone.name
            copy_rotation(pose_bone, "xyz", 1.0)
        elif pose_bone.name == ROOT_BONE:
            location = pose_bone.constraints.new("COPY_LOCATION")
            location.target, location.subtarget = source, pose_bone.name
            # Nearly all of the actor's lean is in this one segment -- his hip-to-lower-back axis
            # reads 24 degrees where his hip-to-neck reads 23.7, so the spine above it is almost
            # straight and damping only those bones changed nothing. A bone points along its own
            # local Y, so for this one that axis is the pelvis-up: copy it whole and the character
            # still turns, damp the other two and only the lean comes down.
            pose_bone.rotation_mode = ROOT_ROTATION_MODE
            copy_rotation(pose_bone, "y", 1.0)
            copy_rotation(pose_bone, "xz", SPINE_LEAN_SCALE)
        elif damping and pose_bone.name in SPINE_LEAN_BONES:
            copy_rotation(pose_bone, "xyz", SPINE_LEAN_SCALE)
        else:
            copy_rotation(pose_bone, "xyz", 1.0)

    total = sum(len(pb.constraints) for pb in cleaned.pose.bones)
    targets = {c.target for pb in cleaned.pose.bones for c in pb.constraints}
    expected = len(cleaned.pose.bones) + (2 if damping else 1)
    if total != expected or targets != {source}:
        raise SetupError("Constraint wiring is not the expected pass-through.")
    if damping:
        log("constrained {} bones ({} constraints) to {}; spine bend scaled to {:.0%} on {}"
            .format(len(cleaned.pose.bones), total, source.name, SPINE_LEAN_SCALE,
                    ", ".join((ROOT_BONE + " pitch",) + SPINE_LEAN_BONES)))
    else:
        log("constrained {} bones ({} constraints) to {}; no damping, so the cleaned rest pose is "
            "the only thing shaping the retarget"
            .format(len(cleaned.pose.bones), total, source.name))


def drop_existing_actions():
    """Empties the blend of actions before the model rig is appended.

    The imported clip's action is the only one here, and it is why an appended rest-pose action
    would otherwise arrive renamed "T-Pose.001" and export a take under that name.
    """
    for action in list(bpy.data.actions):
        action.use_fake_user = False
        bpy.data.actions.remove(action)
    bpy.data.orphans_purge(do_recursive=True)


def append_model(rig_blend):
    """Brings the corrected rig and its mesh across in one call, so parenting, the armature
    modifier and the NLA tracks arrive intact."""
    for name in (MODEL_NAME, MODEL_MESH_NAME):
        if name in bpy.data.objects:
            raise SetupError('An object named "{}" is already in the blend, so the appended one '
                             "would be renamed and could not be found.".format(name))

    bpy.ops.wm.append(directory=rig_blend + "/Object/",
                      files=[{"name": MODEL_NAME}, {"name": MODEL_MESH_NAME}])

    model = bpy.data.objects.get(MODEL_NAME)
    if model is None:
        raise SetupError('No object named "{}" in {}.'.format(MODEL_NAME, rig_blend))

    anim = model.animation_data
    rest = next((t for t in anim.nla_tracks if t.name == "T-Pose"), None) if anim else None
    if rest is None or not rest.strips or rest.strips[0].action is None:
        raise SetupError('The appended "{}" has no "T-Pose" NLA track holding an action, so an '
                         "export would lead with a locomotion take.".format(MODEL_NAME))

    for track in [t for t in anim.nla_tracks if t is not rest and not t.strips]:
        log("dropping empty NLA track " + track.name)
        anim.nla_tracks.remove(track)

    if anim.action:
        log('clearing active action "{}" on {}'.format(anim.action.name, model.name))
        anim.action = None

    # The append brings whatever the rig was last posed with, and a purge leaves it behind because
    # it arrives with a fake user. Anything but the rest pose would export as an extra take.
    for action in [a for a in bpy.data.actions if a is not rest.strips[0].action]:
        log("dropping appended action " + action.name)
        action.use_fake_user = False
        bpy.data.actions.remove(action)
    bpy.data.orphans_purge(do_recursive=True)
    return model


def write_bone_list(scene, cleaned, model):
    """Writes the mapping, refusing any name that is not on both rigs.

    Rokoko silently skips a row naming a bone that does not exist, which shows up as a partly
    posed character rather than as an error, so the names are checked here instead.
    """
    missing = []
    for name, _ in BONE_MAP:
        if not cleaned.pose.bones.get(name):
            missing.append('source "{}"'.format(name))
        if not model.pose.bones.get(name):
            missing.append('target "{}"'.format(name))
    mapped = {name for name, _ in BONE_MAP}
    for name in UNMAPPED_TARGET_BONES:
        if not model.pose.bones.get(name) or name in mapped:
            missing.append('an unmapped target bone "{}" that is not on the rig, or is mapped after '
                           "all".format(name))
    if missing:
        raise SetupError("The mapping names bones that are not on the rigs: " + ", ".join(missing))
    if len(model.data.bones) != MODEL_BONE_COUNT:
        raise SetupError("The target rig has {} bones, expected {}."
                         .format(len(model.data.bones), MODEL_BONE_COUNT))

    scene.rsl_retargeting_bone_list.clear()
    for name, key in BONE_MAP:
        item = scene.rsl_retargeting_bone_list.add()
        item.bone_name_source = name
        item.bone_name_target = name
        item.bone_name_key = key
        item.is_custom = False
    return len(BONE_MAP)


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
    constrain(cleaned, source, damping=not args.no_damping)
    drop_existing_actions()
    model = append_model(rig_blend)
    log('model "{}" appended, {} bones'.format(model.name, len(model.data.bones)))

    # Both armature pointers clear the bone list when assigned, so the mapping is written last.
    scene.rsl_retargeting_armature_source = cleaned
    scene.rsl_retargeting_armature_target = model
    # The converter writes the actor at his own size and the target rig keeps its own, so
    # auto-scaling has nothing to correct and would only float the feet. It defaults to True, so
    # this line is load-bearing. Matches both existing setups.
    scene.rsl_retargeting_auto_scaling = False
    scene.rsl_retargeting_use_pose = "REST"

    rows = write_bone_list(scene, cleaned, model)
    log("{} identity rows; target {} and source {} left unmapped on purpose"
        .format(rows, ", ".join(UNMAPPED_TARGET_BONES), ", ".join(UNMAPPED_SOURCE_BONES)))

    if args.live:
        live_retarget(cleaned, model)

    clips = add_clips(source, [resolve(c) for c in args.clips])
    if clips:
        log("{} clip(s) on {}; the cleaned skeleton follows whichever track is unmuted"
            .format(len(clips), source.name))

    out = resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    log("wrote " + out)
    return out, clips


def verify(path, clips=(), live=False):
    """Re-opens the saved file and resolves it with batch_retarget's own code.

    Checking the blend in memory would prove nothing about what reached the disk, and reproducing
    the resolver here would let the two drift; this is the code the batch actually runs.
    """
    bpy.ops.wm.open_mainfile(filepath=path)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import batch_retarget

    scene = bpy.context.scene
    if live:
        return verify_live(scene)

    setup = batch_retarget.resolve_setup(scene)
    batch_retarget.check_bone_list(setup)
    batch_retarget.order_rest_take_first(setup)
    log('resolved: source "{}" -> cleaned "{}" -> model "{}" ({} mapped bones)'
        .format(setup.source.name, setup.target.name, setup.model.name, len(setup.bone_map)))

    problems = []
    if (scene.render.fps, scene.render.fps_base) != (SCENE_FPS, 1.0):
        problems.append("scene runs at {}/{}".format(scene.render.fps, scene.render.fps_base))
    if {a.name for a in bpy.data.actions} != {"T-Pose"} | set(clips):
        problems.append("actions are " + ", ".join(sorted(a.name for a in bpy.data.actions)))
    if scene.rsl_retargeting_auto_scaling:
        problems.append("Rokoko auto-scaling is on")
    if scene.rsl_retargeting_use_pose != "REST":
        problems.append("Rokoko pose mode is " + scene.rsl_retargeting_use_pose)
    if len(setup.bone_map) != len(BONE_MAP):
        problems.append("{} bone rows, expected {}".format(len(setup.bone_map), len(BONE_MAP)))
    if any(source != target for source, target, _, _ in setup.bone_map):
        problems.append("the mapping is not an identity")
    armatures = [o.name for o in scene.objects if o.type == "ARMATURE"]
    if len(armatures) != 3:
        problems.append("armatures are " + ", ".join(sorted(armatures)))
    if "QUATERNION" in {pb.rotation_mode for pb in setup.target.pose.bones}:
        problems.append("the cleaned skeleton has quaternion pose bones")
    for bone in setup.target.data.bones:
        original = setup.source.data.bones.get(bone.name)
        if original is None or (bone.head_local - original.head_local).length > 1e-9:
            problems.append("the cleaned rest differs from the source at " + bone.name)
            break

    if problems:
        raise SetupError("The saved blend is not usable: " + "; ".join(problems))
    log("verified: {} fps, {} armatures, {} identity rows, auto-scaling off, T-Pose track first"
        .format(scene.render.fps, len(armatures), len(setup.bone_map)))


def verify_live(scene):
    """Checks the live rig, which resolve_setup cannot look at: it wants exactly one armature
    carrying bone constraints, and a live blend has two."""
    cleaned = bpy.data.objects[CLEANED_NAME]
    model = bpy.data.objects[MODEL_NAME]
    problems = []

    # The point of the whole rig: every helper must sit at its model bone's rest, in world space.
    # A helper that does not is a bone the character will follow to the wrong place.
    worst = 0.0
    for name, _ in BONE_MAP:
        helper = cleaned.data.bones.get(HELPER_PREFIX + name)
        if helper is None:
            problems.append("no helper for " + name)
            continue
        if helper.parent is None or helper.parent.name != name:
            problems.append(HELPER_PREFIX + name + " is not parented to " + name)
            continue
        want = (model.matrix_world @ model.data.bones[name].matrix_local).to_3x3().normalized()
        got = (cleaned.matrix_world @ helper.matrix_local).to_3x3().normalized()
        worst = max(worst, max(abs(want[i][j] - got[i][j]) for i in range(3) for j in range(3)))

    if worst > 1e-5:
        problems.append("helper rest orientations differ from the model's by {:.2e}".format(worst))
    driven = sum(1 for pb in model.pose.bones for c in pb.constraints)
    if driven != len(BONE_MAP) + 1:
        problems.append("model carries {} constraints, expected {}".format(driven, len(BONE_MAP) + 1))
    if any(b.use_deform for b in cleaned.data.bones if b.name.startswith(HELPER_PREFIX)):
        problems.append("a helper bone would deform the mesh")

    if problems:
        raise SetupError("The live rig is not usable: " + "; ".join(problems))
    log("verified live rig: {} helpers match the model rest to {:.1e}, {} constraints on {}"
        .format(len(BONE_MAP), worst, driven, model.name))


def main():
    args = parse_args(sys.argv)
    out = resolve(args.out)
    clips = [os.path.splitext(os.path.basename(c))[0] for c in args.clips]

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

    verify(out, clips, args.live)
    log("done")


if __name__ == "__main__":
    try:
        main()
    except SetupError as error:
        log("ERROR: " + str(error))
        sys.exit(1)
