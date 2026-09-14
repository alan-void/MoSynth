"""Overlays a converted Edinburgh BVH on the source point cloud it was solved from.

    blender --background --python Tools/Retargeting/edinburgh_solve_preview.py -- \
        --npz External/LFS/edinburgh_locomotion_mocap_dataset/data/edinburgh_locomotion_test.npz \
        --bvh Assets/LFS/Animation/Edinburgh/bvh/edin_test_00.bvh \
        --index 0 --out Assets/LFS/Retargeting/edinburgh_solve_preview.blend

The release is 21 joint positions per frame and nothing else, so the only way to judge the solved
skeleton by eye is against those positions. This builds a blend holding both, in the same space
and on the same frames: the armature should sit inside the points throughout.

The points are raw source data -- only re-oriented, scaled and decimated the way the converter
does it -- so anything the two disagree about is the solve, not the reading.

Orientation is detected across whatever --npz files are passed, exactly as the converter does. Pass
the same set it was run with, or a split that happens to choose a different quarter turn would lay
the points at right angles to the armature.
"""

import argparse
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edinburgh_npz_to_bvh as converter

REPO_ROOT = converter.REPO_ROOT

POINTS_NAME = "source_points"
MARKER_NAME = "source_marker"
MARKER_RADIUS = 0.018


def log(message):
    print("[preview] " + message, flush=True)


def to_blender(positions):
    """Centimetres in BVH axes to metres in Blender's, the way the importer does it.

    Blender's BVH importer, with the axis_forward="Z" and axis_up="Y" flags batch_retarget uses,
    lands BVH (x, y, z) at Blender (-x, z, y); global_scale=0.01 does the rest.
    """
    return np.stack([-positions[..., 0], positions[..., 2], positions[..., 1]], axis=-1) * 0.01


def build_points(positions):
    """One vertex per Edinburgh joint, edged into its skeleton and keyed on every frame."""
    parents = converter.SOURCE_PARENTS
    edges = [(i, p) for i, p in enumerate(parents) if p >= 0]

    mesh = bpy.data.meshes.new(POINTS_NAME)
    mesh.from_pydata([tuple(p) for p in positions[0]], edges, [])
    mesh.update()

    obj = bpy.data.objects.new(POINTS_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)

    for frame, pose in enumerate(positions, start=1):
        for vertex, point in zip(mesh.vertices, pose):
            vertex.co = point
            vertex.keyframe_insert("co", frame=frame)

    # A vertex draws nothing in object mode, so the object instances a small sphere on each one.
    marker = bpy.data.meshes.new(MARKER_NAME)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=MARKER_RADIUS, segments=12, ring_count=8)
    sphere = bpy.context.object
    sphere.name = MARKER_NAME
    sphere.data.name = MARKER_NAME
    bpy.data.meshes.remove(marker)

    sphere.parent = obj
    obj.instance_type = "VERTS"
    obj.show_instancer_for_viewport = True
    obj.color = (1.0, 0.45, 0.1, 1.0)
    sphere.color = (1.0, 0.45, 0.1, 1.0)
    return obj


# The flags batch_retarget.import_bvh_action uses; the overlay is only honest if the BVH arrives
# in the same place the pipeline puts it.
BVH_IMPORT = dict(global_scale=0.01, rotate_mode="NATIVE", axis_forward="Z", axis_up="Y",
                  use_fps_scale=False, update_scene_fps=False, update_scene_duration=False)


def import_bvh(path):
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=path, **BVH_IMPORT)
    arm = [o for o in bpy.data.objects if o not in before][0]
    arm.color = (0.25, 0.6, 1.0, 1.0)
    arm.show_in_front = True
    arm.data.display_type = "OCTAHEDRAL"
    # Roll is the part of the solve positions cannot pin down, so the bone axes are what anyone
    # correcting this by hand needs to see.
    arm.data.show_axes = True
    return arm


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, nargs="+", help="The .npz the clip came from.")
    parser.add_argument("--bvh", required=True, help="The converted BVH to overlay.")
    parser.add_argument("--split", type=int, default=0,
                        help="Which --npz the clip belongs to, by position.")
    parser.add_argument("--index", type=int, default=0, help="Clip index within that split.")
    parser.add_argument("--scale", type=float, default=converter.DEFAULT_SCALE_CM)
    parser.add_argument("--out", required=True, help="Blend to write.")
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def main():
    args = parse_args(sys.argv)
    loaded = [converter.load_clips(converter.resolve(p)) for p in args.npz]
    sample = np.concatenate([c[:, ::10, :63].reshape(-1, 21, 3) for c in loaded])
    orientation = converter.detect_orientation(sample)

    clips = loaded[args.split]
    if not 0 <= args.index < len(clips):
        raise SystemExit("Clip {} is outside the split's {} clips.".format(args.index, len(clips)))
    positions = converter.prepare(clips[args.index:args.index + 1], orientation, args.scale,
                                  False)[0]
    log("clip {} of split {}: {} frames".format(args.index, args.split, len(positions)))

    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = converter.OUTPUT_FPS, 1.0
    scene.frame_start, scene.frame_end = 1, len(positions)

    arm = import_bvh(converter.resolve(args.bvh))
    points = build_points(to_blender(positions))
    log("armature {} ({} bones), points {} ({} joints)".format(
        arm.name, len(arm.data.bones), points.name, len(points.data.vertices)))

    out = converter.resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    log("wrote " + out)
    log("done")


if __name__ == "__main__":
    main()
