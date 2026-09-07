import numpy as np

from Skeleton import Skeleton


class PoseSet:
    """
    A pose database as the Python side sees it: the frames of every animation clip,
    concatenated, sharing one skeleton and one frame rate.

    The Python mirror of the C# ``PoseSet``, read from ``.mmpose`` by
    ``pose_set_importer.deserialize_pose_set``.

    Frames are indexed globally across the database, so callers must not assume frame
    ``i + 1`` continues frame ``i`` -- across a clip boundary it does not. ``clips`` is
    what maps global indices back to individual animations.

    :param skeleton: the rig every frame is posed on
    :param frame_time: seconds per frame; the velocities below are per-second rates,
        so converting one to a single frame's delta means multiplying by this
    :param local_pos: (n_frames, n_bones, 3) parent-local joint positions -- except bone
        0, the rig's root, which carries the clip's own world position
    :param local_quats: (n_frames, n_bones, 4) parent-local joint rotations, xyzw, bone 0
        again in world space
    :param foot_contacts: (n_frames, 2) left/right foot contact flags
    :param local_vel: (n_frames, n_bones, 3) linear velocities, m/s, per-channel finite
        differences of the two above
    :param local_angular_vel: (n_frames, n_bones, 3) angular velocities as rotation
        vectors, rad/s
    :param clips: per-clip frame ranges, as dicts with ``start`` and ``end``
    :param phase: (n_frames,) gait phase in radians in [0, 2*pi), or None for a database
        that carries none
    :param phase_rate: (n_frames,) its rate in rad/s. **Zero marks a frame with no
        measurable gait cycle**, which is the sentinel training uses to drop it

    Phase is evaluated in Unity from each clip's authored footfalls and read off the file
    rather than reconstructed here, so that what the clip editor draws is what a model
    trains on. Two implementations of one rule is how the two halves drift apart.

    The character frame these poses are matched in is not stored. Its definition is
    structural -- joint 0, faced along the forward axis of its rest rotation -- and
    ``pose_set_importer`` attaches it (``sim_frame_bone_index``, ``sim_frame_forward``)
    for :mod:`simulation_frame` to derive the per-pose frames from.
    """

    def __init__(self,
                 skeleton: Skeleton,
                 frame_time: float,
                 local_pos,
                 local_quats,
                 foot_contacts,
                 local_vel,
                 local_angular_vel,
                 clips,
                 phase=None,
                 phase_rate=None,
                 ):
        self.skeleton: Skeleton = skeleton
        self.frameTime: float = frame_time
        self.local_positions: np.ndarray = local_pos
        self.local_rotations: np.ndarray = local_quats
        self.local_velocities: np.ndarray = local_vel
        self.local_angular_velocities: np.ndarray = local_angular_vel
        self.foot_contacts: np.ndarray = foot_contacts
        self.clips: list = clips

        n_frames = len(local_pos)
        self.phase: np.ndarray = np.zeros(n_frames, dtype=np.float32) if phase is None \
            else np.asarray(phase, dtype=np.float32)
        self.phase_rate: np.ndarray = np.zeros(n_frames, dtype=np.float32) if phase_rate is None \
            else np.asarray(phase_rate, dtype=np.float32)
