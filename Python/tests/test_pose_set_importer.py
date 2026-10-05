"""
Reading a ``.mmpose``.

The fixtures here write the file the way ``PoseSerializer.Serialize`` does -- the layout written
up in ``openwiki/animation-tools/on-disk-formats.md`` -- so that the Python reader is checked
against the format rather than against itself. The gait phase block matters most: a disagreement
about where it sits would surface as a silently wrong phase rather than as an error.
"""

import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formats.pose_set_importer import deserialize_pose_set  # noqa: E402

FRAME_TIME = 1.0 / 30.0


def _csharp_string(text: str) -> bytes:
    """``BinaryWriter.Write(string)``: a 7-bit encoded length, then UTF-8."""
    payload = text.encode('utf-8')
    length, prefix = len(payload), bytearray()
    while length >= 0x80:
        prefix.append((length & 0x7F) | 0x80)
        length >>= 7
    prefix.append(length)
    return bytes(prefix) + payload


def write_mmpose(path: str, n_poses: int, bone_names, parents, phase, phase_rate,
                 contacts=None, contact_bone_names=('foot_l', 'foot_r')) -> None:
    """
    A minimal database in the on-disk layout, with distinct values everywhere.

    :param contacts: per pose, one flag per entry of ``contact_bone_names``; all clear if None.
    """
    n_bones = len(bone_names)
    n_contacts = len(contact_bone_names)
    out = bytearray()

    out += struct.pack('<I', n_bones)
    for bone in range(n_bones):
        out += _csharp_string(bone_names[bone])
        out += struct.pack('<i', parents[bone])
        out += struct.pack('<3f', 0.0, float(bone), 0.0)
        out += struct.pack('<4f', 0.0, 0.0, 0.0, 1.0)

    out += struct.pack('<I', n_contacts)
    for name in contact_bone_names:
        out += _csharp_string(name)

    out += struct.pack('<I', 1)
    out += struct.pack('<IIf', 0, n_poses, FRAME_TIME)

    out += struct.pack('<III', n_poses, n_bones, 0)
    for pose in range(n_poses):
        for block, width in (('pos', 3), ('rot', 4), ('vel', 3), ('ang', 3)):
            for bone in range(n_bones):
                seed = pose * 10 + bone
                values = [float(seed + i) for i in range(width)]
                if block == 'rot':
                    values = [0.0, 0.0, 0.0, 1.0]
                out += struct.pack(f'<{width}f', *values)
        flags = [0] * n_contacts if contacts is None else contacts[pose]
        out += struct.pack(f'<{n_contacts}I', *(int(flag) for flag in flags))

    for pose in range(n_poses):
        out += struct.pack('<2f', float(phase[pose]), float(phase_rate[pose]))

    with open(path, 'wb') as handle:
        handle.write(out)


class PhaseBlockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.n_poses = 6
        self.phase = np.linspace(0.0, 5.0, self.n_poses).astype(np.float32)
        self.phase_rate = np.arange(self.n_poses, dtype=np.float32) * 0.25
        write_mmpose(os.path.join(self.directory, 'db.mmpose'), self.n_poses,
                     ['root', 'spine', 'foot'], [-1, 0, 1], self.phase, self.phase_rate,
                     contacts=[(p % 2, (p + 1) % 2) for p in range(self.n_poses)])

    def test_phase_reads_back_exactly(self):
        pose_set = deserialize_pose_set(self.directory, 'db')

        np.testing.assert_allclose(pose_set.phase, self.phase)
        np.testing.assert_allclose(pose_set.phase_rate, self.phase_rate)

    def test_the_block_sits_after_the_contacts_and_not_before(self):
        # Reading it a record too early would take a contact flag's bits as a float, so the
        # contacts and the phase have to both come back right for the offset to be correct.
        pose_set = deserialize_pose_set(self.directory, 'db')

        expected = np.array([[p % 2, (p + 1) % 2] for p in range(self.n_poses)], dtype=bool)
        np.testing.assert_array_equal(pose_set.foot_contacts, expected)
        np.testing.assert_allclose(pose_set.phase, self.phase)
        self.assertEqual(pose_set.contact_bone_names, ['foot_l', 'foot_r'])

    def test_a_file_that_stops_before_the_phase_block_is_refused(self):
        # The format carries no version, so a file missing the block has to be caught by its
        # content being absent rather than by a header saying so.
        path = os.path.join(self.directory, 'stale.mmpose')
        write_mmpose(path, self.n_poses, ['root', 'spine', 'foot'], [-1, 0, 1],
                     self.phase, self.phase_rate)
        with open(path, 'rb') as handle:
            whole = handle.read()
        with open(path, 'wb') as handle:
            handle.write(whole[:-self.n_poses * 8])

        with self.assertRaises(ValueError) as raised:
            deserialize_pose_set(self.directory, 'stale')
        self.assertIn('gait phase', str(raised.exception))


class ContactBlockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.n_poses = 5
        self.phase = np.linspace(0.0, 4.0, self.n_poses).astype(np.float32)
        self.phase_rate = np.full(self.n_poses, 0.5, dtype=np.float32)

    def test_three_contact_bones_read_back_by_slot(self):
        names = ('toe_l', 'toe_r', 'hand_l')
        contacts = [(p % 2, (p + 1) % 2, int(p >= 3)) for p in range(self.n_poses)]
        write_mmpose(os.path.join(self.directory, 'db.mmpose'), self.n_poses,
                     ['root', 'spine', 'foot'], [-1, 0, 1], self.phase, self.phase_rate,
                     contacts=contacts, contact_bone_names=names)

        pose_set = deserialize_pose_set(self.directory, 'db')

        self.assertEqual(pose_set.contact_bone_names, list(names))
        np.testing.assert_array_equal(pose_set.foot_contacts, np.array(contacts, dtype=bool))
        # The phase block sits after three flags per pose rather than two.
        np.testing.assert_allclose(pose_set.phase, self.phase)
        np.testing.assert_allclose(pose_set.phase_rate, self.phase_rate)

    def test_no_contact_bones_means_no_contact_columns(self):
        write_mmpose(os.path.join(self.directory, 'db.mmpose'), self.n_poses,
                     ['root', 'spine', 'foot'], [-1, 0, 1], self.phase, self.phase_rate,
                     contact_bone_names=())

        pose_set = deserialize_pose_set(self.directory, 'db')

        self.assertEqual(pose_set.contact_bone_names, [])
        self.assertEqual(pose_set.foot_contacts.shape, (self.n_poses, 0))
        np.testing.assert_allclose(pose_set.phase, self.phase)


if __name__ == '__main__':
    unittest.main()
