"""
Reading a ``.mmfeatures`` file.

The fixtures write the bytes by hand rather than going through Unity, which is the point:
these tests pin the wire format the C# writer and this reader have to agree on, and a
change to either that the other does not follow shows up here.
"""

import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formats.feature_set_importer import StaleFeatureFileError, read_feature_set  # noqa: E402

# (name, floats per sample, samples), trajectory features first.
SCHEMA = [('FuturePosition', 2, 3), ('FutureDirection', 2, 3),
          ('LeftFootPosition', 3, 1), ('HipsVelocity', 3, 1)]
N_TRAJECTORY = 2
FEATURE_SIZE = 2 * 3 + 2 * 3 + 3 + 3


def _csharp_string(value: str) -> bytes:
    """A string the way ``BinaryWriter.Write(string)`` writes it: LEB128 length, then UTF-8."""
    encoded = value.encode('utf-8')
    length = len(encoded)
    prefix = bytearray()
    while True:
        byte = length & 0x7F
        length >>= 7
        prefix.append(byte | (0x80 if length else 0x00))
        if not length:
            break
    return bytes(prefix) + encoded


def write_features(directory: str, name: str, features: np.ndarray, valid: np.ndarray,
                   mean: np.ndarray, std: np.ndarray, schema=None, n_trajectory=N_TRAJECTORY,
                   declared_size=None) -> str:
    """
    Write a ``.mmfeatures`` file. ``declared_size`` overrides the header's vector width, so a
    test can produce a header that disagrees with its own schema block.
    """
    schema = SCHEMA if schema is None else schema
    path = os.path.join(directory, f'{name}.mmfeatures')

    with open(path, 'wb') as f:
        f.write(struct.pack('<IIII', features.shape[0],
                            features.shape[1] if declared_size is None else declared_size,
                            n_trajectory, len(schema) - n_trajectory))
        for feature_name, width, count in schema:
            f.write(_csharp_string(feature_name))
            f.write(struct.pack('<II', width, count))

        statistics = np.empty(mean.size * 2, dtype=np.float32)
        statistics[0::2] = mean
        statistics[1::2] = std
        f.write(statistics.tobytes())

        for row, is_valid in zip(features, valid):
            f.write(struct.pack('<I', 1 if is_valid else 0))
            f.write(np.asarray(row, dtype=np.float32).tobytes())

    return path


def sample_features(n_frames: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distinct values everywhere, so a misaligned read cannot pass by coincidence."""
    features = np.arange(n_frames * FEATURE_SIZE, dtype=np.float32).reshape(n_frames, FEATURE_SIZE)
    valid = np.ones(n_frames, dtype=bool)
    valid[-1] = False
    mean = np.linspace(-1.0, 1.0, FEATURE_SIZE).astype(np.float32)
    std = np.linspace(0.5, 2.0, FEATURE_SIZE).astype(np.float32)
    return features, valid, mean, std


class ReadTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name
        self.features, self.valid, self.mean, self.std = sample_features()
        write_features(self.directory, 'db', self.features, self.valid, self.mean, self.std)

    def tearDown(self):
        self._directory.cleanup()

    def test_vectors_and_statistics_come_back_unchanged(self):
        feature_set = read_feature_set(self.directory, 'db')

        np.testing.assert_allclose(feature_set.features, self.features)
        np.testing.assert_array_equal(feature_set.valid, self.valid)
        np.testing.assert_allclose(feature_set.mean, self.mean)
        np.testing.assert_allclose(feature_set.std, self.std)

    def test_the_schema_carries_offsets_and_the_trajectory_split(self):
        feature_set = read_feature_set(self.directory, 'db')

        self.assertEqual([entry.name for entry in feature_set.schema],
                         [name for name, _, _ in SCHEMA])
        self.assertEqual([entry.offset for entry in feature_set.schema], [0, 6, 12, 15])
        self.assertEqual([entry.is_trajectory for entry in feature_set.schema],
                         [True, True, False, False])

    def test_the_pose_block_starts_where_the_trajectory_block_ends(self):
        feature_set = read_feature_set(self.directory, 'db')

        self.assertEqual(feature_set.pose_offset, 12)

    def test_denormalizing_undoes_what_unity_applied(self):
        feature_set = read_feature_set(self.directory, 'db')

        np.testing.assert_allclose(feature_set.denormalized(),
                                   self.features * self.std + self.mean, rtol=1e-5)

    def test_a_feature_can_be_found_by_its_authored_name(self):
        feature_set = read_feature_set(self.directory, 'db')

        self.assertEqual(feature_set.find('FutureDirection').offset, 6)
        with self.assertRaises(KeyError):
            feature_set.find('NotAFeature')

    def test_a_missing_file_is_reported_as_missing(self):
        with self.assertRaises(FileNotFoundError):
            read_feature_set(self.directory, 'absent')


class StalenessTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name

    def tearDown(self):
        self._directory.cleanup()

    def test_a_header_that_disagrees_with_its_own_schema_is_refused(self):
        features, valid, mean, std = sample_features()
        write_features(self.directory, 'db', features, valid, mean, std,
                       declared_size=FEATURE_SIZE + 1)

        with self.assertRaises(StaleFeatureFileError):
            read_feature_set(self.directory, 'db')

    def test_a_truncated_file_is_refused_rather_than_read_short(self):
        features, valid, mean, std = sample_features(n_frames=8)
        path = write_features(self.directory, 'db', features, valid, mean, std)

        with open(path, 'rb') as f:
            data = f.read()
        with open(path, 'wb') as f:
            f.write(data[:len(data) // 2])

        with self.assertRaises(StaleFeatureFileError):
            read_feature_set(self.directory, 'db')


if __name__ == '__main__':
    unittest.main()
