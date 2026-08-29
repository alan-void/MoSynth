"""
Reads the ``.mmfeatures`` file Unity writes beside a pose database.

The Python mirror of C# ``FeatureSerializer``. A feature vector is what motion matching
searches against, and it is also the query vector a learned motion matching projector is
trained to map into -- so a training set built from this database needs the same floats
the search sees, in the same order, with the same normalisation applied.

The file describes itself: its header carries the trajectory/pose split and a schema block
naming every feature and its width. Nothing here needs the Unity asset, and a file written
for a different feature configuration is refused by :func:`read_feature_set` rather than
read off the wrong offsets -- the same contract the C# reader enforces, since neither side
has a version byte to fall back on.

Conventions match the rest of this package: little-endian float32, no padding, strings
LEB128-length-prefixed the way ``System.IO.BinaryWriter`` writes them.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

from binary_reading import read_csharp_string


@dataclass(frozen=True)
class FeatureSchema:
    """
    One feature's account of itself.

    :param name: the authored name, which is how a control input addresses the feature.
    :param floats_per_prediction: width of one sample, 1 to 3 after axis masking.
    :param prediction_count: how many samples the feature stores; 1 for a pose feature.
    :param is_trajectory: trajectory features come first in the vector, pose features after.
    :param offset: float offset of the feature within a vector.
    """

    name: str
    floats_per_prediction: int
    prediction_count: int
    is_trajectory: bool
    offset: int

    @property
    def float_count(self) -> int:
        return self.floats_per_prediction * self.prediction_count


@dataclass
class FeatureSet:
    """
    Every frame's matching feature vector, plus what is needed to interpret one.

    :param features: (n_frames, feature_size) float32, **normalised** -- Unity subtracts
        the mean and divides by the standard deviation before writing.
    :param valid: (n_frames,) bool. False where the frame's sampling window left its own
        clip, at either end. Those rows are all zero and were excluded from the statistics,
        so they must be excluded from training too.
    :param mean: (feature_size,) float32 normalisation mean.
    :param std: (feature_size,) float32 normalisation standard deviation. Shared across the
        floats of one feature rather than per float, which is what keeps a feature's axes
        commensurable with each other.
    :param schema: one :class:`FeatureSchema` per feature, in vector order.
    """

    features: np.ndarray
    valid: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    schema: list[FeatureSchema]

    @property
    def n_frames(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_size(self) -> int:
        return int(self.features.shape[1])

    @property
    def pose_offset(self) -> int:
        """Float offset where the pose block starts, i.e. the end of the trajectory block."""
        for entry in self.schema:
            if not entry.is_trajectory:
                return entry.offset
        return self.feature_size

    def denormalized(self) -> np.ndarray:
        """The feature vectors back in their original units, for inspection and plotting."""
        return self.features * self.std + self.mean

    def find(self, name: str) -> FeatureSchema:
        """The schema entry for an authored feature name."""
        for entry in self.schema:
            if entry.name == name:
                return entry
        raise KeyError(f"No feature named {name!r}; have {[e.name for e in self.schema]}")


class StaleFeatureFileError(Exception):
    """The file on disk does not describe the feature set that was asked for."""


def _read_exact(f, n_bytes: int, what: str) -> bytes:
    """
    Read exactly ``n_bytes``, or refuse the file.

    Every block below is sized from the header, so a short read is never a partial record
    to be salvaged -- it means the header describes a file that is not this one.
    """
    data = f.read(n_bytes)
    if len(data) != n_bytes:
        raise StaleFeatureFileError(
            f"Ran out of bytes reading {what}: wanted {n_bytes}, got {len(data)}. "
            "Regenerate the databases.")
    return data


def _read_schema(f, n_trajectory: int, n_pose: int) -> list[FeatureSchema]:
    schema = []
    offset = 0
    for index in range(n_trajectory + n_pose):
        name = read_csharp_string(f)
        floats_per_prediction, prediction_count = struct.unpack('<II', f.read(8))
        schema.append(FeatureSchema(name=name,
                                    floats_per_prediction=int(floats_per_prediction),
                                    prediction_count=int(prediction_count),
                                    is_trajectory=index < n_trajectory,
                                    offset=offset))
        offset += int(floats_per_prediction) * int(prediction_count)
    return schema


def read_feature_set(path: str, file_name: str) -> FeatureSet:
    """
    Read ``<path>/<file_name>.mmfeatures``.

    :raises FileNotFoundError: no such file.
    :raises StaleFeatureFileError: the header and the schema block disagree about the vector
        width, or a block sized from the header runs off the end. Both mean the file was not
        written for the database being read and has to be regenerated from Unity -- there is
        no version byte that would have said so sooner.
    """
    features_path = os.path.join(path, f"{file_name}.mmfeatures")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Could not find {file_name}.mmfeatures at {path}")

    with open(features_path, 'rb') as f:
        n_vectors, feature_size, n_trajectory, n_pose = struct.unpack('<IIII', f.read(16))
        schema = _read_schema(f, int(n_trajectory), int(n_pose))

        described = sum(entry.float_count for entry in schema)
        if described != feature_size:
            raise StaleFeatureFileError(
                f"{file_name}.mmfeatures says its vectors are {feature_size} floats but its "
                f"{len(schema)} features describe {described}. Regenerate the databases.")

        statistics = np.frombuffer(
            _read_exact(f, int(feature_size) * 8, f"{file_name}.mmfeatures normalisation statistics"),
            dtype=np.float32)
        mean = np.ascontiguousarray(statistics[0::2])
        std = np.ascontiguousarray(statistics[1::2])

        # One uint validity flag then feature_size floats, per vector: a single read and a
        # reshape, since both are four bytes wide.
        stride = int(feature_size) + 1
        raw = np.frombuffer(
            _read_exact(f, int(n_vectors) * stride * 4, f"the {n_vectors} vectors of {file_name}.mmfeatures"),
            dtype=np.uint32)

        rows = raw.reshape(int(n_vectors), stride)
        valid = rows[:, 0] != 0
        features = rows[:, 1:].copy().view(np.float32)

    return FeatureSet(features=features, valid=valid, mean=mean, std=std, schema=schema)
