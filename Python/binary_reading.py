"""
Primitives for reading what ``System.IO.BinaryWriter`` writes.

The Python counterpart of C# ``BinarySerializerExtensions``. Every MoSynth binary format --
``.mmpose``, ``.mmfeatures`` -- is components in order, little-endian, no padding and no
length prefixes, with one exception: strings, which C# prefixes with a LEB128 byte count.
That exception is the whole reason this module exists.
"""

from __future__ import annotations


def read_csharp_string(f) -> str:
    """
    Read one string written by ``BinaryWriter.Write(string)``.

    The length is a 7-bit encoded (LEB128) byte count, low group first, with the high bit
    of each byte set while more groups follow. An empty read is treated as an empty string
    so a truncated file surfaces as a short result rather than an exception here.
    """
    count = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            return ""
        b = b[0]
        count |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break

    if count == 0:
        return ""
    return f.read(count).decode('utf-8')
