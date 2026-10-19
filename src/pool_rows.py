# Created: 2026-10-06
# Last modified: 2026-10-06
"""Read rows [start, stop) of one array in a pool .npz without loading the rest.

np.load(pool)['measurements'] reads the whole array, even for a slice or just its shape, so
every script that touched a FORMAL pool also read the sealed rows [17M, 19M) into memory.
These functions read only the requested rows:

  n_rows(path, key)               row count, from the array header alone
  rows(path, key, start, stop)    rows [start, stop) as a new array
  scalar(path, key)               a 0-d array (e.g. n_det of a compact pool)

Pools written with np.savez are uncompressed (ZIP_STORED): the reader finds the array's
byte offset in the file and seeks straight to row `start`, so no byte of any other row is
read. A compressed member (np.savez_compressed) is read as a stream from its start up to row
`stop` and never beyond it.

trace: optional list; every read appends (key, first_byte, last_byte_exclusive) in the
array's own data coordinates (byte 0 = row 0). With the environment variable POOL_ROWS_TRACE
set to a file path, every rows() call also appends "<pool> <key> <start> <stop>" to that
file, so a whole training or scoring run can be checked (tests/pool_rows_test.py).
"""
import os
import struct
import zipfile

import numpy as np

_MEMBER_HEADER = struct.Struct('<4s2B4HL2L2H')     # zip local file header (30 bytes)


def _open_member(path, key):
    """(file object positioned at the .npy start, is_stored, zip handle or None)."""
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(key + '.npy')
    if info.compress_type == zipfile.ZIP_STORED:
        fh = open(path, 'rb')
        fh.seek(info.header_offset)
        h = _MEMBER_HEADER.unpack(fh.read(_MEMBER_HEADER.size))
        name_len, extra_len = h[10], h[11]
        fh.seek(info.header_offset + _MEMBER_HEADER.size + name_len + extra_len)
        zf.close()
        return fh, True, None
    return zf.open(info), False, zf


def _header(fh):
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    if fortran:
        raise ValueError('Fortran-ordered arrays are not supported')
    return shape, dtype


def n_rows(path, key):
    fh, _, zf = _open_member(path, key)
    try:
        shape, _ = _header(fh)
    finally:
        fh.close()
        if zf is not None:
            zf.close()
    return int(shape[0])


def rows(path, key, start, stop, trace=None):
    fh, stored, zf = _open_member(path, key)
    try:
        shape, dtype = _header(fh)
        start, stop = int(start), int(stop)
        if not 0 <= start <= stop <= shape[0]:
            raise ValueError(f'{key}: rows [{start}, {stop}) outside [0, {shape[0]})')
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        first, last = start * row_bytes, stop * row_bytes
        if stored:
            fh.seek(fh.tell() + first)
        else:
            skip = first
            while skip:                      # stream forward; nothing past `stop` is read
                skip -= len(fh.read(min(skip, 1 << 24)))
        buf = fh.read(last - first)
        if len(buf) != last - first:
            raise IOError(f'{key}: short read ({len(buf)} of {last - first} bytes)')
        if trace is not None:
            trace.append((key, first, last))
        if os.environ.get('POOL_ROWS_TRACE'):
            with open(os.environ['POOL_ROWS_TRACE'], 'a') as t:
                t.write(f'{os.path.abspath(path)} {key} {start} {stop}\n')
        return np.frombuffer(buf, dtype=dtype).reshape((stop - start,) + tuple(shape[1:])).copy()
    finally:
        fh.close()
        if zf is not None:
            zf.close()


def scalar(path, key):
    fh, _, zf = _open_member(path, key)
    try:
        shape, dtype = _header(fh)
        n = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        return np.frombuffer(fh.read(n), dtype=dtype).reshape(shape).copy()
    finally:
        fh.close()
        if zf is not None:
            zf.close()


def keys(path):
    with zipfile.ZipFile(path) as zf:
        return [n[:-4] for n in zf.namelist() if n.endswith('.npy')]
