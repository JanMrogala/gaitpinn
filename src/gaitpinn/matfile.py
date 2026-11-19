"""Read the MATLAB `table` objects in the Camargo .mat files with scipy only.

scipy cannot decode MATLAB objects (MCOS); it leaves them as opaque bytes in
`__function_workspace__`. That blob is itself a MAT v5 stream. We walk its elements, collect
every cell array, and pick out the table's variable names (a cell of strings starting with
'Header') and its columns (a cell of equal-length arrays or string cells). The result is
checked strictly: if the structure is not exactly what we expect, it raises an error rather
than guessing.
"""

import struct
import zlib
from pathlib import Path

import numpy as np
import scipy.io

_DTYPES = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    9: "f8",
    12: "i8",
    13: "u8",
    16: "u1",
    17: "u2",
    18: "u4",
}
_MI_MATRIX, _MI_COMPRESSED = 14, 15
_CELL, _STRUCT, _CHAR, _OBJECT, _OPAQUE = 1, 2, 4, 3, 17


def _tag(b: bytes, o: int) -> tuple[int, int, int, int]:
    """Return (type, nbytes, data_offset, next_offset) of the element at o."""
    t, n = struct.unpack_from("<II", b, o)
    if t >> 16:  # small data element: type and size packed in 4 bytes
        return t & 0xFFFF, t >> 16, o + 4, o + 8
    return t, n, o + 8, o + 8 + ((n + 7) // 8) * 8


def _elements(b: bytes, o: int, end: int):
    while o < end:
        t, n, d, nxt = _tag(b, o)
        yield t, n, d
        o = nxt


def _parse(b: bytes, o: int, end: int) -> list:
    out = []
    for t, n, d in _elements(b, o, end):
        if t == _MI_COMPRESSED:
            z = zlib.decompress(b[d : d + n])
            out += _parse(z, 0, len(z))
            continue
        if t != _MI_MATRIX:
            continue
        sub = list(_elements(b, d, d + n))
        if not sub:
            out.append(None)
            continue
        cls = b[sub[0][2]]
        if cls == _CELL:
            # flags, dims, name, then one matrix element per cell
            out.append([_one(b, d2 - 8, d2 + n2) for _, n2, d2 in sub[3:]])
        elif cls == _STRUCT:
            # flags, dims, name, field-name length, field names, then the field values
            out.append([_one(b, d2 - 8, d2 + n2) for _, n2, d2 in sub[5:]])
        elif cls in (_OPAQUE, _OBJECT):
            vals = []
            for _, n2, d2 in sub[3:]:
                vals += _parse(b, d2 - 8, d2 + n2)
            out.append(vals)
        elif cls == _CHAR:
            t2, n2, d2 = sub[3]
            out.append("".join(map(chr, np.frombuffer(b[d2 : d2 + n2], _DTYPES[t2]))))
        else:
            t2, n2, d2 = sub[3]
            out.append(np.frombuffer(b[d2 : d2 + n2], _DTYPES.get(t2, "u1")).copy())
    return out


def _one(b: bytes, o: int, end: int):
    r = _parse(b, o, end)
    return r[0] if r else None


def _lists(x):
    """Yield every list in the parsed tree."""
    if isinstance(x, list):
        yield x
        for y in x:
            yield from _lists(y)


def _is_column(c) -> bool:
    return isinstance(c, np.ndarray) or (isinstance(c, list) and all(isinstance(s, str) for s in c))


def load_tables(path: str | Path, header: str = "Header") -> list[dict]:
    """Return every MATLAB table in a .mat file as {column name: values}.

    A table is recognised by its variable-name list, which must start with `header`. Each
    name list is paired, in file order, with the next cell of equal-length columns that has
    the same number of entries.
    """
    raw = scipy.io.loadmat(path)["__function_workspace__"].tobytes()
    tree = _parse(raw, 8, len(raw))  # skip the 8-byte subsystem header
    lists = list(_lists(tree))
    names = [
        x
        for x in lists
        if x and all(isinstance(s, str) for s in x) and x[0] == header and len(set(x)) == len(x)
    ]
    cols = [
        x for x in lists if x and all(_is_column(c) for c in x) and len({len(c) for c in x}) == 1
    ]
    tables, j = [], 0
    for nm in names:
        while j < len(cols) and len(cols[j]) != len(nm):
            j += 1
        if j == len(cols):
            raise ValueError(f"{path}: no columns for table {nm[:3]}")
        tables.append(dict(zip(nm, cols[j], strict=True)))
        j += 1
    if not tables:
        raise ValueError(f"{path}: no table with first column {header!r}")
    for tb in tables:
        for k, v in tb.items():
            is_text = isinstance(v, list)
            if k == "Label" and not is_text:
                raise ValueError(f"{path}: column Label is not text")
        if header == "Header":
            h = np.asarray(tb["Header"], dtype=float)
            if h.ndim != 1 or len(h) < 2 or not np.all(np.diff(h) > 0):
                raise ValueError(f"{path}: Header is not an increasing time vector")
    return tables


def load_table(path: str | Path, header: str = "Header") -> dict:
    """The single MATLAB table in a .mat file; raises if there is not exactly one."""
    tables = load_tables(path, header)
    if len(tables) != 1:
        raise ValueError(f"{path}: expected one table, found {len(tables)}")
    return tables[0]


def load_struct(path: str | Path) -> dict:
    """Plain (non-object) variables of a .mat file, e.g. the conditions files."""
    m = scipy.io.loadmat(path, squeeze_me=True)
    return {k: v for k, v in m.items() if not k.startswith("__")}
