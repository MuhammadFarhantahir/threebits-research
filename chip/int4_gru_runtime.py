"""Zero-PyTorch pure-NumPy + ctypes runtime helpers for Threebits C99 int4 GRU (`chip/int4_gru.c`)."""
import ctypes
import math
from pathlib import Path

import numpy as np


class Int4GRUModelC(ctypes.Structure):
    _fields_ = [
        ("input_size", ctypes.c_int32),
        ("hidden_size", ctypes.c_int32),
        ("output_size", ctypes.c_int32),
        ("w_ih_packed", ctypes.POINTER(ctypes.c_uint8)),
        ("w_hh_packed", ctypes.POINTER(ctypes.c_uint8)),
        ("w_out_packed", ctypes.POINTER(ctypes.c_uint8)),
        ("b_ih_q12", ctypes.POINTER(ctypes.c_int32)),
        ("b_hh_q12", ctypes.POINTER(ctypes.c_int32)),
        ("b_out_q12", ctypes.POINTER(ctypes.c_int32)),
        ("ih_mult", ctypes.c_int32),
        ("ih_shift", ctypes.c_int32),
        ("hh_mult", ctypes.c_int32),
        ("hh_shift", ctypes.c_int32),
        ("out_mult", ctypes.c_int32),
        ("out_shift", ctypes.c_int32),
        ("sigmoid_lut", ctypes.POINTER(ctypes.c_int16)),
        ("tanh_lut", ctypes.POINTER(ctypes.c_int16)),
    ]


def build_luts():
    xs = np.linspace(-8.0, 8.0, 257, dtype=np.float64)
    sig = np.round((1.0 / (1.0 + np.exp(-xs))) * 32767.0).astype(np.int16)
    th = np.round(np.tanh(xs) * 32767.0).astype(np.int16)
    return np.ascontiguousarray(sig), np.ascontiguousarray(th)


def float_to_mult_shift(real_mult: float):
    """Convert positive float multiplier to (int32 mult in [2^30, 2^31-1], shift >= 0)."""
    if real_mult <= 0.0:
        return 0, 0
    m, exp = math.frexp(real_mult)
    q_mult = int(round(m * (1 << 31)))
    if q_mult == (1 << 31):
        q_mult //= 2
        exp += 1
    shift = 31 - exp
    assert shift >= 0, f"Unexpected large multiplier {real_mult}"
    return int(q_mult), int(shift)


def pack_int4_matrix(w_int: np.ndarray) -> np.ndarray:
    """Pack 2D signed int4 array [-8, +7] (rows, cols) with even cols into uint8 (rows, cols//2)."""
    rows, cols = w_int.shape
    assert cols % 2 == 0
    lo = (w_int[:, 0::2] & 0x0F).astype(np.uint8)
    hi = ((w_int[:, 1::2] & 0x0F) << 4).astype(np.uint8)
    return np.ascontiguousarray(lo | hi)


def find_int4_gru_library(search_dirs):
    """Locate compiled libint4_gru.{so,dylib} across candidate directories."""
    for d in search_dirs:
        p = Path(d)
        for name in ("libint4_gru.so", "libint4_gru.dylib"):
            cand = p / name
            if cand.exists():
                return cand
    return None
