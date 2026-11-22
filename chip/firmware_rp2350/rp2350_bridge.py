"""Host-side USB-CDC / POSIX PTY serial bridge for Threebits RP2350 pure-integer C99 firmware.

Speaks the CRC-16/CCITT-FALSE line protocol defined in `chip/firmware_rp2350/rp2350_protocol.h`.
Works with BOTH:
  - Physical Raspberry Pi Pico 2 (RP2350) over USB-CDC (`/dev/cu.usbmodem*`, `/dev/ttyACM0`)
  - Host emulator (`rp2350_host_emu --stdio`) attached to a POSIX PTY (`/dev/ttys*` / `/dev/pts/*`)

Uses `pyserial` when installed for real UART/USB ports, and falls back cleanly to zero-dependency
POSIX raw `termios` + `select` I/O on macOS/Linux so PTY loopback tests need zero extra packages.
"""
import os
import select
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) matching rp2350_protocol.h."""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def verify_crc_line(line: str) -> str:
    """Verify trailing ` crc=xxxx` and return the verified payload prefix."""
    s = line.strip()
    marker = " crc="
    pos = s.rfind(marker)
    if pos < 0:
        raise ValueError(f"Missing CRC field in MCU reply: {s!r}")
    payload = s[:pos]
    got_hex = s[pos + len(marker):].strip()
    expected = f"{crc16_ccitt(payload.encode('ascii')):04x}"
    if got_hex.lower() != expected:
        raise ValueError(f"CRC mismatch: got {got_hex!r}, expected {expected!r} for {payload!r}")
    return payload


def _parse_kv(payload: str) -> Tuple[str, Dict[str, str]]:
    parts = payload.split()
    if not parts:
        raise ValueError("Empty MCU reply")
    cmd = parts[0]
    kv = {}
    for tok in parts[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return cmd, kv


class _PosixRawPort:
    """Zero-dependency raw POSIX serial/PTY transport."""

    def __init__(self, path: str, timeout: float = 1.5):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self.timeout = timeout
        self._buf = bytearray()
        if termios is not None and tty is not None:
            try:
                attrs = termios.tcgetattr(self.fd)
                attrs[0] &= ~(termios.ICRNL | termios.IXON)
                attrs[1] &= ~termios.OPOST
                attrs[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
                termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            except termios.error:
                pass

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        total = 0
        deadline = time.monotonic() + self.timeout
        while total < len(data):
            _, w, _ = select.select([], [self.fd], [], max(0.01, deadline - time.monotonic()))
            if not w:
                raise TimeoutError("Timed out writing to RP2350 serial port")
            n = os.write(self.fd, view[total:])
            if n <= 0:
                raise OSError("Short write to RP2350 serial port")
            total += n

    def readline(self) -> str:
        deadline = time.monotonic() + self.timeout
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line = bytes(self._buf[:nl + 1])
                del self._buf[:nl + 1]
                return line.decode("ascii", errors="replace")
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError("Timed out reading line from RP2350 serial port")
            r, _, _ = select.select([self.fd], [], [], rem)
            if not r:
                raise TimeoutError("Timed out waiting for RP2350 reply")
            chunk = os.read(self.fd, 1024)
            if not chunk:
                raise EOFError("RP2350 serial port closed")
            self._buf.extend(chunk)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class RP2350Bridge:
    """Thread-safe host client for Threebits RP2350 USB-CDC / PTY firmware."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._lock = threading.Lock()
        self._transport = None
        self.board = "unknown"
        self.proto_version = 0
        self.rom_bytes = 0
        self.sram_bytes = 0
        self.frames_gru = 0
        self.frames_local = 0
        self.last_gru_us: Optional[int] = None
        self.last_local_us: Optional[int] = None
        self.last_out_q12: Optional[List[int]] = None
        self.last_local_pred: Optional[int] = None

        use_pyserial = False
        if not port.startswith(("/dev/ttys", "/dev/pts/")):
            try:
                import serial  # type: ignore
                self._transport = serial.Serial(port, baudrate=baudrate, timeout=timeout)
                use_pyserial = True
            except Exception:
                use_pyserial = False
        if not use_pyserial:
            self._transport = _PosixRawPort(port, timeout=timeout)

        # Verify live firmware handshake via PING/PONG
        info = self.ping()
        self.board = info["board"]
        self.proto_version = info["proto"]
        self.rom_bytes = info["rom_b"]
        self.sram_bytes = info["sram_b"]

    def _xfer(self, cmd: str, expected_prefix: str) -> Tuple[str, Dict[str, str]]:
        with self._lock:
            self._transport.write((cmd.strip() + "\n").encode("ascii"))
            # Skip any unsolicited READY boot banner if present
            for _ in range(4):
                raw = self._transport.readline()
                if isinstance(raw, bytes):
                    raw = raw.decode("ascii", errors="replace")
                if not raw.strip():
                    continue
                payload = verify_crc_line(raw)
                tag, kv = _parse_kv(payload)
                if tag == "READY":
                    continue
                if tag == "ERR":
                    raise RuntimeError(f"RP2350 returned error for {cmd!r}: {payload}")
                if expected_prefix and tag != expected_prefix:
                    raise RuntimeError(f"Expected {expected_prefix}, got {payload!r}")
                return tag, kv
            raise TimeoutError(f"No {expected_prefix} reply from RP2350")

    def ping(self) -> Dict[str, object]:
        _, kv = self._xfer("PING", "PONG")
        return {
            "proto": int(kv["proto"]),
            "board": kv["board"],
            "rom_b": int(kv["rom_b"]),
            "sram_b": int(kv["sram_b"]),
            "gru_dim": [int(x) for x in kv["gru"].split(",")],
            "local_dim": [int(x) for x in kv["local"].split(",")],
        }

    def reset(self) -> None:
        self._xfer("RESET", "OK")

    def bench(self, steps: int = 200) -> Dict[str, int]:
        _, kv = self._xfer(f"BENCH {int(steps)}", "BENCH_OUT")
        return {k: int(v) for k, v in kv.items()}

    def step_gru(self, x_q8: np.ndarray) -> Dict[str, object]:
        arr = np.asarray(x_q8, dtype=np.int8).reshape(-1)
        if arr.size != 176:
            raise ValueError(f"Expected 176 int8 inputs, got {arr.size}")
        hex_in = arr.view(np.uint8).tobytes().hex()
        _, kv = self._xfer(f"GRU {hex_in}", "GRU_OUT")
        out_q12 = [int(x) for x in kv["out"].split(",")]
        us = int(kv["us"])
        seq = int(kv["seq"])
        self.frames_gru += 1
        self.last_gru_us = us
        self.last_out_q12 = out_q12
        return {"seq": seq, "us": us, "out_q12": out_q12}

    def step_local(self, x_q8: np.ndarray, reward_bit: int = -1) -> Dict[str, object]:
        arr = np.asarray(x_q8, dtype=np.int8).reshape(-1)
        if arr.size != 176:
            raise ValueError(f"Expected 176 int8 inputs, got {arr.size}")
        hex_in = arr.view(np.uint8).tobytes().hex()
        _, kv = self._xfer(f"LOCAL {int(reward_bit)} {hex_in}", "LOCAL_OUT")
        wb_bytes = bytes.fromhex(kv["wb"])
        wb_q8 = np.frombuffer(wb_bytes, dtype=np.int8).tolist()
        us = int(kv["us"])
        pred = int(kv["pred"])
        self.frames_local += 1
        self.last_local_us = us
        self.last_local_pred = pred
        return {
            "seq": int(kv["seq"]),
            "us": us,
            "pred": pred,
            "margin": int(kv["margin"]),
            "updates": int(kv["updates"]),
            "wb_q8": wb_q8,
        }

    def status(self) -> Dict[str, object]:
        return {
            "connected": True,
            "port": self.port,
            "board": self.board,
            "proto_version": self.proto_version,
            "rom_bytes": self.rom_bytes,
            "sram_bytes": self.sram_bytes,
            "frames_gru": self.frames_gru,
            "frames_local": self.frames_local,
            "last_gru_us": self.last_gru_us,
            "last_local_us": self.last_local_us,
            "last_out_q12": self.last_out_q12,
            "last_local_pred": self.last_local_pred,
        }

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
