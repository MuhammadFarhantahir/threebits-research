"""Automated C99 Firmware, PTY Wire-Protocol, EvalAI Decoder, Pilot API & Reproducibility Test Suite.

Run with standard library unittest (zero extra packages required):
    .venv/bin/python -m unittest discover -s tests -v
"""
import ctypes
import json
import os
import pickle
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import termios
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "chip"))
sys.path.insert(0, str(ROOT / "chip" / "firmware_rp2350"))

from chip.int4_gru_runtime import Int4GRUModelC, build_luts, float_to_mult_shift
from chip.firmware_rp2350.rp2350_bridge import RP2350Bridge, crc16_ccitt, verify_crc_line


def _cc_bin():
    for cand in ("cc", "gcc", "clang"):
        p = shutil.which(cand)
        if p:
            return p
    raise RuntimeError("No C compiler (cc/gcc/clang) found on PATH")


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestC99Kernels(unittest.TestCase):
    """Strict -std=c99 -O3 -Wall -Wextra -Werror build and numerical tests for C99 kernels."""

    def test_int4_gru_c99_compile_and_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            so_ext = ".dylib" if platform.system() == "Darwin" else ".so"
            so_path = Path(td) / f"libint4_gru{so_ext}"
            cmd = [
                _cc_bin(), "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror",
                "-shared", "-fPIC", str(ROOT / "chip" / "int4_gru.c"), "-o", str(so_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, f"C99 compile failed:\n{proc.stderr}")

            lib = ctypes.CDLL(str(so_path))
            lib.gru_int4_step.argtypes = [
                ctypes.POINTER(Int4GRUModelC), ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ]
            pkl_path = ROOT / "falcon_submit" / "h1_int4_bundle.pkl"
            self.assertTrue(pkl_path.exists(), "falcon_submit/h1_int4_bundle.pkl missing")
            with open(pkl_path, "rb") as fh:
                payload = pickle.load(fh)
            core, ad = payload["core"], payload["adapters"]["default"]
            sig, th = build_luts()
            ih_m, ih_s = float_to_mult_shift(core["s_ih"] * core["x_scale"] * 4096.0)
            hh_m, hh_s = float_to_mult_shift(core["s_hh"] * (1.0 / 128.0) * 4096.0)
            out_m, out_s = float_to_mult_shift(ad["s_out"] * (1.0 / 128.0) * 4096.0)
            model = Int4GRUModelC(
                core["I"], core["H"], core["O"],
                core["w_ih_packed"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                core["w_hh_packed"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ad["w_out_packed"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                core["b_ih_q12"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                core["b_hh_q12"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ad["b_out_q12"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ih_m, ih_s, hh_m, hh_s, out_m, out_s,
                sig.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                th.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            )
            rng = np.random.default_rng(123)
            x = np.ascontiguousarray(rng.integers(-40, 40, size=core["I"], dtype=np.int8))
            h1 = np.zeros(core["H"], dtype=np.int8)
            h2 = np.zeros(core["H"], dtype=np.int8)
            o1 = np.zeros(core["O"], dtype=np.int32)
            o2 = np.zeros(core["O"], dtype=np.int32)
            sc = np.zeros(6 * core["H"], dtype=np.int32)
            lib.gru_int4_step(ctypes.byref(model), x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                              h1.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                              o1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                              sc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            lib.gru_int4_step(ctypes.byref(model), x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                              h2.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                              o2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                              sc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            np.testing.assert_array_equal(o1, o2)
            np.testing.assert_array_equal(h1, h2)
            self.assertGreater(int(np.abs(o1).sum()), 0)

    def test_c99_local_plasticity_compile_and_self_organization(self):
        class C99LocalChipC(ctypes.Structure):
            _fields_ = [
                ("n_in", ctypes.c_int32), ("n_codes", ctypes.c_int32), ("d_lat", ctypes.c_int32),
                ("lr_shift", ctypes.c_int32), ("v_ema_q8", ctypes.c_int32), ("n_updates", ctypes.c_int32),
                ("w_read_q8", ctypes.POINTER(ctypes.c_int16)), ("w_write_q8", ctypes.POINTER(ctypes.c_int16)),
                ("last_x_q8", ctypes.POINTER(ctypes.c_int8)), ("usage_q8", ctypes.POINTER(ctypes.c_int32)),
                ("last_pred", ctypes.c_int32), ("last_margin", ctypes.c_int32),
            ]

        with tempfile.TemporaryDirectory() as td:
            so_ext = ".dylib" if platform.system() == "Darwin" else ".so"
            so_path = Path(td) / f"libc99_local{so_ext}"
            cmd = [
                _cc_bin(), "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror",
                "-shared", "-fPIC", str(ROOT / "chip" / "c99_local_plasticity.c"), "-o", str(so_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, f"C99 local plasticity compile failed:\n{proc.stderr}")

            lib = ctypes.CDLL(str(so_path))
            lib.c99_local_forward.argtypes = [
                ctypes.POINTER(C99LocalChipC), ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32),
            ]
            lib.c99_local_forward.restype = ctypes.c_int32
            lib.c99_local_learn_rpe.argtypes = [ctypes.POINTER(C99LocalChipC), ctypes.c_int32, ctypes.c_int32]

            I, C, D = 176, 8, 16
            rng = np.random.default_rng(7)
            w_read = np.ascontiguousarray(rng.integers(-32, 32, size=C * I, dtype=np.int16))
            w_write = np.zeros(C * D, dtype=np.int16)
            for c in range(C):
                w_write[c * D + 2 * c] = 224
                w_write[c * D + 2 * c + 1] = 224
            last_x = np.zeros(I, dtype=np.int8)
            usage = np.full(C, (256 * 16) // C, dtype=np.int32)
            chip = C99LocalChipC(
                I, C, D, 4, 32, 0,
                w_read.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                w_write.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                last_x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                usage.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                0, 0,
            )
            wb = np.zeros(D, dtype=np.int8)
            scores = np.zeros(C, dtype=np.int32)

            # 8 disjoint assemblies (20 active channels each)
            patterns = []
            for k in range(C):
                x = np.zeros(I, dtype=np.int8)
                x[k * 20:(k + 1) * 20] = 64
                patterns.append(x)

            for step in range(400):
                k = step % C
                pred = lib.c99_local_forward(
                    ctypes.byref(chip),
                    patterns[k].ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    wb.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                )
                self.assertGreaterEqual(int(wb.min()), -112)
                self.assertLessEqual(int(wb.max()), 112)
                order = np.argsort(scores)
                runner = int(order[-2]) if int(order[-1]) == pred else int(order[-1])
                lib.c99_local_learn_rpe(ctypes.byref(chip), 1, runner)

            self.assertEqual(chip.n_updates, 400)
            assigned = [
                int(lib.c99_local_forward(
                    ctypes.byref(chip),
                    patterns[k].ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    wb.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ))
                for k in range(C)
            ]
            # Label-free local plasticity must cleanly separate distinct assemblies (>= 7/8 unique codes)
            self.assertGreaterEqual(len(set(assigned)), 7, f"Expected >=7 distinct codes, got {assigned}")


class TestRP2350FirmwareAndPTY(unittest.TestCase):
    """Compile RP2350 firmware under -Werror and test end-to-end POSIX PTY wire protocol + server."""

    def test_crc16_and_pty_loopback_bit_exact_parity(self):
        # Verify CRC-16 detects single-character corruption
        msg = "PONG proto=1 board=host_emu"
        good_line = f"{msg} crc={crc16_ccitt(msg.encode('ascii')):04x}"
        self.assertEqual(verify_crc_line(good_line), msg)
        with self.assertRaises(ValueError):
            verify_crc_line(good_line[:-1] + ("0" if good_line[-1] != "0" else "1"))

        with tempfile.TemporaryDirectory() as td:
            emu_bin = Path(td) / "rp2350_host_emu"
            so_ext = ".dylib" if platform.system() == "Darwin" else ".so"
            so_path = Path(td) / f"libint4_gru{so_ext}"
            cc = _cc_bin()
            r1 = subprocess.run([
                cc, "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-DHOST_TEST",
                str(ROOT / "chip" / "firmware_rp2350" / "main.c"),
                str(ROOT / "chip" / "int4_gru.c"),
                str(ROOT / "chip" / "c99_local_plasticity.c"),
                "-o", str(emu_bin),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(r1.returncode, 0, f"RP2350 firmware compile failed:\n{r1.stderr}")

            # 1. Verify --bench mode
            rb = subprocess.run([str(emu_bin), "--bench", "100"], capture_output=True, text=True, check=True)
            self.assertIn("Flash ROM: 62944 B", rb.stdout)
            self.assertIn("Static SRAM: 6796 B", rb.stdout)

            # 2. Compile reference shared lib for bit-exact parity check
            subprocess.run([
                cc, "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                str(ROOT / "chip" / "int4_gru.c"), "-o", str(so_path),
            ], check=True)
            lib = ctypes.CDLL(str(so_path))
            lib.gru_int4_step.argtypes = [
                ctypes.POINTER(Int4GRUModelC), ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ]
            with open(ROOT / "falcon_submit" / "h1_int4_bundle.pkl", "rb") as fh:
                payload = pickle.load(fh)
            core, ad = payload["core"], payload["adapters"]["default"]
            sig, th = build_luts()
            ih_m, ih_s = float_to_mult_shift(core["s_ih"] * core["x_scale"] * 4096.0)
            hh_m, hh_s = float_to_mult_shift(core["s_hh"] * (1.0 / 128.0) * 4096.0)
            out_m, out_s = float_to_mult_shift(ad["s_out"] * (1.0 / 128.0) * 4096.0)
            c_model = Int4GRUModelC(
                core["I"], core["H"], core["O"],
                core["w_ih_packed"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                core["w_hh_packed"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ad["w_out_packed"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                core["b_ih_q12"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                core["b_hh_q12"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ad["b_out_q12"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ih_m, ih_s, hh_m, hh_s, out_m, out_s,
                sig.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                th.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            )

            # 3. Real POSIX PTY loopback: rp2350_host_emu on master FD <-> RP2350Bridge on slave PTY path
            master_fd, slave_fd = os.openpty()
            slave_path = os.ttyname(slave_fd)
            # Disable ECHO & canonical processing on PTY slave so lines do not echo back
            attrs = termios.tcgetattr(slave_fd)
            attrs[0] &= ~(termios.ICRNL | termios.IXON)
            attrs[1] &= ~termios.OPOST
            attrs[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

            emu_proc = subprocess.Popen(
                [str(emu_bin), "--stdio"],
                stdin=master_fd, stdout=master_fd, stderr=subprocess.PIPE,
                close_fds=True,
            )
            os.close(master_fd)
            bridge = None
            try:
                bridge = RP2350Bridge(slave_path, timeout=2.0)
                self.assertEqual(bridge.board, "host_emu")
                self.assertEqual(bridge.rom_bytes, 62944)
                self.assertEqual(bridge.sram_bytes, 6796)

                # Verify bit-for-bit parity over 5 consecutive recurrent GRU steps across PTY!
                bridge.reset()
                rng = np.random.default_rng(2026)
                h_ref = np.zeros(core["H"], dtype=np.int8)
                out_ref = np.zeros(core["O"], dtype=np.int32)
                scratch = np.zeros(6 * core["H"], dtype=np.int32)
                for step in range(5):
                    x_q8 = np.ascontiguousarray(rng.integers(-50, 50, size=176, dtype=np.int8))
                    lib.gru_int4_step(
                        ctypes.byref(c_model),
                        x_q8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        h_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        out_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    )
                    reply = bridge.step_gru(x_q8)
                    self.assertEqual(reply["seq"], step + 1)
                    self.assertEqual(reply["out_q12"], out_ref.tolist())

                # Verify on-chip local plasticity command over PTY
                x_loc = np.zeros(176, dtype=np.int8)
                x_loc[:16] = 64
                lrep = bridge.step_local(x_loc, reward_bit=1)
                self.assertEqual(lrep["updates"], 1)
                self.assertEqual(len(lrep["wb_q8"]), 16)
                self.assertTrue(all(-112 <= v <= 112 for v in lrep["wb_q8"]))

                # Verify BENCH over PTY
                brep = bridge.bench(50)
                self.assertEqual(brep["steps"], 50)
                self.assertGreaterEqual(brep["gru_total_us"], 0)
            finally:
                if bridge is not None:
                    bridge.close()
                os.close(slave_fd)
                emu_proc.terminate()
                emu_proc.wait(timeout=3)
                if emu_proc.stderr is not None:
                    emu_proc.stderr.close()

            # 4. End-to-end level3_app/server.py --serial <pty_slave> HTTP telemetry test (when local EEG replay npz exists)
            wm_npz = ROOT / "data" / "tmseeg_ds008037" / "wm_features" / "sub-020_wm.npz"
            if wm_npz.exists():
                m2, s2 = os.openpty()
                s2_path = os.ttyname(s2)
                attrs2 = termios.tcgetattr(s2)
                attrs2[0] &= ~(termios.ICRNL | termios.IXON)
                attrs2[1] &= ~termios.OPOST
                attrs2[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
                termios.tcsetattr(s2, termios.TCSANOW, attrs2)

                emu2 = subprocess.Popen([str(emu_bin), "--stdio"], stdin=m2, stdout=m2, stderr=subprocess.PIPE, close_fds=True)
                os.close(m2)
                port = _free_port()
                srv = subprocess.Popen(
                    [sys.executable, "-u", str(ROOT / "level3_app" / "server.py"), "--port", str(port), "--serial", s2_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                try:
                    hw = None
                    for _ in range(40):
                        time.sleep(0.15)
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/hardware_status", timeout=2) as resp:
                                hw = json.loads(resp.read().decode())
                                break
                        except Exception:
                            continue
                    self.assertIsNotNone(hw, "level3_app/server.py failed to start with --serial PTY")
                    self.assertTrue(hw["ok"])
                    self.assertTrue(hw["mcu"]["connected"])
                    self.assertEqual(hw["mcu"]["board"], "host_emu")
                    self.assertGreaterEqual(hw["mcu"]["frames_gru"], 1)

                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/mcu_step",
                        data=json.dumps({"mode": "local", "reward_bit": 1}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        step_res = json.loads(resp.read().decode())
                    self.assertTrue(step_res["ok"])
                    self.assertEqual(step_res["mode"], "local")
                    self.assertEqual(step_res["updates"], 1)
                finally:
                    srv.terminate()
                    srv.wait(timeout=3)
                    if srv.stdout is not None:
                        srv.stdout.close()
                    if srv.stderr is not None:
                        srv.stderr.close()
                    os.close(s2)
                    emu2.terminate()
                    emu2.wait(timeout=3)
                    if emu2.stderr is not None:
                        emu2.stderr.close()


class TestFalconEvalAIDecoder(unittest.TestCase):
    """Verify falcon_submit/decoder.py runs with ZERO PyTorch installed (matching Dockerfile)."""

    def test_decoder_runs_without_torch(self):
        with tempfile.TemporaryDirectory() as td:
            so_ext = ".dylib" if platform.system() == "Darwin" else ".so"
            lib_out = Path(td) / f"libint4_gru{so_ext}"
            subprocess.run([
                _cc_bin(), "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                str(ROOT / "chip" / "int4_gru.c"), "-o", str(lib_out),
            ], check=True)
            # Copy decoder + bundle + runtime into isolated temp dir mimicking Docker container
            shutil.copy(ROOT / "falcon_submit" / "decoder.py", Path(td) / "decoder.py")
            shutil.copy(ROOT / "falcon_submit" / "h1_int4_bundle.pkl", Path(td) / "h1_int4_bundle.pkl")
            (Path(td) / "chip").mkdir()
            shutil.copy(ROOT / "chip" / "int4_gru_runtime.py", Path(td) / "chip" / "int4_gru_runtime.py")

            smoke = f"""
import sys, numpy as np
sys.modules['torch'] = None  # Prove EvalAI container works with ZERO PyTorch!
sys.path.insert(0, {str(td)!r})
import decoder
dec = decoder.Int4AnchoredGRUDecoder(task_config=None, model_path={str(Path(td) / 'h1_int4_bundle.pkl')!r}, batch_size=2)
dec.reset(['default', '2023-08-29'])
obs = np.random.default_rng(0).uniform(0.0, 2.0, size=(2, 176)).astype(np.float32)
dec.observe(obs)
pred1 = dec.predict(obs)
assert pred1.shape == (2, 7) and np.all(np.isfinite(pred1)), pred1.shape
dec.on_done(np.array([True, False]))
assert np.all(dec.h_states[0] == 0) and np.any(dec.h_states[1] != 0)
print('FALCON_DECODER_OK', pred1.shape, float(np.abs(pred1).mean()))
"""
            proc = subprocess.run([sys.executable, "-c", smoke], capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, f"EvalAI zero-torch smoke failed:\n{proc.stderr}")
            self.assertIn("FALCON_DECODER_OK", proc.stdout)


class TestPilotAPIAndAnalysis(unittest.TestCase):
    """End-to-end validation of pilot_api_server.py and analyze_human_gym_pilot.py."""

    def test_pilot_validation_storage_summary_and_analysis(self):
        import pilot_api_server as api
        import analyze_human_gym_pilot as an

        def make_session(pid, coproc_err, sham_err, un_err):
            rows = []
            arms = ["unassisted"] * 6 + ["sham_code"] * 6 + ["latent_coproc"] * 6
            for idx, arm in enumerate(arms, start=1):
                err = coproc_err if arm == "latent_coproc" else (sham_err if arm == "sham_code" else un_err)
                target = float((idx * 29) % 360)
                reported = (target + err) % 360.0
                rows.append({
                    "participant_id": pid, "trial_idx": idx, "arm": arm, "set_size": 12,
                    "rule_deg": 0, "target_deg": target, "reported_deg": reported,
                    "err_deg": err, "rt_ms": 1400, "eff_bps": 2.4 if arm == "latent_coproc" else 0.6,
                    "ts": 1760000000.0 + idx,
                })
            return {"participant_id": pid, "rows": rows}

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # Strict validation rejects bad arm / too few rows
            with self.assertRaises(ValueError):
                api.validate_pilot_submission({"participant_id": "BAD", "rows": [{"arm": "bogus"}]})

            for pid, ce, se, ue in [("P-TEST1", 8.0, 74.0, 68.0), ("P-TEST2", 11.0, 81.0, 72.0)]:
                clean_pid, clean_rows = api.validate_pilot_submission(make_session(pid, ce, se, ue))
                api.save_pilot_csv(d, clean_pid, clean_rows)

            summ = api.compute_pilot_cohort_summary(d)
            self.assertEqual(summ["n_sessions"], 2)
            self.assertEqual(summ["n_trials"], 36)
            self.assertEqual(summ["arms_pooled"]["latent_coproc"]["acc30"], 100.0)
            self.assertEqual(summ["arms_pooled"]["sham_code"]["acc30"], 0.0)

            csv_bytes = api.export_combined_pilot_csv(d)
            self.assertIn(b"P-TEST1", csv_bytes)
            self.assertIn(b"P-TEST2", csv_bytes)

            loaded = an.load_pilot_csvs(d)
            self.assertEqual(len(loaded), 36)
            res = an.analyze(loaded, out_fig=None)
            self.assertEqual(res["n_participants"], 2)
            self.assertGreater(res["arms"]["latent_coproc"]["acc30_pct"], res["arms"]["sham_code"]["acc30_pct"])


class TestReproducibilityInvariants(unittest.TestCase):
    """Guard key empirical invariants across saved JSON outputs and live web JS syntax."""

    def test_saved_benchmark_invariants(self):
        ft_path = ROOT / "validate_ccep_ftract_external.json"
        if ft_path.exists():
            ft = json.loads(ft_path.read_text())
            b = ft["part_b_utrecht_to_ftract_zeroshot"]
            self.assertGreater(
                b["prob_all"]["Utrecht Bio+CCEP Twin (ZS)"]["pooled_rho"],
                b["prob_all"]["fMRI-tuned Hopf"]["pooled_rho"] + 0.15,
            )
            self.assertGreater(
                b["prob_unseen_in_utrecht"]["Utrecht Bio-Only (47p ZS)"]["pooled_partial"],
                b["prob_unseen_in_utrecht"]["Permuted-Bio-Only Ctrl (3p)"]["pooled_partial"] + 0.05,
            )
            self.assertLessEqual(
                b["bootstraps"]["bio_only_vs_perm_bio_unseen_prob"]["delta_partial"]["p_two_sided"],
                0.01,
            )

        g_path = ROOT / "personalized_50brain_rescue.json"
        if g_path.exists():
            g = json.loads(g_path.read_text())
            self.assertEqual(g["n_patients"], 50)
            self.assertGreater(g["aggregate"]["healthy_adapted_r101"]["mean"], 95.0)
            self.assertGreater(g["aggregate"]["lesion_rescued_r101"]["mean"], 80.0)
            self.assertEqual(
                sum(1 for p in g["patients"] if p["lesion_rescued_r101"] > p["lesion_bio_alone"]),
                50,
            )

        bench_path = ROOT / "chip" / "int4_gru_benchmark.json"
        if bench_path.exists():
            cb = json.loads(bench_path.read_text())
            self.assertGreater(float(cb["minival_r2_c99_w4a8_int"]), 0.80)
            self.assertLess(abs(float(cb["minival_gap"])), 0.01)

        lpo_path = ROOT / "validate_patient_level_ccep.json"
        if lpo_path.exists():
            lpo = json.loads(lpo_path.read_text())
            self.assertEqual(lpo["n_evaluated_patients"], 74)
            self.assertGreater(lpo["summary"]["Multi-Scale Bio+CCEP + Bipolar VTA (LPO)"]["mean_auc"], 0.80)
            self.assertEqual(
                lpo["wilcoxon_tests"]["Multi-Scale Bio+CCEP + Bipolar VTA (LPO) vs fMRI-tuned Hopf"]["wins_auc"],
                74,
            )

        lat_path = ROOT / "ccep_latency_waveform_model.json"
        if lat_path.exists():
            lat = json.loads(lat_path.read_text())
            self.assertGreater(lat["part_b_utrecht_developmental_and_transfer"]["speed_vs_age_rho"], 0.30)
            self.assertLess(lat["part_b_utrecht_developmental_and_transfer"]["lpo_latency_cv"]["wilcoxon_p"], 0.05)

        sub_path = ROOT / "subcortical_clinical_pipeline.json"
        if sub_path.exists():
            sub = json.loads(sub_path.read_text())
            self.assertEqual(sub["n_regions"], 114)
            self.assertTrue(sub["brl_lyapunov_228_state"]["certified_exponentially_stable"])
            self.assertGreater(sub["cortical_ccep_preservation"]["ccep_114_rho"], 0.70)

    def test_web_js_syntax(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        bd = ROOT / "website" / "brain" / "brain_data.js"
        if bd.exists():
            r = subprocess.run([node, "--check", str(bd)], capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 0, f"brain_data.js syntax error:\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
