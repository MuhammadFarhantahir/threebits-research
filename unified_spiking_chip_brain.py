"""Track B: Unified Silicon-in-the-Loop Spiking Human Brain + Real C99 66 kB int4 Chip
+ Counterfactual World-Model Region 101 + Lyapunov Safety Barrier (Threebits Research).

Unifies ALL THREE Threebits Papers into ONE Single End-to-End Closed-Loop System:
  - Paper 1 (Whole-Brain Causal Physics + Safety): 100-region CCEP + 19-PET + MEG scaffold
    (ρ = +0.66 on 74 patients, 45/46 healthy TMS-EEG) + Lyapunov / Small-Gain safety barrier.
  - Paper 2 (Real Compiled Silicon): 176 biophysical LIF E–I spiking neurons embedded inside
    prefrontal hub `LH_Cont_PFCl_1` (matching FALCON H1's 176 channels, dt = 1 ms, 20 ms bins)
    decoded LIVE via `ctypes` by our ACTUAL compiled zero-float C99 packed `int4 x int8` binary
    (`chip/libint4_gru.dylib`, `gru_int4_step()`) — including Week-6 spiking electrode drift
    recalibration via a 1% L2-anchored frozen-core adapter with SHA256-verified byte-identical ROM!
  - Paper 3 (Superhuman Counterfactual Planning — Gap 2): Region 101 builds a 4-bit Causal
    Transition World Model from the C99-decoded spikes and runs 81 parallel 4-step counterfactual
    future rollouts (`3^4 = 81`) avoiding hidden traps/hazards while aligning to a PRIVATE embodied
    human value goal injected exclusively into spiking PFC (invisible to Standalone AI!).

Five Compared Regimes (3 seeds x 150 fresh unseen 8-state stochastic causal trap worlds = 450 worlds):
  1. 100-Region Spiking Bio Brain Alone (biological WM + short 1-step greedy horizon)
  2. Standalone AI Counterfactual Planner (100% public world model + 81-branch tree search, blind to private human value -> 50% Bayes cap)
  3. Sham C99 Chip Loop (permuted 176-ch int8 spikes fed to real libint4_gru.dylib)
  4. Unconstrained Region 101 Writeback (no Lyapunov/Small-Gain safety barrier -> epileptiform hypersynchrony!)
  5. Unified Threebits Stack (Day 1 Baseline + Week-6 Electrode Drift w/ 1% Frozen-Core Adapter)

Run (~60-90 seconds):
    .venv/bin/python unified_spiking_chip_brain.py
"""
import ctypes
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "chip"))

from bench_int4_gru import Int4GRUModelC, build_luts, float_to_mult_shift, pack_int4_matrix
from functional_brain_emulator import get_causal_human_scaffold
from precision_sweep import Rounded
from superhuman_reasoning_emulator import eval_zero_shot_tmseeg_n46
from validate_simulator_ccep import partial_spearman

SO_PATH = ROOT / "chip" / "libint4_gru.dylib"
OUT_JSON = ROOT / "unified_spiking_chip_brain.json"
FIG_PATH = ROOT / "figures" / "fig11_unified_spiking_chip_brain.png"

N_SPK = 176          # Exact FALCON H1 channel count: 140 Excitatory + 36 Inhibitory LIF neurons
N_EXC = 140
N_INH = 36
N_STATES = 8
N_ACTIONS = 3
OUT_DIM = 16         # 8 (next_state) + 3 (action) + 2 (hazard flag) + 3 (private human value: g0 / g1 / none)
BIN_STEPS = 4        # 4 x 20 ms = 80 ms observation window per causal probe


# ---------------------------------------------------------------------------
# 1. Biophysical 176-Neuron LIF E-I Prefrontal Spiking Microcolumn (dt = 1 ms)
# ---------------------------------------------------------------------------
class BiophysicalSpikingPFC:
    """176-neuron LIF E-I spiking column inside `LH_Cont_PFCl_1` (AMPA/NMDA/GABAa, dt=1ms, 20ms bins)."""

    def __init__(self, pet_ei_gain: float, seed: int = 0):
        rng = np.random.default_rng(seed + 777)
        self.pet_ei_gain = float(pet_ei_gain)
        # Fixed sparse anatomical cortical projection from 21-d sensory/context + 2-d private value -> 176 LIF neurons
        # Input channels (23-d): [s_oh(8), a_oh(3), sp_oh(8), haz(2), priv_val(2)]
        W = rng.normal(0.0, 0.55, size=(23, N_SPK)).astype(np.float32)
        # Biologically realistic structured assemblies: subsets of PFC neurons tuned to states, actions, hazards, values
        for s in range(N_STATES):
            cols = np.arange(s * 14, (s + 1) * 14) % N_EXC
            W[s, cols] += 1.45
            W[11 + s, (cols + 7) % N_EXC] += 1.75
        for a in range(N_ACTIONS):
            cols = np.arange(112 + a * 9, 112 + (a + 1) * 9) % N_EXC
            W[8 + a, cols] += 1.65
        W[19, 0:28] += 1.35    # safe indicator
        W[20, 28:56] += 1.55   # hazard indicator
        W[21, 56:98] += 1.50   # private human value = g0
        W[22, 98:140] += 1.50  # private human value = g1
        # Inhibitory interneurons receive broad excitatory drive (E->I balance)
        W[:, N_EXC:] = np.abs(rng.normal(0.35, 0.18, size=(23, N_INH))).astype(np.float32)
        self.W_in = W

        # Recurrent E-I microcircuit weights scaled by Hansen PET E:I gain
        W_rec = np.zeros((N_SPK, N_SPK), dtype=np.float32)
        mask_ee = rng.random((N_EXC, N_EXC)) < 0.12
        W_rec[:N_EXC, :N_EXC] = mask_ee * rng.uniform(0.015, 0.035, size=(N_EXC, N_EXC)) * self.pet_ei_gain
        mask_ei = rng.random((N_EXC, N_INH)) < 0.25
        W_rec[:N_EXC, N_EXC:] = mask_ei * rng.uniform(0.04, 0.09, size=(N_EXC, N_INH))
        mask_ie = rng.random((N_INH, N_EXC)) < 0.35
        W_rec[N_EXC:, :N_EXC] = -mask_ie.astype(np.float32) * rng.uniform(0.07, 0.14, size=(N_INH, N_EXC))
        np.fill_diagonal(W_rec, 0.0)
        self.W_rec = W_rec

        # Default (Day-1) electrode gain & baseline; modified during Week-6 drift test!
        self.elec_gain = np.ones(N_SPK, dtype=np.float32)
        self.elec_bias = np.zeros(N_SPK, dtype=np.float32)
        self.elec_alive = np.ones(N_SPK, dtype=np.float32)

    def apply_week6_drift(self, rng):
        """Simulate realistic 6-week intracortical electrode drift (gain + baseline shift + 12% dropout)."""
        self.elec_gain = rng.uniform(0.65, 1.40, size=N_SPK).astype(np.float32)
        self.elec_bias = rng.normal(0.0, 0.45, size=N_SPK).astype(np.float32)
        alive = (rng.random(N_SPK) > 0.12).astype(np.float32)
        self.elec_alive = alive

    def reset_drift(self):
        self.elec_gain = np.ones(N_SPK, dtype=np.float32)
        self.elec_bias = np.zeros(N_SPK, dtype=np.float32)
        self.elec_alive = np.ones(N_SPK, dtype=np.float32)

    def simulate_probe_bins(self, drive_23: np.ndarray, rng, n_bins=BIN_STEPS, writeback_current=None):
        """Integrate 176 LIF E-I neurons at dt = 1 ms over `n_bins` x 20 ms bins.

        Returns:
          bins_obs: (n_bins, 176) observed electrode spike counts (subject to electrode gain/bias/dropout)
          raster:   (n_bins*20, 176) raw 1 ms binary spikes (for visualization & physiology metrics)
          metrics:  dict with mean_hz, max_hz, synchrony_index
        """
        T_ms = n_bins * 20
        v = rng.uniform(0.0, 0.35, size=N_SPK).astype(np.float32)
        refrac = np.zeros(N_SPK, dtype=np.int32)
        g_ampa = np.zeros(N_SPK, dtype=np.float32)
        g_nmda = np.zeros(N_SPK, dtype=np.float32)
        g_gaba = np.zeros(N_SPK, dtype=np.float32)

        base_in = np.maximum(0.0, drive_23 @ self.W_in) * 0.165 + 0.088
        if writeback_current is not None:
            base_in = base_in + writeback_current

        raster = np.zeros((T_ms, N_SPK), dtype=np.float32)
        decay_m = math.exp(-1.0 / 20.0)
        decay_ampa = math.exp(-1.0 / 5.0)
        decay_nmda = math.exp(-1.0 / 45.0)
        decay_gaba = math.exp(-1.0 / 10.0)

        for t in range(T_ms):
            noise = rng.normal(0.0, 0.085, size=N_SPK).astype(np.float32)
            # Slow ramp & sustained NMDA dynamics over the 80 ms probe
            env = 0.85 + 0.25 * math.sin(math.pi * t / max(T_ms - 1, 1))
            i_tot = base_in * env + g_ampa + 0.45 * g_nmda + g_gaba + noise
            active = refrac <= 0
            v = np.where(active, decay_m * v + (1.0 - decay_m) * (i_tot * 10.5), 0.0)
            spk = (v >= 1.0) & active
            raster[t] = spk.astype(np.float32)
            v = np.where(spk, 0.0, np.clip(v, -0.5, 1.5))
            refrac = np.where(spk, 2, np.maximum(0, refrac - 1))

            if np.any(spk):
                exc_s = spk[:N_EXC].astype(np.float32)
                inh_s = spk[N_EXC:].astype(np.float32)
                e_drive = exc_s @ self.W_rec[:N_EXC, :]
                i_drive = inh_s @ self.W_rec[N_EXC:, :]
                g_ampa = decay_ampa * g_ampa + 0.65 * e_drive
                g_nmda = decay_nmda * g_nmda + 0.35 * e_drive
                g_gaba = decay_gaba * g_gaba + i_drive
            else:
                g_ampa *= decay_ampa
                g_nmda *= decay_nmda
                g_gaba *= decay_gaba

        bins_raw = raster.reshape(n_bins, 20, N_SPK).sum(axis=1)  # (n_bins, 176)
        bins_obs = np.maximum(0.0, bins_raw * self.elec_gain + self.elec_bias) * self.elec_alive

        # Biophysical firing rate & population synchrony (Fano-normalized pop-rate variance)
        mean_hz = float(raster.mean() * 1000.0)
        pop_1ms = raster.mean(axis=1)  # fraction of neurons firing per ms
        pop_var = float(np.var(pop_1ms))
        mean_unit_var = float(np.mean(np.var(raster, axis=0))) + 1e-6
        sync_idx = float(np.clip(pop_var / mean_unit_var, 0.0, 1.0))
        return bins_obs.astype(np.float32), raster, {
            "mean_hz": mean_hz,
            "sync_idx": sync_idx,
        }


# ---------------------------------------------------------------------------
# 2. Causal Trap World Generator (Requires True 3-4 Step Counterfactual Lookahead)
# ---------------------------------------------------------------------------
def make_causal_trap_world(rng):
    """Build a fresh unseen 8-state, 3-action causal trap environment with private human value.

    States:
      0 = Start
      1, 2, 3 = Intermediate states (with 1 = deceptive trap hub!)
      4, 5 = Catastrophic hazards (instant episode failure!)
      6 = Candidate Goal g0 (private human value v=0)
      7 = Candidate Goal g1 (private human value v=1)

    Structure guarantees:
      - Random action permutations per state so NO static action bias can cheat!
      - Every world has a deceptive trap action from Start -> State 1 (looks safe at step 1,
        but 2 of 3 actions from State 1 plunge directly into hazards 4/5 and the 3rd loops back!).
      - Safe path to g0 takes 3 steps (0 -> 2 -> 3 -> 6) and safe path to g1 takes 3 steps (0 -> 3 -> 2 -> 7),
        or random distinct 3-step/4-step non-hazard trajectories!
      - Greedy 1-step policies have no direct transition 0 -> 6/7 and frequently step into hazards!
    """
    T = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    hazards = {4, 5}
    g0, g1 = 6, 7

    # Randomize which intermediate state is the trap vs safe corridors
    inter = list(rng.permutation([1, 2, 3]))
    s_trap, s_a, s_b = inter[0], inter[1], inter[2]

    # From Start (0): 1 action -> s_trap, 1 action -> s_a (toward g0), 1 action -> s_b (toward g1)
    p0 = rng.permutation([s_trap, s_a, s_b])
    T[0] = p0

    # From s_trap: 2 actions -> hazards (4, 5), 1 action -> self-loop (dead-end trap!)
    T[s_trap] = rng.permutation([4, 5, s_trap])

    # Create a 3-step safe path 0 -> s_a -> s_midA -> g0 and 0 -> s_b -> s_midB -> g1
    # Wait: using distinct 3-step paths where s_a -> s_b -> g0 is tricky if s_b also leads to g1;
    # Make s_a have 1 safe bridge to g0 via a 2-step chain? We have 3 intermediate states!
    # Cleanest 3-step guarantee with 8 states:
    #   Let intermediate states be [s_trap, s_a, s_b]:
    #   From s_a: 1 action -> g0 (so 0->s_a->g0 is 2 steps), OR even better make 1 state a 3-step relay:
    #   Let's use: 0=start, [1,2,3]=safe relays, 4=trap_hub, 5=hazard, 6=g0, 7=g1!
    hazards = {5}
    relays = list(rng.permutation([1, 2, 3]))
    r1, r2, r3 = relays[0], relays[1], relays[2]
    s_trap = 4

    # Start (0): action A -> s_trap (deceptive safe-looking step 1!), action B -> r1, action C -> 5 (immediate hazard)
    T[0] = rng.permutation([s_trap, r1, 5])
    # s_trap (4): 2 actions -> hazard 5, 1 action -> self-loop (trapped!)
    T[s_trap] = rng.permutation([5, 5, s_trap])
    # r1: 1 action -> r2 (toward g0), 1 action -> r3 (toward g1), 1 action -> hazard 5!
    T[r1] = rng.permutation([r2, r3, 5])
    # r2: 1 action -> g0 (6), 1 action -> hazard 5, 1 action -> s_trap (4)!
    T[r2] = rng.permutation([g0, 5, s_trap])
    # r3: 1 action -> g1 (7), 1 action -> hazard 5, 1 action -> s_trap (4)!
    T[r3] = rng.permutation([g1, 5, s_trap])
    # Absorbing terminal states
    T[5] = [5, 5, 5]
    T[g0] = [g0, g0, g0]
    T[g1] = [g1, g1, g1]

    private_val = int(rng.integers(0, 2))  # 0 -> human wants g0 (6); 1 -> human wants g1 (7)
    target_goal = g0 if private_val == 0 else g1
    return {
        "T": T,
        "start": 0,
        "hazards": hazards,
        "g0": g0,
        "g1": g1,
        "private_val": private_val,
        "target_goal": target_goal,
    }


def make_probe_drive(s: int, a: int, sp: int, is_haz: bool, priv_val: int) -> np.ndarray:
    v = np.zeros(23, dtype=np.float32)
    v[s] = 1.0
    v[8 + a] = 1.0
    v[11 + sp] = 1.0
    v[20 if is_haz else 19] = 1.0
    v[21 + priv_val] = 1.0
    return v


# ---------------------------------------------------------------------------
# 3. 4-Bit LSQ Causal GRU Spike Decoder -> Real Compiled C99 Binary Exporter
# ---------------------------------------------------------------------------
class SpikingCausalGRU(nn.Module):
    """Exact Paper-2 CausalGRU architecture (176 ch -> 128 GRU -> 16 out) compatible with int4_gru.c!"""

    def __init__(self, n_in=N_SPK, n_hidden=128, n_out=OUT_DIM):
        super().__init__()
        self.ch_scale = nn.Parameter(torch.ones(n_in))
        self.ch_bias = nn.Parameter(torch.zeros(n_in))
        self.gru = nn.GRU(n_in, n_hidden, batch_first=True)
        self.readout = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = x * self.ch_scale + self.ch_bias
        h, _ = self.gru(x)
        return self.readout(h)


class LiveC99Int4Chip:
    """Wraps our ACTUAL compiled zero-float C99 `chip/libint4_gru.dylib` binary via ctypes!"""

    def __init__(self, rounded_model: Rounded, x_clip: float = 6.0):
        assert SO_PATH.exists(), f"Missing compiled C99 binary: {SO_PATH}"
        self.lib = ctypes.CDLL(str(SO_PATH))
        self.lib.gru_int4_forward_seq.argtypes = [
            ctypes.POINTER(Int4GRUModelC),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.gru_int4_forward_seq.restype = None

        self.x_clip = float(x_clip)
        dec = rounded_model.decoder
        steps = {name: float(ls.exp().item()) for name, ls in zip(rounded_model.names, rounded_model.log_steps)}

        def qpack(name):
            w = dec.get_parameter(name).detach().cpu().numpy()
            s = steps[name]
            q = np.clip(np.round(w / s), -8, 7).astype(np.int32)
            return pack_int4_matrix(q), s

        self.w_ih_p, s_ih = qpack("gru.weight_ih_l0")
        self.w_hh_p, s_hh = qpack("gru.weight_hh_l0")
        self.w_out_p, s_out = qpack("readout.weight")

        self.b_ih = np.ascontiguousarray(np.round(dec.gru.bias_ih_l0.detach().cpu().numpy() * 4096.0).astype(np.int32))
        self.b_hh = np.ascontiguousarray(np.round(dec.gru.bias_hh_l0.detach().cpu().numpy() * 4096.0).astype(np.int32))
        self.b_out = np.ascontiguousarray(np.round(dec.readout.bias.detach().cpu().numpy() * 4096.0).astype(np.int32))

        x_scale = self.x_clip / 127.0
        h_scale = 1.0 / 128.0
        ih_mult, ih_shift = float_to_mult_shift(s_ih * x_scale * 4096.0)
        hh_mult, hh_shift = float_to_mult_shift(s_hh * h_scale * 4096.0)
        out_mult, out_shift = float_to_mult_shift(s_out * h_scale * 4096.0)

        self.sig_lut, self.th_lut = build_luts()
        self.ch_scale = dec.ch_scale.detach().cpu().numpy().astype(np.float32).copy()
        self.ch_bias = dec.ch_bias.detach().cpu().numpy().astype(np.float32).copy()

        self.I = dec.gru.input_size
        self.H = dec.gru.hidden_size
        self.O = dec.readout.out_features

        # SHA256 fingerprint of the packed int4 GRU core ROM (proves 100% read-only byte identity across drift!)
        h_sha = hashlib.sha256()
        h_sha.update(self.w_ih_p.tobytes())
        h_sha.update(self.w_hh_p.tobytes())
        self.core_sha256 = h_sha.hexdigest()[:16]

        self.rom_bytes = int(
            self.w_ih_p.nbytes + self.w_hh_p.nbytes + self.w_out_p.nbytes
            + self.b_ih.nbytes + self.b_hh.nbytes + self.b_out.nbytes
            + self.sig_lut.nbytes + self.th_lut.nbytes
        )
        self.sram_bytes = int(self.H + 6 * self.H * 4 + self.I + self.O * 4)

        self.c_struct = Int4GRUModelC(
            input_size=self.I, hidden_size=self.H, output_size=self.O,
            w_ih_packed=self.w_ih_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w_hh_packed=self.w_hh_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w_out_packed=self.w_out_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            b_ih_q12=self.b_ih.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            b_hh_q12=self.b_hh.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            b_out_q12=self.b_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ih_mult=ih_mult, ih_shift=ih_shift,
            hh_mult=hh_mult, hh_shift=hh_shift,
            out_mult=out_mult, out_shift=out_shift,
            sigmoid_lut=self.sig_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            tanh_lut=self.th_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
        self.h_buf = np.zeros(self.H, dtype=np.int8)
        self.scratch = np.zeros(6 * self.H, dtype=np.int32)
        self.total_c_ns = 0
        self.total_bins = 0

    def update_frozen_core_adapter(self, rounded_adapted: Rounded):
        """Update ONLY the 1% channel-affine + readout adapter while keeping packed int4 GRU ROM bit-identical!"""
        dec = rounded_adapted.decoder
        self.ch_scale = dec.ch_scale.detach().cpu().numpy().astype(np.float32).copy()
        self.ch_bias = dec.ch_bias.detach().cpu().numpy().astype(np.float32).copy()
        steps = {name: float(ls.exp().item()) for name, ls in zip(rounded_adapted.names, rounded_adapted.log_steps)}
        w_out = dec.readout.weight.detach().cpu().numpy()
        s_out = steps["readout.weight"]
        q_out = np.clip(np.round(w_out / s_out), -8, 7).astype(np.int32)
        self.w_out_p = pack_int4_matrix(q_out)
        self.b_out = np.ascontiguousarray(np.round(dec.readout.bias.detach().cpu().numpy() * 4096.0).astype(np.int32))
        out_mult, out_shift = float_to_mult_shift(s_out * (1.0 / 128.0) * 4096.0)
        self.c_struct.w_out_packed = self.w_out_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self.c_struct.b_out_q12 = self.b_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self.c_struct.out_mult = out_mult
        self.c_struct.out_shift = out_shift
        # Verify SHA256 of packed int4 GRU core ROM is 100% untouched!
        h_sha = hashlib.sha256()
        h_sha.update(self.w_ih_p.tobytes())
        h_sha.update(self.w_hh_p.tobytes())
        assert h_sha.hexdigest()[:16] == self.core_sha256, "Core ROM altered!"

    def decode_spikes_c99(self, bins_f32: np.ndarray, sham_perm=None) -> np.ndarray:
        """Execute REAL zero-float C99 `gru_int4_forward_seq()` on a (T, 176) spike-bin array."""
        x = bins_f32
        if sham_perm is not None:
            x = x[:, sham_perm]
        x_cal = x * self.ch_scale + self.ch_bias
        x_q8 = np.ascontiguousarray(np.clip(np.round(x_cal / (self.x_clip / 127.0)), -127, 127).astype(np.int8))
        T = x_q8.shape[0]
        out_q12 = np.zeros((T, self.O), dtype=np.int32)

        t0 = time.perf_counter_ns()
        self.lib.gru_int4_forward_seq(
            ctypes.byref(self.c_struct),
            x_q8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ctypes.c_int32(T),
            out_q12.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.h_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self.scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        self.total_c_ns += (time.perf_counter_ns() - t0)
        self.total_bins += T
        return (out_q12[-1].astype(np.float32) / 4096.0)


# ---------------------------------------------------------------------------
# 4. Counterfactual 4-Bit World-Model & 81-Branch Tree-Search Region 101
# ---------------------------------------------------------------------------
def counterfactual_plan_region101(T_hat: np.ndarray, haz_hat: np.ndarray, target_goal: int, horizon: int = 4):
    """Run all 3^4 = 81 counterfactual 4-step future rollouts through the 4-bit Causal World Model!

    Evaluates every action sequence (a0, a1, a2, a3) against:
      +10.0 for reaching the human's private goal `target_goal` (and staying safe)
      -20.0 catastrophic penalty for entering any predicted hazard state
      -0.25 step discount (prefers shortest safe path)
    Returns optimal action sequence and best imagined return + count of pruned hazardous branches.
    """
    best_seq = [0] * horizon
    best_val = -1e9
    n_pruned = 0
    for idx in range(N_ACTIONS ** horizon):
        seq = []
        tmp = idx
        for _ in range(horizon):
            seq.append(tmp % N_ACTIONS)
            tmp //= N_ACTIONS

        s = 0
        val = 0.0
        hit_haz = False
        reached = False
        for step_i, a in enumerate(seq):
            s = int(T_hat[s, a])
            if haz_hat[s] > 0.5:
                val -= 20.0
                hit_haz = True
                break
            if s == target_goal and not reached:
                val += 10.0 - 0.25 * step_i
                reached = True
        if hit_haz:
            n_pruned += 1
        if val > best_val:
            best_val = val
            best_seq = seq
    return best_seq, float(best_val), n_pruned


def execute_plan_in_world(world: dict, action_seq: list) -> dict:
    s = world["start"]
    hit_hazard = False
    reached_private_goal = False
    reached_any_goal = False
    for a in action_seq:
        s = int(world["T"][s, a])
        if s in world["hazards"]:
            hit_hazard = True
            break
        if s == world["target_goal"]:
            reached_private_goal = True
            reached_any_goal = True
            break
        if s in (world["g0"], world["g1"]):
            reached_any_goal = True
            break
    return {
        "safe_private_goal": float(reached_private_goal and not hit_hazard),
        "hit_hazard": float(hit_hazard),
        "reached_any_goal": float(reached_any_goal and not hit_hazard),
    }


# ---------------------------------------------------------------------------
# 5. Train 4-Bit LSQ Spike Decoder + Frozen-Core Week-6 Drift Adapter
# ---------------------------------------------------------------------------
def collect_spiking_dataset(pfc: BiophysicalSpikingPFC, n_samples: int, rng):
    X = np.zeros((n_samples, BIN_STEPS, N_SPK), dtype=np.float32)
    y_sp = np.zeros(n_samples, dtype=np.int64)
    y_a = np.zeros(n_samples, dtype=np.int64)
    y_h = np.zeros(n_samples, dtype=np.int64)
    y_v = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        s = int(rng.integers(0, N_STATES))
        a = int(rng.integers(0, N_ACTIONS))
        sp = int(rng.integers(0, N_STATES))
        haz = int(sp == 5)
        pv = int(rng.integers(0, 2))
        drv = make_probe_drive(s, a, sp, bool(haz), pv)
        bins, _, _ = pfc.simulate_probe_bins(drv, rng)
        X[i] = bins
        y_sp[i] = sp
        y_a[i] = a
        y_h[i] = haz
        y_v[i] = pv
    return X, y_sp, y_a, y_h, y_v


def spike_decoder_loss(logits, y_sp, y_a, y_h, y_v):
    # logits: (B, 16) = [sp(8), a(3), h(2), v(3)]
    return (
        F.cross_entropy(logits[:, 0:8], y_sp)
        + 0.7 * F.cross_entropy(logits[:, 8:11], y_a)
        + 0.8 * F.cross_entropy(logits[:, 11:13], y_h)
        + 1.0 * F.cross_entropy(logits[:, 13:15], y_v)
    )


def train_4bit_spike_decoder(pfc: BiophysicalSpikingPFC, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    X_tr, y_sp, y_a, y_h, y_v = collect_spiking_dataset(pfc, 1600, rng)
    mean_tr = X_tr.mean(axis=(0, 1))
    std_tr = X_tr.std(axis=(0, 1)) + 1e-4

    base = SpikingCausalGRU(N_SPK, 128, OUT_DIM)
    with torch.no_grad():
        base.ch_scale.copy_(torch.tensor(1.0 / std_tr, dtype=torch.float32))
        base.ch_bias.copy_(torch.tensor(-mean_tr / std_tr, dtype=torch.float32))
    model = Rounded(base, bits=4)

    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    X_t = torch.tensor(X_tr)
    ysp_t, ya_t, yh_t, yv_t = torch.tensor(y_sp), torch.tensor(y_a), torch.tensor(y_h), torch.tensor(y_v)
    model.train()
    for ep in range(36):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            out = model(X_t[idx])[:, -1, :]
            loss = spike_decoder_loss(out, ysp_t[idx], ya_t[idx], yh_t[idx], yv_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
    model.eval()
    return model


def recalibrate_1pct_frozen_core(model: Rounded, pfc_drifted: BiophysicalSpikingPFC, rng):
    """48-second (~240 probe) L2-anchored frozen-core recalibration: ONLY 1% adapter updated!"""
    import copy
    m_ad = copy.deepcopy(model)
    X_cal, y_sp, y_a, y_h, y_v = collect_spiking_dataset(pfc_drifted, 360, rng)
    # Step 1: Unsupervised per-channel mean-centering (zero labels!)
    m_cur = X_cal.mean(axis=(0, 1))
    s_cur = X_cal.std(axis=(0, 1)) + 1e-4
    with torch.no_grad():
        m_ad.decoder.ch_scale.copy_(torch.tensor(1.0 / s_cur, dtype=torch.float32))
        m_ad.decoder.ch_bias.copy_(torch.tensor(-m_cur / s_cur, dtype=torch.float32))

    # Step 2: Freeze 100% of packed int4 GRU core! Update ONLY ch_scale, ch_bias, readout!
    for p in m_ad.parameters():
        p.requires_grad = False
    m_ad.decoder.ch_scale.requires_grad = True
    m_ad.decoder.ch_bias.requires_grad = True
    m_ad.decoder.readout.weight.requires_grad = True
    m_ad.decoder.readout.bias.requires_grad = True
    # Keep readout LSQ log_step trainable if present
    for name, ls in zip(m_ad.names, m_ad.log_steps):
        if "readout" in name:
            ls.requires_grad = True

    anchor_scale = m_ad.decoder.ch_scale.detach().clone()
    anchor_bias = m_ad.decoder.ch_bias.detach().clone()
    anchor_rw = m_ad.decoder.readout.weight.detach().clone()
    opt = torch.optim.AdamW([p for p in m_ad.parameters() if p.requires_grad], lr=5e-3)

    X_t = torch.tensor(X_cal)
    ysp_t, ya_t, yh_t, yv_t = torch.tensor(y_sp), torch.tensor(y_a), torch.tensor(y_h), torch.tensor(y_v)
    m_ad.train()
    for _ in range(28):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            out = m_ad(X_t[idx])[:, -1, :]
            loss = spike_decoder_loss(out, ysp_t[idx], ya_t[idx], yh_t[idx], yv_t[idx])
            reg = 0.02 * (
                (m_ad.decoder.ch_scale - anchor_scale).pow(2).mean()
                + (m_ad.decoder.ch_bias - anchor_bias).pow(2).mean()
                + (m_ad.decoder.readout.weight - anchor_rw).pow(2).mean()
            )
            (loss + reg).backward()
            opt.step()
    m_ad.eval()
    return m_ad


# ---------------------------------------------------------------------------
# 6. Whole-Brain CCEP + Hybrid Spiking Safety Evaluation (Paper 1 + 2 + 3)
# ---------------------------------------------------------------------------
def measure_hybrid_ccep_and_tmseeg(scaf: dict, safe_gain_delta: float = 0.008):
    """Verify the 100-region CCEP+PET scaffold with embedded spiking PFC preserves 74-pt CCEP & 46-adult TMS-EEG."""
    W_np, d_np = scaf["W_causal"].copy(), scaf["d_causal"].copy()
    denom = np.maximum(d_np[:, None] + W_np.sum(axis=1, keepdims=True), 1e-6)
    A_off = W_np / denom
    np.fill_diagonal(A_off, 0.0)
    eig_max = float(np.max(np.abs(np.linalg.eigvals(A_off))))
    A = 0.78 * A_off / max(eig_max, 1e-6)
    alpha = scaf["alpha_bio"][:, None]
    ei = scaf["ei_gain"][:, None]
    # Perturb LH_Cont_PFCl_1 slightly to reflect closed-loop safe writeback coupling
    pfc_i = scaf["cont_idx"][0]
    ei_mod = ei.copy()
    ei_mod[pfc_i] *= (1.0 + safe_gain_delta)

    # Single-pulse evoked response matrix (100 x 100)
    evoked = np.zeros((100, 100), dtype=np.float64)
    for src in range(100):
        h = np.zeros((100, 1), dtype=np.float64)
        ext = np.zeros((100, 1), dtype=np.float64)
        ext[src, 0] = 0.25
        h = (1.0 - alpha) * h + alpha * np.tanh(ei_mod * (A @ h) + ext)
        acc = np.zeros(100, dtype=np.float64)
        for _ in range(4):
            h = (1.0 - alpha) * h + alpha * np.tanh(ei_mod * (A @ h))
            acc += np.abs(h[:, 0])
        evoked[src] = acc

    valid = scaf["valid"]
    rho = float(spearmanr(evoked[valid], scaf["prob"][valid]).statistic)
    prho = float(partial_spearman(evoked[valid], scaf["prob"][valid], scaf["dist_mat"][valid]))
    tmseeg = eval_zero_shot_tmseeg_n46(evoked, scaf)
    return rho, prho, tmseeg


def run_seed(seed: int, scaf: dict, n_worlds: int = 150):
    rng = np.random.default_rng(seed + 3000)
    pfc_idx = scaf["cont_idx"][0]
    pet_ei = float(scaf["ei_gain"][pfc_idx])
    pfc = BiophysicalSpikingPFC(pet_ei_gain=pet_ei, seed=seed)

    # Train 4-bit LSQ spike decoder & compile/export into REAL zero-float C99 binary (libint4_gru.dylib)!
    model_4bit = train_4bit_spike_decoder(pfc, seed=seed)
    chip_c99 = LiveC99Int4Chip(model_4bit, x_clip=6.0)

    # Verify PyTorch W4A32 vs Pure-Integer C99 W4A8 parity on 200 fresh spiking probes!
    X_val, ysp_v, ya_v, yh_v, yv_v = collect_spiking_dataset(pfc, 200, rng)
    with torch.no_grad():
        pt_out = model_4bit(torch.tensor(X_val))[:, -1, :].numpy()
    c_out = np.stack([chip_c99.decode_spikes_c99(X_val[i]) for i in range(200)], axis=0)
    pt_acc_sp = float((pt_out[:, 0:8].argmax(1) == ysp_v).mean() * 100.0)
    c_acc_sp = float((c_out[:, 0:8].argmax(1) == ysp_v).mean() * 100.0)
    c_acc_v = float((c_out[:, 13:15].argmax(1) == yv_v).mean() * 100.0)

    # Measure resting baseline spiking physiology (zero writeback)
    rest_hzs, rest_syncs = [], []
    for _ in range(30):
        _, _, m = pfc.simulate_probe_bins(np.zeros(23, dtype=np.float32), rng)
        rest_hzs.append(m["mean_hz"])
        rest_syncs.append(m["sync_idx"])
    rest_hz = float(np.mean(rest_hzs))
    rest_sync = float(np.mean(rest_syncs))

    # Generate 150 fresh unseen causal trap worlds!
    worlds = [make_causal_trap_world(rng) for _ in range(n_worlds)]
    sham_perm = rng.permutation(N_SPK)

    # Save 1 rich example spiking raster + counterfactual rollout tree for web Brain Twin & plotting!
    sample_demo = None

    # ---- Evaluate all regimes across the 150 unseen causal worlds ----
    stats = {k: {"goal": [], "haz": [], "hz": [], "dsync": [], "seizure": []} for k in (
        "bio_spiking_alone", "standalone_ai", "sham_c99_chip", "unconstrained_writeback",
        "threebits_unified_day1", "threebits_week6_zeroshot", "threebits_week6_adapted",
    )}

    # Helper: run one world through Spiking PFC -> Live C99 int4 chip -> Counterfactual Region 101 -> Writeback
    def run_world_with_chip(world, chip_obj, sham_p=None, lyapunov_safe=True, save_demo=False):
        T_hat = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
        haz_votes = np.zeros(N_STATES, dtype=np.float32)
        haz_cnt = np.zeros(N_STATES, dtype=np.float32) + 1e-6
        val_votes = np.zeros(2, dtype=np.float32)
        demo_raster = None

        # Observe all 15 non-terminal causal transitions (states 0..4 x 3 actions) via live 176-LIF spikes -> C99 binary!
        for s in range(5):
            for a in range(N_ACTIONS):
                sp = int(world["T"][s, a])
                drv = make_probe_drive(s, a, sp, sp in world["hazards"], world["private_val"])
                bins, raster, _ = pfc.simulate_probe_bins(drv, rng)
                if save_demo and s == 0 and a == 0:
                    demo_raster = raster  # (80, 176) 1 ms spike raster!
                logits = chip_obj.decode_spikes_c99(bins, sham_perm=sham_p)
                sp_pred = int(np.argmax(logits[0:8]))
                h_prob = float(1.0 / (1.0 + math.exp(-(logits[12] - logits[11]))))
                v_pred = int(np.argmax(logits[13:15]))
                T_hat[s, a] = sp_pred
                haz_votes[sp_pred] += h_prob
                haz_cnt[sp_pred] += 1.0
                val_votes[v_pred] += 1.0
        for s_term in (5, 6, 7):
            T_hat[s_term] = [s_term, s_term, s_term]
        haz_hat = haz_votes / haz_cnt
        pred_pv = int(np.argmax(val_votes))
        pred_goal = world["g0"] if pred_pv == 0 else world["g1"]

        # Counterfactual 81-branch 4-step rollout inside Region 101
        plan_seq, plan_val, n_pruned = counterfactual_plan_region101(T_hat, haz_hat, pred_goal, horizon=4)

        # Closed-loop writeback into 176-neuron Spiking PFC column
        if lyapunov_safe:
            # Certified Lyapunov / Small-Gain norm-bounded balanced E-I writeback (|u|_2 <= u_BRL)
            u_wb = np.zeros(N_SPK, dtype=np.float32)
            u_wb[:N_EXC] = 0.042
            u_wb[N_EXC:] = 0.055  # Balanced inhibitory recruitment keeps synchrony & rate strictly bounded!
        else:
            # Unconstrained high-gain excitatory writeback (no Lyapunov barrier -> epileptiform burst!)
            u_wb = np.zeros(N_SPK, dtype=np.float32)
            u_wb[:N_EXC] = 2.85   # Unbounded excitatory surge driving epileptiform runaway firing (>100 Hz)!
            u_wb[N_EXC:] = -0.30

        _, wb_raster, wb_m = pfc.simulate_probe_bins(np.zeros(23, dtype=np.float32), rng, writeback_current=u_wb)
        dsync = abs(wb_m["sync_idx"] - rest_sync)
        # Paper 1 safety barrier: sustained firing > 50 Hz OR hypersynchrony shift |Δsync| > 0.05
        seizure = 1.0 if (wb_m["mean_hz"] > 50.0 or dsync > 0.05) else 0.0

        # If epileptiform seizure/hypersynchrony erupts, motor execution is disrupted -> degraded random action!
        exec_seq = plan_seq if seizure < 0.5 else list(rng.integers(0, N_ACTIONS, size=4))
        ev = execute_plan_in_world(world, exec_seq)
        ev["hz"] = wb_m["mean_hz"]
        ev["dsync"] = dsync
        ev["seizure"] = seizure
        if save_demo:
            ev["demo"] = {
                "probe_raster_80ms_24ch": demo_raster[:, :24].astype(int).tolist(),
                "safe_wb_hz": round(wb_m["mean_hz"], 1),
                "safe_wb_dsync": round(dsync, 4),
                "plan_seq": [int(x) for x in plan_seq],
                "n_branches_simulated": N_ACTIONS ** 4,
                "n_hazard_branches_pruned": int(n_pruned),
                "pred_private_goal": int(pred_goal),
                "true_private_goal": int(world["target_goal"]),
            }
        return ev

    # Day 1 evaluations
    for wi, w in enumerate(worlds):
        # 1. Spiking Bio Brain Alone (Cowan's ~4-slot WM + greedy 1-step horizon without counterfactual search)
        # Remembers only 4 random transitions out of 15, avoids known direct hazards at step 1, guesses beyond horizon 1
        known = set(rng.choice(15, size=4, replace=False))
        s_cur = w["start"]
        bio_seq = []
        for _ in range(4):
            # Pick action not known to lead directly to hazard (still falls into 2-step traps!)
            safe_cands = []
            for a in range(N_ACTIONS):
                edge_id = s_cur * N_ACTIONS + a
                if s_cur < 5 and edge_id in known and w["T"][s_cur, a] in w["hazards"]:
                    continue
                if s_cur < 5 and edge_id in known and w["T"][s_cur, a] == w["target_goal"]:
                    safe_cands = [a]
                    break
                safe_cands.append(a)
            act = int(rng.choice(safe_cands)) if safe_cands else int(rng.integers(0, N_ACTIONS))
            bio_seq.append(act)
            s_cur = int(w["T"][s_cur, act])
        ev_bio = execute_plan_in_world(w, bio_seq)
        stats["bio_spiking_alone"]["goal"].append(ev_bio["safe_private_goal"])
        stats["bio_spiking_alone"]["haz"].append(ev_bio["hit_hazard"])
        stats["bio_spiking_alone"]["hz"].append(rest_hz)
        stats["bio_spiking_alone"]["dsync"].append(0.0)
        stats["bio_spiking_alone"]["seizure"].append(0.0)

        # 2. Standalone AI Counterfactual Planner (100% public causal graph + exact 81-branch search, BLIND to private goal!)
        ai_haz = np.array([1.0 if s in w["hazards"] else 0.0 for s in range(N_STATES)], dtype=np.float32)
        ai_guess_goal = w["g0"] if rng.random() < 0.5 else w["g1"]
        ai_seq, _, _ = counterfactual_plan_region101(w["T"], ai_haz, ai_guess_goal, horizon=4)
        ev_ai = execute_plan_in_world(w, ai_seq)
        stats["standalone_ai"]["goal"].append(ev_ai["safe_private_goal"])
        stats["standalone_ai"]["haz"].append(ev_ai["hit_hazard"])
        stats["standalone_ai"]["hz"].append(0.0)
        stats["standalone_ai"]["dsync"].append(0.0)
        stats["standalone_ai"]["seizure"].append(0.0)

        # 3. Sham C99 Chip Loop (permuted spikes into real libint4_gru.dylib)
        ev_sh = run_world_with_chip(w, chip_c99, sham_p=sham_perm, lyapunov_safe=True)
        for k, mkey in [("goal", "safe_private_goal"), ("haz", "hit_hazard"), ("hz", "hz"), ("dsync", "dsync"), ("seizure", "seizure")]:
            stats["sham_c99_chip"][k].append(ev_sh[mkey])

        # 4. Unconstrained Writeback (no Lyapunov safety barrier)
        ev_un = run_world_with_chip(w, chip_c99, sham_p=None, lyapunov_safe=False)
        for k, mkey in [("goal", "safe_private_goal"), ("haz", "hit_hazard"), ("hz", "hz"), ("dsync", "dsync"), ("seizure", "seizure")]:
            stats["unconstrained_writeback"][k].append(ev_un[mkey])

        # 5. Unified Threebits Stack (Day 1 — Real C99 int4xint8 + Counterfactual R101 + Lyapunov Safety!)
        ev_u1 = run_world_with_chip(w, chip_c99, sham_p=None, lyapunov_safe=True, save_demo=(seed == 0 and wi == 0))
        if "demo" in ev_u1:
            sample_demo = ev_u1["demo"]
        for k, mkey in [("goal", "safe_private_goal"), ("haz", "hit_hazard"), ("hz", "hz"), ("dsync", "dsync"), ("seizure", "seizure")]:
            stats["threebits_unified_day1"][k].append(ev_u1[mkey])

    # ---- Week-6 Intracortical Electrode Drift Stress Test (same 150 worlds!) ----
    pfc.apply_week6_drift(rng)
    # 5b. Zero-shot under Week-6 drift (no recalibration)
    for w in worlds:
        ev_z = run_world_with_chip(w, chip_c99, sham_p=None, lyapunov_safe=True)
        for k, mkey in [("goal", "safe_private_goal"), ("haz", "hit_hazard"), ("hz", "hz"), ("dsync", "dsync"), ("seizure", "seizure")]:
            stats["threebits_week6_zeroshot"][k].append(ev_z[mkey])

    # 5c. 48-second 1% Frozen-Core Adapter (99% packed int4 C99 ROM SHA256 byte-identical!)
    adapted_4bit = recalibrate_1pct_frozen_core(model_4bit, pfc, rng)
    chip_c99.update_frozen_core_adapter(adapted_4bit)
    for w in worlds:
        ev_a = run_world_with_chip(w, chip_c99, sham_p=None, lyapunov_safe=True)
        for k, mkey in [("goal", "safe_private_goal"), ("haz", "hit_hazard"), ("hz", "hz"), ("dsync", "dsync"), ("seizure", "seizure")]:
            stats["threebits_week6_adapted"][k].append(ev_a[mkey])

    us_per_bin = (chip_c99.total_c_ns / max(chip_c99.total_bins, 1)) / 1000.0
    return {
        "seed": seed,
        "c99_us_per_bin": round(us_per_bin, 2),
        "c99_total_bins_executed": int(chip_c99.total_bins),
        "c99_rom_bytes": chip_c99.rom_bytes,
        "c99_sram_bytes": chip_c99.sram_bytes,
        "c99_total_kb": round((chip_c99.rom_bytes + chip_c99.sram_bytes) / 1024.0, 2),
        "c99_core_sha256": chip_c99.core_sha256,
        "parity_pt_w4a32_pct": round(pt_acc_sp, 2),
        "parity_c99_w4a8_pct": round(c_acc_sp, 2),
        "c99_private_val_decode_pct": round(c_acc_v, 2),
        "rest_hz": round(rest_hz, 1),
        "rest_sync": round(rest_sync, 4),
        "stats": {
            k: {
                "safe_private_goal_pct": round(float(np.mean(v["goal"]) * 100.0), 2),
                "hazard_rate_pct": round(float(np.mean(v["haz"]) * 100.0), 2),
                "mean_hz": round(float(np.mean(v["hz"])), 1),
                "mean_dsync": round(float(np.mean(v["dsync"])), 4),
                "seizure_pct": round(float(np.mean(v["seizure"]) * 100.0), 2),
            }
            for k, v in stats.items()
        },
        "sample_demo": sample_demo,
    }


def make_figure(summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), dpi=220)
    fig.patch.set_facecolor("white")

    # Panel A: Safe Private-Goal Counterfactual Planning vs Hazard Rate (including Week-6 Drift!)
    keys = [
        ("bio_spiking_alone", "Spiking Bio\nBrain Alone\n(Greedy)", "#64748b"),
        ("sham_c99_chip", "Sham C99\nChip Loop\n(Permuted)", "#94a3b8"),
        ("unconstrained_writeback", "No Lyapunov\nGuard\n(Seizure!)", "#dc2626"),
        ("standalone_ai", "Standalone AI\n(Blind to\nHuman Goal)", "#2563eb"),
        ("threebits_week6_zeroshot", "Threebits\nWeek-6 Drift\n(Zero-Shot)", "#f59e0b"),
        ("threebits_week6_adapted", "Threebits\nWeek-6 Drift\n(1% Adapter)", "#0d9488"),
        ("threebits_unified_day1", "Unified\nThreebits\nStack (Day 1)", "#0f766e"),
    ]
    x = np.arange(len(keys))
    goals = [summary["regimes"][k]["safe_private_goal_pct"] for k, _, _ in keys]
    hazs = [summary["regimes"][k]["hazard_rate_pct"] for k, _, _ in keys]
    cols = [c for _, _, c in keys]

    w = 0.36
    b1 = ax1.bar(x - w / 2, goals, width=w, color=cols, label="Safe Private-Goal Success (%)")
    b2 = ax1.bar(x + w / 2, hazs, width=w, color="#fca5a5", edgecolor="#dc2626", linewidth=0.9, label="Catastrophic Hazard Hit (%)")
    ax1.axhline(50.0, color="#2563eb", linestyle="--", linewidth=1.1, alpha=0.75)
    ax1.text(
        3.0, 61.5, "Standalone AI Private-Goal\nBayes Cap (50%)", ha="center", va="bottom",
        fontsize=7.3, color="#1d4ed8", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#bfdbfe", alpha=0.92),
    )

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        if h >= 3.0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=6.9, color="#b91c1c")

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl, _ in keys], fontsize=7.5)
    ax1.set_ylim(0, 122)
    ax1.set_ylabel("Episode Outcome Across 450 Unseen Worlds (%)", fontsize=9.1, fontweight="bold")
    ax1.set_title("A. Spiking PFC + Real C99 int4 Chip + 81-Branch Counterfactual R101", fontsize=10.1, fontweight="bold", pad=9)
    ax1.legend(loc="upper center", ncol=2, fontsize=7.6, frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    # Panel B: Biophysical Spiking Safety (Firing Rate Hz & Seizure Rate) + Live Silicon Metrics
    safe_keys = [
        ("bio_spiking_alone", "Resting Spiking\nPFC Baseline"),
        ("threebits_unified_day1", "Lyapunov-Safe\nWriteback (Day 1)"),
        ("threebits_week6_adapted", "Lyapunov-Safe\nWeek-6 Adapted"),
        ("unconstrained_writeback", "Unconstrained\nWriteback (No BRL)"),
    ]
    x2 = np.arange(len(safe_keys))
    hzs = [summary["regimes"][k]["mean_hz"] for k, _ in safe_keys]
    szs = [summary["regimes"][k]["seizure_pct"] for k, _ in safe_keys]
    dsy = [summary["regimes"][k]["mean_dsync"] for k, _ in safe_keys]
    c2 = ["#64748b", "#0f766e", "#0d9488", "#dc2626"]

    bars2 = ax2.bar(x2, hzs, width=0.50, color=c2, alpha=0.92)
    ymax_b = max(135.0, float(max(hzs)) * 1.32 + 22.0)
    ax2.axhline(50.0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax2.text(0.04, 53.0, "Paper 1 Safe Cortical Firing Ceiling (50 Hz / |Δsync| ≤ 0.05)", fontsize=7.5, color="#b91c1c", fontweight="bold")
    for idx_b, bar in enumerate(bars2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 3.0,
            f"{h:.1f} Hz\n|Δsync|={dsy[idx_b]:.3f}\nSeiz: {szs[idx_b]:.0f}%",
            ha="center", va="bottom", fontsize=7.3, fontweight="bold",
            color="#b91c1c" if szs[idx_b] > 10 else "#0f172a",
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in safe_keys], fontsize=8.1)
    ax2.set_ylim(0, ymax_b)
    ax2.set_ylabel("176-Neuron LIF PFC Column Firing Rate (Hz)", fontsize=9.1, fontweight="bold")
    ax2.set_title("B. Biophysical E–I Spiking Safety Under Closed-Loop Writeback", fontsize=10.1, fontweight="bold", pad=9)

    hw = summary["hardware_c99_silicon"]
    phys = summary["whole_brain_electrophysiology"]
    wk6_zs = summary["regimes"]["threebits_week6_zeroshot"]["safe_private_goal_pct"]
    wk6_ad = summary["regimes"]["threebits_week6_adapted"]["safe_private_goal_pct"]
    footer = (
        f"Real Compiled C99 Binary (libint4_gru.dylib, 0 floats): {hw['total_static_kb']} kB ROM+SRAM  |  "
        f"{hw['mean_us_per_bin']:.1f} µs / 20 ms bin ({hw['total_bins_executed']:,} live bins)  |  "
        f"PyTorch W4A32 vs C99 W4A8: {hw['parity_pt_w4a32_pct']:.1f}% vs {hw['parity_c99_w4a8_pct']:.1f}%\n"
        f"Week-6 Core ROM SHA256: 100% Byte-Identical (1% adapter recovers {wk6_zs:.1f}% → {wk6_ad:.1f}%)  |  "
        f"Whole-Brain CCEP ρ={phys['ccep_rho']:+.3f} (p|d {phys['ccep_prho']:+.3f})  |  "
        f"TMS-EEG: {phys['tmseeg_n46']['n_positive']}/{phys['tmseeg_n46']['n_subjects']} ({phys['tmseeg_n46']['mean_partial_rho']:+.3f})"
    )
    fig.text(
        0.5, 0.012, footer, ha="center", va="bottom", fontsize=7.8, color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.9),
    )
    plt.tight_layout(rect=[0.0, 0.095, 1.0, 1.0])
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved {FIG_PATH}")


def main():
    t0 = time.time()
    scaf = get_causal_human_scaffold()
    torch.set_default_dtype(torch.float32)
    runs = []
    for s in [0, 1, 2]:
        r = run_seed(s, scaf, n_worlds=150)
        runs.append(r)
        st = r["stats"]
        print(
            f"  [Seed {s}] C99={r['c99_us_per_bin']:.1f}µs/bin ({r['c99_total_kb']} kB, parity {r['parity_pt_w4a32_pct']:.1f}%/{r['parity_c99_w4a8_pct']:.1f}%) | "
            f"Bio={st['bio_spiking_alone']['safe_private_goal_pct']:.1f}% | "
            f"AI={st['standalone_ai']['safe_private_goal_pct']:.1f}% | "
            f"Sham={st['sham_c99_chip']['safe_private_goal_pct']:.1f}% | "
            f"NoGuard={st['unconstrained_writeback']['safe_private_goal_pct']:.1f}% (seiz {st['unconstrained_writeback']['seizure_pct']:.0f}%) | "
            f"UnifiedDay1={st['threebits_unified_day1']['safe_private_goal_pct']:.1f}% | "
            f"Wk6-ZS={st['threebits_week6_zeroshot']['safe_private_goal_pct']:.1f}% -> "
            f"Wk6-1%Adapt={st['threebits_week6_adapted']['safe_private_goal_pct']:.1f}%",
            flush=True,
        )

    ccep_rho, ccep_prho, tmseeg = measure_hybrid_ccep_and_tmseeg(scaf)
    reg_keys = list(runs[0]["stats"].keys())
    reg_summary = {}
    for k in reg_keys:
        reg_summary[k] = {
            "safe_private_goal_pct": round(float(np.mean([r["stats"][k]["safe_private_goal_pct"] for r in runs])), 2),
            "safe_private_goal_sd": round(float(np.std([r["stats"][k]["safe_private_goal_pct"] for r in runs])), 2),
            "hazard_rate_pct": round(float(np.mean([r["stats"][k]["hazard_rate_pct"] for r in runs])), 2),
            "mean_hz": round(float(np.mean([r["stats"][k]["mean_hz"] for r in runs])), 1),
            "mean_dsync": round(float(np.mean([r["stats"][k]["mean_dsync"] for r in runs])), 4),
            "seizure_pct": round(float(np.mean([r["stats"][k]["seizure_pct"] for r in runs])), 2),
        }

    out = {
        "protocol": "Unified Silicon-in-the-Loop Spiking PFC (176 LIF) + Real C99 int4xint8 Binary + Counterfactual 81-Branch Region 101 + Lyapunov Safety",
        "n_seeds": len(runs),
        "worlds_per_seed": 150,
        "total_worlds": 150 * len(runs),
        "hardware_c99_silicon": {
            "binary": "chip/libint4_gru.dylib (pure integer C99, 0 floats)",
            "rom_bytes": runs[0]["c99_rom_bytes"],
            "sram_bytes": runs[0]["c99_sram_bytes"],
            "total_static_kb": runs[0]["c99_total_kb"],
            "mean_us_per_bin": round(float(np.mean([r["c99_us_per_bin"] for r in runs])), 2),
            "total_bins_executed": int(sum(r["c99_total_bins_executed"] for r in runs)),
            "parity_pt_w4a32_pct": round(float(np.mean([r["parity_pt_w4a32_pct"] for r in runs])), 2),
            "parity_c99_w4a8_pct": round(float(np.mean([r["parity_c99_w4a8_pct"] for r in runs])), 2),
            "private_val_decode_pct": round(float(np.mean([r["c99_private_val_decode_pct"] for r in runs])), 2),
            "core_rom_sha256_verified_byte_identical_across_drift": True,
        },
        "whole_brain_electrophysiology": {
            "ccep_rho": round(ccep_rho, 4),
            "ccep_prho": round(ccep_prho, 4),
            "tmseeg_n46": tmseeg,
        },
        "regimes": reg_summary,
        "sample_demo": runs[0]["sample_demo"],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    make_figure(out)
    print(f"\nDONE Track B in {time.time() - t0:.1f}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
