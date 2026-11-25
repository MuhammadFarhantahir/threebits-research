"""Empirical Test of Behavioral Reverse-Training vs. Anatomical & Causal Constraints on Real Human Data.

Directly tests the inverse-problem / non-identifiability question on real human datasets:
  Can reverse-training a 100-region whole-brain network purely from REAL human behavioral outputs
  (OpenNeuro ds008037: 15 healthy adults, 3,744 working-memory trials across K = 2, 4, 6)
  recover true human cortical wiring, causal electrical propagation, and scalp EEG topography?

Four controlled regimes (identical 100-region recurrent cortical dynamics & behavioral loss, 3 seeds):
  1. Pure Behavior Reverse-Training (Free random W, zero anatomy / zero CCEP)
  2. Anatomy-Initialized Free Reverse-Training (W initialized at HCP dMRI, unconstrained backprop)
  3. Anatomy-Anchored Behavior Training (HCP white-matter mask + L2 anchor)
  4. Causal-CCEP + 19-PET Anchored Behavior Training (W_causal mask + L2 anchor)

Evaluated strictly zero-shot on 4 real human benchmarks:
  A. Held-out real human behavioral recall curve (ds008037 K = 2, 4, 6; R^2 / MAE)
  B. Real 74-patient intracranial CCEP propagation (ds004080, 3,219 non-SOZ pairs; rho & partial rho | d)
  C. Real 46-adult healthy non-invasive TMS-EEG topography (ds008037; partial rho | coil dist & #/46)
  D. Real 15-adult 62-channel scalp WM Load ERP (K = 6 - K = 2, 0-250 ms; Spearman rho & #/15 adults)

Run (~18 s):
    .venv/bin/python -u test_behavior_reverse_training.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from functional_brain_emulator import get_causal_human_scaffold
from human_brain_model import load as load_hcp, untuned
from laminar_spiking_sleep_eeg import eval_zero_shot_tmseeg_n46
from validate_simulator_ccep import partial_spearman
from validate_simulator_tmseeg import load_eeg_1005_mni

ROOT = Path(__file__).resolve().parent
WM_EEG_DIR = ROOT / "data" / "tmseeg_ds008037" / "wm_features"
OUT_JSON = ROOT / "test_behavior_reverse_training.json"


def load_real_human_wm_behavior_and_eeg(coords):
    """Load real 15-adult ds008037 continuous behavioral recall trials + 62-ch scalp WM Load ERP."""
    wm_files = sorted(glob.glob(str(WM_EEG_DIR / "sub-*_wm.npz")))
    all_k, all_acc, all_err = [], [], []
    emp_erp_diff_0 = []
    chs = None
    for f in wm_files:
        d = np.load(f, allow_pickle=True)
        ok = d["ok"].astype(bool) & np.isfinite(d["errs"])
        ss = d["set_sizes"][ok].astype(np.int64)
        errs = np.abs(d["errs"][ok].astype(np.float64))
        # Recall accuracy (< 30 deg precision) & normalized continuous precision cos(err)
        acc = (errs < 30.0).astype(np.float64)
        all_k.append(ss)
        all_acc.append(acc)
        all_err.append(errs)

        erp_abs = np.abs(d["erp"][ok])  # (N, 6, 62)
        chs = [str(c) for c in d["channels"]]
        m2, m6 = (ss == 2), (ss == 6)
        emp_erp_diff_0.append(erp_abs[m6, 0].mean(axis=0) - erp_abs[m2, 0].mean(axis=0))

    K_vec = np.concatenate(all_k)
    acc_vec = np.concatenate(all_acc)
    err_vec = np.concatenate(all_err)

    elec_map = load_eeg_1005_mni()
    keep = [i for i, c in enumerate(chs) if c.upper() in elec_map]
    e_coords = np.array([elec_map[chs[i].upper()] for i in keep], dtype=np.float64)
    d_ec = np.linalg.norm(e_coords[:, None, :] - coords[None, :, :], axis=-1)
    L = 1.0 / (d_ec ** 2 + 24.0 ** 2)
    L = L / L.sum(axis=1, keepdims=True)
    emp_erp_subjs = [arr[keep] for arr in emp_erp_diff_0]
    emp_erp_mean = np.mean(emp_erp_subjs, axis=0)
    return {
        "n_adults": len(wm_files),
        "n_trials": int(len(K_vec)),
        "K": K_vec,
        "acc": acc_vec,
        "err": err_vec,
        "emp_acc_by_k": {int(k): float(acc_vec[K_vec == k].mean()) for k in (2, 4, 6)},
        "L": L,
        "emp_erp_mean": emp_erp_mean,
        "emp_erp_subjs": emp_erp_subjs,
    }


class ReverseTrainedBrain100(nn.Module):
    """100-region recurrent cortical model reverse-trained from real human WM behavioral trials."""

    def __init__(self, regime, W_hcp, W_causal, d_causal, vis_idx, dors_idx, motor_idx, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.regime = regime
        self.vis_idx = vis_idx
        self.dors_idx = dors_idx
        self.motor_idx = motor_idx
        self.n = 100

        # Normalize base connectivity scales so all 4 regimes start with stable recurrent gain
        w_hcp_t = torch.tensor(W_hcp, dtype=torch.float32)
        w_ccep_t = torch.tensor(W_causal, dtype=torch.float32)
        d_ccep_t = torch.tensor(d_causal, dtype=torch.float32)
        self.register_buffer("W_hcp", w_hcp_t)
        self.register_buffer("W_ccep", w_ccep_t)
        self.register_buffer("d_ccep", d_ccep_t)
        self.register_buffer("mask_hcp", (w_hcp_t > 0).float())
        self.register_buffer("mask_ccep", (w_ccep_t > 0).float())

        if regime == "behavior_free_random":
            # Completely free 100x100 synaptic matrix (zero anatomy, zero CCEP)
            self.W_param = nn.Parameter(torch.randn(100, 100) * 0.08)
            self.log_d = nn.Parameter(torch.zeros(100))
        elif regime == "behavior_free_hcp_init":
            # Initialized at real HCP dMRI wiring, then freely modified by behavioral gradients
            self.W_param = nn.Parameter(w_hcp_t.clone())
            self.log_d = nn.Parameter(torch.zeros(100))
        elif regime == "behavior_anatomy_anchored":
            # Multiplicative plasticity constrained to real HCP white-matter tracts + L2 anchor
            self.delta_g = nn.Parameter(torch.zeros(100, 100))
            self.log_d = nn.Parameter(torch.zeros(100))
        elif regime == "behavior_ccep_anchored":
            # Multiplicative plasticity constrained to CCEP+19-PET causal connectome + L2 anchor
            self.delta_g = nn.Parameter(torch.zeros(100, 100))
            self.log_d = nn.Parameter(torch.log(d_ccep_t.clamp(min=0.1)))
        else:
            raise ValueError(regime)

        # Sensory encoder (stimulus set size K + interference noise -> visual/dorsal cortex) & motor readout
        self.sens_in = nn.Linear(3, len(vis_idx) + len(dors_idx))
        self.motor_head = nn.Sequential(nn.Linear(len(motor_idx), 16), nn.Tanh(), nn.Linear(16, 1))
        self.in_idx = list(vis_idx) + list(dors_idx)

    def effective_W_and_d(self):
        eye_mask = 1.0 - torch.eye(100, device=self.W_hcp.device)
        if self.regime in ("behavior_free_random", "behavior_free_hcp_init"):
            W = F.softplus(self.W_param) * eye_mask
            d = F.softplus(self.log_d) + 0.5
        elif self.regime == "behavior_anatomy_anchored":
            W = self.W_hcp * torch.exp(self.delta_g.clamp(-2.0, 2.0)) * self.mask_hcp * eye_mask
            d = F.softplus(self.log_d) + 0.5
        else:
            W = self.W_ccep * torch.exp(self.delta_g.clamp(-2.0, 2.0)) * self.mask_ccep * eye_mask
            d = torch.exp(self.log_d.clamp(-3.0, 3.0))
        return W, d

    def forward(self, k_oh, trial_noise):
        # 4-step recurrent cortical propagation from Visual+Dorsal encoding -> Somatomotor behavioral recall logit
        B = k_oh.shape[0]
        W, d = self.effective_W_and_d()
        # Diffusive stability-normalized recurrent step (matches Hopf linear operator K = diag(W1) - W + diag(d))
        deg = W.sum(dim=1)
        scale = 1.0 / (deg + d + 1e-4)
        h = torch.zeros(B, 100, device=k_oh.device)
        drive = self.sens_in(k_oh)
        for step in range(4):
            ext = torch.zeros(B, 100, device=k_oh.device)
            if step < 2:
                ext[:, self.in_idx] = drive + trial_noise[:, step, :]
            # Diffusive cortical propagation step
            h_next = (h @ W.T + ext) * scale[None, :]
            h = 0.4 * h + 0.6 * torch.tanh(h_next)
        logit = self.motor_head(h[:, self.motor_idx]).squeeze(-1)
        return logit, h

    def causal_greens_matrix(self):
        with torch.no_grad():
            W, d = self.effective_W_and_d()
            Wn = W.cpu().numpy().astype(np.float64)
            dn = d.cpu().numpy().astype(np.float64)
            K = np.diag(Wn.sum(1)) - Wn + np.diag(dn)
            G = np.linalg.inv(K).T  # [src, dst]
            np.fill_diagonal(G, 0.0)
            return G


def run_regime(regime, scaf, wm_data, W_hcp, seeds=(0, 1, 2), steps=450):
    k_vals = wm_data["K"]
    acc_vals = wm_data["acc"]
    k_map = {2: 0, 4: 1, 6: 2}
    k_idx_all = np.array([k_map[int(k)] for k in k_vals], dtype=np.int64)

    # Fixed 80/20 held-out behavioral split across all 3,744 real human trials
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(len(k_vals))
    n_te = len(k_vals) // 5
    te_i, tr_i = perm[:n_te], perm[n_te:]

    vis_idx = list(scaf["vis_idx"])
    dors_idx = list(scaf["dors_idx"])
    motor_idx = list(scaf["mot_idx"])
    in_dim = len(vis_idx) + len(dors_idx)

    metrics = []
    for sd in seeds:
        torch.manual_seed(sd)
        rng = np.random.default_rng(sd + 100)
        model = ReverseTrainedBrain100(
            regime, W_hcp, scaf["W_causal"], scaf["d_causal"],
            vis_idx, dors_idx, motor_idx, seed=sd,
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.02)

        for step in range(steps):
            bi = rng.choice(tr_i, size=128, replace=True)
            k_oh = F.one_hot(torch.tensor(k_idx_all[bi]), num_classes=3).float()
            y = torch.tensor(acc_vals[bi], dtype=torch.float32)
            # Load-proportional sensory encoding noise (K=2 low, K=4 med, K=6 high)
            k_scale = torch.tensor(k_vals[bi] / 6.0, dtype=torch.float32)[:, None, None]
            tnoise = torch.randn(128, 2, in_dim) * (0.25 * k_scale)

            opt.zero_grad()
            logit, _ = model(k_oh, tnoise)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if regime in ("behavior_anatomy_anchored", "behavior_ccep_anchored"):
                loss = loss + 0.05 * (model.delta_g ** 2).mean()
            loss.backward()
            opt.step()

        # 1. Held-out real human behavioral recall curve match (K = 2, 4, 6)
        model.eval()
        with torch.no_grad():
            k_oh_te = F.one_hot(torch.tensor(k_idx_all[te_i]), num_classes=3).float()
            k_sc_te = torch.tensor(k_vals[te_i] / 6.0, dtype=torch.float32)[:, None, None]
            tnoise_te = torch.randn(len(te_i), 2, in_dim) * (0.25 * k_sc_te)
            probs_te = torch.sigmoid(model(k_oh_te, tnoise_te)[0]).numpy()
        pred_by_k = {k: float(probs_te[k_vals[te_i] == k].mean()) for k in (2, 4, 6)}
        true_by_k = {k: float(acc_vals[te_i][k_vals[te_i] == k].mean()) for k in (2, 4, 6)}
        beh_mae_pp = float(np.mean([abs(pred_by_k[k] - true_by_k[k]) * 100.0 for k in (2, 4, 6)]))

        # 2. Zero-shot 74-patient human CCEP causal propagation (ds004080, 3,219 non-SOZ pairs)
        G = model.causal_greens_matrix()
        valid = scaf["valid"]
        ccep_rho = float(spearmanr(G[valid], scaf["prob"][valid]).statistic)
        ccep_part = float(partial_spearman(G[valid], scaf["prob"][valid], scaf["dist_mat"][valid]))

        # 3. Zero-shot 46-adult healthy non-invasive TMS-EEG topography (ds008037)
        tms = eval_zero_shot_tmseeg_n46(G, scaf)

        # 4. Zero-shot 15-adult 62-ch scalp WM Load ERP topography (K = 6 - K = 2)
        vis_mask = np.zeros(100); vis_mask[vis_idx] = 1.0
        dors_mask = np.zeros(100); dors_mask[dors_idx] = 1.0
        sim_parcel_erp = G.T @ (vis_mask + 0.5 * dors_mask)
        sim_scalp_erp = wm_data["L"] @ sim_parcel_erp
        erp_rho = float(spearmanr(sim_scalp_erp, wm_data["emp_erp_mean"]).statistic)
        erp_pos = int(sum(spearmanr(sim_scalp_erp, s).statistic > 0 for s in wm_data["emp_erp_subjs"]))

        metrics.append({
            "beh_mae_pp": beh_mae_pp,
            "pred_k2": pred_by_k[2] * 100.0,
            "pred_k4": pred_by_k[4] * 100.0,
            "pred_k6": pred_by_k[6] * 100.0,
            "ccep_rho": ccep_rho,
            "ccep_partial": ccep_part,
            "tmseeg_partial": float(tms["mean_partial_rho"]),
            "tmseeg_pos_46": int(tms["n_positive"]),
            "wm_erp_rho": erp_rho,
            "wm_erp_pos_15": erp_pos,
        })

    return {
        "beh_mae_pp": round(float(np.mean([m["beh_mae_pp"] for m in metrics])), 2),
        "pred_curve_pct": [
            round(float(np.mean([m["pred_k2"] for m in metrics])), 1),
            round(float(np.mean([m["pred_k4"] for m in metrics])), 1),
            round(float(np.mean([m["pred_k6"] for m in metrics])), 1),
        ],
        "ccep_rho": round(float(np.mean([m["ccep_rho"] for m in metrics])), 4),
        "ccep_partial": round(float(np.mean([m["ccep_partial"] for m in metrics])), 4),
        "tmseeg_partial": round(float(np.mean([m["tmseeg_partial"] for m in metrics])), 4),
        "tmseeg_pos_46": round(float(np.mean([m["tmseeg_pos_46"] for m in metrics])), 1),
        "wm_erp_rho": round(float(np.mean([m["wm_erp_rho"] for m in metrics])), 4),
        "wm_erp_pos_15": round(float(np.mean([m["wm_erp_pos_15"] for m in metrics])), 1),
    }


def main():
    torch.set_default_dtype(torch.float32)
    scaf = get_causal_human_scaffold()
    wm_data = load_real_human_wm_behavior_and_eeg(scaf["coords"])
    wiring, real_fc, _ = load_hcp(str(ROOT / "data" / "hcp_connectome"))
    g0 = untuned(wiring, real_fc, np.triu_indices(100, 1))
    W_hcp = g0 * wiring

    print(
        f"=== Behavioral Reverse-Training Experiment ({wm_data['n_adults']} healthy adults, "
        f"{wm_data['n_trials']:,} real ds008037 WM trials) ===",
        flush=True,
    )
    ek = wm_data["emp_acc_by_k"]
    print(f"  Empirical human recall (<30 deg): K=2 {ek[2]*100:.1f}% | K=4 {ek[4]*100:.1f}% | K=6 {ek[6]*100:.1f}%\n", flush=True)

    regimes = [
        ("behavior_free_random", "1. Pure Behavior Reverse-Train (Free random W, 0 anatomy)"),
        ("behavior_free_hcp_init", "2. Behavior Reverse-Train (HCP-init, free unconstrained W)"),
        ("behavior_anatomy_anchored", "3. Anatomy-Anchored Behavior Train (HCP tracts + L2)"),
        ("behavior_ccep_anchored", "4. Causal-CCEP + 19-PET Anchored Behavior Train"),
    ]
    out = {"empirical_human_curve_pct": [round(ek[2]*100, 1), round(ek[4]*100, 1), round(ek[6]*100, 1)], "regimes": {}}
    for key, label in regimes:
        res = run_regime(key, scaf, wm_data, W_hcp)
        out["regimes"][key] = res
        print(
            f"  {label:<58}\n"
            f"     Behavior MAE={res['beh_mae_pp']:4.2f} pp {res['pred_curve_pct']} | "
            f"74-pt CCEP rho={res['ccep_rho']:+.4f} (part {res['ccep_partial']:+.4f}) | "
            f"46-adult TMS-EEG={res['tmseeg_partial']:+.4f} ({res['tmseeg_pos_46']:.0f}/46) | "
            f"15-adult 62-ch WM ERP rho={res['wm_erp_rho']:+.4f} ({res['wm_erp_pos_15']:.0f}/15)",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
