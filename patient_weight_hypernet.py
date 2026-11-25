"""Graph-Transformer Connectome Weight Hypernetwork (Threebits Research).

Tests whether a Graph-Attention / Transformer Hypernetwork conditioned on real human multi-scale
cortical biology (19 PET receptors, 6-band HCP MEG, Mesulam hierarchy, 3D anatomy, white-matter
topology) + individual patient age (ages 4–51) can GENERATE patient-specific causal brain weights
(W_patient, d_patient, velocity_patient) in a single forward pass — beating static group models!

Strictly evaluated across ALL 74/74 unseen neurosurgical patients (5-fold Leave-Patients-Out CV,
OpenNeuro ds004080, 277,638 raw non-SOZ electrode pairs) AND strictly zero-shot out-of-hospital on
780-patient multi-center F-TRACT SEEG + 46-adult healthy TMS-EEG (OpenNeuro ds008037):
  1. Static Baseline: fMRI-tuned Hopf (observational)
  2. Static Linear Baseline (v1.3): 47-param Multi-Scale Bio+CCEP + Bipolar VTA (LPO)
  3. Graph-Transformer HyperNet (Utrecht-only LPO) — generates age-conditioned W(age), d(age), v(age)
  4. Graph-Transformer HyperNet (780-pt F-TRACT Pretrained + Utrecht LPO) — multi-hospital foundation pretrain
  5. Permuted-Biology Graph-Transformer Control (LPO) — identical architecture, shuffled regional biology

Run (~65 s):
    .venv/bin/python -u patient_weight_hypernet.py
"""
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from ccep_latency_waveform_model import compute_graph_hops_and_dist, load_ftract_360
from human_brain_model import load as load_hcp, tune, untuned
from laminar_spiking_sleep_eeg import eval_zero_shot_tmseeg_n46
from multiscale_brain_twin import fit_multiscale, load_biology_features
from validate_patient_level_ccep import load_per_patient_ccep, score_patient
from validate_simulator_ccep import greens_function, load_centroids, partial_spearman

torch.set_default_dtype(torch.float32)
ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "patient_weight_hypernet.json"
OUT_FIG = ROOT / "figures" / "fig_weight_hypernet.png"


class GraphTransformerBlock(nn.Module):
    """Pre-norm Multi-Head Self-Attention with learnable structural + spatial pair bias (Graphormer-style)."""

    def __init__(self, d_model=48, n_heads=4, dropout=0.10):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias):
        # x: (N, d_model), attn_bias: (n_heads, N, N)
        N = x.shape[0]
        h = self.norm1(x)
        qkv = self.qkv(h).view(N, 3, self.n_heads, self.d_head)
        q = qkv[:, 0].transpose(0, 1)  # (H, N, Dh)
        k = qkv[:, 1].transpose(0, 1)
        v = qkv[:, 2].transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head) + attn_bias
        attn = self.drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(0, 1).contiguous().view(N, self.d_model)
        x = x + self.drop(self.proj(out))
        x = x + self.ffn(self.norm2(x))
        return x


class ConnectomeWeightHyperNet(nn.Module):
    """Graph-Transformer Hypernetwork generating patient-specific (W_patient, d_patient, speed_patient)."""

    def __init__(self, wiring, g0, node_feats, edge_feats, pair_bias_feats, d_model=48, n_heads=4, n_layers=2,
                 use_residuals=True):
        super().__init__()
        self.n = len(wiring)
        self.use_residuals = use_residuals
        r, c = np.nonzero((wiring > 0) & ~np.eye(self.n, dtype=bool))
        self.register_buffer("rows", torch.tensor(r, dtype=torch.long))
        self.register_buffer("cols", torch.tensor(c, dtype=torch.long))
        self.register_buffer("base_w", torch.tensor(wiring[r, c], dtype=torch.float32))
        self.register_buffer("node_feats", torch.tensor(node_feats, dtype=torch.float32))      # (100, F_n)
        self.register_buffer("edge_feats", torch.tensor(edge_feats[r, c], dtype=torch.float32))  # (E, F_e)
        self.register_buffer("pair_bias_feats", torch.tensor(pair_bias_feats, dtype=torch.float32))  # (100, 100, F_p)
        self.log_g0 = float(np.log(g0))

        # Node token projection: [node_bio (F_n), age_norm, inv_age] -> d_model
        self.node_in = nn.Sequential(
            nn.Linear(node_feats.shape[1] + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Learnable spatial + structural graph attention bias (F_p -> n_heads)
        self.pair_bias_proj = nn.Sequential(
            nn.Linear(pair_bias_feats.shape[-1], n_heads * 2),
            nn.GELU(),
            nn.Linear(n_heads * 2, n_heads),
        )
        self.blocks = nn.ModuleList([GraphTransformerBlock(d_model, n_heads, dropout=0.10) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # Weight-generation heads (strictly physical: directed edge gain, regional damping, stim susceptibility)
        self.g_offset = nn.Parameter(torch.tensor(self.log_g0, dtype=torch.float32))
        self.edge_head = nn.Sequential(
            nn.Linear(2 * d_model + edge_feats.shape[-1], d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.damp_head = nn.Linear(d_model, 1)
        self.stim_head = nn.Linear(d_model, 1)

        # Patient-specific developmental axonal conduction-velocity & synaptic-delay head!
        # Outputs per-region effective conduction speed v_i(age) (m/s) & synaptic onset tau_i(age) (ms)
        self.vel_head = nn.Sequential(nn.Linear(d_model, 24), nn.GELU(), nn.Linear(24, 1))
        self.tau_head = nn.Sequential(nn.Linear(d_model, 24), nn.GELU(), nn.Linear(24, 1))
        self.hop_delay = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.wdist_delay = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        # Optional L2-anchored residual table (shared structural refinement on top of generated weights)
        if use_residuals:
            self.delta_g = nn.Parameter(torch.zeros(len(r), dtype=torch.float32))
            self.delta_d = nn.Parameter(torch.zeros(self.n, dtype=torch.float32))
        else:
            self.register_parameter("delta_g", None)
            self.register_parameter("delta_d", None)

        self.logit_scale = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        self.logit_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def encode_brain(self, age_years=25.0):
        """Run 2-layer Graph Transformer conditioned on patient age to get contextual brain embeddings Z."""
        age_norm = (float(age_years) - 20.0) / 15.0
        inv_age = (1.0 / (float(age_years) + 5.0) - 0.04) / 0.03
        age_tok = torch.tensor([[age_norm, inv_age]], dtype=torch.float32, device=self.node_feats.device).expand(self.n, 2)
        x = self.node_in(torch.cat([self.node_feats, age_tok], dim=-1))
        # Pair attention bias: (N, N, H) -> (H, N, N)
        pb = self.pair_bias_proj(self.pair_bias_feats).permute(2, 0, 1)
        for blk in self.blocks:
            x = blk(x, pb)
        return self.final_norm(x)

    def generate_patient_weights(self, age_years=25.0):
        """Generate patient-specific physical connectome (W, d, stim_gain, G, v_region, tau_region)."""
        Z = self.encode_brain(age_years)
        # Edge (rows=dst, cols=src): source embedding Z[cols], target embedding Z[rows], edge_feats
        e_in = torch.cat([Z[self.cols], Z[self.rows], self.edge_feats], dim=-1)
        log_gain = self.g_offset + 1.5 * torch.tanh(self.edge_head(e_in).squeeze(-1))
        if self.use_residuals:
            log_gain = log_gain + self.delta_g
        log_gain = log_gain.clamp(-10.0, 4.0)

        W = torch.zeros(self.n, self.n, dtype=torch.float32, device=Z.device)
        W[self.rows, self.cols] = self.base_w * torch.exp(log_gain)

        log_d = 1.5 * torch.tanh(self.damp_head(Z).squeeze(-1))
        if self.use_residuals:
            log_d = log_d + self.delta_d
        d = torch.exp(log_d.clamp(-3.5, 3.5))

        stim_gain = torch.exp((1.2 * torch.tanh(self.stim_head(Z).squeeze(-1))).clamp(-2.0, 2.0))
        K = torch.diag(W.sum(1)) - W + torch.diag(d)
        G = torch.linalg.inv(K).T * stim_gain[:, None]  # G[src, dst]

        # Regional conduction velocity (m/s in [0.6, 8.0]) & synaptic delay (ms in [6, 35])
        v_reg = 0.6 + 7.4 * torch.sigmoid(self.vel_head(Z).squeeze(-1) - 1.2)
        tau_reg = 6.0 + 29.0 * torch.sigmoid(self.tau_head(Z).squeeze(-1) - 0.8)
        return W, d, G, v_reg, tau_reg

    def predict_prob_matrix(self, age_years=25.0):
        _, _, G, v_reg, tau_reg = self.generate_patient_weights(age_years)
        p = torch.sigmoid(self.logit_scale * torch.log(G.clamp(min=1e-7)) + self.logit_bias)
        return p, G, v_reg, tau_reg

    def predict_latencies(self, src_roi, dst_roi, dist_mm, hops_vec, wdist_vec, v_reg, tau_reg):
        # Symmetric harmonic/mean tract speed between source and target region
        v_pair = 0.5 * (v_reg[src_roi] + v_reg[dst_roi])
        tau_syn = 0.5 * (tau_reg[src_roi] + tau_reg[dst_roi])
        return tau_syn + dist_mm / v_pair.clamp(min=0.4) + F.softplus(self.hop_delay) * hops_vec + F.softplus(self.wdist_delay) * wdist_vec


def build_brain_graph_features(wiring, coords, hemi, F_bio, rec_cos, hier_diff):
    """Build strictly anatomical/biological node, edge, and pair-bias features (zero CCEP leakage!)."""
    dist_mm = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    hops, wdist = compute_graph_hops_and_dist(wiring, dist_mm)
    coords_z = (coords - coords.mean(0, keepdims=True)) / np.maximum(coords.std(0, keepdims=True), 1e-6)
    hemi_num = (hemi == "R").astype(np.float64)[:, None]
    deg_z = np.log1p(wiring.sum(1))
    deg_z = (deg_z - deg_z.mean()) / max(deg_z.std(), 1e-6)
    node_feats = np.column_stack([F_bio, coords_z, deg_z[:, None], hemi_num])  # (100, 16)

    log_w = np.log1p(wiring * 50.0)
    log_w = (log_w - log_w.mean()) / max(log_w.std(), 1e-6)
    d_norm = dist_mm / 100.0
    same_h = (hemi[:, None] == hemi[None, :]).astype(np.float64)
    hops_n = np.clip(hops, 0.0, 5.0) / 5.0
    edge_feats = np.stack([log_w, d_norm, rec_cos, hier_diff, same_h, hops_n], axis=-1)  # (100, 100, 6)
    pair_bias_feats = np.stack([log_w, -d_norm, rec_cos, same_h, -hops_n], axis=-1)      # (100, 100, 5)
    return node_feats, edge_feats, pair_bias_feats, dist_mm, hops, wdist


def load_ftract_schaefer100(s_coords):
    """Load 780-patient multi-center F-TRACT probability + peak latency mapped to Schaefer-100."""
    g_labels = (ROOT / "data" / "hcp_connectome" / "strucLabels_ctx_glasser_360.csv").read_text().strip().split(",")
    xyz = np.vstack([nib.load(ROOT / "data" / "hcp_connectome" / f"fsa5_{h}.gii").darrays[0].data for h in ("lh", "rh")])
    g_par = np.loadtxt(ROOT / "data" / "hcp_connectome" / "glasser_360_fsa5.csv").astype(int)
    g_cent = np.array([xyz[g_par == v].mean(0) for v in range(1, 361)])
    g_to_s = np.concatenate([cdist(g_cent[:180], s_coords[:50]).argmin(1), cdist(g_cent[180:], s_coords[50:]).argmin(1) + 50])

    def map360(M360, min_val=-1e9):
        sums = np.zeros((100, 100), dtype=np.float64)
        cnts = np.zeros((100, 100), dtype=np.int64)
        rr, cc = np.nonzero(np.isfinite(M360) & (M360 > min_val) & ~np.eye(360, dtype=bool))
        for i, j in zip(rr, cc):
            si, sj = g_to_s[i], g_to_s[j]
            if si != sj:
                sums[si, sj] += M360[i, j]
                cnts[si, sj] += 1
        out = np.full((100, 100), np.nan)
        ok = cnts > 0
        out[ok] = sums[ok] / cnts[ok]
        return out

    ft_p = map360(load_ftract_360("probability", g_labels))
    ft_lat = map360(load_ftract_360("peak_latency", g_labels), min_val=2.0)
    return ft_p, ft_lat


def train_hypernet_fold(
    wiring, g0, node_feats, edge_feats, pair_bias_feats, hops100, wdist100, s_dist,
    tr_pats, pretrain_ftract=None, pretrain_mode="joint", steps=260, seed=0, use_residuals=True,
):
    """Train ConnectomeWeightHyperNet on training patients (with optional F-TRACT pretraining).

    pretrain_mode:
      - "joint": pretrain both probability & latency on F-TRACT (tests naive cross-protocol transfer)
      - "latency_only": split-head pretraining using ONLY universal millisecond F-TRACT latencies,
        leaving the ECoG probability weights untouched until Utrecht training!
    """
    torch.manual_seed(seed)
    model = ConnectomeWeightHyperNet(
        wiring, g0, node_feats, edge_feats, pair_bias_feats,
        d_model=48, n_heads=4, n_layers=2, use_residuals=use_residuals,
    )

    # Optional Stage 0: Multi-Hospital Foundation Pretraining on independent 780-pt F-TRACT (ZERO Utrecht data!)
    if pretrain_ftract is not None:
        ft_p, ft_lat = pretrain_ftract
        m_fp = np.isfinite(ft_p) & ~np.eye(100, dtype=bool)
        m_fl = np.isfinite(ft_lat) & ~np.eye(100, dtype=bool)
        from scipy.stats import rankdata
        y_fp = np.zeros((100, 100), dtype=np.float32)
        y_fp[m_fp] = 0.05 + 0.90 * (rankdata(ft_p[m_fp]) / m_fp.sum())
        m_fp_t = torch.tensor(m_fp)
        y_fp_t = torch.tensor(y_fp[m_fp], dtype=torch.float32)

        rr_l, cc_l = np.nonzero(m_fl)
        sr_l = torch.tensor(rr_l, dtype=torch.long)
        dr_l = torch.tensor(cc_l, dtype=torch.long)
        dd_l = torch.tensor(s_dist[rr_l, cc_l], dtype=torch.float32)
        hp_l = torch.tensor(hops100[rr_l, cc_l], dtype=torch.float32)
        wd_l = torch.tensor(wdist100[rr_l, cc_l], dtype=torch.float32)
        y_fl_t = torch.tensor(ft_lat[rr_l, cc_l], dtype=torch.float32)

        opt_pre = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=2e-3)
        model.train()
        for _ in range(110):
            opt_pre.zero_grad()
            if pretrain_mode == "latency_only":
                # Split-head pretrain: ONLY update shared backbone + conduction velocity/delay heads on F-TRACT ms!
                Z = model.encode_brain(age_years=30.0)
                v_reg = 0.6 + 7.4 * torch.sigmoid(model.vel_head(Z).squeeze(-1) - 1.2)
                tau_reg = 6.0 + 29.0 * torch.sigmoid(model.tau_head(Z).squeeze(-1) - 0.8)
                pred_lat = model.predict_latencies(sr_l, dr_l, dd_l, hp_l, wd_l, v_reg, tau_reg)
                loss = F.smooth_l1_loss(pred_lat, y_fl_t, beta=5.0) * 0.05
            else:
                p_mat, _, v_reg, tau_reg = model.predict_prob_matrix(age_years=30.0)
                pm = p_mat[m_fp_t].clamp(1e-5, 1 - 1e-5)
                loss_p = F.binary_cross_entropy(pm, y_fp_t)
                pred_lat = model.predict_latencies(sr_l, dr_l, dd_l, hp_l, wd_l, v_reg, tau_reg)
                loss_l = F.smooth_l1_loss(pred_lat, y_fl_t, beta=5.0) * 0.03
                loss = loss_p + loss_l
                if use_residuals:
                    loss = loss + 0.12 * (model.delta_g ** 2).mean() + 0.12 * (model.delta_d ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt_pre.step()

    # Build age-stratified training cohorts (pediatric <13, adolescent 13–21, adult >21) + pooled cohort
    # so the Hypernetwork explicitly learns how W(age), d(age), and velocity(age) vary across ages!
    bins = [
        ("ped", 9.0, [p for p in tr_pats if np.isfinite(p["age"]) and p["age"] < 14.0]),
        ("adol", 17.5, [p for p in tr_pats if np.isfinite(p["age"]) and 14.0 <= p["age"] < 23.0]),
        ("adult", 34.0, [p for p in tr_pats if np.isfinite(p["age"]) and p["age"] >= 23.0]),
    ]
    bin_tensors = []
    for _, rep_age, plist in bins:
        if not plist:
            continue
        tr_t = np.sum([p["trials"] for p in plist], axis=0)
        tr_h = np.sum([p["hits"] for p in plist], axis=0)
        mask = (tr_t >= 5) & ~np.eye(100, dtype=bool)
        if mask.sum() < 50:
            continue
        prob = np.divide(tr_h, tr_t, out=np.zeros((100, 100)), where=tr_t > 0)
        wp = np.log1p(tr_t[mask])
        wp = wp / wp.mean()

        # Latency samples from this age cohort
        ls, ld, le, lh, lw, ly = [], [], [], [], [], []
        for p in plist:
            ok = np.isfinite(p["lat_ms"]) & (p["lat_ms"] > 12.0) & (p["lat_ms"] < 105.0) & (p["elec_dist"] >= 10.0)
            if ok.sum() > 0:
                ls.append(p["src_roi"][ok]); ld.append(p["dst_roi"][ok])
                le.append(p["elec_dist"][ok]); ly.append(p["lat_ms"][ok])
                lh.append(hops100[p["src_roi"][ok], p["dst_roi"][ok]])
                lw.append(wdist100[p["src_roi"][ok], p["dst_roi"][ok]])
        bin_tensors.append({
            "age": float(np.mean([p["age"] for p in plist])),
            "mask": torch.tensor(mask),
            "y": torch.tensor(prob[mask], dtype=torch.float32),
            "w": torch.tensor(wp, dtype=torch.float32),
            "lat": (
                torch.tensor(np.concatenate(ls), dtype=torch.long),
                torch.tensor(np.concatenate(ld), dtype=torch.long),
                torch.tensor(np.concatenate(le), dtype=torch.float32),
                torch.tensor(np.concatenate(lh), dtype=torch.float32),
                torch.tensor(np.concatenate(lw), dtype=torch.float32),
                torch.tensor(np.concatenate(ly), dtype=torch.float32),
            ) if ls else None,
        })

    # Also build pooled training mask (strong high-SNR target across all 59 training patients)
    tr_trials_all = np.sum([p["trials"] for p in tr_pats], axis=0)
    tr_hits_all = np.sum([p["hits"] for p in tr_pats], axis=0)
    m_all = (tr_trials_all >= 10) & ~np.eye(100, dtype=bool)
    p_all = np.divide(tr_hits_all, tr_trials_all, out=np.zeros((100, 100)), where=tr_trials_all > 0)
    w_all = np.log1p(tr_trials_all[m_all]); w_all = w_all / w_all.mean()
    m_all_t = torch.tensor(m_all)
    y_all_t = torch.tensor(p_all[m_all], dtype=torch.float32)
    w_all_t = torch.tensor(w_all, dtype=torch.float32)
    mean_age = float(np.nanmean([p["age"] for p in tr_pats]))

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-4)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        # Pooled CCEP propagation loss at cohort mean age
        pm_all, _, _, _ = model.predict_prob_matrix(age_years=mean_age)
        pm_c = pm_all[m_all_t].clamp(1e-5, 1 - 1e-5)
        bce_all = -(y_all_t * torch.log(pm_c) + (1.0 - y_all_t) * torch.log(1.0 - pm_c))
        loss = (w_all_t * bce_all).mean()

        # Age-conditioned patient propagation + millisecond latency loss across pediatric / adolescent / adult bins
        for b in bin_tensors:
            pm_b, _, v_b, tau_b = model.predict_prob_matrix(age_years=b["age"])
            pb_c = pm_b[b["mask"]].clamp(1e-5, 1 - 1e-5)
            bce_b = -(b["y"] * torch.log(pb_c) + (1.0 - b["y"]) * torch.log(1.0 - pb_c))
            loss = loss + 0.25 * (b["w"] * bce_b).mean()
            if b["lat"] is not None:
                sr, dr, ed, hp, wd, yl = b["lat"]
                pred_l = model.predict_latencies(sr, dr, ed, hp, wd, v_b, tau_b)
                loss = loss + 0.035 * F.smooth_l1_loss(pred_l, yl, beta=4.0)

        loss = loss + 0.02 * (model.g_offset - model.log_g0) ** 2
        if use_residuals:
            loss = loss + 0.08 * (model.delta_g ** 2).mean() + 0.08 * (model.delta_d ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

    model.eval()
    return model


def eval_patient_latency_hypernet(model, pat, hops100, wdist100):
    ok = np.isfinite(pat["lat_ms"]) & (pat["lat_ms"] > 12.0) & (pat["lat_ms"] < 105.0) & (pat["elec_dist"] >= 10.0)
    if ok.sum() < 20:
        return None
    age = float(pat["age"]) if np.isfinite(pat["age"]) else 22.0
    with torch.no_grad():
        _, _, _, v_reg, tau_reg = model.generate_patient_weights(age_years=age)
        sr = torch.tensor(pat["src_roi"][ok], dtype=torch.long)
        dr = torch.tensor(pat["dst_roi"][ok], dtype=torch.long)
        ed = torch.tensor(pat["elec_dist"][ok], dtype=torch.float32)
        hp = torch.tensor(hops100[pat["src_roi"][ok], pat["dst_roi"][ok]], dtype=torch.float32)
        wd = torch.tensor(wdist100[pat["src_roi"][ok], pat["dst_roi"][ok]], dtype=torch.float32)
        pred = model.predict_latencies(sr, dr, ed, hp, wd, v_reg, tau_reg).numpy()
    y = pat["lat_ms"][ok]
    return {"mae_ms": float(np.mean(np.abs(pred - y))), "rho": float(spearmanr(pred, y).statistic)}


def main():
    print("=== Graph-Transformer Connectome Weight Hypernetwork (74-Patient LPO + F-TRACT + TMS-EEG) ===", flush=True)
    labels, coords, hemi = load_centroids()
    patients = load_per_patient_ccep(coords, hemi, exclude_soz=True)
    wiring, real_fc, _ = load_hcp(str(ROOT / "data" / "hcp_connectome"))
    upper = np.triu_indices(100, 1)
    g0 = untuned(wiring, real_fc, upper)
    F_bio, rec_cos, hier_diff, *_ = load_biology_features()
    node_f, edge_f, pb_f, s_dist, hops100, wdist100 = build_brain_graph_features(wiring, coords, hemi, F_bio, rec_cos, hier_diff)

    # Permuted biology features for matched-capacity negative control
    perm = np.random.default_rng(101).permutation(100)
    node_fp, edge_fp, pb_fp, _, _, _ = build_brain_graph_features(
        wiring, coords, hemi, F_bio[perm], rec_cos[np.ix_(perm, perm)], hier_diff[np.ix_(perm, perm)],
    )

    ft_prob_100, ft_lat_100 = load_ftract_schaefer100(coords)

    # Same deterministic 5-fold patient split as validate_patient_level_ccep.py!
    rng = np.random.default_rng(42)
    order = rng.permutation(len(patients))
    fold_of = np.zeros(len(patients), dtype=int)
    for k, idx in enumerate(order):
        fold_of[idx] = k % 5

    # Baseline observational fMRI-tuned Hopf
    torch.set_default_dtype(torch.float64)
    m_fmri = tune(wiring, real_fc, upper, np.random.default_rng(0))
    with torch.no_grad():
        G_fmri, _ = greens_function(m_fmri.weights().numpy(), m_fmri.log_distance.exp().numpy())
    torch.set_default_dtype(torch.float32)

    keys = [
        "fMRI-tuned Hopf",
        "Linear Bio+CCEP + VTA (v1.3 LPO)",
        "Permuted-Bio HyperNet + VTA (LPO)",
        "Graph-Transformer HyperNet + VTA (Utrecht LPO)",
        "Graph-Transformer HyperNet + VTA (F-TRACT Joint Pretrain)",
        "Graph-Transformer HyperNet + VTA (F-TRACT Split-Head Lat)",
    ]
    lpo_scores = {k: [] for k in keys}
    lat_scores = {
        "Linear Age-Conditioned Delay (v1.3 LPO)": [],
        "Graph-Transformer HyperNet Latency (Utrecht LPO)": [],
        "Graph-Transformer HyperNet Latency (F-TRACT Joint Pretrain)": [],
        "Graph-Transformer HyperNet Latency (F-TRACT Split-Head Lat)": [],
    }
    scored_ids = []

    for f in range(5):
        tr_pats = [p for i, p in enumerate(patients) if fold_of[i] != f]
        te_pats = [p for i, p in enumerate(patients) if fold_of[i] == f]

        # Linear v1.3 baseline (MultiScaleBioHopf in float64)
        tr_trials = np.sum([p["trials"] for p in tr_pats], axis=0)
        tr_hits = np.sum([p["hits"] for p in tr_pats], axis=0)
        tr_prob = np.divide(tr_hits, tr_trials, out=np.full((100, 100), np.nan), where=tr_trials > 0)
        tr_mask = (tr_trials >= 10) & ~np.eye(100, dtype=bool)
        torch.set_default_dtype(torch.float64)
        m_lin = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, tr_prob, tr_trials, tr_mask, residual_edges=True)
        with torch.no_grad():
            G_lin = m_lin.green().numpy()
        torch.set_default_dtype(torch.float32)

        # Linear age-conditioned latency OLS baseline (matching ccep_latency_waveform_model.py)
        d_tr, h_tr, wd_tr, a_tr, y_tr = [], [], [], [], []
        for p in tr_pats:
            ok = np.isfinite(p["lat_ms"]) & (p["lat_ms"] > 12.0) & (p["lat_ms"] < 105.0) & (p["elec_dist"] >= 10.0)
            if ok.sum() > 0 and np.isfinite(p["age"]):
                d_tr.append(p["elec_dist"][ok]); y_tr.append(p["lat_ms"][ok])
                h_tr.append(hops100[p["src_roi"][ok], p["dst_roi"][ok]])
                wd_tr.append(wdist100[p["src_roi"][ok], p["dst_roi"][ok]])
                a_tr.append(np.full(ok.sum(), p["age"]))
        d_tr, h_tr, wd_tr, a_tr, y_tr = map(np.concatenate, (d_tr, h_tr, wd_tr, a_tr, y_tr))
        ia_tr = 1.0 / (a_tr + 5.0)
        beta_lat = np.linalg.lstsq(np.column_stack([np.ones(len(d_tr)), d_tr, d_tr * ia_tr, ia_tr, h_tr, wd_tr]), y_tr, rcond=None)[0]

        # Train our 4 Graph-Transformer Connectome Weight Hypernetworks!
        m_perm = train_hypernet_fold(
            wiring, g0, node_fp, edge_fp, pb_fp, hops100, wdist100, s_dist,
            tr_pats, pretrain_ftract=None, steps=240, seed=f + 10, use_residuals=True,
        )
        m_ut = train_hypernet_fold(
            wiring, g0, node_f, edge_f, pb_f, hops100, wdist100, s_dist,
            tr_pats, pretrain_ftract=None, steps=240, seed=f + 20, use_residuals=True,
        )
        m_ft = train_hypernet_fold(
            wiring, g0, node_f, edge_f, pb_f, hops100, wdist100, s_dist,
            tr_pats, pretrain_ftract=(ft_prob_100, ft_lat_100), pretrain_mode="joint", steps=240, seed=f + 30, use_residuals=True,
        )
        m_sp = train_hypernet_fold(
            wiring, g0, node_f, edge_f, pb_f, hops100, wdist100, s_dist,
            tr_pats, pretrain_ftract=(ft_prob_100, ft_lat_100), pretrain_mode="latency_only", steps=240, seed=f + 40, use_residuals=True,
        )

        print(f"  Fold {f+1}/5: trained HyperNets on {len(tr_pats)} patients, scoring {len(te_pats)} unseen patients...", flush=True)
        for p in te_pats:
            age = float(p["age"]) if np.isfinite(p["age"]) else 22.0
            with torch.no_grad():
                G_p = m_perm.generate_patient_weights(age)[2].numpy().astype(np.float64)
                G_u = m_ut.generate_patient_weights(age)[2].numpy().astype(np.float64)
                G_f = m_ft.generate_patient_weights(age)[2].numpy().astype(np.float64)
                G_s = m_sp.generate_patient_weights(age)[2].numpy().astype(np.float64)

            s_fmri = score_patient(p, G_fmri, use_vta=False)
            s_lin = score_patient(p, G_lin, use_vta=True)
            s_perm = score_patient(p, G_p, use_vta=True)
            s_ut = score_patient(p, G_u, use_vta=True)
            s_ft = score_patient(p, G_f, use_vta=True)
            s_sp = score_patient(p, G_s, use_vta=True)
            if None not in (s_fmri, s_lin, s_perm, s_ut, s_ft, s_sp):
                scored_ids.append(p["id"])
                lpo_scores["fMRI-tuned Hopf"].append(s_fmri)
                lpo_scores["Linear Bio+CCEP + VTA (v1.3 LPO)"].append(s_lin)
                lpo_scores["Permuted-Bio HyperNet + VTA (LPO)"].append(s_perm)
                lpo_scores["Graph-Transformer HyperNet + VTA (Utrecht LPO)"].append(s_ut)
                lpo_scores["Graph-Transformer HyperNet + VTA (F-TRACT Joint Pretrain)"].append(s_ft)
                lpo_scores["Graph-Transformer HyperNet + VTA (F-TRACT Split-Head Lat)"].append(s_sp)

            # Latency evaluation on this unseen patient
            ok_l = np.isfinite(p["lat_ms"]) & (p["lat_ms"] > 12.0) & (p["lat_ms"] < 105.0) & (p["elec_dist"] >= 10.0)
            if ok_l.sum() >= 20 and np.isfinite(p["age"]):
                ia = 1.0 / (age + 5.0)
                d_p = p["elec_dist"][ok_l]
                h_p = hops100[p["src_roi"][ok_l], p["dst_roi"][ok_l]]
                wd_p = wdist100[p["src_roi"][ok_l], p["dst_roi"][ok_l]]
                pl_lin = np.column_stack([np.ones(len(d_p)), d_p, d_p * ia, np.full(len(d_p), ia), h_p, wd_p]) @ beta_lat
                lat_scores["Linear Age-Conditioned Delay (v1.3 LPO)"].append({
                    "mae_ms": float(np.mean(np.abs(pl_lin - p["lat_ms"][ok_l]))),
                    "rho": float(spearmanr(pl_lin, p["lat_ms"][ok_l]).statistic),
                })
                lat_scores["Graph-Transformer HyperNet Latency (Utrecht LPO)"].append(eval_patient_latency_hypernet(m_ut, p, hops100, wdist100))
                lat_scores["Graph-Transformer HyperNet Latency (F-TRACT Joint Pretrain)"].append(eval_patient_latency_hypernet(m_ft, p, hops100, wdist100))
                lat_scores["Graph-Transformer HyperNet Latency (F-TRACT Split-Head Lat)"].append(eval_patient_latency_hypernet(m_sp, p, hops100, wdist100))

    n_eval = len(scored_ids)
    print(f"\n=== 74-Patient Strict Leave-Patients-Out Summary (n = {n_eval} unseen patients) ===", flush=True)
    ccep_summ = {}
    for k in keys:
        aucs = np.array([x["auc"] for x in lpo_scores[k]])
        parts = np.array([x["partial"] for x in lpo_scores[k]])
        ccep_summ[k] = {
            "mean_auc": round(float(aucs.mean()), 4),
            "sd_auc": round(float(aucs.std(ddof=1)), 4),
            "mean_partial": round(float(parts.mean()), 4),
            "sd_partial": round(float(parts.std(ddof=1)), 4),
            "n_partial_pos": int((parts > 0).sum()),
        }
        s = ccep_summ[k]
        print(f"  {k:<58} AUROC={s['mean_auc']:.4f}±{s['sd_auc']:.4f} | part_rho|d_elec={s['mean_partial']:+.4f} ({s['n_partial_pos']}/{n_eval}>0)", flush=True)

    # Paired Wilcoxon tests
    pairs_to_test = [
        ("Graph-Transformer HyperNet + VTA (Utrecht LPO)", "Linear Bio+CCEP + VTA (v1.3 LPO)"),
        ("Graph-Transformer HyperNet + VTA (F-TRACT Split-Head Lat)", "Linear Bio+CCEP + VTA (v1.3 LPO)"),
        ("Graph-Transformer HyperNet + VTA (F-TRACT Split-Head Lat)", "Graph-Transformer HyperNet + VTA (F-TRACT Joint Pretrain)"),
        ("Graph-Transformer HyperNet + VTA (F-TRACT Joint Pretrain)", "Graph-Transformer HyperNet + VTA (Utrecht LPO)"),
        ("Graph-Transformer HyperNet + VTA (F-TRACT Split-Head Lat)", "fMRI-tuned Hopf"),
    ]
    w_tests = {}
    print("\n=== Paired Patient-Level Wilcoxon Signed-Rank Tests ===", flush=True)
    for a, b in pairs_to_test:
        aa = np.array([x["auc"] for x in lpo_scores[a]])
        ab = np.array([x["auc"] for x in lpo_scores[b]])
        pa = np.array([x["partial"] for x in lpo_scores[a]])
        pb = np.array([x["partial"] for x in lpo_scores[b]])
        wa = wilcoxon(aa - ab)
        wp = wilcoxon(pa - pb)
        w_tests[f"{a} vs {b}"] = {
            "delta_auc": round(float((aa - ab).mean()), 4), "wins_auc": int((aa > ab).sum()), "p_auc": float(wa.pvalue),
            "delta_partial": round(float((pa - pb).mean()), 4), "wins_partial": int((pa > pb).sum()), "p_partial": float(wp.pvalue),
        }
        t = w_tests[f"{a} vs {b}"]
        print(f"  {a} vs {b}:\n     dAUC={t['delta_auc']:+.4f} ({t['wins_auc']}/{n_eval}, p={t['p_auc']:.2e}) | dPart={t['delta_partial']:+.4f} ({t['wins_partial']}/{n_eval}, p={t['p_partial']:.2e})", flush=True)

    # Latency summary + paired Wilcoxon
    print("\n=== 74-Patient Strict LPO Millisecond Latency Summary ===", flush=True)
    lat_summ = {}
    for k, vals in lat_scores.items():
        maes = np.array([x["mae_ms"] for x in vals])
        rhos = np.array([x["rho"] for x in vals])
        lat_summ[k] = {"n_patients": len(vals), "mean_mae_ms": round(float(maes.mean()), 2), "mean_rho": round(float(rhos.mean()), 4)}
        print(f"  {k:<58} MAE={maes.mean():.2f} ms | mean patient rho={rhos.mean():+.4f}", flush=True)
    l_lin = np.array([x["mae_ms"] for x in lat_scores["Linear Age-Conditioned Delay (v1.3 LPO)"]])
    l_ut = np.array([x["mae_ms"] for x in lat_scores["Graph-Transformer HyperNet Latency (Utrecht LPO)"]])
    l_ft = np.array([x["mae_ms"] for x in lat_scores["Graph-Transformer HyperNet Latency (F-TRACT Joint Pretrain)"]])
    l_sp = np.array([x["mae_ms"] for x in lat_scores["Graph-Transformer HyperNet Latency (F-TRACT Split-Head Lat)"]])
    w_lu = wilcoxon(l_lin - l_ut)
    w_lf = wilcoxon(l_lin - l_ft)
    w_ls = wilcoxon(l_lin - l_sp)
    lat_wilcoxon = {
        "utrecht_hypernet_vs_linear": {"delta_mae_ms": round(float((l_ut - l_lin).mean()), 2), "wins": int((l_ut < l_lin).sum()), "p": float(w_lu.pvalue)},
        "ftract_joint_hypernet_vs_linear": {"delta_mae_ms": round(float((l_ft - l_lin).mean()), 2), "wins": int((l_ft < l_lin).sum()), "p": float(w_lf.pvalue)},
        "ftract_split_head_hypernet_vs_linear": {"delta_mae_ms": round(float((l_sp - l_lin).mean()), 2), "wins": int((l_sp < l_lin).sum()), "p": float(w_ls.pvalue)},
    }
    print(f"  Latency Wilcoxon (Utrecht HyperNet vs Linear):          dMAE={lat_wilcoxon['utrecht_hypernet_vs_linear']['delta_mae_ms']:+.2f} ms ({lat_wilcoxon['utrecht_hypernet_vs_linear']['wins']}/{len(l_lin)}, p={w_lu.pvalue:.2e})", flush=True)
    print(f"  Latency Wilcoxon (F-TRACT Joint HyperNet vs Linear):    dMAE={lat_wilcoxon['ftract_joint_hypernet_vs_linear']['delta_mae_ms']:+.2f} ms ({lat_wilcoxon['ftract_joint_hypernet_vs_linear']['wins']}/{len(l_lin)}, p={w_lf.pvalue:.2e})", flush=True)
    print(f"  Latency Wilcoxon (F-TRACT Split-Head Lat vs Linear):    dMAE={lat_wilcoxon['ftract_split_head_hypernet_vs_linear']['delta_mae_ms']:+.2f} ms ({lat_wilcoxon['ftract_split_head_hypernet_vs_linear']['wins']}/{len(l_lin)}, p={w_ls.pvalue:.2e})", flush=True)

    # Full-Utrecht HyperNet (zero F-TRACT fitting!) tested ZERO-SHOT on 780-pt F-TRACT & 46-adult TMS-EEG!
    print("\n=== Full Utrecht-Only HyperNet Zero-Shot External Transfer (F-TRACT 780 pts + TMS-EEG 46 adults) ===", flush=True)
    m_ut_full = train_hypernet_fold(
        wiring, g0, node_f, edge_f, pb_f, hops100, wdist100, s_dist,
        patients, pretrain_ftract=None, steps=260, seed=99, use_residuals=True,
    )
    m_ut_bioonly = train_hypernet_fold(
        wiring, g0, node_f, edge_f, pb_f, hops100, wdist100, s_dist,
        patients, pretrain_ftract=None, steps=260, seed=99, use_residuals=False,
    )
    all_tr_u = np.sum([p["trials"] for p in patients], axis=0)
    valid_u = (all_tr_u >= 10) & ~np.eye(100, dtype=bool)
    m_fp_all = np.isfinite(ft_prob_100) & ~np.eye(100, dtype=bool)
    m_fp_unseen = m_fp_all & ~valid_u

    from functional_brain_emulator import get_causal_human_scaffold
    scaf = get_causal_human_scaffold()
    zs_ext = {}
    for nm, mdl in [("HyperNet Full (Utrecht-only ZS)", m_ut_full), ("HyperNet Pure-Gen 0-Residuals (Utrecht-only ZS)", m_ut_bioonly)]:
        with torch.no_grad():
            G_zs = mdl.generate_patient_weights(age_years=30.0)[2].numpy().astype(np.float64)
            G_zs_nod = G_zs.copy(); np.fill_diagonal(G_zs_nod, 0.0)
        tms = eval_zero_shot_tmseeg_n46(G_zs_nod, scaf)
        zs_ext[nm] = {
            "ftract_all_4307_rho": round(float(spearmanr(G_zs[m_fp_all], ft_prob_100[m_fp_all]).statistic), 4),
            "ftract_all_4307_partial": round(float(partial_spearman(G_zs[m_fp_all], ft_prob_100[m_fp_all], s_dist[m_fp_all])), 4),
            "ftract_unseen_1681_rho": round(float(spearmanr(G_zs[m_fp_unseen], ft_prob_100[m_fp_unseen]).statistic), 4),
            "ftract_unseen_1681_partial": round(float(partial_spearman(G_zs[m_fp_unseen], ft_prob_100[m_fp_unseen], s_dist[m_fp_unseen])), 4),
            "tmseeg_n46_partial": round(float(tms["mean_partial_rho"]), 4),
            "tmseeg_n46_pos": int(tms["n_positive"]),
        }
        z = zs_ext[nm]
        print(
            f"  {nm:<48} F-TRACT all={z['ftract_all_4307_rho']:+.4f} (part {z['ftract_all_4307_partial']:+.4f}) | "
            f"unseen={z['ftract_unseen_1681_rho']:+.4f} (part {z['ftract_unseen_1681_partial']:+.4f}) | "
            f"TMS-EEG={z['tmseeg_n46_partial']:+.4f} ({z['tmseeg_n46_pos']}/46)",
            flush=True,
        )

    out = {
        "n_evaluated_patients": n_eval,
        "n_params_hypernet_pure": int(sum(p.numel() for p in m_ut_bioonly.parameters())),
        "n_params_hypernet_full": int(sum(p.numel() for p in m_ut_full.parameters())),
        "lpo_ccep_summary": ccep_summ,
        "lpo_ccep_wilcoxon": w_tests,
        "lpo_latency_summary": lat_summ,
        "lpo_latency_wilcoxon": lat_wilcoxon,
        "zero_shot_external": zs_ext,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Saved: {OUT_JSON}", flush=True)
    make_figure(out, lpo_scores, lat_scores)


def make_figure(out, lpo_scores, lat_scores):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)
    n_eval = out["n_evaluated_patients"]
    s = out["lpo_ccep_summary"]

    # Panel A: 74-Patient LPO Per-Patient Electrode N1 AUROC
    ax = axes[0]
    order_a = [
        "fMRI-tuned Hopf",
        "Graph-Transformer HyperNet + VTA (F-TRACT Joint Pretrain)",
        "Permuted-Bio HyperNet + VTA (LPO)",
        "Linear Bio+CCEP + VTA (v1.3 LPO)",
        "Graph-Transformer HyperNet + VTA (Utrecht LPO)",
        "Graph-Transformer HyperNet + VTA (F-TRACT Split-Head Lat)",
    ]
    short_a = [
        "fMRI-tuned\nHopf",
        "HyperNet\n(F-TRACT Joint)",
        "Permuted-Bio\nHyperNet",
        "Linear Bio+CCEP\n+ VTA (v1.3)",
        "HyperNet\n(Utrecht LPO)",
        "HyperNet\n(Split-Head Lat)",
    ]
    y = np.arange(len(order_a))
    means = [s[k]["mean_auc"] for k in order_a]
    sems = [s[k]["sd_auc"] / np.sqrt(n_eval) for k in order_a]
    cols = ["#d95f02", "#cc79a7", "#b0a99f", "#4c956c", "#2b59c3", "#0b6e4f"]
    ax.barh(y, means, xerr=sems, height=0.58, color=cols, alpha=0.92, capsize=3)
    for i, m in enumerate(means):
        ax.text(m + sems[i] + 0.008, i, f"{m:.4f}", va="center", fontsize=8.0, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(short_a, fontsize=8.2)
    ax.set_xlim(0.50, 0.875)
    ax.set_xlabel("Per-Patient Electrode N1 AUROC (Mean ± SEM)")
    ax.set_title(f"A. Unseen-Patient CCEP Detection (n = {n_eval})\n(Generated Patient Weights vs. Static v1.3)", fontweight="bold", fontsize=9.8)
    ax.grid(axis="x", alpha=0.25)

    # Panel B: Distance-Controlled Partial rho | Exact Millimeter Electrode Distance
    ax = axes[1]
    pmeans = [s[k]["mean_partial"] for k in order_a]
    psems = [s[k]["sd_partial"] / np.sqrt(n_eval) for k in order_a]
    ax.barh(y, pmeans, xerr=psems, height=0.58, color=cols, alpha=0.92, capsize=3)
    for i, pm in enumerate(pmeans):
        ax.text(pm + psems[i] + 0.002, i, f"{pm:+.4f}", va="center", fontsize=8.0, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(short_a, fontsize=8.2)
    ax.set_xlim(0.0, max(pmeans) + 0.025)
    ax.set_xlabel("Partial Spearman ρ | Electrode Distance (mm)")
    ax.set_title("B. Distance-Controlled Individual Accuracy\n(Controlling Exact Contact Distance)", fontweight="bold", fontsize=9.8)
    ax.grid(axis="x", alpha=0.25)

    # Panel C: Unseen-Patient Millisecond N1 Latency MAE (ms)
    ax = axes[2]
    lk = [
        "Linear Age-Conditioned Delay (v1.3 LPO)",
        "Graph-Transformer HyperNet Latency (Utrecht LPO)",
        "Graph-Transformer HyperNet Latency (F-TRACT Joint Pretrain)",
        "Graph-Transformer HyperNet Latency (F-TRACT Split-Head Lat)",
    ]
    lshort = ["Linear Age-Aware\nDelay (v1.3)", "HyperNet\n(Utrecht)", "HyperNet\n(F-TRACT Joint)", "HyperNet\n(Split-Head Lat)"]
    yl = np.arange(len(lk))
    lmeans = [out["lpo_latency_summary"][k]["mean_mae_ms"] for k in lk]
    lrhos = [out["lpo_latency_summary"][k]["mean_rho"] for k in lk]
    lcols = ["#4c956c", "#2b59c3", "#cc79a7", "#0b6e4f"]
    ax.barh(yl, lmeans, height=0.52, color=lcols, alpha=0.92)
    for i, (lm, lr) in enumerate(zip(lmeans, lrhos)):
        ax.text(lm + 0.18, i, f"{lm:.2f} ms (ρ={lr:+.3f})", va="center", fontsize=8.0, fontweight="bold")
    ax.set_yticks(yl)
    ax.set_yticklabels(lshort, fontsize=8.3)
    ax.set_xlim(0, max(lmeans) + 7.2)
    ax.set_xlabel("Per-Patient Held-Out N1 Latency MAE (ms, lower = better)")
    ax.set_title("C. Unseen-Patient Conduction Timing\n(Generated Regional Myelination Speed v(age))", fontweight="bold", fontsize=9.8)
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Graph-Transformer Connectome Weight Hypernetwork vs. Static Models "
        f"(5-Fold Leave-Patients-Out Across {n_eval}/74 Unseen Human Patients, ds004080)",
        fontsize=10.3,
        fontweight="bold",
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
