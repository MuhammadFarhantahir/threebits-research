"""Multi-Scale Biological Whole-Brain Digital Twin (Threebits Research).

Tests whether "filling in the cellular/molecular gaps" of macro whole-brain models using
real open human cortical biology (19 PET neurotransmitter receptors from Hansen et al. 2022
Nat. Neurosci., 6-band HCP MEG spectral power, and 4-level Mesulam laminar cytoarchitecture)
improves STRICTLY HELD-OUT causal electrical-stimulation accuracy over uniform-region Hopf models.

Four falsifiable benchmarks with strict matched-capacity negative controls (permuted biology):
  1. 5-fold Unseen Stimulated Cortical Regions (74 human CCEP patients, 3,219 non-SOZ pairs)
  2. Leave-One-Functional-Network-Out (entire Yeo-7 cognitive network never stimulated!)
  3. 5-fold Unseen Directed Pairs (same 3,219 non-SOZ human CCEP pairs)
  4. Zero-Shot Healthy-Human Non-Invasive TMS-EEG (n=46 clean adults, OpenNeuro ds008037)

Run (~75 s):
    .venv/bin/python multiscale_brain_twin.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from human_brain_model import TARGET, load as load_hcp, untuned
from validate_simulator_ccep import build_ccep_matrix, greens_function, load_centroids, partial_spearman
from causal_brain_simulator import fit_causal

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "multiscale_brain_twin.json"
ATLAS_NPZ = ROOT / "data" / "multiscale_atlas" / "schaefer100_multiscale_biology.npz"


def load_biology_features():
    """Load real 100-region human PET receptors + MEG power + Mesulam hierarchy and build compact basis."""
    d = np.load(ATLAS_NPZ, allow_pickle=True)
    rec = d["receptors"].astype(np.float64)            # (100, 19) z-scored
    rec_names = [str(x) for x in d["receptor_names"]]
    meg = d["meg_power"].astype(np.float64)            # (100, 6)
    meg_z = (meg - meg.mean(0, keepdims=True)) / np.maximum(meg.std(0, keepdims=True), 1e-8)
    mes = d["mesulam_hierarchy"].astype(np.float64)    # (100,) in {1,2,3,4}
    mes_z = (mes - mes.mean()) / max(mes.std(), 1e-8)

    # Receptor PCA (top 6 PCs capture ~85% of variance across 19 PET receptors, preventing overfitting)
    u, s, vt = np.linalg.svd(rec - rec.mean(0, keepdims=True), full_matrices=False)
    rec_pcs = u[:, :6] * s[:6]
    rec_pcs = rec_pcs / np.maximum(rec_pcs.std(0, keepdims=True), 1e-8)
    var_exp = float((s[:6] ** 2).sum() / (s ** 2).sum())

    # Explicit E:I ratio proxy: excitatory (NMDA + mGluR5)/2 - inhibitory GABAa
    idx_nmda = rec_names.index("NMDA")
    idx_mglu = rec_names.index("mGluR5")
    idx_gaba = rec_names.index("GABAa")
    ei = 0.5 * (rec[:, idx_nmda] + rec[:, idx_mglu]) - rec[:, idx_gaba]
    ei_z = (ei - ei.mean()) / max(ei.std(), 1e-8)

    # MEG PCA (top 3 PCs: slow/alpha/fast spectral gradient)
    um, sm, _ = np.linalg.svd(meg_z, full_matrices=False)
    meg_pcs = um[:, :3] * sm[:3]
    meg_pcs = meg_pcs / np.maximum(meg_pcs.std(0, keepdims=True), 1e-8)

    # Compact 11-feature multi-scale biological descriptor per cortical region:
    # [6 receptor PCs, E:I balance, 3 MEG spectral PCs, Mesulam laminar hierarchy]
    F = np.column_stack([rec_pcs, ei_z, meg_pcs, mes_z])  # (100, 11)
    feat_names = [f"Rec_PC{k+1}" for k in range(6)] + ["EI_NMDA_mGluR5_minus_GABAa"] + [f"MEG_PC{k+1}" for k in range(3)] + ["Mesulam_Hierarchy"]

    # Pairwise receptor cosine similarity (100, 100) and laminar hierarchy difference h_dst - h_src
    rec_norm = rec / np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-8)
    rec_cos = rec_norm @ rec_norm.T
    rec_cos = (rec_cos - rec_cos.mean()) / max(rec_cos.std(), 1e-8)
    hier_diff = mes_z[None, :] - mes_z[:, None]  # [src, dst] = dst_hier - src_hier (feedforward > 0)

    return F, rec_cos, hier_diff, feat_names, rec, rec_names, var_exp


class MultiScaleBioHopf(torch.nn.Module):
    """Multi-scale whole-brain Hopf twin parameterized by human receptors, MEG spectra, and laminar hierarchy.

    Two operating modes:
      - residual_edges=False ("Bio-Prior Only"): ONLY ~37 shared biological parameters across the entire brain.
      - residual_edges=True  ("Full Multi-Scale Twin"): biological prior + L2-anchored per-edge/node residuals.
    """

    def __init__(self, wiring, g0, F_bio, rec_cos, hier_diff, residual_edges=True):
        super().__init__()
        self.n = len(wiring)
        self.residual_edges = residual_edges
        r, c = np.nonzero(wiring > 0)
        keep = r != c
        self.rows = torch.tensor(r[keep], dtype=torch.long)
        self.cols = torch.tensor(c[keep], dtype=torch.long)
        self.base_w = torch.tensor(wiring[r[keep], c[keep]], dtype=torch.float64)
        self.log_g0 = float(np.log(g0))

        self.register_buffer("F", torch.tensor(F_bio, dtype=torch.float64))             # (100, D)
        self.register_buffer("rcos_e", torch.tensor(rec_cos[r[keep], c[keep]], dtype=torch.float64))
        self.register_buffer("hdiff_e", torch.tensor(hier_diff[r[keep], c[keep]], dtype=torch.float64))
        D = F_bio.shape[1]

        # Shared biological laws (transfer to completely unseen stimulated cortical regions!)
        self.g_offset = torch.nn.Parameter(torch.tensor(self.log_g0))
        self.w_d = torch.nn.Parameter(torch.zeros(D))        # regional damping / excitability from biology
        self.w_src = torch.nn.Parameter(torch.zeros(D))      # outgoing gain from source region biology
        self.w_dst = torch.nn.Parameter(torch.zeros(D))      # incoming gain from target region biology
        self.w_rcos = torch.nn.Parameter(torch.tensor(0.0))  # receptor similarity coupling boost
        self.w_hier = torch.nn.Parameter(torch.tensor(0.0))  # laminar feedforward vs feedback asymmetry
        # Also source-region excitability scaling of stimulation input (real biology: GABA/NMDA/cytoarchitecture affects evoked N1!)
        self.w_stim = torch.nn.Parameter(torch.zeros(D))

        if residual_edges:
            self.delta_g = torch.nn.Parameter(torch.zeros(len(self.rows)))
            self.delta_d = torch.nn.Parameter(torch.zeros(self.n))
        else:
            self.register_parameter("delta_g", None)
            self.register_parameter("delta_d", None)

        self.scale = torch.nn.Parameter(torch.tensor(1.5))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def weights_and_d(self):
        # Note: in K = diag(W.sum(1)) - W + diag(d), W[dst, src] couples src -> dst!
        # Here rows=dst, cols=src in K; let's keep exact convention of DirectedHopf:
        # DirectedHopf: W[rows, cols], K = diag(W.sum(1)) - W + diag(d), G = inv(K).T so G[src, dst] = inv(K)[dst, src]
        # Thus edge (rows=dst, cols=src) has source=cols, target=rows!
        bio_edge = (
            self.g_offset
            + (self.F[self.cols] @ self.w_src)
            + (self.F[self.rows] @ self.w_dst)
            + self.w_rcos * self.rcos_e
            + self.w_hier * self.hdiff_e
        )
        if self.residual_edges:
            bio_edge = bio_edge + self.delta_g
        bio_edge = bio_edge.clamp(-12.0, 4.0)
        W = torch.zeros(self.n, self.n)
        W[self.rows, self.cols] = self.base_w * bio_edge.exp()

        log_d = self.F @ self.w_d
        if self.residual_edges:
            log_d = log_d + self.delta_d
        d = log_d.clamp(-4.0, 4.0).exp()
        return W, d

    def green(self):
        W, d = self.weights_and_d()
        K = torch.diag(W.sum(1)) - W + torch.diag(d)
        G = torch.linalg.inv(K).T  # [src, dst]
        # Multiply each stimulated source row by biological stimulation susceptibility exp(w_stim · F_src)
        stim_gain = (self.F @ self.w_stim).clamp(-2.0, 2.0).exp()
        return G * stim_gain[:, None]

    def predict_prob(self):
        G = self.green().clamp(min=1e-9)
        return torch.sigmoid(self.scale * torch.log(G) + self.bias), G


def fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob_mat, trials_mat, train_mask,
                   residual_edges=True, lam_bio=0.05, lam_w=0.06, lam_d=0.06, steps=650):
    model = MultiScaleBioHopf(wiring, g0, F_bio, rec_cos, hier_diff, residual_edges=residual_edges)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    m_idx = torch.tensor(train_mask)
    y = torch.tensor(np.nan_to_num(prob_mat[train_mask], nan=0.0))
    w_pair = torch.tensor(np.log1p(trials_mat[train_mask]))
    w_pair = w_pair / w_pair.mean()
    for _ in range(steps):
        opt.zero_grad()
        p, _ = model.predict_prob()
        pm = p[m_idx].clamp(1e-6, 1 - 1e-6)
        bce = -(y * torch.log(pm) + (1 - y) * torch.log(1 - pm))
        loss = (w_pair * bce).mean()
        bio_reg = (
            (model.w_d ** 2).mean() + (model.w_src ** 2).mean() + (model.w_dst ** 2).mean()
            + (model.w_stim ** 2).mean() + model.w_rcos ** 2 + model.w_hier ** 2
        )
        loss = loss + lam_bio * bio_reg + 0.02 * (model.g_offset - model.log_g0) ** 2
        if residual_edges:
            loss = loss + lam_w * (model.delta_g ** 2).mean() + lam_d * (model.delta_d ** 2).mean()
        loss.backward()
        opt.step()
    return model


def eval_scores(pred_vec, y_vec, d_vec):
    return {
        "spearman": round(float(spearmanr(pred_vec, y_vec).statistic), 4),
        "pearson": round(float(np.corrcoef(pred_vec, y_vec)[0, 1]), 4),
        "partial_spearman": round(float(partial_spearman(pred_vec, y_vec, d_vec)), 4),
    }


def eval_tmseeg_zeroshot(G_mat, coords):
    """Evaluate zero-shot on all n=46 clean healthy adults in OpenNeuro ds008037 (PFC + Parietal vs Sham)."""
    import csv
    cache = ROOT / "data" / "tmseeg_ds008037" / "tms_cache_n46.npz"
    if not cache.exists():
        return None
    d = np.load(cache, allow_pickle=True)
    # Inspect keys dynamically so we match replicate_tmseeg_g.py cache format
    return {k: d[k].shape for k in d.files}


def main():
    labels, coords, hemi = load_centroids()
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    trials, hits, prob, _, n_pat, _, n_raw = build_ccep_matrix(coords, hemi, exclude_soz=True)
    wiring, real_fc, _ = load_hcp("data/hcp_connectome")
    upper = np.triu_indices(100, 1)
    g0 = untuned(wiring, real_fc, upper)
    G_untuned, _ = greens_function(g0 * wiring, np.ones(100))

    F_bio, rec_cos, hier_diff, feat_names, rec_raw, rec_names, var_exp = load_biology_features()
    valid = (trials >= 10) & (~np.eye(100, dtype=bool))
    valid_idx = np.argwhere(valid)
    y = prob[valid]
    d_vec = dist_mat[valid]
    print(f"=== Multi-Scale Biological Digital Twin ({n_pat} CCEP patients, {n_raw:,} trials, {len(valid_idx)} pairs) ===")
    print(f"    Human biology basis: 19 PET receptors (6 PCs={var_exp*100:.1f}% var + E:I) + 6-band MEG + Mesulam hierarchy -> {F_bio.shape}", flush=True)

    # Permuted biology helper (exact same architecture & degrees of freedom, shuffled across 100 regions)
    def permute_bio(seed):
        p = np.random.default_rng(seed).permutation(100)
        return F_bio[p], rec_cos[np.ix_(p, p)], hier_diff[np.ix_(p, p)]

    results = {}

    # ---------------- BENCHMARK 1: 5-FOLD UNSEEN STIMULATED REGIONS ----------------
    print("\n--- Benchmark 1: 5-Fold COMPLETELY UNSEEN Stimulated Regions ---", flush=True)
    src_regions = np.unique(valid_idx[:, 0])
    src_folds = np.random.default_rng(2).permutation(len(src_regions)) % 5
    src_fold_map = dict(zip(src_regions, src_folds))

    oof_base_src = np.full((100, 100), np.nan)
    oof_bio_only_src = np.full((100, 100), np.nan)
    oof_bio_full_src = np.full((100, 100), np.nan)
    oof_shuf_only_src = np.zeros((3, 100, 100))
    oof_shuf_full_src = np.zeros((3, 100, 100))

    for f in range(5):
        tr_m = np.zeros((100, 100), dtype=bool)
        te_m = np.zeros((100, 100), dtype=bool)
        for r, c in valid_idx:
            (te_m if src_fold_map[r] == f else tr_m)[r, c] = True

        m_base = fit_causal(wiring, g0, prob, trials, tr_m)
        m_bo = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob, trials, tr_m, residual_edges=False)
        m_bf = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob, trials, tr_m, residual_edges=True)
        with torch.no_grad():
            oof_base_src[te_m] = m_base.green().numpy()[te_m]
            oof_bio_only_src[te_m] = m_bo.green().numpy()[te_m]
            oof_bio_full_src[te_m] = m_bf.green().numpy()[te_m]

        for pi in range(3):
            Fp, rcp, hdp = permute_bio(100 + pi)
            m_so = fit_multiscale(wiring, g0, Fp, rcp, hdp, prob, trials, tr_m, residual_edges=False, steps=450)
            m_sf = fit_multiscale(wiring, g0, Fp, rcp, hdp, prob, trials, tr_m, residual_edges=True, steps=450)
            with torch.no_grad():
                oof_shuf_only_src[pi, te_m] = m_so.green().numpy()[te_m]
                oof_shuf_full_src[pi, te_m] = m_sf.green().numpy()[te_m]

    b1 = {
        "untuned_hopf": eval_scores(G_untuned[valid], y, d_vec),
        "uniform_ccep_hopf": eval_scores(oof_base_src[valid], y, d_vec),
        "multiscale_bio_only": eval_scores(oof_bio_only_src[valid], y, d_vec),
        "multiscale_bio_full": eval_scores(oof_bio_full_src[valid], y, d_vec),
        "permuted_bio_only_ctrl": {
            k: round(float(np.mean([eval_scores(oof_shuf_only_src[pi][valid], y, d_vec)[k] for pi in range(3)])), 4)
            for k in ("spearman", "pearson", "partial_spearman")
        },
        "permuted_bio_full_ctrl": {
            k: round(float(np.mean([eval_scores(oof_shuf_full_src[pi][valid], y, d_vec)[k] for pi in range(3)])), 4)
            for k in ("spearman", "pearson", "partial_spearman")
        },
    }
    for k, v in b1.items():
        print(f"  {k:<28} rho={v['spearman']:+.4f}  r={v['pearson']:+.4f}  partial|d={v['partial_spearman']:+.4f}", flush=True)
    results["unseen_stim_regions_5fold"] = b1

    # ---------------- BENCHMARK 2: LEAVE-ONE-YEO-NETWORK-OUT ----------------
    print("\n--- Benchmark 2: Leave-One-Yeo-7-Network-Out (Entire Cognitive Network Never Stimulated!) ---", flush=True)
    nets = [lbl.split("_")[2] for lbl in labels]  # Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
    uniq_nets = sorted(set(nets))
    oof_net_base = np.full((100, 100), np.nan)
    oof_net_bo = np.full((100, 100), np.nan)
    oof_net_bf = np.full((100, 100), np.nan)
    oof_net_shuf = np.full((100, 100), np.nan)
    Fp0, rcp0, hdp0 = permute_bio(777)

    net_breakdown = {}
    for net in uniq_nets:
        tr_m = np.zeros((100, 100), dtype=bool)
        te_m = np.zeros((100, 100), dtype=bool)
        for r, c in valid_idx:
            (te_m if nets[r] == net else tr_m)[r, c] = True
        if te_m.sum() < 20:
            continue
        mb = fit_causal(wiring, g0, prob, trials, tr_m)
        mbo = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob, trials, tr_m, residual_edges=False)
        mbf = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob, trials, tr_m, residual_edges=True)
        ms = fit_multiscale(wiring, g0, Fp0, rcp0, hdp0, prob, trials, tr_m, residual_edges=True, steps=500)
        with torch.no_grad():
            oof_net_base[te_m] = mb.green().numpy()[te_m]
            oof_net_bo[te_m] = mbo.green().numpy()[te_m]
            oof_net_bf[te_m] = mbf.green().numpy()[te_m]
            oof_net_shuf[te_m] = ms.green().numpy()[te_m]
        net_breakdown[net] = {
            "n_pairs": int(te_m.sum()),
            "uniform_ccep": eval_scores(oof_net_base[te_m], prob[te_m], dist_mat[te_m]),
            "bio_only": eval_scores(oof_net_bo[te_m], prob[te_m], dist_mat[te_m]),
            "bio_full": eval_scores(oof_net_bf[te_m], prob[te_m], dist_mat[te_m]),
            "shuf_ctrl": eval_scores(oof_net_shuf[te_m], prob[te_m], dist_mat[te_m]),
        }
        nb = net_breakdown[net]
        print(f"  Held-out {net:<11} ({nb['n_pairs']:4d}p): uniform={nb['uniform_ccep']['spearman']:+.3f} (p{nb['uniform_ccep']['partial_spearman']:+.3f}) | "
              f"bio_only={nb['bio_only']['spearman']:+.3f} (p{nb['bio_only']['partial_spearman']:+.3f}) | "
              f"bio_full={nb['bio_full']['spearman']:+.3f} (p{nb['bio_full']['partial_spearman']:+.3f}) | "
              f"shuf={nb['shuf_ctrl']['spearman']:+.3f}", flush=True)

    b2 = {
        "overall_uniform_ccep": eval_scores(oof_net_base[valid], y, d_vec),
        "overall_bio_only": eval_scores(oof_net_bo[valid], y, d_vec),
        "overall_bio_full": eval_scores(oof_net_bf[valid], y, d_vec),
        "overall_shuf_ctrl": eval_scores(oof_net_shuf[valid], y, d_vec),
        "per_network": net_breakdown,
    }
    print(f"  OVERALL LOO-Net: uniform={b2['overall_uniform_ccep']} | bio_only={b2['overall_bio_only']} | bio_full={b2['overall_bio_full']} | shuf={b2['overall_shuf_ctrl']}", flush=True)
    results["leave_one_network_out"] = b2

    # ---------------- BENCHMARK 3: 5-FOLD UNSEEN DIRECTED PAIRS ----------------
    print("\n--- Benchmark 3: 5-Fold Unseen Directed Pairs ---", flush=True)
    folds_p = np.random.default_rng(0).permutation(len(valid_idx)) % 5
    oof_p_base = np.full((100, 100), np.nan)
    oof_p_bo = np.full((100, 100), np.nan)
    oof_p_bf = np.full((100, 100), np.nan)
    oof_p_shuf = np.full((100, 100), np.nan)
    for f in range(5):
        tr_m = np.zeros((100, 100), dtype=bool)
        te_m = np.zeros((100, 100), dtype=bool)
        for k, (r, c) in enumerate(valid_idx):
            (te_m if folds_p[k] == f else tr_m)[r, c] = True
        mb = fit_causal(wiring, g0, prob, trials, tr_m)
        mbo = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob, trials, tr_m, residual_edges=False)
        mbf = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob, trials, tr_m, residual_edges=True)
        ms = fit_multiscale(wiring, g0, Fp0, rcp0, hdp0, prob, trials, tr_m, residual_edges=True, steps=500)
        with torch.no_grad():
            oof_p_base[te_m] = mb.green().numpy()[te_m]
            oof_p_bo[te_m] = mbo.green().numpy()[te_m]
            oof_p_bf[te_m] = mbf.green().numpy()[te_m]
            oof_p_shuf[te_m] = ms.green().numpy()[te_m]

    b3 = {
        "uniform_ccep_hopf": eval_scores(oof_p_base[valid], y, d_vec),
        "multiscale_bio_only": eval_scores(oof_p_bo[valid], y, d_vec),
        "multiscale_bio_full": eval_scores(oof_p_bf[valid], y, d_vec),
        "permuted_bio_full_ctrl": eval_scores(oof_p_shuf[valid], y, d_vec),
    }
    for k, v in b3.items():
        print(f"  {k:<28} rho={v['spearman']:+.4f}  r={v['pearson']:+.4f}  partial|d={v['partial_spearman']:+.4f}", flush=True)
    results["unseen_pairs_5fold"] = b3

    # ---------------- FULL FIT INTERPRETABILITY + ZERO-SHOT HEALTHY TMS-EEG ----------------
    print("\n--- Full Multi-Scale Biological Laws Learned from 74 Human CCEP Patients ---", flush=True)
    m_full_bo = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob, trials, valid, residual_edges=False)
    m_full_bf = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob, trials, valid, residual_edges=True)
    m_full_base = fit_causal(wiring, g0, prob, trials, valid)

    with torch.no_grad():
        G_bo = m_full_bo.green().numpy()
        G_bf = m_full_bf.green().numpy()
        G_base = m_full_base.green().numpy()
        laws = {
            "receptor_similarity_weight": round(float(m_full_bo.w_rcos.item()), 4),
            "laminar_hierarchy_direction_weight": round(float(m_full_bo.w_hier.item()), 4),
            "feature_weights_damping": {feat_names[j]: round(float(m_full_bo.w_d[j].item()), 4) for j in range(len(feat_names))},
            "feature_weights_stim_susceptibility": {feat_names[j]: round(float(m_full_bo.w_stim[j].item()), 4) for j in range(len(feat_names))},
            "feature_weights_outgoing_edge": {feat_names[j]: round(float(m_full_bo.w_src[j].item()), 4) for j in range(len(feat_names))},
            "feature_weights_incoming_edge": {feat_names[j]: round(float(m_full_bo.w_dst[j].item()), 4) for j in range(len(feat_names))},
        }
        # Also compute direct correlation of each of the 19 individual PET receptors with empirical CCEP outgoing excitability!
        out_mean = np.array([prob[i, valid[i]].mean() if valid[i].sum() >= 5 else np.nan for i in range(100)])
        in_mean = np.array([prob[valid[:, i], i].mean() if valid[:, i].sum() >= 5 else np.nan for i in range(100)])
        ok_o, ok_i = np.isfinite(out_mean), np.isfinite(in_mean)
        rec_corr = {}
        for j, rn in enumerate(rec_names):
            rec_corr[rn] = {
                "outgoing_ccep_rho": round(float(spearmanr(rec_raw[ok_o, j], out_mean[ok_o]).statistic), 3),
                "incoming_ccep_rho": round(float(spearmanr(rec_raw[ok_i, j], in_mean[ok_i]).statistic), 3),
            }
        laws["individual_receptor_ccep_correlations"] = rec_corr

    print(f"  Receptor cosine similarity weight (w_rcos): {laws['receptor_similarity_weight']:+.4f}")
    print(f"  Mesulam laminar feedforward weight (w_hier): {laws['laminar_hierarchy_direction_weight']:+.4f}")
    print("  Top receptors correlated with empirical CCEP outgoing spread:")
    for rn, rc in sorted(rec_corr.items(), key=lambda kv: abs(kv[1]["outgoing_ccep_rho"]), reverse=True)[:6]:
        print(f"    {rn:<8} outgoing rho={rc['outgoing_ccep_rho']:+.3f}  incoming rho={rc['incoming_ccep_rho']:+.3f}")
    results["learned_biological_laws"] = laws

    # Benchmark 4: Zero-Shot Healthy-Human Non-Invasive TMS-EEG Transfer (n=46 clean adults, zero EEG params fit!)
    from human_intelligence_wm_eeg import PRE_TMS_MS, TMS25_CACHE
    from replicate_tmseeg_g import REP_CACHE
    from validate_simulator_tmseeg import load_eeg_1005_mni

    elec_map = load_eeg_1005_mni()
    tms_mni = {"pfc": np.array([-46.0, 33.0, 34.0]), "par": np.array([-26.0, -69.0, 63.0])}
    site_idx = {s: int(np.argmin(np.linalg.norm(coords - m, axis=1))) for s, m in tms_mni.items()}
    models = {
        "untuned_hopf": G_untuned,
        "uniform_ccep_hopf": G_base,
        "multiscale_bio_only": G_bo,
        "multiscale_bio_full": G_bf,
    }
    w_qc = slice(PRE_TMS_MS + 35, PRE_TMS_MS + 180)
    w_tep = slice(PRE_TMS_MS + 35, PRE_TMS_MS + 100)
    all_subs = {}
    ref_chs = None
    for cp in (TMS25_CACHE, REP_CACHE):
        if cp.exists():
            z = np.load(cp, allow_pickle=True)
            ref_chs = [str(c) for c in z["channels"]]
            for s, arr in zip(z["subjects"], z["erps"]):
                all_subs[str(s)] = arr

    tms_res = {}
    if all_subs and ref_chs is not None:
        keep_ch = [i for i, c in enumerate(ref_chs) if c.upper() in elec_map]
        e_coords = np.array([elec_map[ref_chs[i].upper()] for i in keep_ch], dtype=np.float64)
        # Fixed physical inverse-square cortical-to-scalp leadfield (zero parameters fitted!)
        d_ec = np.linalg.norm(e_coords[:, None, :] - coords[None, :, :], axis=-1)
        L = 1.0 / (d_ec ** 2 + 20.0 ** 2)
        L = L / L.sum(axis=1, keepdims=True)

        per_sub = {k: [] for k in models}
        for s, arr in sorted(all_subs.items()):
            cleaned, ok = [], True
            for ci in range(3):  # pfc, par, ctrl(shoulder sham)
                e = arr[ci].astype(np.float64).copy()
                e -= e[:PRE_TMS_MS - 15].mean(axis=0, keepdims=True)
                good = np.sqrt(np.mean(e[w_qc] ** 2, axis=0)) < 40.0
                if good.sum() < 45:
                    ok = False
                    break
                e[:, good] -= e[:, good].mean(axis=1, keepdims=True)
                e[:, ~good] = np.nan
                cleaned.append(e)
            if not ok:
                continue
            pfc, par, ctrl = cleaned
            sub_scores = {k: [] for k in models}
            for cond, cond_e in (("pfc", pfc), ("par", par)):
                net = np.sqrt(np.nanmean((cond_e[w_tep] - ctrl[w_tep]) ** 2, axis=0))[keep_ch]
                good = np.isfinite(net)
                if good.sum() < 30:
                    continue
                coil_d = np.linalg.norm(e_coords[good] - tms_mni[cond], axis=1)
                si = site_idx[cond]
                for mname, Gm in models.items():
                    pred_scalp = L[good] @ Gm[si]
                    sub_scores[mname].append(partial_spearman(pred_scalp, net[good], coil_d))
            for mname in models:
                if len(sub_scores[mname]) == 2:
                    per_sub[mname].append(float(np.mean(sub_scores[mname])))
        print(f"\n--- Benchmark 4: Zero-Shot Healthy-Human TMS-EEG Transfer (n={len(per_sub['uniform_ccep_hopf'])} clean adults) ---", flush=True)
        for mname, arr in per_sub.items():
            if arr:
                a = np.array(arr)
                tms_res[mname] = {"n": len(a), "mean_partial_rho": round(float(a.mean()), 4), "pos_count": int((a > 0).sum())}
                print(f"  {mname:<24} mean partial rho|d = {a.mean():+.4f} ({int((a>0).sum())}/{len(a)} > 0)", flush=True)
    results["tmseeg_zeroshot"] = tms_res

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
