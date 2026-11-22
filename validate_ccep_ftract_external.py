"""Independent Second-Hospital / Multi-Center Out-of-Hospital CCEP Validation (Threebits Research).

Validates Paper 1's core findings strictly out-of-hospital on the independent multi-center
F-TRACT human intracranial SEEG atlas (v2112, 780 patients aged >= 15 across European/international
epilepsy centers; David et al. 2013, Trebaul et al. 2018, Lemarechal et al. 2022) — completely
separate from Paper 1's single-hospital UMCU Utrecht ECoG cohort (OpenNeuro ds004080, 74 patients).

Two complementary validations (zero F-TRACT parameters fitted anywhere):
  Part A — Native Glasser-360 F-TRACT Replication (26,771 prob pairs; 15,772 amplitude pairs):
    Replicates the resting-fMRI vs causal-perturbation trade-off natively on 360 cortical regions
    with zero atlas remapping, reporting pooled + per-site (>=10 targets) Spearman rho,
    distance-controlled partial rho | d, and transparent intra- vs inter-hemispheric breakdown.
  Part B — Strict Out-of-Hospital Zero-Shot Transfer (Utrecht ds004080 -> Multi-Center F-TRACT):
    Evaluates models fitted ONLY on Utrecht ds004080 (74 ECoG patients) + open human cellular priors
    (19 PET neurotransmitter receptors, 6-band HCP MEG, Mesulam hierarchy) ZERO-SHOT on F-TRACT
    mapped to Schaefer-100 across:
      1. All 4,307 measured F-TRACT directed response-probability pairs,
      2. The 1,681 F-TRACT directed pairs NEVER measured in Utrecht ds004080,
      3. All 3,006 F-TRACT continuous evoked-amplitude pairs (cross-observable transfer),
    including matched-capacity Permuted-Biology negative controls (3 seeds) and 1,000-resample
    stimulated-region cluster bootstrap CIs and paired p-values.

Run (~35 s):
    .venv/bin/python validate_ccep_ftract_external.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from causal_brain_simulator import fit_causal
from human_brain_model import correlations, load as load_hcp, pearson, rewire, tune, untuned
from multiscale_brain_twin import fit_multiscale, load_biology_features
from validate_simulator_ccep import build_ccep_matrix, greens_function, load_centroids, partial_spearman

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "validate_ccep_ftract_external.json"
OUT_FIG = ROOT / "figures" / "fig_ftract_external.png"


def load_ftract_360(kind, g_labels):
    path = glob.glob(str(ROOT / "data" / "f_tract" / "**" / kind / f"{kind}.csv"), recursive=True)[0]
    lines = [l for l in open(path) if not l.startswith("#")]
    names = lines[0].strip().split(",")[1:]
    mat = np.array([[float(v) for v in l.strip().split(",")[1:]] for l in lines[1:]], dtype=np.float64)
    order = [names.index(lbl) for lbl in g_labels]
    return mat[np.ix_(order, order)]


def eval_matrix(P, Y, D, mask, min_targets=10):
    rho = float(spearmanr(P[mask], Y[mask]).statistic)
    part = float(partial_spearman(P[mask], Y[mask], D[mask]))
    sp, spp = [], []
    for j in range(len(Y)):
        tg = np.flatnonzero(mask[j])
        if len(tg) >= min_targets and np.ptp(Y[j, tg]) > 0 and np.ptp(P[j, tg]) > 0:
            sp.append(float(spearmanr(P[j, tg], Y[j, tg]).statistic))
            spp.append(float(partial_spearman(P[j, tg], Y[j, tg], D[j, tg])))
    return {
        "n_pairs": int(mask.sum()),
        "pooled_rho": round(rho, 4),
        "pooled_partial": round(part, 4),
        "n_sites": len(sp),
        "site_rho": round(float(np.nanmean(sp)), 4) if sp else None,
        "site_partial": round(float(np.nanmean(spp)), 4) if spp else None,
        "site_pos_frac": round(float(np.mean(np.array(sp) > 0)), 4) if sp else None,
    }


def cluster_bootstrap_diff(P_a, P_b, Y, D, mask, n_boot=1000, seed=42):
    """Stimulated-region cluster bootstrap comparing model A minus model B (rho & partial rho | d)."""
    src_rows = np.unique(np.nonzero(mask)[0])
    # Pre-index pairs per stimulated source region for fast exact resampling
    row_slices = []
    for s in src_rows:
        tg = np.flatnonzero(mask[s])
        row_slices.append((P_a[s, tg], P_b[s, tg], Y[s, tg], D[s, tg]))

    rng = np.random.default_rng(seed)
    d_rho = np.empty(n_boot, dtype=np.float64)
    d_part = np.empty(n_boot, dtype=np.float64)
    n_src = len(src_rows)
    for b in range(n_boot):
        idx = rng.integers(0, n_src, size=n_src)
        pa = np.concatenate([row_slices[i][0] for i in idx])
        pb = np.concatenate([row_slices[i][1] for i in idx])
        yy = np.concatenate([row_slices[i][2] for i in idx])
        dd = np.concatenate([row_slices[i][3] for i in idx])
        d_rho[b] = spearmanr(pa, yy).statistic - spearmanr(pb, yy).statistic
        d_part[b] = partial_spearman(pa, yy, dd) - partial_spearman(pb, yy, dd)

    def summarize(arr):
        p_two = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
        p_two = max(p_two, 1.0 / (n_boot + 1))
        return {
            "mean_diff": round(float(arr.mean()), 4),
            "ci95_lo": round(float(np.percentile(arr, 2.5)), 4),
            "ci95_hi": round(float(np.percentile(arr, 97.5)), 4),
            "p_two_sided": round(min(p_two, 1.0), 4),
        }

    return {"delta_rho": summarize(d_rho), "delta_partial": summarize(d_part)}


def run_part_a_glasser360():
    """Part A: Native 360-region Glasser-MMP1 F-TRACT validation (0 atlas remapping)."""
    print("=== Part A: Native Glasser-360 F-TRACT (780 multi-center patients) ===", flush=True)
    read = lambda k: np.loadtxt(ROOT / "data" / "hcp_connectome" / f"{k}_ctx_glasser_360.csv", delimiter=",")
    g_labels = (ROOT / "data" / "hcp_connectome" / "strucLabels_ctx_glasser_360.csv").read_text().strip().split(",")
    log_w = read("strucMatrix")
    wiring = np.where(log_w > 0, np.exp(log_w), 0.0)
    wiring = wiring / wiring.max()
    real_fc = read("funcMatrix")
    dist = np.load(ROOT / "data" / "hcp_connectome" / "glasser_360_centre_distance.npy")

    ft_prob = load_ftract_360("probability", g_labels)
    ft_amp = load_ftract_360("amplitude", g_labels)
    hemi = np.array(["L" if l.startswith("L_") else "R" for l in g_labels])
    same_h = hemi[:, None] == hemi[None, :]

    n = len(g_labels)
    upper = np.triu_indices(n, 1)
    g0 = untuned(wiring, real_fc, upper)
    G_untuned = np.linalg.inv(np.diag(g0 * wiring.sum(1)) - g0 * wiring + np.eye(n))
    r_un = float(pearson(correlations(g0 * torch.tensor(wiring), torch.ones(n)).numpy()[upper], real_fc[upper]))

    m_fmri = tune(wiring, real_fc, upper, np.random.default_rng(0))
    with torch.no_grad():
        r_fmri = float(pearson(m_fmri().numpy()[upper], real_fc[upper]))
        w, d = m_fmri.weights().numpy(), m_fmri.log_distance.exp().numpy()
    G_fmri = np.linalg.inv(np.diag(w.sum(1)) - w + np.diag(d))

    m_rew = tune(rewire(wiring, np.random.default_rng(1)), real_fc, upper, np.random.default_rng(2))
    with torch.no_grad():
        wr, dr = m_rew.weights().numpy(), m_rew.log_distance.exp().numpy()
    G_rew = np.linalg.inv(np.diag(wr.sum(1)) - wr + np.diag(dr))

    predictors = {
        "HCP dMRI wiring": wiring,
        "HCP resting fMRI": real_fc,
        "Untuned Hopf (1p)": G_untuned,
        "fMRI-tuned Hopf": G_fmri,
        "Rewired Hopf": G_rew,
    }

    mp = np.isfinite(ft_prob) & ~np.eye(n, dtype=bool)
    ma = np.isfinite(ft_amp) & ~np.eye(n, dtype=bool)
    out = {
        "n_regions": n,
        "fmri_r_untuned": round(r_un, 4),
        "fmri_r_tuned": round(r_fmri, 4),
        "prob_pairs_total": int(mp.sum()),
        "prob_pairs_intra": int((mp & same_h).sum()),
        "prob_pairs_inter": int((mp & ~same_h).sum()),
        "prob_ceiling_frac_eq_1": round(float((ft_prob[mp] == 1.0).mean()), 4),
        "amp_pairs_total": int(ma.sum()),
        "amp_pairs_intra": int((ma & same_h).sum()),
        "prob_all": {},
        "prob_intra": {},
        "prob_inter": {},
        "amp_all": {},
        "amp_intra": {},
    }
    for name, P in predictors.items():
        out["prob_all"][name] = eval_matrix(P, ft_prob, dist, mp)
        out["prob_intra"][name] = eval_matrix(P, ft_prob, dist, mp & same_h)
        out["prob_inter"][name] = eval_matrix(P, ft_prob, dist, mp & ~same_h, min_targets=5)
        out["amp_all"][name] = eval_matrix(P, ft_amp, dist, ma)
        out["amp_intra"][name] = eval_matrix(P, ft_amp, dist, ma & same_h)
        print(
            f"  [360] {name:<18} Prob rho={out['prob_all'][name]['pooled_rho']:+.4f} "
            f"(site {out['prob_all'][name]['site_rho']:+.4f}, part {out['prob_all'][name]['pooled_partial']:+.4f}) | "
            f"Amp rho={out['amp_all'][name]['pooled_rho']:+.4f} (site {out['amp_all'][name]['site_rho']:+.4f})",
            flush=True,
        )

    out["bootstrap_untuned_vs_fmri_prob"] = cluster_bootstrap_diff(G_untuned, G_fmri, ft_prob, dist, mp, n_boot=1000, seed=10)
    out["bootstrap_untuned_vs_fmri_amp"] = cluster_bootstrap_diff(G_untuned, G_fmri, ft_amp, dist, ma, n_boot=1000, seed=11)
    return out, g_labels, ft_prob, ft_amp


def run_part_b_cross_hospital(g_labels, ft_prob_360, ft_amp_360):
    """Part B: Strict Zero-Shot Cross-Hospital Transfer (Utrecht ds004080 -> Multi-Center F-TRACT)."""
    print("\n=== Part B: Strict Out-of-Hospital Zero-Shot Transfer (Utrecht -> F-TRACT) ===", flush=True)
    s_labels, s_coords, s_hemi = load_centroids()
    s_dist = np.linalg.norm(s_coords[:, None, :] - s_coords[None, :, :], axis=-1)
    trials_u, _, prob_u, _, n_pat, _, n_raw = build_ccep_matrix(s_coords, s_hemi, exclude_soz=True)
    valid_u = (trials_u >= 10) & ~np.eye(100, dtype=bool)

    # Same-hemisphere nearest-centroid mapping Glasser-360 (fsa5) -> Schaefer-100 (MNI152)
    xyz = np.vstack([nib.load(ROOT / "data" / "hcp_connectome" / f"fsa5_{h}.gii").darrays[0].data for h in ("lh", "rh")])
    g_par = np.loadtxt(ROOT / "data" / "hcp_connectome" / "glasser_360_fsa5.csv").astype(int)
    g_cent = np.array([xyz[g_par == v].mean(0) for v in range(1, 361)])
    D_lh = cdist(g_cent[:180], s_coords[:50])
    D_rh = cdist(g_cent[180:], s_coords[50:])
    g_to_s = np.concatenate([D_lh.argmin(1), D_rh.argmin(1) + 50])
    nn_dist_med = float(np.median(np.concatenate([D_lh.min(1), D_rh.min(1)])))

    def map_360_to_100(M360):
        sums = np.zeros((100, 100), dtype=np.float64)
        cnts = np.zeros((100, 100), dtype=np.int64)
        rr, cc = np.nonzero(np.isfinite(M360) & ~np.eye(360, dtype=bool))
        for i, j in zip(rr, cc):
            si, sj = g_to_s[i], g_to_s[j]
            if si != sj:
                sums[si, sj] += M360[i, j]
                cnts[si, sj] += 1
        out = np.full((100, 100), np.nan)
        ok = cnts > 0
        out[ok] = sums[ok] / cnts[ok]
        return out

    ft_prob_100 = map_360_to_100(ft_prob_360)
    ft_amp_100 = map_360_to_100(ft_amp_360)
    m_p_all = np.isfinite(ft_prob_100) & ~np.eye(100, dtype=bool)
    m_p_unseen = m_p_all & ~valid_u
    m_a_all = np.isfinite(ft_amp_100) & ~np.eye(100, dtype=bool)
    m_shared_p = m_p_all & valid_u
    m_shared_a = m_a_all & valid_u

    emp_prob = eval_matrix(prob_u, ft_prob_100, s_dist, m_shared_p)
    emp_amp = eval_matrix(prob_u, ft_amp_100, s_dist, m_shared_a)
    print(
        f"  Empirical inter-hospital agreement (Utrecht ECoG vs F-TRACT SEEG): "
        f"Prob ({m_shared_p.sum()} pairs) rho={emp_prob['pooled_rho']:+.4f} part={emp_prob['pooled_partial']:+.4f} | "
        f"Amp ({m_shared_a.sum()} pairs) rho={emp_amp['pooled_rho']:+.4f} part={emp_amp['pooled_partial']:+.4f}",
        flush=True,
    )

    # Fit all Schaefer-100 models STRICTLY on HCP + Utrecht ds004080 (ZERO F-TRACT data!)
    wiring, real_fc, _ = load_hcp(str(ROOT / "data" / "hcp_connectome"))
    upper = np.triu_indices(100, 1)
    g0 = untuned(wiring, real_fc, upper)
    G_untuned, _ = greens_function(g0 * wiring, np.ones(100))

    m_fmri = tune(wiring, real_fc, upper, np.random.default_rng(0))
    with torch.no_grad():
        G_fmri, _ = greens_function(m_fmri.weights().numpy(), m_fmri.log_distance.exp().numpy())

    m_rew = tune(rewire(wiring, np.random.default_rng(1)), real_fc, upper, np.random.default_rng(2))
    with torch.no_grad():
        G_rew, _ = greens_function(m_rew.weights().numpy(), m_rew.log_distance.exp().numpy())

    m_ccep = fit_causal(wiring, g0, prob_u, trials_u, valid_u)
    F_bio, rec_cos, hier_diff, *_ = load_biology_features()
    m_bio_only = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob_u, trials_u, valid_u, residual_edges=False)
    m_bio_full = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, prob_u, trials_u, valid_u, residual_edges=True)
    with torch.no_grad():
        G_ccep = m_ccep.green().numpy()
        G_bio_only = m_bio_only.green().numpy()
        G_bio_full = m_bio_full.green().numpy()

    # Matched-capacity Permuted-Biology negative controls (3 independent region permutations)
    G_perm_only_list, G_perm_full_list = [], []
    for seed in (101, 102, 103):
        perm = np.random.default_rng(seed).permutation(100)
        Fp, rcp, hdp = F_bio[perm], rec_cos[np.ix_(perm, perm)], hier_diff[np.ix_(perm, perm)]
        mp_o = fit_multiscale(wiring, g0, Fp, rcp, hdp, prob_u, trials_u, valid_u, residual_edges=False)
        mp_f = fit_multiscale(wiring, g0, Fp, rcp, hdp, prob_u, trials_u, valid_u, residual_edges=True)
        with torch.no_grad():
            G_perm_only_list.append(mp_o.green().numpy())
            G_perm_full_list.append(mp_f.green().numpy())

    models = {
        "HCP dMRI wiring": wiring,
        "HCP resting fMRI": real_fc,
        "Untuned Hopf (1p)": G_untuned,
        "fMRI-tuned Hopf": G_fmri,
        "Rewired Hopf": G_rew,
        "Utrecht CCEP-Hopf (ZS)": G_ccep,
        "Utrecht Bio-Only (47p ZS)": G_bio_only,
        "Utrecht Bio+CCEP Twin (ZS)": G_bio_full,
    }

    def eval_perm_mean(G_list, Y, mask):
        vals = [eval_matrix(G, Y, s_dist, mask) for G in G_list]
        return {
            "n_pairs": vals[0]["n_pairs"],
            "pooled_rho": round(float(np.mean([v["pooled_rho"] for v in vals])), 4),
            "pooled_partial": round(float(np.mean([v["pooled_partial"] for v in vals])), 4),
            "site_rho": round(float(np.mean([v["site_rho"] for v in vals])), 4),
            "site_partial": round(float(np.mean([v["site_partial"] for v in vals])), 4),
        }

    out = {
        "mapping_nn_median_mm": round(nn_dist_med, 2),
        "schaefer_parcels_covered": int(len(np.unique(g_to_s))),
        "utrecht_patients": int(n_pat),
        "utrecht_trials_non_soz": int(n_raw),
        "empirical_inter_hospital_prob": emp_prob,
        "empirical_inter_hospital_amp": emp_amp,
        "prob_all": {},
        "prob_unseen_in_utrecht": {},
        "amp_all": {},
    }
    for name, P in models.items():
        out["prob_all"][name] = eval_matrix(P, ft_prob_100, s_dist, m_p_all)
        out["prob_unseen_in_utrecht"][name] = eval_matrix(P, ft_prob_100, s_dist, m_p_unseen)
        out["amp_all"][name] = eval_matrix(P, ft_amp_100, s_dist, m_a_all)
        print(
            f"  [100 ZS] {name:<28} ALL({m_p_all.sum()}) rho={out['prob_all'][name]['pooled_rho']:+.4f} "
            f"part={out['prob_all'][name]['pooled_partial']:+.4f} | "
            f"UNSEEN({m_p_unseen.sum()}) rho={out['prob_unseen_in_utrecht'][name]['pooled_rho']:+.4f} "
            f"part={out['prob_unseen_in_utrecht'][name]['pooled_partial']:+.4f} | "
            f"AMP({m_a_all.sum()}) rho={out['amp_all'][name]['pooled_rho']:+.4f}",
            flush=True,
        )

    out["prob_all"]["Permuted-Bio-Only Ctrl (3p)"] = eval_perm_mean(G_perm_only_list, ft_prob_100, m_p_all)
    out["prob_unseen_in_utrecht"]["Permuted-Bio-Only Ctrl (3p)"] = eval_perm_mean(G_perm_only_list, ft_prob_100, m_p_unseen)
    out["amp_all"]["Permuted-Bio-Only Ctrl (3p)"] = eval_perm_mean(G_perm_only_list, ft_amp_100, m_a_all)

    out["prob_all"]["Permuted-Bio-Full Ctrl (3p)"] = eval_perm_mean(G_perm_full_list, ft_prob_100, m_p_all)
    out["prob_unseen_in_utrecht"]["Permuted-Bio-Full Ctrl (3p)"] = eval_perm_mean(G_perm_full_list, ft_prob_100, m_p_unseen)
    out["amp_all"]["Permuted-Bio-Full Ctrl (3p)"] = eval_perm_mean(G_perm_full_list, ft_amp_100, m_a_all)

    print(
        f"  [100 ZS] {'Permuted-Bio-Only Ctrl (3p)':<28} "
        f"ALL rho={out['prob_all']['Permuted-Bio-Only Ctrl (3p)']['pooled_rho']:+.4f} "
        f"part={out['prob_all']['Permuted-Bio-Only Ctrl (3p)']['pooled_partial']:+.4f} | "
        f"UNSEEN rho={out['prob_unseen_in_utrecht']['Permuted-Bio-Only Ctrl (3p)']['pooled_rho']:+.4f} "
        f"part={out['prob_unseen_in_utrecht']['Permuted-Bio-Only Ctrl (3p)']['pooled_partial']:+.4f}",
        flush=True,
    )
    print(
        f"  [100 ZS] {'Permuted-Bio-Full Ctrl (3p)':<28} "
        f"ALL rho={out['prob_all']['Permuted-Bio-Full Ctrl (3p)']['pooled_rho']:+.4f} "
        f"part={out['prob_all']['Permuted-Bio-Full Ctrl (3p)']['pooled_partial']:+.4f} | "
        f"UNSEEN rho={out['prob_unseen_in_utrecht']['Permuted-Bio-Full Ctrl (3p)']['pooled_rho']:+.4f} "
        f"part={out['prob_unseen_in_utrecht']['Permuted-Bio-Full Ctrl (3p)']['pooled_partial']:+.4f}",
        flush=True,
    )

    # Cluster-bootstrap paired statistical tests (1,000 stimulated-site resamples)
    G_perm_only_avg = np.mean(G_perm_only_list, axis=0)
    out["bootstraps"] = {
        "bio_full_vs_fmri_all_prob": cluster_bootstrap_diff(G_bio_full, G_fmri, ft_prob_100, s_dist, m_p_all, seed=21),
        "bio_full_vs_untuned_all_prob": cluster_bootstrap_diff(G_bio_full, G_untuned, ft_prob_100, s_dist, m_p_all, seed=22),
        "bio_full_vs_uniform_ccep_all_prob": cluster_bootstrap_diff(G_bio_full, G_ccep, ft_prob_100, s_dist, m_p_all, seed=23),
        "bio_only_vs_perm_bio_unseen_prob": cluster_bootstrap_diff(G_bio_only, G_perm_only_avg, ft_prob_100, s_dist, m_p_unseen, seed=24),
        "bio_only_vs_fmri_unseen_prob": cluster_bootstrap_diff(G_bio_only, G_fmri, ft_prob_100, s_dist, m_p_unseen, seed=25),
        "bio_full_vs_fmri_amp": cluster_bootstrap_diff(G_bio_full, G_fmri, ft_amp_100, s_dist, m_a_all, seed=26),
    }
    print("\n  Cluster-bootstrap paired tests (1,000 stimulated-region resamples):", flush=True)
    for k, v in out["bootstraps"].items():
        dr, dp = v["delta_rho"], v["delta_partial"]
        print(
            f"    {k:<36} d_rho={dr['mean_diff']:+.4f} [{dr['ci95_lo']:+.4f},{dr['ci95_hi']:+.4f}] p={dr['p_two_sided']:.4f} | "
            f"d_part={dp['mean_diff']:+.4f} [{dp['ci95_lo']:+.4f},{dp['ci95_hi']:+.4f}] p={dp['p_two_sided']:.4f}",
            flush=True,
        )
    return out


def make_figure(res_a, res_b):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), constrained_layout=True)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Panel A: Native Glasser-360 F-TRACT (780 patients, 0 atlas remapping)
    ax = axes[0]
    names_a = ["Rewired Hopf", "fMRI-tuned Hopf", "HCP resting fMRI", "HCP dMRI wiring", "Untuned Hopf (1p)"]
    short_a = ["Rewired\nHopf", "fMRI-tuned\nHopf", "Resting\nfMRI", "dMRI\nwiring", "Untuned\nHopf (1p)"]
    y_pos = np.arange(len(names_a))
    prob_vals = [res_a["prob_all"][k]["pooled_rho"] for k in names_a]
    amp_vals = [res_a["amp_all"][k]["pooled_rho"] for k in names_a]
    h = 0.36
    ax.barh(y_pos + h / 2, prob_vals, height=h, color="#3a7ca5", label="Response Prob (26,771 pairs)")
    ax.barh(y_pos - h / 2, amp_vals, height=h, color="#e07a5f", label="Evoked Amplitude (15,772 pairs)")
    for i, (pv, av) in enumerate(zip(prob_vals, amp_vals)):
        ax.text(pv + 0.005, i + h / 2, f"+{pv:.3f}", va="center", fontsize=8.0, color="#1f4e6b", fontweight="bold")
        ax.text(av + 0.005, i - h / 2, f"+{av:.3f}", va="center", fontsize=8.0, color="#9e3d25", fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_a, fontsize=8.8)
    ax.set_xlim(0, 0.365)
    ax.set_ylim(-1.15, len(names_a) - 0.35)
    ax.set_xlabel("Pooled Spearman ρ vs. Multi-Center F-TRACT (780 pts)")
    ax.set_title("A. Native Glasser-360 F-TRACT Replication\n(fMRI tuning degrades both Prob & Amplitude)", fontweight="bold", fontsize=10.2)
    ax.legend(loc="lower right", frameon=True, fontsize=7.9, facecolor="white", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)

    # Panel B: Strict Zero-Shot Cross-Hospital Transfer (Utrecht -> F-TRACT, ALL 4,307 & UNSEEN 1,681)
    ax = axes[1]
    names_b = [
        "Rewired Hopf",
        "fMRI-tuned Hopf",
        "Untuned Hopf (1p)",
        "Utrecht CCEP-Hopf (ZS)",
        "Permuted-Bio-Only Ctrl (3p)",
        "Utrecht Bio-Only (47p ZS)",
        "Utrecht Bio+CCEP Twin (ZS)",
    ]
    short_b = [
        "Rewired Hopf",
        "fMRI-tuned Hopf",
        "Untuned Hopf (1p)",
        "Utrecht CCEP-Hopf",
        "Permuted-Bio Ctrl",
        "Utrecht Bio-Only (47p)",
        "Utrecht Bio+CCEP Twin",
    ]
    yb = np.arange(len(names_b))
    v_all = [res_b["prob_all"][k]["pooled_rho"] for k in names_b]
    v_unseen_part = [res_b["prob_unseen_in_utrecht"][k]["pooled_partial"] for k in names_b]
    colors_all = ["#999999", "#d95f02", "#7570b3", "#4c956c", "#b0a99f", "#1b9e77", "#0b6e4f"]
    ax.barh(yb + 0.18, v_all, height=0.35, color=colors_all, alpha=0.92)
    ax.barh(yb - 0.18, v_unseen_part, height=0.35, color="#2b59c3", alpha=0.80)
    emp_ceil = res_b["empirical_inter_hospital_prob"]["pooled_rho"]
    ax.axvline(emp_ceil, color="#222222", linestyle="--", linewidth=1.2)
    for i, (va, vu) in enumerate(zip(v_all, v_unseen_part)):
        ax.text(va + 0.006, i + 0.18, f"+{va:.3f}", va="center", fontsize=7.8, fontweight="bold")
        ax.text(vu + 0.006, i - 0.18, f"+{vu:.3f}", va="center", fontsize=7.6, color="#1d3f8f")
    ax.set_yticks(yb)
    ax.set_yticklabels(short_b, fontsize=8.4)
    ax.set_xlim(0, 0.445)
    ax.set_ylim(-1.65, len(names_b) - 0.35)
    ax.set_xlabel("Zero-Shot Correlation on 780-Patient F-TRACT")
    ax.set_title("B. Zero-Shot Out-of-Hospital Transfer\n(Trained ONLY on Utrecht ds004080, 74 pts)", fontweight="bold", fontsize=10.2)
    leg_b = [
        Patch(facecolor="#0b6e4f", label="All 4,307 pairs (Spearman ρ)"),
        Patch(facecolor="#2b59c3", alpha=0.80, label="1,681 Utrecht-unseen (partial ρ|d)"),
        Line2D([0], [0], color="#222222", linestyle="--", lw=1.2, label=f"Utrecht↔F-TRACT emp. ({emp_ceil:+.3f})"),
    ]
    ax.legend(handles=leg_b, loc="lower right", frameon=True, fontsize=7.5, facecolor="white", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)

    # Panel C: Cross-Observable Transfer to Continuous Evoked Amplitude (3,006 pairs)
    ax = axes[2]
    names_c = [
        "Rewired Hopf",
        "HCP resting fMRI",
        "fMRI-tuned Hopf",
        "HCP dMRI wiring",
        "Untuned Hopf (1p)",
        "Utrecht CCEP-Hopf (ZS)",
        "Utrecht Bio-Only (47p ZS)",
        "Utrecht Bio+CCEP Twin (ZS)",
    ]
    short_c = [
        "Rewired Hopf",
        "HCP resting fMRI",
        "fMRI-tuned Hopf",
        "HCP dMRI wiring",
        "Untuned Hopf (1p)",
        "Utrecht CCEP-Hopf",
        "Utrecht Bio-Only (47p)",
        "Utrecht Bio+CCEP Twin",
    ]
    yc = np.arange(len(names_c))
    vc_rho = [res_b["amp_all"][k]["pooled_rho"] for k in names_c]
    vc_part = [res_b["amp_all"][k]["pooled_partial"] for k in names_c]
    col_c = ["#999999", "#8da0cb", "#d95f02", "#66c2a5", "#7570b3", "#4c956c", "#1b9e77", "#0b6e4f"]
    ax.barh(yc + 0.18, vc_rho, height=0.35, color=col_c, alpha=0.92)
    ax.barh(yc - 0.18, vc_part, height=0.35, color="#e07a5f", alpha=0.82)
    emp_a = res_b["empirical_inter_hospital_amp"]["pooled_rho"]
    ax.axvline(emp_a, color="#222222", linestyle="--", linewidth=1.2)
    for i, (vr, vp) in enumerate(zip(vc_rho, vc_part)):
        ax.text(vr + 0.005, i + 0.18, f"+{vr:.3f}", va="center", fontsize=7.8, fontweight="bold")
        ax.text(vp + 0.005, i - 0.18, f"+{vp:.3f}", va="center", fontsize=7.5, color="#9e3d25")
    ax.set_yticks(yc)
    ax.set_yticklabels(short_c, fontsize=8.3)
    ax.set_xlim(0, 0.365)
    ax.set_ylim(-1.85, len(names_c) - 0.35)
    ax.set_xlabel("Zero-Shot Correlation on F-TRACT Amplitude (3,006 pairs)")
    ax.set_title("C. Cross-Observable Generalization\n(Binary Utrecht N1 Prob → F-TRACT z-Amplitude)", fontweight="bold", fontsize=10.2)
    leg_c = [
        Patch(facecolor="#0b6e4f", label="Pooled Spearman ρ"),
        Patch(facecolor="#e07a5f", alpha=0.82, label="Partial ρ | distance"),
        Line2D([0], [0], color="#222222", linestyle="--", lw=1.2, label=f"Utrecht↔F-TRACT emp. ({emp_a:+.3f})"),
    ]
    ax.legend(handles=leg_c, loc="lower right", frameon=True, fontsize=7.5, facecolor="white", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)

    p_bf = res_b["bootstraps"]["bio_full_vs_fmri_all_prob"]["delta_rho"]["p_two_sided"]
    p_bo = res_b["bootstraps"]["bio_only_vs_perm_bio_unseen_prob"]["delta_partial"]["p_two_sided"]
    fig.suptitle(
        f"Independent Out-of-Hospital Validation on Multi-Center F-TRACT SEEG (780 Patients, Zero F-TRACT Fitting) | "
        f"Bio+CCEP vs fMRI-tuned p={p_bf:.4f}; Bio-Only vs Permuted-Bio on Unseen Pairs (partial) p={p_bo:.4f}",
        fontsize=10.5,
        fontweight="bold",
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {OUT_FIG}", flush=True)


def main():
    res_a, g_labels, ft_prob_360, ft_amp_360 = run_part_a_glasser360()
    res_b = run_part_b_cross_hospital(g_labels, ft_prob_360, ft_amp_360)
    payload = {"part_a_glasser360_ftract": res_a, "part_b_utrecht_to_ftract_zeroshot": res_b}
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Saved results: {OUT_JSON}", flush=True)
    make_figure(res_a, res_b)


if __name__ == "__main__":
    main()
