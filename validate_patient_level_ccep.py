"""Strict 74-Patient Leave-Patients-Out (LPO) CCEP Validation & Per-Patient Bipolar E-Field VTA Test.

Evaluates whole-brain causal stimulation models STRICTLY on completely unseen neurosurgical
patients (OpenNeuro ds004080, UMC Utrecht, 74 patients, 277,638 non-SOZ electrode trials)
at the INDIVIDUAL ELECTRODE-PAIR level — controlling for exact millimeter electrode-to-electrode
Euclidean distance (not just parcel centroid distance):
  1. 5-fold Leave-Patients-Out CV (zero electrodes/trials from test patients ever seen in training):
     - Per-patient electrode-pair N1 Detection AUROC
     - Per-patient Spearman rho & distance-controlled partial rho | electrode distance
     - Paired patient-level Wilcoxon signed-rank tests (Bio+CCEP vs fMRI-tuned, Untuned, Uniform CCEP, Permuted-Bio)
  2. Discrete Nearest-Centroid Parcel Assignment vs. Continuous Bipolar Dipole E-Field VTA Mapping:
     - Uses each bipolar pair's exact MNI152 anode/cathode coordinates (e1, e2) to project stimulation
       through physical volume-conduction decay into Schaefer-100 cortical parcels.

Run (~50 s):
    .venv/bin/python -u validate_patient_level_ccep.py
"""
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, wilcoxon

from causal_brain_simulator import fit_causal
from human_brain_model import load as load_hcp, tune, untuned
from multiscale_brain_twin import fit_multiscale, load_biology_features
from validate_simulator_ccep import greens_function, load_centroids, partial_spearman

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "validate_patient_level_ccep.json"
OUT_FIG = ROOT / "figures" / "fig_patient_lpo_ccep.png"
MAX_DIST_MM = 25.0


def load_patient_ages():
    ages = {}
    with open(ROOT / "data" / "ccep_ds004080" / "participants.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                ages[row["participant_id"].strip()] = float(row["age"])
            except (KeyError, ValueError):
                pass
    return ages


def bipolar_efield_weights(p1, p2, coords, hemi_mask, sigma_mm=15.0):
    """Continuous same-hemisphere bipolar E-field recruitment weights over Schaefer-100 parcels.

    Models localized cortical recruitment around bipolar contact pair (p1, p2) via regularized
    inverse-distance dipole field magnitude combined with Gaussian cortical-volume smoothing.
    Returns normalized weight vector w (100,) summing to 1 over same-hemisphere parcels.
    """
    r1 = coords - p1[None, :]
    r2 = coords - p2[None, :]
    d1 = np.linalg.norm(r1, axis=1)
    d2 = np.linalg.norm(r2, axis=1)
    # Regularized dipole field strength |r1/(d1^2+r0^2)^1.5 - r2/(d2^2+r0^2)^1.5|
    r0_sq = 6.0 ** 2  # 6 mm regularization radius (~cortical sheet thickness + contact radius)
    e_vec = r1 / ((d1 ** 2 + r0_sq) ** 1.5)[:, None] - r2 / ((d2 ** 2 + r0_sq) ** 1.5)[:, None]
    e_mag = np.linalg.norm(e_vec, axis=1)
    mid = 0.5 * (p1 + p2)
    d_mid = np.linalg.norm(coords - mid[None, :], axis=1)
    w = e_mag * np.exp(-0.5 * (d_mid / sigma_mm) ** 2)
    w = np.where(hemi_mask, w, 0.0)
    s = w.sum()
    if s <= 1e-12:
        cand = np.flatnonzero(hemi_mask)
        best = cand[int(np.argmin(d_mid[cand]))]
        w = np.zeros(len(coords), dtype=np.float64)
        w[best] = 1.0
        return w
    return w / s


def load_per_patient_ccep(coords, hemi, exclude_soz=True):
    """Extract per-patient non-SOZ electrode-pair records and parcel-aggregated trial/hit tables."""
    mat = sio.loadmat(str(ROOT / "data" / "ccep_ds004080" / "ccepData_V2.mat"), simplify_cells=True)["ccepData"]
    ages = load_patient_ages()
    elec_files = {}
    for p in glob.glob(str(ROOT / "data" / "ccep_ds004080" / "electrodes" / "*.tsv")):
        sub = Path(p).name.split("_")[0]
        elec_files.setdefault(sub, []).append(p)

    patients = []
    for pat in mat:
        sub = str(pat["id"])
        efiles = elec_files.get(sub, [])
        if not efiles:
            continue
        ses = pat.get("ses", "")
        ef = next((p for p in efiles if f"_{ses}_" in p), efiles[0])
        elec_xyz, elec_roi, elec_soz = {}, {}, {}
        with open(ef) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                name = row["name"].strip()
                try:
                    xyz = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
                except ValueError:
                    continue
                if not np.all(np.isfinite(xyz)):
                    continue
                h = row.get("hemisphere", "").strip().upper()
                if h not in ("L", "R"):
                    h = "L" if xyz[0] < 0 else "R"
                cand = np.flatnonzero(hemi == h)
                dists = np.linalg.norm(coords[cand] - xyz[None, :], axis=1)
                bi = int(np.argmin(dists))
                elec_xyz[name] = (xyz, h)
                elec_soz[name] = (row.get("soz", "no").strip().lower() == "yes")
                if dists[bi] <= MAX_DIST_MM:
                    elec_roi[name] = int(cand[bi])

        trials = np.zeros((100, 100), dtype=np.int64)
        hits = np.zeros((100, 100), dtype=np.int64)
        # Raw electrode-pair records for exact patient-level electrode evaluation
        pair_src_roi, pair_dst_roi, pair_dist, pair_hit, pair_lat = [], [], [], [], []
        pair_src_w_list = []
        stim_w_cache = {}

        runs = pat["run"] if isinstance(pat["run"], list) else [pat["run"]]
        for run in runs:
            if not isinstance(run, dict) or "n1_peak_sample" not in run:
                continue
            n1 = np.asarray(run["n1_peak_sample"], dtype=float)
            if n1.ndim == 1:
                n1 = n1[:, None]
            ch_names = [str(x).strip() for x in np.atleast_1d(run["channel_names"])]
            sp_names = [str(x).strip() for x in np.atleast_1d(run["stimpair_names"])]
            good_idx = np.atleast_1d(np.asarray(run["good_channels"], dtype=int)) - 1
            tt = np.asarray(run["tt"], dtype=float)

            rec_rois = {}
            for ci in good_idx:
                if 0 <= ci < len(ch_names):
                    cn = ch_names[ci]
                    if cn in elec_roi and (not exclude_soz or not elec_soz.get(cn, False)):
                        rec_rois[ci] = (cn, elec_roi[cn], elec_xyz[cn][0])

            for sj, sp in enumerate(sp_names):
                if sj >= n1.shape[1] or "-" not in sp:
                    continue
                e1, e2 = [x.strip() for x in sp.split("-", 1)]
                if e1 not in elec_xyz or e2 not in elec_xyz:
                    continue
                if exclude_soz and (elec_soz.get(e1, False) or elec_soz.get(e2, False)):
                    continue
                p1, h1 = elec_xyz[e1]
                p2, _ = elec_xyz[e2]
                mid = 0.5 * (p1 + p2)
                hemi_mask = (hemi == h1)
                cand = np.flatnonzero(hemi_mask)
                dists = np.linalg.norm(coords[cand] - mid[None, :], axis=1)
                bi = int(np.argmin(dists))
                if dists[bi] > MAX_DIST_MM:
                    continue
                src_roi = int(cand[bi])
                if sp not in stim_w_cache:
                    stim_w_cache[sp] = bipolar_efield_weights(p1, p2, coords, hemi_mask)
                src_w = stim_w_cache[sp]

                for ci, (cn, dst_roi, dst_xyz) in rec_rois.items():
                    if cn in (e1, e2) or src_roi == dst_roi:
                        continue
                    edist = float(np.linalg.norm(dst_xyz - mid))
                    val = n1[ci, sj]
                    hit = 1 if np.isfinite(val) else 0
                    lat_ms = np.nan
                    if hit:
                        sidx = int(round(val)) - 1
                        if 0 <= sidx < len(tt):
                            lat_ms = float(tt[sidx] * 1000.0)
                    trials[src_roi, dst_roi] += 1
                    hits[src_roi, dst_roi] += hit
                    pair_src_roi.append(src_roi)
                    pair_dst_roi.append(dst_roi)
                    pair_dist.append(edist)
                    pair_hit.append(hit)
                    pair_lat.append(lat_ms)
                    pair_src_w_list.append(sp)

        if len(pair_hit) > 0:
            patients.append({
                "id": sub,
                "age": ages.get(sub, np.nan),
                "trials": trials,
                "hits": hits,
                "src_roi": np.array(pair_src_roi, dtype=np.int32),
                "dst_roi": np.array(pair_dst_roi, dtype=np.int32),
                "elec_dist": np.array(pair_dist, dtype=np.float64),
                "hit": np.array(pair_hit, dtype=np.int32),
                "lat_ms": np.array(pair_lat, dtype=np.float64),
                "stim_w_keys": pair_src_w_list,
                "stim_w_cache": stim_w_cache,
            })
    return patients


def score_patient(pat, G_mat, use_vta=False):
    """Score a model matrix G_mat[src, dst] on an individual patient's raw electrode pairs."""
    y = pat["hit"]
    d = pat["elec_dist"]
    if len(y) < 30 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    if not use_vta:
        pred = G_mat[pat["src_roi"], pat["dst_roi"]]
    else:
        # Compute bipolar VTA-weighted prediction: sum_k w_k * G[k, dst_roi]
        pred = np.empty(len(y), dtype=np.float64)
        # Group by stim_pair key for fast matrix-vector product
        by_sp = {}
        for idx, sp in enumerate(pat["stim_w_keys"]):
            by_sp.setdefault(sp, []).append(idx)
        for sp, idxs in by_sp.items():
            idxs_arr = np.array(idxs, dtype=np.int64)
            vta_resp = pat["stim_w_cache"][sp] @ G_mat  # (100,)
            pred[idxs_arr] = vta_resp[pat["dst_roi"][idxs_arr]]

    # Aggregate repeated stimulations of identical (src_roi, dst_roi) is NOT done here:
    # we score EVERY raw electrode pair directly!
    if np.ptp(pred) <= 1e-12:
        return None
    auc = float(roc_auc_score(y, pred))
    rho = float(spearmanr(pred, y).statistic)
    part = float(partial_spearman(pred, y, d))
    return {"auc": auc, "rho": rho, "partial": part, "n_pairs": int(len(y)), "hit_rate": float(y.mean())}


def main():
    labels, coords, hemi = load_centroids()
    patients = load_per_patient_ccep(coords, hemi, exclude_soz=True)
    print(f"=== Strict 74-Patient Leave-Patients-Out (LPO) CCEP Validation ({len(patients)} patients) ===", flush=True)

    wiring, real_fc, _ = load_hcp(str(ROOT / "data" / "hcp_connectome"))
    upper = np.triu_indices(100, 1)
    g0 = untuned(wiring, real_fc, upper)
    G_untuned, _ = greens_function(g0 * wiring, np.ones(100))

    m_fmri = tune(wiring, real_fc, upper, np.random.default_rng(0))
    with torch.no_grad():
        G_fmri, _ = greens_function(m_fmri.weights().numpy(), m_fmri.log_distance.exp().numpy())

    F_bio, rec_cos, hier_diff, *_ = load_biology_features()
    perm = np.random.default_rng(101).permutation(100)
    Fp, rcp, hdp = F_bio[perm], rec_cos[np.ix_(perm, perm)], hier_diff[np.ix_(perm, perm)]

    # Assign all patients deterministically to 5 patient-disjoint folds
    rng = np.random.default_rng(42)
    order = rng.permutation(len(patients))
    fold_of = np.zeros(len(patients), dtype=int)
    for k, idx in enumerate(order):
        fold_of[idx] = k % 5

    model_names = [
        "HCP dMRI wiring",
        "HCP resting fMRI",
        "Untuned Hopf (1p)",
        "fMRI-tuned Hopf",
        "Uniform CCEP-Hopf (LPO)",
        "Permuted-Bio Twin (LPO)",
        "Multi-Scale Bio-Only (47p LPO)",
        "Multi-Scale Bio+CCEP Twin (LPO)",
        "Multi-Scale Bio+CCEP + Bipolar VTA (LPO)",
    ]
    per_patient_scores = {nm: [] for nm in model_names}
    scored_patient_ids = []
    scored_patient_ages = []
    scored_patient_n = []

    for f in range(5):
        tr_pats = [p for i, p in enumerate(patients) if fold_of[i] != f]
        te_pats = [p for i, p in enumerate(patients) if fold_of[i] == f]

        tr_trials = np.sum([p["trials"] for p in tr_pats], axis=0)
        tr_hits = np.sum([p["hits"] for p in tr_pats], axis=0)
        tr_prob = np.divide(tr_hits, tr_trials, out=np.full((100, 100), np.nan), where=tr_trials > 0)
        tr_mask = (tr_trials >= 10) & ~np.eye(100, dtype=bool)

        m_ccep = fit_causal(wiring, g0, tr_prob, tr_trials, tr_mask)
        m_perm = fit_multiscale(wiring, g0, Fp, rcp, hdp, tr_prob, tr_trials, tr_mask, residual_edges=True)
        m_bo = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, tr_prob, tr_trials, tr_mask, residual_edges=False)
        m_bf = fit_multiscale(wiring, g0, F_bio, rec_cos, hier_diff, tr_prob, tr_trials, tr_mask, residual_edges=True)

        with torch.no_grad():
            G_ccep = m_ccep.green().numpy()
            G_perm = m_perm.green().numpy()
            G_bo = m_bo.green().numpy()
            G_bf = m_bf.green().numpy()

        fold_models = {
            "HCP dMRI wiring": (wiring, False),
            "HCP resting fMRI": (real_fc, False),
            "Untuned Hopf (1p)": (G_untuned, False),
            "fMRI-tuned Hopf": (G_fmri, False),
            "Uniform CCEP-Hopf (LPO)": (G_ccep, False),
            "Permuted-Bio Twin (LPO)": (G_perm, False),
            "Multi-Scale Bio-Only (47p LPO)": (G_bo, False),
            "Multi-Scale Bio+CCEP Twin (LPO)": (G_bf, False),
            "Multi-Scale Bio+CCEP + Bipolar VTA (LPO)": (G_bf, True),
        }
        print(f"  Fold {f+1}/5: trained on {len(tr_pats)} patients ({int(tr_mask.sum())} pairs), testing {len(te_pats)} unseen patients...", flush=True)
        for p in te_pats:
            res_map = {}
            ok = True
            for nm, (G, vta) in fold_models.items():
                sc = score_patient(p, G, use_vta=vta)
                if sc is None:
                    ok = False
                    break
                res_map[nm] = sc
            if ok:
                scored_patient_ids.append(p["id"])
                scored_patient_ages.append(None if np.isnan(p["age"]) else float(p["age"]))
                scored_patient_n.append(res_map["Multi-Scale Bio+CCEP Twin (LPO)"]["n_pairs"])
                for nm in model_names:
                    per_patient_scores[nm].append(res_map[nm])

    n_eval = len(scored_patient_ids)
    print(f"\n=== Individual Unseen-Patient Results (n = {n_eval} / {len(patients)} eligible patients, raw electrode pairs) ===", flush=True)
    summary = {}
    for nm in model_names:
        aucs = np.array([x["auc"] for x in per_patient_scores[nm]])
        rhos = np.array([x["rho"] for x in per_patient_scores[nm]])
        parts = np.array([x["partial"] for x in per_patient_scores[nm]])
        summary[nm] = {
            "mean_auc": round(float(aucs.mean()), 4),
            "sd_auc": round(float(aucs.std(ddof=1)), 4),
            "median_auc": round(float(np.median(aucs)), 4),
            "frac_auc_gt_05": round(float((aucs > 0.5).mean()), 4),
            "n_auc_gt_05": int((aucs > 0.5).sum()),
            "mean_rho": round(float(rhos.mean()), 4),
            "mean_partial": round(float(parts.mean()), 4),
            "sd_partial": round(float(parts.std(ddof=1)), 4),
            "n_partial_pos": int((parts > 0.0).sum()),
        }
        s = summary[nm]
        print(
            f"  {nm:<42} AUROC={s['mean_auc']:.4f}±{s['sd_auc']:.4f} ({s['n_auc_gt_05']}/{n_eval}>0.5) | "
            f"part_rho|d_elec={s['mean_partial']:+.4f}±{s['sd_partial']:.4f} ({s['n_partial_pos']}/{n_eval}>0)",
            flush=True,
        )

    # Paired patient-level Wilcoxon signed-rank tests
    comparisons = [
        ("Multi-Scale Bio+CCEP Twin (LPO)", "fMRI-tuned Hopf"),
        ("Multi-Scale Bio+CCEP Twin (LPO)", "Untuned Hopf (1p)"),
        ("Multi-Scale Bio+CCEP Twin (LPO)", "Uniform CCEP-Hopf (LPO)"),
        ("Multi-Scale Bio+CCEP Twin (LPO)", "Permuted-Bio Twin (LPO)"),
        ("Multi-Scale Bio+CCEP + Bipolar VTA (LPO)", "Multi-Scale Bio+CCEP Twin (LPO)"),
        ("Multi-Scale Bio+CCEP + Bipolar VTA (LPO)", "fMRI-tuned Hopf"),
        ("Untuned Hopf (1p)", "fMRI-tuned Hopf"),
    ]
    wilcoxon_tests = {}
    print("\n=== Paired Patient-Level Wilcoxon Signed-Rank Tests (n = %d unseen patients) ===" % n_eval, flush=True)
    for a, b in comparisons:
        auc_a = np.array([x["auc"] for x in per_patient_scores[a]])
        auc_b = np.array([x["auc"] for x in per_patient_scores[b]])
        part_a = np.array([x["partial"] for x in per_patient_scores[a]])
        part_b = np.array([x["partial"] for x in per_patient_scores[b]])
        w_auc = wilcoxon(auc_a - auc_b)
        w_part = wilcoxon(part_a - part_b)
        key = f"{a} vs {b}"
        wilcoxon_tests[key] = {
            "delta_auc": round(float((auc_a - auc_b).mean()), 4),
            "wins_auc": int((auc_a > auc_b).sum()),
            "p_auc": float(w_auc.pvalue),
            "delta_partial": round(float((part_a - part_b).mean()), 4),
            "wins_partial": int((part_a > part_b).sum()),
            "p_partial": float(w_part.pvalue),
        }
        t = wilcoxon_tests[key]
        print(
            f"  {key:<74} dAUC={t['delta_auc']:+.4f} ({t['wins_auc']}/{n_eval}, p={t['p_auc']:.2e}) | "
            f"dPart={t['delta_partial']:+.4f} ({t['wins_partial']}/{n_eval}, p={t['p_partial']:.2e})",
            flush=True,
        )

    out = {
        "n_total_patients": len(patients),
        "n_evaluated_patients": n_eval,
        "total_electrode_pairs_scored": int(sum(scored_patient_n)),
        "summary": summary,
        "wilcoxon_tests": wilcoxon_tests,
        "per_patient": [
            {
                "id": scored_patient_ids[i],
                "age": scored_patient_ages[i],
                "n_pairs": int(scored_patient_n[i]),
                "fmri_auc": round(per_patient_scores["fMRI-tuned Hopf"][i]["auc"], 4),
                "untuned_auc": round(per_patient_scores["Untuned Hopf (1p)"][i]["auc"], 4),
                "bio_ccep_auc": round(per_patient_scores["Multi-Scale Bio+CCEP Twin (LPO)"][i]["auc"], 4),
                "bio_ccep_vta_auc": round(per_patient_scores["Multi-Scale Bio+CCEP + Bipolar VTA (LPO)"][i]["auc"], 4),
                "fmri_partial": round(per_patient_scores["fMRI-tuned Hopf"][i]["partial"], 4),
                "bio_ccep_partial": round(per_patient_scores["Multi-Scale Bio+CCEP Twin (LPO)"][i]["partial"], 4),
                "bio_ccep_vta_partial": round(per_patient_scores["Multi-Scale Bio+CCEP + Bipolar VTA (LPO)"][i]["partial"], 4),
            }
            for i in range(n_eval)
        ],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Saved: {OUT_JSON}", flush=True)
    make_figure(out, per_patient_scores)


def make_figure(out, per_patient_scores):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)
    n_eval = out["n_evaluated_patients"]
    summ = out["summary"]

    # Panel A: Per-Patient Electrode-Pair N1 Detection AUROC (bar + IQR/SD)
    ax = axes[0]
    order_a = [
        "HCP resting fMRI",
        "fMRI-tuned Hopf",
        "HCP dMRI wiring",
        "Untuned Hopf (1p)",
        "Uniform CCEP-Hopf (LPO)",
        "Permuted-Bio Twin (LPO)",
        "Multi-Scale Bio+CCEP Twin (LPO)",
        "Multi-Scale Bio+CCEP + Bipolar VTA (LPO)",
    ]
    short_a = [
        "Resting fMRI",
        "fMRI-tuned Hopf",
        "dMRI wiring",
        "Untuned Hopf",
        "Uniform CCEP (LPO)",
        "Permuted-Bio (LPO)",
        "Bio+CCEP Twin (LPO)",
        "Bio+CCEP + VTA (LPO)",
    ]
    y = np.arange(len(order_a))
    means = [summ[k]["mean_auc"] for k in order_a]
    sds = [summ[k]["sd_auc"] / np.sqrt(n_eval) for k in order_a]
    cols = ["#8da0cb", "#d95f02", "#66c2a5", "#7570b3", "#4c956c", "#b0a99f", "#1b9e77", "#0b6e4f"]
    ax.barh(y, means, xerr=sds, height=0.62, color=cols, alpha=0.92, capsize=3)
    ax.axvline(0.50, color="#555555", linestyle="--", lw=1.0, label="Chance (0.50)")
    for i, m in enumerate(means):
        ax.text(m + 0.012, i, f"{m:.3f}", va="center", fontsize=8.0, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(short_a, fontsize=8.4)
    ax.set_xlim(0.45, 0.88)
    ax.set_xlabel("Per-Patient Electrode-Pair N1 AUROC (Mean ± SEM)")
    ax.set_title(f"A. Unseen-Patient N1 Detection AUROC\n(5-Fold Leave-Patients-Out, n = {n_eval} patients)", fontweight="bold", fontsize=10.0)
    ax.grid(axis="x", alpha=0.25)

    # Panel B: Distance-Controlled Partial rho | Exact Millimeter Electrode Distance
    ax = axes[1]
    pmeans = [summ[k]["mean_partial"] for k in order_a]
    psems = [summ[k]["sd_partial"] / np.sqrt(n_eval) for k in order_a]
    ax.barh(y, pmeans, xerr=psems, height=0.62, color=cols, alpha=0.92, capsize=3)
    ax.axvline(0.0, color="#555555", linestyle="--", lw=1.0)
    for i, pm in enumerate(pmeans):
        ax.text(pm + psems[i] + 0.002, i, f"{pm:+.3f}", va="center", fontsize=8.0, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(short_a, fontsize=8.4)
    ax.set_xlim(-0.005, 0.068)
    ax.set_xlabel("Partial Spearman ρ | Electrode Distance (mm)")
    ax.set_title("B. Distance-Controlled Individual Accuracy\n(Controlling Exact Contact-to-Contact Distance)", fontweight="bold", fontsize=10.0)
    ax.grid(axis="x", alpha=0.25)

    # Panel C: Paired Per-Patient Scatter — Bio+CCEP+VTA vs fMRI-tuned Hopf
    ax = axes[2]
    x_fmri = np.array([r["fmri_auc"] for r in out["per_patient"]])
    y_best = np.array([r["bio_ccep_vta_auc"] for r in out["per_patient"]])
    wins = int((y_best > x_fmri).sum())
    p_val = out["wilcoxon_tests"]["Multi-Scale Bio+CCEP + Bipolar VTA (LPO) vs fMRI-tuned Hopf"]["p_auc"]
    ax.scatter(x_fmri, y_best, c="#0b6e4f", alpha=0.78, s=36, edgecolors="white", linewidths=0.5)
    lims = [min(x_fmri.min(), y_best.min()) - 0.03, max(x_fmri.max(), y_best.max()) + 0.03]
    ax.plot(lims, lims, "--", color="#666666", lw=1.1, label="Parity (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("fMRI-Tuned Hopf AUROC (per patient)")
    ax.set_ylabel("Bio+CCEP + Bipolar VTA AUROC (per patient)")
    ax.set_title(f"C. Individual Patient Win Rate\n({wins}/{n_eval} patients improved, Wilcoxon p = {p_val:.1e})", fontweight="bold", fontsize=10.0)
    ax.legend(loc="lower right", frameon=True, fontsize=8.2)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Strict 5-Fold Leave-Patients-Out Validation Across {n_eval} Unseen Human Neurosurgical Patients "
        f"({out['total_electrode_pairs_scored']:,} Raw Non-SOZ Electrode Pairs, OpenNeuro ds004080)",
        fontsize=10.5,
        fontweight="bold",
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
