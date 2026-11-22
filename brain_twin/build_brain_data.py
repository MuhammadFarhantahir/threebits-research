"""Export compact real 100-region human brain twin data into website/brain/brain_data.js.

Bundles:
  - 100 real Schaefer parcels (MNI 3D coords, Yeo-7 networks, MEG timescales, Mesulam hierarchy,
    19 human PET neurotransmitter receptor densities from Hansen et al. 2022)
  - 74-patient empirical human CCEP response matrix (OpenNeuro ds004080, 277,638 non-SOZ trials)
  - Threebits CCEP + 19-PET + MEG Multi-Scale Twin causal propagation matrix
  - Resting-fMRI-tuned Hopf propagation matrix (showing observational vs causal failure live!)
  - Top causal white-matter tracts for 3D animated pulse propagation
  - Verified benchmark tables (Working Memory K=2..16, Multi-Hop Fluid Reasoning D=1..8*, Continual 3-Task)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from functional_brain_emulator import get_causal_human_scaffold
from human_brain_model import load as load_hcp, tune
from validate_simulator_ccep import greens_function, partial_spearman

OUT_JS = Path(__file__).resolve().parent / "brain_data.js"
BIO_NPZ = ROOT / "data" / "multiscale_atlas" / "schaefer100_multiscale_biology.npz"


def _quantize_mat(M, valid_mask=None):
    """Quantize non-negative matrix rows to compact integers 0..1000 (-1 where invalid)."""
    out = []
    for i in range(M.shape[0]):
        row = M[i]
        m = valid_mask[i] if valid_mask is not None else np.ones_like(row, dtype=bool)
        vals = row[m]
        hi = float(np.percentile(vals, 99)) if len(vals) and np.max(vals) > 0 else 1.0
        hi = max(hi, 1e-6)
        qrow = []
        for j in range(M.shape[1]):
            if valid_mask is not None and not valid_mask[i, j]:
                qrow.append(-1)
            else:
                qrow.append(int(round(min(max(float(row[j]) / hi, 0.0), 1.0) * 1000)))
        out.append(qrow)
    return out


def main():
    scaf = get_causal_human_scaffold()
    bio = np.load(BIO_NPZ, allow_pickle=True)
    rec_names = [str(r) for r in bio["receptor_names"]]
    rec_raw = bio["receptors"].astype(np.float64)
    # Z-score receptors across the 100 regions for intuitive biological inspection
    rec_z = (rec_raw - rec_raw.mean(axis=0, keepdims=True)) / np.maximum(rec_raw.std(axis=0, keepdims=True), 1e-6)
    key_recs = ["A4B2", "VAChT", "GABAa", "NMDA", "mGluR5", "D1", "5HT2a", "NAT"]
    key_idx = [rec_names.index(k) for k in key_recs if k in rec_names]
    key_recs = [rec_names[i] for i in key_idx]

    coords = scaf["coords"]  # (100, 3) MNI mm
    labels = scaf["labels"]
    prob = scaf["prob"]      # empirical 74-patient CCEP [src, dst]
    valid = scaf["valid"]
    dist_mat = scaf["dist_mat"]

    # 1. Multi-Scale CCEP + 19-PET Twin causal Green's function (evoked 50..200 ms response)
    W_c = scaf["W_causal"].astype(np.float64)
    d_c = scaf["d_causal"].astype(np.float64)
    K_mat = np.diag(d_c + W_c.sum(axis=1)) - W_c
    G_twin = np.linalg.inv(K_mat).T  # [src, dst]
    np.fill_diagonal(G_twin, 0.0)

    # 2. Exact Paper 1 Resting-fMRI-Tuned Hopf simulator (zero CCEP seen -> observational vs causal failure!)
    wiring, real_fc, _ = load_hcp(str(ROOT / "data" / "hcp_connectome"))
    upper = np.triu_indices(100, 1)
    tuned = tune(wiring, real_fc, upper, np.random.default_rng(0))
    with torch.no_grad():
        G_fmri, _ = greens_function(tuned.weights().numpy(), tuned.log_distance.exp().numpy())
    np.fill_diagonal(G_fmri, 0.0)

    # Compute exact global + per-region Spearman rho & partial rho | dist
    twin_global_rho = float(spearmanr(G_twin[valid], prob[valid]).statistic)
    twin_global_prho = float(partial_spearman(G_twin[valid], prob[valid], dist_mat[valid]))
    fmri_global_rho = float(spearmanr(G_fmri[valid], prob[valid]).statistic)
    fmri_global_prho = float(partial_spearman(G_fmri[valid], prob[valid], dist_mat[valid]))
    print(f"Global CCEP check: Twin rho={twin_global_rho:+.3f} (p|d {twin_global_prho:+.3f}) | fMRI-tuned rho={fmri_global_rho:+.3f} (p|d {fmri_global_prho:+.3f})")

    # Build 100 region records
    mesulam = bio["mesulam_hierarchy"].astype(np.float64)
    meg_p = bio["meg_power"].astype(np.float64)
    meg_slow_ratio = (meg_p[:, 0] + meg_p[:, 1]) / np.maximum(meg_p[:, 4] + meg_p[:, 5], 1e-6)
    meg_slow_z = (meg_slow_ratio - meg_slow_ratio.mean()) / max(meg_slow_ratio.std(), 1e-6)

    regions = []
    for i in range(100):
        raw_lbl = str(labels[i])
        lbl = raw_lbl.replace("7Networks_", "")
        parts = lbl.split("_")
        hemi = "L" if parts[0].startswith("LH") else "R"
        net = parts[1] if len(parts) > 1 else "Other"
        short = f"{hemi} {' '.join(parts[1:])}"
        m = valid[i] & (np.arange(100) != i)
        n_meas = int(m.sum())
        r_twin = float(spearmanr(G_twin[i, m], prob[i, m]).statistic) if n_meas >= 8 else None
        r_fmri = float(spearmanr(G_fmri[i, m], prob[i, m]).statistic) if n_meas >= 8 else None
        pr_twin = float(partial_spearman(G_twin[i, m], prob[i, m], dist_mat[i, m])) if n_meas >= 8 else None
        pr_fmri = float(partial_spearman(G_fmri[i, m], prob[i, m], dist_mat[i, m])) if n_meas >= 8 else None
        regions.append({
            "id": i,
            "name": lbl,
            "short": short,
            "hemi": hemi,
            "net": net,
            "xyz": [round(float(v), 1) for v in coords[i]],
            "alpha": round(float(scaf["alpha_bio"][i]), 3),
            "ei": round(float(scaf["ei_gain"][i]), 3),
            "mesulam": round(float(mesulam[i]), 2),
            "meg_slow_z": round(float(meg_slow_z[i]), 2),
            "rec": {k: round(float(rec_z[i, idx]), 2) for k, idx in zip(key_recs, key_idx)},
            "n_ccep": n_meas,
            "rho_twin": round(r_twin, 3) if (r_twin is not None and np.isfinite(r_twin)) else None,
            "prho_twin": round(pr_twin, 3) if (pr_twin is not None and np.isfinite(pr_twin)) else None,
            "rho_fmri": round(r_fmri, 3) if (r_fmri is not None and np.isfinite(r_fmri)) else None,
            "prho_fmri": round(pr_fmri, 3) if (pr_fmri is not None and np.isfinite(pr_fmri)) else None,
            "in_r101_iface": bool(i in scaf["cont_idx"] or i in scaf["dors_idx"]),
            "is_cont": bool(i in scaf["cont_idx"]),
            "is_vis": bool(i in scaf["vis_idx"]),
            "is_mot": bool(i in scaf["mot_idx"]),
        })

    # Top 260 directed causal white-matter edges for 3D visual tracts
    W_off = W_c.copy()
    np.fill_diagonal(W_off, 0.0)
    flat_idx = np.argsort(W_off.ravel())[::-1][:260]
    w_max = float(W_off.max()) or 1.0
    tracts = []
    for fi in flat_idx:
        dst, src = divmod(int(fi), 100)
        tracts.append([src, dst, round(float(W_off[dst, src] / w_max), 3)])

    # Load saved JSON benchmarks (wm, reasoning, continual if present)
    def load_opt(fname):
        p = ROOT / fname
        return json.loads(p.read_text()) if p.exists() else None

    payload = {
        "meta": {
            "n_regions": 100,
            "n_ccep_patients": 74,
            "n_ccep_trials": 277638,
            "n_pet_receptors": 19,
            "n_tmseeg_adults": 46,
            "twin_global_rho": round(twin_global_rho, 3),
            "twin_global_prho": round(twin_global_prho, 3),
            "fmri_global_rho": round(fmri_global_rho, 3),
            "fmri_global_prho": round(fmri_global_prho, 3),
            "receptor_keys": key_recs,
        },
        "regions": regions,
        "tracts": tracts,
        "mat_twin": _quantize_mat(G_twin),
        "mat_fmri": _quantize_mat(G_fmri),
        "mat_emp": _quantize_mat(prob, valid_mask=valid),
        "bench_wm": load_opt("functional_brain_emulator.json"),
        "bench_reasoning": load_opt("superhuman_reasoning_emulator.json"),
        "bench_continual": load_opt("continual_multitask_brain.json"),
        "bench_autonomous": load_opt("autonomous_brain_bg_thalamus.json"),
        "bench_unified": load_opt("unified_spiking_chip_brain.json"),
        "bench_coadapt": load_opt("zero_backprop_coadapt.json"),
        "bench_arc": load_opt("arc_program_brain.json"),
        "bench_patients50": load_opt("personalized_50brain_rescue.json"),
        "bench_scaling1e9": load_opt("scaling_law_1e9x.json"),
        "bench_collective8": load_opt("collective_multibrain_superintelligence.json"),
        "bench_laminar_sleep": load_opt("laminar_spiking_sleep_eeg.json"),
    }

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    OUT_JS.write_text("window.THREEBITS_BRAIN_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"Wrote {OUT_JS} ({OUT_JS.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
