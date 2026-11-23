"""Real Human Axonal Conduction-Delay Physics, Developmental Velocity (Ages 4–51),
Zero-Shot Cross-Hospital Latency Transfer, and Biophysical Waveform/Frequency Transfer.

Directly addresses the conduction-delay and pulse-waveform physics gap using TWO independent
real human intracranial latency datasets + real clinical RAM waveform settings on disk:
  Part A — 780-Patient Multi-Center F-TRACT Glasser-360 Latency Physics
           (15,725 onset latency pairs & 15,772 peak latency pairs; fiber_distance + multi-hop dMRI graph)
  Part B — 74-Patient UMCU Utrecht ds004080 Developmental Myelination (ages 4–51)
           + 5-fold Leave-Patients-Out millisecond latency prediction
           + Strict Zero-Shot Cross-Hospital Latency Transfer (Utrecht ds004080 -> F-TRACT 780 pts)
  Part C — Biophysical Charge-Recruitment & Tsodyks-Markram Frequency-Dependent Pulse Operator
           (1 Hz SPES, 10–50 Hz rhythmic/theta-burst, 100–200 Hz HFS/DBS) profiled across all 40
           real clinical DARPA RAM sessions (ds005557/ds005558, 4,567 stim events).

Run (~20 s):
    .venv/bin/python -u ccep_latency_waveform_model.py
"""
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from validate_patient_level_ccep import load_per_patient_ccep
from validate_simulator_ccep import load_centroids

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "ccep_latency_waveform_model.json"
OUT_FIG = ROOT / "figures" / "fig_latency_waveform.png"


def load_ftract_360(kind, g_labels):
    path = glob.glob(str(ROOT / "data" / "f_tract" / "**" / kind / f"{kind}.csv"), recursive=True)[0]
    lines = [l for l in open(path) if not l.startswith("#")]
    names = lines[0].strip().split(",")[1:]
    mat = np.array([[float(v) for v in l.strip().split(",")[1:]] for l in lines[1:]], dtype=np.float64)
    order = [names.index(lbl) for lbl in g_labels]
    return mat[np.ix_(order, order)]


def compute_graph_hops_and_dist(wiring, dist_mm):
    """Compute white-matter shortest-path hop count and weighted path length on a connectome."""
    n = len(wiring)
    # Keep structural edges > 0; weight edges by physical length / log-strength proxy
    pos = (wiring > 0) & ~np.eye(n, dtype=bool)
    adj_hops = csr_matrix(pos.astype(np.float64))
    w_len = np.where(pos, dist_mm / np.maximum(np.log1p(wiring * 100.0), 0.25), 0.0)
    adj_wlen = csr_matrix(w_len)
    hops = dijkstra(adj_hops, directed=False, unweighted=True)
    wdist = dijkstra(adj_wlen, directed=False)
    # Replace unreachable inf with max finite + 1
    fin_h = hops[np.isfinite(hops)]
    hops = np.where(np.isfinite(hops), hops, (fin_h.max() + 1.0) if len(fin_h) else 4.0)
    fin_w = wdist[np.isfinite(wdist)]
    wdist = np.where(np.isfinite(wdist), wdist, (fin_w.max() * 1.2) if len(fin_w) else 200.0)
    return hops, wdist


def fit_ols_cv(X, y, src_ids, n_folds=5, seed=42):
    """Strict 5-fold unseen stimulated-region CV for linear biophysical delay models."""
    uniq = np.unique(src_ids)
    folds = np.random.default_rng(seed).permutation(len(uniq)) % n_folds
    fmap = dict(zip(uniq, folds))
    fold_of = np.array([fmap[s] for s in src_ids])

    pred = np.zeros_like(y)
    coefs = []
    for f in range(n_folds):
        tr = fold_of != f
        te = fold_of == f
        A_tr = np.column_stack([np.ones(tr.sum()), X[tr]])
        A_te = np.column_stack([np.ones(te.sum()), X[te]])
        beta = np.linalg.lstsq(A_tr, y[tr], rcond=None)[0]
        pred[te] = A_te @ beta
        coefs.append(beta)
    beta_full = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    err = np.abs(pred - y)
    return {
        "spearman": round(float(spearmanr(pred, y).statistic), 4),
        "pearson": round(float(np.corrcoef(pred, y)[0, 1]), 4),
        "mae_ms": round(float(err.mean()), 2),
        "rmse_ms": round(float(np.sqrt(np.mean((pred - y) ** 2))), 2),
        "beta": [round(float(b), 5) for b in beta_full],
        "pred": pred,
    }


def run_part_a_ftract_latency():
    """Part A: 780-Patient Multi-Center F-TRACT Glasser-360 Onset & Peak Latency Physics."""
    print("=== Part A: 780-Patient Multi-Center F-TRACT Latency Physics (Glasser-360) ===", flush=True)
    g_labels = (ROOT / "data" / "hcp_connectome" / "strucLabels_ctx_glasser_360.csv").read_text().strip().split(",")
    log_w = np.loadtxt(ROOT / "data" / "hcp_connectome" / "strucMatrix_ctx_glasser_360.csv", delimiter=",")
    wiring = np.where(log_w > 0, np.exp(log_w), 0.0)
    wiring = wiring / wiring.max()
    euc_cent = np.load(ROOT / "data" / "hcp_connectome" / "glasser_360_centre_distance.npy")

    ft_onset = load_ftract_360("onset_latency", g_labels)
    ft_peak = load_ftract_360("peak_latency", g_labels)
    ft_fiber = load_ftract_360("fiber_distance", g_labels)
    ft_euc = load_ftract_360("euclidian_distance", g_labels)
    hops, wdist = compute_graph_hops_and_dist(wiring, euc_cent)
    hemi = np.array([0.0 if l.startswith("L_") else 1.0 for l in g_labels])
    inter_hemi = (hemi[:, None] != hemi[None, :]).astype(np.float64)
    # Structural node strength (proxy for myelinated hub degree)
    node_str = np.log1p(wiring.sum(1))
    src_str = np.broadcast_to(node_str[:, None], wiring.shape)
    dst_str = np.broadcast_to(node_str[None, :], wiring.shape)

    results = {}
    for target_name, Y in [("onset_latency", ft_onset), ("peak_latency", ft_peak)]:
        m = (
            np.isfinite(Y)
            & np.isfinite(ft_fiber)
            & np.isfinite(ft_euc)
            & (Y > 2.0)
            & (Y < 150.0)
            & ~np.eye(360, dtype=bool)
        )
        rr, cc = np.nonzero(m)
        yv = Y[m]
        models = {
            "Euclidean distance only": ft_euc[m][:, None],
            "Real white-matter fiber length": ft_fiber[m][:, None],
            "Connectome hops + weighted path": np.column_stack([hops[m], wdist[m]]),
            "Biophysical fiber + hops + inter-hemi + hubness": np.column_stack([
                ft_fiber[m], ft_euc[m], hops[m], wdist[m], inter_hemi[m], src_str[m], dst_str[m],
            ]),
        }
        res_t = {"n_pairs": int(m.sum()), "median_ms": round(float(np.median(yv)), 2), "models": {}}
        for nm, X in models.items():
            r = fit_ols_cv(X, yv, rr)
            rec = {k: v for k, v in r.items() if k != "pred"}
            # Extract effective conduction velocity from single-distance linear fit: slope is ms/mm => v = 1/slope (m/s)
            if nm == "Real white-matter fiber length":
                tau_syn = r["beta"][0]
                v_ms = 1.0 / max(r["beta"][1], 1e-6)  # mm/ms == m/s!
                rec["implied_tau_syn_ms"] = round(float(tau_syn), 2)
                rec["implied_velocity_m_per_s"] = round(float(v_ms), 2)
            res_t["models"][nm] = rec
            extra = (
                f" | tau_syn={rec['implied_tau_syn_ms']:.1f}ms, v={rec['implied_velocity_m_per_s']:.2f}m/s"
                if "implied_velocity_m_per_s" in rec else ""
            )
            print(
                f"  [{target_name:<13}] {nm:<48} rho={rec['spearman']:+.4f} r={rec['pearson']:+.4f} "
                f"MAE={rec['mae_ms']:.2f} ms{extra}",
                flush=True,
            )
        results[target_name] = res_t
    return results, g_labels, ft_onset, ft_peak, ft_fiber


def run_part_b_utrecht_developmental_and_transfer(g_labels, ft_onset_360, ft_peak_360):
    """Part B: 74-Patient Developmental Velocity (Ages 4–51), LPO Latency CV & Zero-Shot Transfer to F-TRACT."""
    print("\n=== Part B: 74-Patient ds004080 Developmental Myelination & Cross-Hospital Latency Transfer ===", flush=True)
    _, s_coords, s_hemi = load_centroids()
    s_dist = np.linalg.norm(s_coords[:, None, :] - s_coords[None, :, :], axis=-1)
    patients = load_per_patient_ccep(s_coords, s_hemi, exclude_soz=True)

    # 1. Per-patient effective transmission velocity vs. real age (4–51 yrs; replicating van Blooijs 2023!)
    TAU_SYN_FIXED = 10.0  # ms physiological synaptic + initial axonal integration delay
    pat_vel_rows = []
    for p in patients:
        ok = np.isfinite(p["lat_ms"]) & (p["lat_ms"] > 12.0) & (p["lat_ms"] < 105.0) & (p["elec_dist"] >= 15.0)
        if ok.sum() < 20 or not np.isfinite(p["age"]):
            continue
        dists = p["elec_dist"][ok]
        lats = p["lat_ms"][ok]
        # Effective velocity (mm/ms == m/s) across all patient electrode pairs
        v_eff = float(np.median(dists / np.maximum(lats - TAU_SYN_FIXED, 4.0)))
        med_lat = float(np.median(lats))
        pat_vel_rows.append({
            "id": p["id"], "age": float(p["age"]), "n_n1": int(ok.sum()),
            "v_eff_m_s": round(v_eff, 3), "med_lat_ms": round(med_lat, 2),
        })

    ages_arr = np.array([r["age"] for r in pat_vel_rows])
    vels_arr = np.array([r["v_eff_m_s"] for r in pat_vel_rows])
    lats_arr = np.array([r["med_lat_ms"] for r in pat_vel_rows])
    rho_v_age, p_v_age = spearmanr(ages_arr, vels_arr)
    rho_l_age, p_l_age = spearmanr(ages_arr, lats_arr)
    ped = vels_arr[ages_arr < 15.0]
    adult = vels_arr[ages_arr >= 25.0]
    u_stat, p_ped_adult = mannwhitneyu(adult, ped, alternative="greater")
    print(
        f"  Developmental myelination (n={len(pat_vel_rows)} patients, ages {ages_arr.min():.0f}–{ages_arr.max():.0f} y): "
        f"Speed vs Age rho={rho_v_age:+.4f} (p={p_v_age:.2e}) | Latency vs Age rho={rho_l_age:+.4f} (p={p_l_age:.2e})",
        flush=True,
    )
    print(
        f"  Pediatric (<15 y, n={len(ped)}) median v={np.median(ped):.2f} m/s vs "
        f"Adult (>=25 y, n={len(adult)}) median v={np.median(adult):.2f} m/s (Mann-Whitney p={p_ped_adult:.2e})",
        flush=True,
    )

    # 2. Strict 5-fold Leave-Patients-Out millisecond N1 latency prediction: Age-Blind vs. Developmental Age-Aware!
    # Load Schaefer-100 structural connectome hops
    log_w100 = np.loadtxt(ROOT / "data" / "hcp_connectome" / "strucMatrix_ctx_schaefer_100.csv", delimiter=",")
    w100 = np.where(log_w100 > 0, np.exp(log_w100), 0.0)
    w100 /= w100.max()
    hops100, wdist100 = compute_graph_hops_and_dist(w100, s_dist)

    elig = []
    for p in patients:
        ok = np.isfinite(p["lat_ms"]) & (p["lat_ms"] > 12.0) & (p["lat_ms"] < 105.0) & (p["elec_dist"] >= 10.0)
        if ok.sum() >= 20 and np.isfinite(p["age"]):
            elig.append({
                "id": p["id"], "age": float(p["age"]),
                "d": p["elec_dist"][ok], "y": p["lat_ms"][ok],
                "h": hops100[p["src_roi"][ok], p["dst_roi"][ok]],
                "wd": wdist100[p["src_roi"][ok], p["dst_roi"][ok]],
            })

    rng = np.random.default_rng(7)
    order = rng.permutation(len(elig))
    fold_of = np.zeros(len(elig), dtype=int)
    for i, idx in enumerate(order):
        fold_of[idx] = i % 5

    mae_blind, mae_age, rho_blind, rho_age = [], [], [], []
    for f in range(5):
        tr = [elig[i] for i in range(len(elig)) if fold_of[i] != f]
        te = [elig[i] for i in range(len(elig)) if fold_of[i] == f]

        d_tr = np.concatenate([p["d"] for p in tr])
        h_tr = np.concatenate([p["h"] for p in tr])
        wd_tr = np.concatenate([p["wd"] for p in tr])
        a_tr = np.concatenate([np.full(len(p["d"]), p["age"]) for p in tr])
        y_tr = np.concatenate([p["y"] for p in tr])
        inv_age_tr = 1.0 / (a_tr + 5.0)  # smooth developmental maturation basis

        # Age-blind biophysical model: [1, d_elec, hops, wdist]
        X_tr_b = np.column_stack([np.ones(len(d_tr)), d_tr, h_tr, wd_tr])
        beta_b = np.linalg.lstsq(X_tr_b, y_tr, rcond=None)[0]

        # Age-conditioned developmental myelination model: [1, d_elec, d_elec/(age+5), 1/(age+5), hops, wdist]
        X_tr_a = np.column_stack([np.ones(len(d_tr)), d_tr, d_tr * inv_age_tr, inv_age_tr, h_tr, wd_tr])
        beta_a = np.linalg.lstsq(X_tr_a, y_tr, rcond=None)[0]

        for p in te:
            ia = 1.0 / (p["age"] + 5.0)
            pb = np.column_stack([np.ones(len(p["d"])), p["d"], p["h"], p["wd"]]) @ beta_b
            pa = np.column_stack([np.ones(len(p["d"])), p["d"], p["d"] * ia, np.full(len(p["d"]), ia), p["h"], p["wd"]]) @ beta_a
            mae_blind.append(float(np.mean(np.abs(pb - p["y"]))))
            mae_age.append(float(np.mean(np.abs(pa - p["y"]))))
            rho_blind.append(float(spearmanr(pb, p["y"]).statistic))
            rho_age.append(float(spearmanr(pa, p["y"]).statistic))

    mae_blind = np.array(mae_blind)
    mae_age = np.array(mae_age)
    w_lat = wilcoxon(mae_blind - mae_age)
    print(
        f"  5-Fold LPO N1 Latency Prediction (n={len(elig)} unseen patients): "
        f"Age-Blind MAE={mae_blind.mean():.2f} ms (rho={np.mean(rho_blind):+.3f}) -> "
        f"Age-Conditioned MAE={mae_age.mean():.2f} ms ({(mae_age < mae_blind).sum()}/{len(elig)} improved, "
        f"Wilcoxon p={w_lat.pvalue:.2e})",
        flush=True,
    )

    # 3. Strict Zero-Shot Cross-Hospital Latency Transfer: Fit ONLY on Utrecht ds004080 (Schaefer-100) -> Test on F-TRACT (780 pts)!
    # Map Glasser-360 F-TRACT onset & peak latency to Schaefer-100
    xyz = np.vstack([nib.load(ROOT / "data" / "hcp_connectome" / f"fsa5_{h}.gii").darrays[0].data for h in ("lh", "rh")])
    g_par = np.loadtxt(ROOT / "data" / "hcp_connectome" / "glasser_360_fsa5.csv").astype(int)
    g_cent = np.array([xyz[g_par == v].mean(0) for v in range(1, 361)])
    g_to_s = np.concatenate([cdist(g_cent[:180], s_coords[:50]).argmin(1), cdist(g_cent[180:], s_coords[50:]).argmin(1) + 50])

    def map_360_to_100(M360):
        sums = np.zeros((100, 100))
        cnts = np.zeros((100, 100), dtype=int)
        rr, cc = np.nonzero(np.isfinite(M360) & (M360 > 2.0) & ~np.eye(360, dtype=bool))
        for i, j in zip(rr, cc):
            si, sj = g_to_s[i], g_to_s[j]
            if si != sj:
                sums[si, sj] += M360[i, j]
                cnts[si, sj] += 1
        out = np.full((100, 100), np.nan)
        ok = cnts > 0
        out[ok] = sums[ok] / cnts[ok]
        return out

    ft_peak_100 = map_360_to_100(ft_peak_360)
    ft_onset_100 = map_360_to_100(ft_onset_360)

    # Fit Utrecht parcel-level adult-calibrated latency model (evaluated at adult age=30 matching F-TRACT ages >= 15)
    d_all = np.concatenate([p["d"] for p in elig])
    h_all = np.concatenate([p["h"] for p in elig])
    wd_all = np.concatenate([p["wd"] for p in elig])
    a_all = np.concatenate([np.full(len(p["d"]), p["age"]) for p in elig])
    y_all = np.concatenate([p["y"] for p in elig])
    ia_all = 1.0 / (a_all + 5.0)
    beta_utrecht = np.linalg.lstsq(
        np.column_stack([np.ones(len(d_all)), d_all, d_all * ia_all, ia_all, h_all, wd_all]),
        y_all, rcond=None,
    )[0]

    ia_adult = 1.0 / (30.0 + 5.0)
    pred_lat_100 = (
        beta_utrecht[0]
        + beta_utrecht[1] * s_dist
        + beta_utrecht[2] * s_dist * ia_adult
        + beta_utrecht[3] * ia_adult
        + beta_utrecht[4] * hops100
        + beta_utrecht[5] * wdist100
    )
    transfer = {}
    for nm, Y100 in [("ftract_peak_latency_100", ft_peak_100), ("ftract_onset_latency_100", ft_onset_100)]:
        m = np.isfinite(Y100) & ~np.eye(100, dtype=bool)
        rho_t = float(spearmanr(pred_lat_100[m], Y100[m]).statistic)
        r_t = float(np.corrcoef(pred_lat_100[m], Y100[m])[0, 1])
        mae_t = float(np.mean(np.abs(pred_lat_100[m] - Y100[m])))
        transfer[nm] = {"n_pairs": int(m.sum()), "spearman": round(rho_t, 4), "pearson": round(r_t, 4), "mae_ms": round(mae_t, 2)}
        print(f"  Zero-shot Utrecht -> {nm} ({int(m.sum())} pairs): rho={rho_t:+.4f}, r={r_t:+.4f}, MAE={mae_t:.2f} ms", flush=True)

    return {
        "n_patients_developmental": len(pat_vel_rows),
        "age_range": [float(ages_arr.min()), float(ages_arr.max())],
        "speed_vs_age_rho": round(float(rho_v_age), 4),
        "speed_vs_age_p": float(p_v_age),
        "latency_vs_age_rho": round(float(rho_l_age), 4),
        "latency_vs_age_p": float(p_l_age),
        "pediatric_lt15_median_m_s": round(float(np.median(ped)), 3),
        "adult_ge25_median_m_s": round(float(np.median(adult)), 3),
        "pediatric_vs_adult_p": float(p_ped_adult),
        "lpo_latency_cv": {
            "n_patients": len(elig),
            "age_blind_mae_ms": round(float(mae_blind.mean()), 2),
            "age_conditioned_mae_ms": round(float(mae_age.mean()), 2),
            "patients_improved": int((mae_age < mae_blind).sum()),
            "wilcoxon_p": float(w_lat.pvalue),
            "mean_patient_rho": round(float(np.mean(rho_age)), 4),
        },
        "zero_shot_utrecht_to_ftract_latency": transfer,
        "patient_velocities": pat_vel_rows,
    }


def biophysical_waveform_transfer(amp_ua, pw_us, freq_hz, n_pulses=10):
    """Biophysical axonal charge recruitment + Tsodyks-Markram short-term synaptic plasticity transfer.

    Inputs match real clinical stimulator registers (OpenNeuro ds005557/ds005558 RAM & DBS/TMS):
      - amp_ua: biphasic pulse current amplitude in microamperes (e.g. 250–2000 uA)
      - pw_us:  phase pulse width in microseconds (e.g. 200–300 us)
      - freq_hz: pulse train repetition frequency in Hz (1–200 Hz)
    Returns:
      - charge_nc: charge per phase in nanocoulombs (I_mA * pw_us)
      - recruit_frac: Hill-curve axonal recruitment fraction [0, 1] (Q50 = 250 nC, Hill n = 2.0)
      - tm_steady_gain: Tsodyks-Markram steady-state synaptic efficacy x_ss * u_ss (facilitatory vs depressive)
      - effective_net_drive: net phasic evoked drive (peaks at 10–50 Hz; depresses / blocks at 130–200 Hz)
    """
    charge_nc = (amp_ua / 1000.0) * pw_us  # mA * us = nC
    q50 = 250.0  # ~0.83 mA @ 300 us half-maximal cortical axonal recruitment
    recruit = (charge_nc ** 2) / (charge_nc ** 2 + q50 ** 2)

    # Standard cortical excitatory Tsodyks-Markram STP parameters:
    # U0 = 0.25, tau_rec = 450 ms (vesicle recovery), tau_fac = 80 ms (calcium facilitation)
    U0 = 0.25
    tau_rec = 0.450
    tau_fac = 0.080
    dt = 1.0 / max(freq_hz, 0.5)
    u = U0
    x = 1.0
    effs = []
    for _ in range(max(int(n_pulses), 1)):
        # Impulse response at pulse arrival
        effs.append(u * x)
        x_post = x * (1.0 - u)
        # Inter-pulse exponential recovery & facilitation decay
        x = 1.0 - (1.0 - x_post) * np.exp(-dt / tau_rec)
        u_dec = U0 + (u - U0) * np.exp(-dt / tau_fac)
        u = u_dec + U0 * (1.0 - u_dec)
    mean_per_pulse = float(np.mean(effs))
    # Net temporal drive over a 500 ms stimulation train with depolarization-block ceiling at >120 Hz
    depol_block = 1.0 / (1.0 + (freq_hz / 140.0) ** 3)
    net_drive = recruit * mean_per_pulse * np.log1p(freq_hz * 0.5) * depol_block
    return {
        "charge_nc": round(float(charge_nc), 1),
        "recruit_frac": round(float(recruit), 4),
        "tm_per_pulse_efficacy": round(mean_per_pulse, 4),
        "depol_block_factor": round(float(depol_block), 4),
        "effective_net_drive": round(float(net_drive), 4),
    }


def run_part_c_waveform_ram_audit():
    """Part C: Profile Biophysical Waveform/Frequency Transfer Across Real Clinical RAM Sessions."""
    print("\n=== Part C: Biophysical Waveform/Frequency Operator Profiled on 40 Real RAM Clinical Sessions ===", flush=True)
    ev_files = sorted(glob.glob(str(ROOT / "data" / "ram_closedloop" / "ds00555*" / "sub-*" / "ses-*" / "ieeg" / "*_events.tsv")))
    combos = {}
    total_events = 0
    for ef in ev_files:
        with open(ef) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("trial_type") == "STIM_ON" or r.get("stimulation") == "1":
                    try:
                        amp = float(r["amplitude"])
                        freq = float(r["pulse_freq"])
                        pw = float(r["pulse_width"])
                        npul = int(float(r.get("n_pulses", 10) or 10))
                    except (KeyError, ValueError):
                        continue
                    if amp > 0 and freq > 0 and pw > 0:
                        total_events += 1
                        key = (int(round(amp)), int(round(pw)), int(round(freq)))
                        combos[key] = combos.get(key, 0) + 1

    freq_sweep = [1, 5, 10, 25, 50, 100, 130, 200]
    sweep_curve = {f"{f}Hz": biophysical_waveform_transfer(1000.0, 300.0, float(f), n_pulses=max(5, int(f * 0.5))) for f in freq_sweep}
    top_clinical = []
    for (amp, pw, freq), cnt in sorted(combos.items(), key=lambda kv: -kv[1])[:8]:
        op = biophysical_waveform_transfer(amp, pw, freq, n_pulses=max(5, int(freq * 0.5)))
        top_clinical.append({"amp_ua": amp, "pw_us": pw, "freq_hz": freq, "n_events": cnt, **op})
        print(
            f"  RAM setting {amp:4d} uA / {pw:3d} us / {freq:3d} Hz (n={cnt:4d}): "
            f"Q={op['charge_nc']:.0f} nC, recruit={op['recruit_frac']:.2f}, TM_eff={op['tm_per_pulse_efficacy']:.3f}, "
            f"net_drive={op['effective_net_drive']:.3f}",
            flush=True,
        )
    return {"n_ram_sessions": len(ev_files), "n_stim_events": total_events, "freq_sweep_1mA_300us": sweep_curve, "top_ram_settings": top_clinical}


def make_figure(part_a, part_b, part_c):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)

    # Panel A: 780-Patient F-TRACT Latency Model Comparison (Unseen Stimulated Regions CV)
    ax = axes[0]
    m_keys = [
        "Euclidean distance only",
        "Connectome hops + weighted path",
        "Real white-matter fiber length",
        "Biophysical fiber + hops + inter-hemi + hubness",
    ]
    short_m = ["Euclidean\ndistance", "dMRI graph\nhops + path", "White-matter\nfiber length", "Full Biophysical\nDelay Model"]
    y = np.arange(len(m_keys))
    onset_rho = [part_a["onset_latency"]["models"][k]["spearman"] for k in m_keys]
    peak_rho = [part_a["peak_latency"]["models"][k]["spearman"] for k in m_keys]
    onset_mae = [part_a["onset_latency"]["models"][k]["mae_ms"] for k in m_keys]
    h = 0.36
    ax.barh(y + h / 2, onset_rho, height=h, color="#2b59c3", label="Onset Latency (15,725 pairs)")
    ax.barh(y - h / 2, peak_rho, height=h, color="#1b9e77", label="Peak Latency (15,772 pairs)")
    for i, (orho, prho, omae) in enumerate(zip(onset_rho, peak_rho, onset_mae)):
        ax.text(orho + 0.01, i + h / 2, f"+{orho:.3f} ({omae:.1f}ms)", va="center", fontsize=7.6, color="#1a3a82", fontweight="bold")
        ax.text(prho + 0.01, i - h / 2, f"+{prho:.3f}", va="center", fontsize=7.6, color="#0b6e4f", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(short_m, fontsize=8.4)
    ax.set_xlim(0, 0.56)
    ax.set_ylim(-1.1, len(m_keys) - 0.35)
    ax.set_xlabel("Held-Out Spearman ρ (5-Fold Unseen Stim Regions)")
    ax.set_title("A. 780-Patient F-TRACT Axonal Delay Model\n(Fiber Length + Synaptic Hops, Glasser-360)", fontweight="bold", fontsize=10.0)
    ax.legend(loc="lower right", frameon=True, fontsize=7.7, facecolor="white")
    ax.grid(axis="x", alpha=0.25)

    # Panel B: 74-Patient ds004080 Developmental Transmission Velocity (Ages 4–51)
    ax = axes[1]
    pvs = part_b["patient_velocities"]
    ages = np.array([r["age"] for r in pvs])
    vels = np.array([r["v_eff_m_s"] for r in pvs])
    ax.scatter(ages, vels, c="#0b6e4f", alpha=0.80, s=36, edgecolors="white", linewidths=0.5)
    # Smooth trend fit: v(age) = a - b/(age+5)
    ag_grid = np.linspace(ages.min(), ages.max(), 100)
    X_fit = np.column_stack([np.ones(len(ages)), 1.0 / (ages + 5.0)])
    coef = np.linalg.lstsq(X_fit, vels, rcond=None)[0]
    ax.plot(ag_grid, coef[0] + coef[1] / (ag_grid + 5.0), color="#d95f02", lw=2.0, label="Developmental fit")
    ax.set_xlabel("Patient Age (years, OpenNeuro ds004080)")
    ax.set_ylabel("Effective Axonal Conduction Velocity (m/s)")
    ax.set_title(
        f"B. Human Myelination Speed Trajectory (n = {len(pvs)})\n"
        f"ρ = {part_b['speed_vs_age_rho']:+.3f} (p = {part_b['speed_vs_age_p']:.1e}) | "
        f"LPO MAE: {part_b['lpo_latency_cv']['age_blind_mae_ms']:.1f}→{part_b['lpo_latency_cv']['age_conditioned_mae_ms']:.1f} ms",
        fontweight="bold", fontsize=9.5,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=8.0)
    ax.grid(alpha=0.25)

    # Panel C: Biophysical Frequency Transfer Curve (1–200 Hz) + Clinical RAM Regimes
    ax = axes[2]
    freqs = [1, 5, 10, 25, 50, 100, 130, 200]
    tm_eff = [part_c["freq_sweep_1mA_300us"][f"{f}Hz"]["tm_per_pulse_efficacy"] for f in freqs]
    net_d = [part_c["freq_sweep_1mA_300us"][f"{f}Hz"]["effective_net_drive"] for f in freqs]
    ax.plot(freqs, tm_eff, "o-", color="#2b59c3", lw=2.0, label="Per-pulse synaptic efficacy (Tsodyks-Markram)")
    ax.plot(freqs, net_d, "s-", color="#d95f02", lw=2.0, label="Net train drive (w/ 140 Hz depol-block)")
    ax.axvspan(100, 200, color="#d95f02", alpha=0.10, label="HFS / DBS depression band (100–200 Hz)")
    ax.axvspan(10, 50, color="#1b9e77", alpha=0.10, label="Rhythmic / burst band (10–50 Hz)")
    ax.set_xscale("log")
    tick_freqs = [1, 5, 10, 25, 50, 100, 200]
    ax.set_xticks(tick_freqs)
    ax.set_xticklabels([str(f) for f in tick_freqs], fontsize=8.0)
    ax.set_xlabel("Stimulation Frequency (Hz, 1.0 mA / 300 µs biphasic)")
    ax.set_ylabel("Normalized Synaptic / Network Transfer")
    ax.set_title("C. Biophysical Waveform & Frequency Transfer\n(Tsodyks-Markram STP + Charge Recruitment)", fontweight="bold", fontsize=9.8)
    ax.legend(loc="upper right", frameon=True, fontsize=7.2, facecolor="white")
    ax.grid(alpha=0.25)

    t_pk = part_b["zero_shot_utrecht_to_ftract_latency"]["ftract_peak_latency_100"]
    fig.suptitle(
        f"Real Human Axonal Conduction-Delay Physics & Biophysical Waveform Transfer | "
        f"Zero-Shot Utrecht→F-TRACT Peak Latency ρ = {t_pk['spearman']:+.3f} (MAE = {t_pk['mae_ms']:.1f} ms)",
        fontsize=10.4,
        fontweight="bold",
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {OUT_FIG}", flush=True)


def main():
    part_a, g_labels, ft_onset, ft_peak, _ = run_part_a_ftract_latency()
    part_b = run_part_b_utrecht_developmental_and_transfer(g_labels, ft_onset, ft_peak)
    part_c = run_part_c_waveform_ram_audit()
    payload = {
        "part_a_ftract_360_latency": part_a,
        "part_b_utrecht_developmental_and_transfer": part_b,
        "part_c_waveform_ram": part_c,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {OUT_JSON}", flush=True)
    make_figure(part_a, part_b, part_c)


if __name__ == "__main__":
    main()
