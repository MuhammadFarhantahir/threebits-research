"""Living-Human Pilot Study Statistical Analyzer (Threebits Level-3A Sensory-Substitution Co-Adaptation).

Reads one or more exported CSV files from https://threebits.si/gym/?tab=pilot
(threebits_pilot_*.csv) and computes publication-grade within-subject statistics:
  - For N = 1 (single-subject self-experiment): trial-level Mann-Whitney U + 10,000-permutation
    p-values comparing Real 4-bit Co-Processor vs. Unassisted & vs. Sham Latent Code, plus
    co-adaptation learning slope across trials.
  - For N >= 2 (multi-participant study): within-subject participant-level means, paired
    Wilcoxon signed-rank p-values, paired Cohen's d_z, responder count, and publication figure.

Usage:
    .venv/bin/python analyze_human_gym_pilot.py path/to/pilot_csvs/ --out-fig figures/fig_human_pilot.png
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def perm_test_diff(a, b, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs - 1e-12:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def load_pilot_csvs(folder_or_file: Path):
    files = [folder_or_file] if folder_or_file.is_file() else sorted(folder_or_file.glob("*.csv"))
    rows = []
    for f in files:
        with open(f, newline="") as fp:
            for r in csv.DictReader(fp):
                if not r.get("arm") or not r.get("err_deg"):
                    continue
                rows.append({
                    "participant_id": r.get("participant_id") or f.stem,
                    "trial_idx": int(float(r.get("trial_idx", 0))),
                    "arm": r["arm"].strip(),
                    "set_size": int(float(r.get("set_size", 12))),
                    "rule_deg": int(float(r.get("rule_deg", 0))),
                    "err_deg": float(r["err_deg"]),
                    "rt_ms": float(r["rt_ms"]),
                    "eff_bps": float(r.get("eff_bps", 0.0)),
                })
    return rows


def analyze(rows, out_fig: Path | None = None):
    arms = ("unassisted", "sham_code", "latent_coproc")
    pids = sorted({r["participant_id"] for r in rows})
    by_pid = {p: {a: [r for r in rows if r["participant_id"] == p and r["arm"] == a] for a in arms} for p in pids}

    summary = {"n_participants": len(pids), "n_trials_total": len(rows), "arms": {}, "comparisons": {}}
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        if not sub:
            continue
        errs = np.array([r["err_deg"] for r in sub])
        rts = np.array([r["rt_ms"] for r in sub])
        bps = np.array([r["eff_bps"] for r in sub])
        summary["arms"][a] = {
            "n_trials": len(sub),
            "acc30_pct": round(float((errs < 30.0).mean() * 100.0), 1),
            "mae_deg": round(float(errs.mean()), 2),
            "median_rt_ms": round(float(np.median(rts)), 1),
            "mean_eff_bps": round(float(bps.mean()), 2),
        }

    # Within-subject statistical tests
    if len(pids) == 1:
        p = pids[0]
        e_c = np.array([r["err_deg"] for r in by_pid[p]["latent_coproc"]])
        for ctrl in ("unassisted", "sham_code"):
            e_0 = np.array([r["err_deg"] for r in by_pid[p][ctrl]])
            if len(e_c) and len(e_0):
                u_p = float(mannwhitneyu(e_c, e_0, alternative="two-sided").pvalue)
                perm_p = perm_test_diff(e_c, e_0)
                summary["comparisons"][f"latent_coproc_vs_{ctrl}"] = {
                    "test": "single-subject trial-level Mann-Whitney + 10k perm",
                    "mae_reduction_deg": round(float(e_0.mean() - e_c.mean()), 2),
                    "mannwhitney_p": round(u_p, 5),
                    "perm_p": round(perm_p, 5),
                }
    else:
        for ctrl in ("unassisted", "sham_code"):
            m_c, m_0 = [], []
            for p in pids:
                if by_pid[p]["latent_coproc"] and by_pid[p][ctrl]:
                    m_c.append(np.mean([r["err_deg"] for r in by_pid[p]["latent_coproc"]]))
                    m_0.append(np.mean([r["err_deg"] for r in by_pid[p][ctrl]]))
            m_c, m_0 = np.array(m_c), np.array(m_0)
            if len(m_c) >= 2:
                diff = m_0 - m_c
                dz = float(diff.mean() / max(diff.std(ddof=1), 1e-8))
                w_p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                summary["comparisons"][f"latent_coproc_vs_{ctrl}"] = {
                    "test": "paired participant-level Wilcoxon signed-rank",
                    "n_paired": int(len(m_c)),
                    "improved_count": int((diff > 0).sum()),
                    "mae_reduction_deg": round(float(diff.mean()), 2),
                    "cohens_dz": round(dz, 3),
                    "wilcoxon_p": round(w_p, 5),
                }

    # Co-adaptation slope (1st half vs 2nd half of Co-Processor trials per participant)
    early_all, late_all = [], []
    for p in pids:
        sub = sorted(by_pid[p]["latent_coproc"], key=lambda x: x["trial_idx"])
        if len(sub) >= 4:
            h = len(sub) // 2
            early_all.append(np.mean([r["err_deg"] for r in sub[:h]]))
            late_all.append(np.mean([r["err_deg"] for r in sub[h:]]))
    if early_all:
        summary["coadaptation_slope"] = {
            "first_half_mae_deg": round(float(np.mean(early_all)), 2),
            "second_half_mae_deg": round(float(np.mean(late_all)), 2),
            "delta_deg": round(float(np.mean(late_all) - np.mean(early_all)), 2),
        }

    # Optional publication plot
    if out_fig is not None and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3), dpi=200)
        fig.patch.set_facecolor("#ffffff")
        labels = ["Unassisted\n(K=12)", "Sham Latent\nCode", "Real 4-bit\nCo-Processor"]
        colors = ["#5c8ec4", "#9a9589", "#c8502a"]
        maes = [summary["arms"].get(a, {}).get("mae_deg", 0) for a in arms]
        ax1.bar(range(3), maes, color=colors, width=0.58)
        if len(pids) >= 2:
            for p in pids:
                pts = [np.mean([r["err_deg"] for r in by_pid[p][a]]) if by_pid[p][a] else np.nan for a in arms]
                ax1.plot(range(3), pts, "o-", color="#22201c", alpha=0.35, linewidth=1.0, markersize=3.5)
        for i, m in enumerate(maes):
            ax1.text(i, m + 1.5, f"{m:.1f}°", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax1.set_xticks(range(3))
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("Mean Angular Error (deg, lower = better)")
        ax1.set_title(f"A. Blinded 3-Arm Human Pilot (N={len(pids)}, {len(rows)} trials)", fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        bps_vals = [summary["arms"].get(a, {}).get("mean_eff_bps", 0) for a in arms]
        ax2.bar(range(3), bps_vals, color=colors, width=0.58)
        for i, v in enumerate(bps_vals):
            ax2.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Effective Human+AI Throughput (bits/s)")
        ax2.set_title("B. Non-Verbal Human+AI Throughput", fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=200, facecolor="#ffffff")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Analyze Threebits living-human Neural Gym Pilot CSVs.")
    ap.add_argument("input_path", type=Path, help="Folder of threebits_pilot_*.csv files or single CSV")
    ap.add_argument("--out-fig", type=Path, default=None, help="Optional output PNG figure path")
    ap.add_argument("--out-json", type=Path, default=None, help="Optional output JSON summary path")
    args = ap.parse_args()

    rows = load_pilot_csvs(args.input_path)
    if not rows:
        raise SystemExit(f"No pilot trials found in {args.input_path}")
    summary = analyze(rows, out_fig=args.out_fig)
    print(json.dumps(summary, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
