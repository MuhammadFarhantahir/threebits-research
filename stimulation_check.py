"""Does the brain model predict how real stimulation spreads through the human brain?

Real stimulation: the F-TRACT atlas (v2112, CC BY 4.0; David et al. 2013, Trebaul et al. 2018, Lemarechal et al. 2022):
single electrical pulses applied to the cortex of 780 epilepsy patients through implanted electrodes, with responses
recorded at other sites. For each pair of the 360 regions of the HCP-MMP1 atlas, it gives the share of stimulations of one
region that drew a response in the other (patients aged 15 and over). Only about a fifth of the pairs were measured,
because electrodes go where doctors need them, and the atlas does not say how many measurements lie behind each value.

Model: the whole-brain model of human_brain_model.py on the same 360 regions, from the Human Connectome Project maps in the
ENIGMA toolbox, tuned to resting activity correlations only. It never sees stimulation data. Its prediction for
stimulating region j is each region's steady response to a constant push at j: column j of the inverse of (network
Laplacian + each region's distance from oscillating).

Rivals: the distance between the regions' centres (on the fsaverage5 surface; it matches F-TRACT's own measured distances
at r = 0.93), the wiring map alone, the resting activity correlations alone, and the untuned model (one coupling strength).

Score: for each stimulated region with at least 10 measured targets, the rank correlation (Spearman) between a predictor
and the measured response shares across its targets, averaged over stimulated regions; also with distance taken out.
A similar test with other connectome models: Seguin et al. 2023, Neuron 111:1391.

Run (a few minutes):
    .venv/bin/python stimulation_check.py data
"""
import glob
import sys

import nibabel as nib
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from human_brain_model import correlations, pearson, tune, untuned


def load(root):
    """Wiring, activity correlations, labels and centre distances for the 360 regions, and F-TRACT's response shares."""
    read = lambda kind: np.loadtxt(f"{root}/hcp_connectome/{kind}_ctx_glasser_360.csv", delimiter=",")
    labels = open(f"{root}/hcp_connectome/strucLabels_ctx_glasser_360.csv").read().strip().split(",")
    log_wiring = read("strucMatrix")
    wiring = np.where(log_wiring > 0, np.exp(log_wiring), 0.0)  # undo the log scale, as in human_brain_model.py
    parcel = np.loadtxt(f"{root}/hcp_connectome/glasser_360_fsa5.csv").astype(int)
    xyz = np.vstack([nib.load(f"{root}/hcp_connectome/fsa5_{h}.gii").darrays[0].data for h in ("lh", "rh")])
    centre = np.array([xyz[parcel == v].mean(0) for v in range(1, len(labels) + 1)])
    lines = [line for line in open(glob.glob(f"{root}/f_tract/**/probability/probability.csv", recursive=True)[0]) if not line.startswith("#")]
    names = lines[0].strip().split(",")[1:]
    share = np.array([[float(v) for v in line.strip().split(",")[1:]] for line in lines[1:]])  # rows stimulated, columns recorded
    order = [names.index(label) for label in labels]
    return (wiring / wiring.max(), read("funcMatrix"), labels, np.linalg.norm(centre[:, None] - centre[None], axis=-1),
            share[np.ix_(order, order)], log_wiring)


def site_scores(predictor, share, distance, minimum=10):
    """Per stimulated region: rank correlation of predictor and measured share over its measured targets, plain and with distance taken out."""
    plain, partial = [], []
    for j in range(len(share)):
        targets = np.flatnonzero(np.isfinite(share[j]) & (np.arange(len(share)) != j))
        if len(targets) < minimum:
            continue
        p, s, d = (rankdata(a[j, targets]) for a in (predictor, share, distance))
        plain.append(spearmanr(p, s)[0])
        residual = lambda y: y - np.polyval(np.polyfit(d, y, 1), d)  # remove what distance explains
        partial.append(pearson(residual(p), residual(s)) if np.ptp(residual(p)) > 0 else np.nan)
    return np.array(plain), np.array(partial)


def main(root):
    wiring, real, labels, distance, share, log_wiring = load(root)
    n, upper = len(labels), np.triu_indices(len(labels), 1)
    measured = np.isfinite(share) & ~np.eye(n, dtype=bool)
    print(f"{n} regions | wiring: {int((wiring[upper] > 0).sum())} connected pairs | F-TRACT: {int(measured.sum())} measured "
          f"stimulation-response pairs of {n * (n - 1)}")

    g = untuned(wiring, real, upper)
    plain_model = np.linalg.inv(np.diag(g * wiring.sum(1)) - g * wiring + np.eye(n))
    model = tune(wiring, real, upper, np.random.default_rng(0))
    with torch.no_grad():
        print(f"model tuned to resting activity: matches it at r = {pearson(model().numpy()[upper], real[upper]):.2f} "
              f"(untuned: {pearson(correlations(g * torch.tensor(wiring), torch.ones(n)).numpy()[upper], real[upper]):.2f})")
        w, d = model.weights().numpy(), model.log_distance.exp().numpy()
    tuned_response = np.linalg.inv(np.diag(w.sum(1)) - w + np.diag(d))

    predictors = {"closeness (minus distance)": -distance, "wiring map": log_wiring, "resting activity correlation": real,
                  "untuned model response": plain_model, "tuned model response": tuned_response}
    print("\npredicting which regions respond when a region is stimulated (rank correlation, mean over stimulated regions):")
    for name, predictor in predictors.items():
        plain, partial = site_scores(predictor, share, distance)
        line = f"  {name:<30} {np.nanmean(plain):+.3f} (positive for {np.mean(plain > 0):.0%} of {len(plain)} sites)"
        if not name.startswith("closeness"):
            line += f" | with distance taken out {np.nanmean(partial):+.3f} (positive for {np.nanmean(partial > 0):.0%})"
        print(line)
    pooled = {name: spearmanr(p[measured], share[measured])[0] for name, p in predictors.items()}
    print("\nall measured pairs pooled (rank correlation): " + ", ".join(f"{name} {value:+.3f}" for name, value in pooled.items()))


if __name__ == "__main__":
    main(sys.argv[1])
