"""Predict from a patient's brain activity, while they study a word, whether they will recall it later.

Data: the University of Pennsylvania free-recall study (OpenNeuro ds004789, CC0; Kahana lab, DARPA Restoring Active
Memory). Epilepsy patients with implanted electrodes study lists of 12 words, each shown for 1.6 s, then recall them
aloud. Bipolar intracranial recordings (each channel is the difference of two neighbouring contacts), two sessions of 300
words per patient. A word counts as remembered if the patient said it during that list's recall period.

Features, following Ezzyat et al. 2017 (Current Biology 27:1251) and 2018 (Nature Communications 9:365): for each word
and channel, the power at 8 log-spaced frequencies from 3 to 180 Hz (Morlet wavelets, 5 cycles), averaged over the 1.6 s
the word is on screen, log-transformed and z-scored within each session. Mains interference at 60, 120 and 180 Hz is
removed first.

Decoders, each trained on one session and scored on the other, both ways (AUC: 0.5 is chance, 1 is perfect):
  logistic regression with an L2 penalty, the published approach; penalty chosen within the training session by
                  leaving whole lists out
  neural network  one hidden layer of 32 units, weight penalty chosen the same way
Control: logistic regression trained with the remembered/forgotten labels shuffled, 20 times per training session; the
table gives the mean and the level 95% of the shuffled decoders stay below. One shuffled decoder's AUC varies by about
0.06, so a single patient's score is noisy and the mean over patients is the number to read.

Run (about 5 minutes the first time, while the features are computed and saved; about 1 minute after):
    .venv/bin/python memory_decoder.py data/memory_fr1
"""
import glob
import sys
from pathlib import Path

import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier

FREQUENCIES = np.logspace(np.log10(3), np.log10(180), 8)
WINDOW, BUFFER = 1.6, 1.0  # s the word is shown; s of extra signal either side so the wavelets have data at the edges
DECIM = 5                  # keep every 5th time point of the power (500 -> 100 per second)
SHUFFLES = 20              # shuffled-label decoders per training session, for the chance level


def session(path):
    """Features (words, channels * 8), remembered or not, and list number for each word of one session; computed once,
    then read from features/<session>.npz in the data folder."""
    cache = Path(path).parents[3] / "features" / Path(path).name.replace("_acq-bipolar_ieeg.edf", ".npz")
    if cache.exists():
        saved = np.load(cache)
        return saved["features"], saved["remembered"], saved["lists"]
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    raw.notch_filter([60, 120, 180], verbose="error")
    rate = raw.info["sfreq"]
    events = np.genfromtxt(path.replace("_acq-bipolar_ieeg.edf", "_events.tsv"), delimiter="\t", names=True,
                           dtype=None, encoding="utf-8")
    words = events[events["trial_type"] == "WORD"]
    said = {(int(r["list"]), r["item_name"]) for r in events[events["trial_type"] == "REC_WORD"]}
    remembered = np.array([(int(w["list"]), w["item_name"]) in said for w in words])
    data, before, length = raw.get_data(), int(BUFFER * rate), int((WINDOW + 2 * BUFFER) * rate)
    epochs = np.stack([data[:, s - before:s - before + length] for s in words["sample"].astype(int)])
    power = mne.time_frequency.tfr_array_morlet(epochs, rate, FREQUENCIES, n_cycles=5, output="power", decim=DECIM, verbose="error")
    shown = slice(int(BUFFER * rate / DECIM), int((BUFFER + WINDOW) * rate / DECIM))
    features = np.log10(power[..., shown].mean(-1))
    features = (features - features.mean(0)) / features.std(0)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, features=features.reshape(len(words), -1), remembered=remembered, lists=words["list"].astype(int))
    return session(path)


def fit(model, grid, x, y, lists):
    """Choose the model's penalty by leaving whole lists of the training session out, then fit on all of it."""
    return GridSearchCV(model, grid, scoring="roc_auc", cv=GroupKFold(5)).fit(x, y, groups=lists)


def main(folder):
    patients = sorted({p.split("/sub-")[1].split("/")[0] for p in glob.glob(f"{folder}/sub-*/ses-*/ieeg/*acq-bipolar_ieeg.edf")})
    decoders = {
        "logistic regression": (LogisticRegression(class_weight="balanced", max_iter=5000), {"C": np.logspace(-5, 0, 11)}),
        "neural network": (MLPClassifier((32,), max_iter=2000, early_stopping=True, random_state=0), {"alpha": [0.1, 1, 10, 100]}),
    }
    print(f"{len(patients)} patients; trained on one session, scored on the other, both ways (AUC)")
    print(f"{'patient':<10}{'channels':>9}{'recalled':>10}{'logistic regression':>21}{'neural network':>16}"
          f"{'shuffled control':>18}{'95% below':>14}")
    table = []
    for patient in patients:
        sessions = [session(p) for p in sorted(glob.glob(f"{folder}/sub-{patient}/ses-*/ieeg/*acq-bipolar_ieeg.edf"))[:2]]
        rng = np.random.default_rng(0)
        scores = {name: [] for name in [*decoders, "control"]}
        for train, test in ((0, 1), (1, 0)):
            (x, y, lists), (xt, yt, _) = sessions[train], sessions[test]
            for name, (model, grid) in decoders.items():
                fitted = fit(model, grid, x, y, lists)
                scores[name].append(roc_auc_score(yt, fitted.predict_proba(xt)[:, 1]))
                if name == "logistic regression":
                    for _ in range(SHUFFLES):  # the same decoder trained on shuffled labels: how high chance alone reaches
                        shuffled = LogisticRegression(C=fitted.best_params_["C"], class_weight="balanced", max_iter=5000)
                        scores["control"].append(roc_auc_score(yt, shuffled.fit(x, rng.permutation(y)).predict_proba(xt)[:, 1]))
        row = [np.mean(scores[name]) for name in scores] + [np.percentile(scores["control"], 95)]
        table.append(row)
        recalled = np.mean(np.concatenate([s[1] for s in sessions]))
        print(f"{patient:<10}{sessions[0][0].shape[1] // 8:>9}{recalled:>10.0%}" + "".join(f"{v:>{w}.3f}" for v, w in zip(row, (21, 16, 18, 14))), flush=True)
    print(f"{'mean':<29}" + "".join(f"{v:>{w}.3f}" for v, w in zip(np.mean(table, 0), (21, 16, 18, 14))))


if __name__ == "__main__":
    main(sys.argv[1])
