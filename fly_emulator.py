"""Can a network learn to act like a brain from its inputs and outputs alone, the way a language model learns from text?

The brain is the whole fruit fly brain of fly_brain.py. We know its wiring, so every answer can be checked. Its inputs are
four kinds of taste neurons on the right side of the mouthparts (sugar, water, bitter, Ir94e; 18 to 21 neurons each, IDs
from the authors' figures.ipynb). Its output is the pair of MN9 motor neurons, which extend the mouthparts to feed. At
100 Hz each, only sugar makes MN9 fire, yet water boosts the sugar response and bitter and Ir94e cut it.

An emulator, a small recurrent network (GRU), learns to predict MN9 from the taste input alone. It never sees the wiring
or any other neuron. Three emulators are trained from different random starts on each of two training sets:
  single tastes:  800 one-second recordings with at most one taste on at a time, at changing strengths
  with mixtures:  800 recordings in which two tastes are often on together
All are tested on new recordings of single tastes and of mixtures, against the brain's average over 6 runs of each.

The wiring data are FlyWire's, licence CC BY-NC 4.0 (non-commercial use only).
Run from the project folder (about 8 minutes on 14 cores; the simulated recordings are saved in the data folder and reused):
    python fly_emulator.py data/fly_brain
"""
import copy
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

import fly_brain as fb

TASTES = {
    "sugar": fb.SUGAR,
    "water": [720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053, 720575940622902535,
              720575940616177458, 720575940660292225, 720575940622486922, 720575940613786774, 720575940629852866,
              720575940625861168, 720575940613996959, 720575940617857694, 720575940644965399, 720575940625203504,
              720575940630553415, 720575940635172191, 720575940634796536],
    "bitter": [720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
               720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
               720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
               720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
               720575940604027168],
    "Ir94e": [720575940614211295, 720575940638218173, 720575940628832256, 720575940626016017, 720575940621375231,
              720575940612920386, 720575940614273292, 720575940628198503, 720575940626241636, 720575940619387814,
              720575940624604560, 720575940615274425, 720575940610683315, 720575940627265265, 720575940624079544,
              720575940629211607, 720575940615089369, 720575940631082124],
}
MN9 = [720575940660219265, 720575940645521262]  # left, right
BINS = fb.STEPS // fb.BIN                       # 100 bins of 10 ms per recording
SETTLE = 3                                      # bins skipped at the start of each stretch when averaging a response
SETS = {  # name: (chance that a stretch has no taste / one taste / two tastes, recordings, runs of each recording)
    "train_single": ((0.25, 0.75, 0.0), 800, 1),
    "train_mixed": ((0.2, 0.4, 0.4), 800, 1),
    "test_single": ((0.25, 0.75, 0.0), 100, 6),
    "test_mixed": ((0.2, 0.0, 0.8), 100, 6),
}


def schedule(rng, mix):
    """Taste strengths over one recording, (100 bins, 4 tastes) in Hz: stretches of 100 to 300 ms, each with no taste,
    one or two tastes on (chances `mix`), each taste at a random strength between 20 and 200 Hz."""
    rates, start = np.zeros((BINS, len(TASTES))), 0
    while start < BINS:
        n, length = rng.choice(3, p=mix), rng.integers(10, 31)
        rates[start:start + length, rng.choice(len(TASTES), n, replace=False)] = rng.uniform(20, 200, n)
        start += length
    return rates


def run(args):
    """MN9 spikes per 10 ms bin, (100, 2), for one run of one recording."""
    rates, seed = args
    ids, starts, receivers, counts = fb.BRAIN
    stimulated = np.searchsorted(ids, np.concatenate(list(TASTES.values())))
    taste_of = np.repeat(np.arange(len(TASTES)), [len(group) for group in TASTES.values()])
    return fb.trial(starts, receivers, counts * fb.W_SYN, stimulated, seed, rates[:, taste_of], np.searchsorted(ids, MN9))[1]


def simulate(folder):
    """Every recording's taste input and MN9 spikes, simulated once and then read back from the data folder."""
    path = Path(folder) / "emulator_recordings.npz"
    if path.exists():
        return dict(np.load(path))
    data = {}
    with Pool(initializer=fb.setup, initargs=(folder,)) as pool:
        for k, (name, (mix, n, runs)) in enumerate(SETS.items()):
            rng = np.random.default_rng(k)
            rates = np.stack([schedule(rng, mix) for _ in range(n)])
            jobs = [(r, 1_000_000 * (k + 1) + 10 * i + j) for i, r in enumerate(rates) for j in range(runs)]
            data[f"{name}_rates"] = rates
            data[f"{name}_spikes"] = np.array(pool.map(run, jobs, chunksize=4)).reshape(n, runs, BINS, len(MN9))
            print(f"simulated {name}: {n} recordings, {runs} run(s) each", flush=True)
    np.savez_compressed(path, **data)
    return data


class Emulator(nn.Module):
    """Taste strengths (recordings, bins, 4) in Hz -> expected MN9 spikes per bin (recordings, bins, 2)."""

    def __init__(self):
        super().__init__()
        self.gru, self.out = nn.GRU(len(TASTES), 128, batch_first=True), nn.Linear(128, len(MN9))

    def forward(self, rates):
        return nn.functional.softplus(self.out(self.gru(rates / 200)[0]))


def train(rates, spikes, seed):
    """Fit an emulator to recordings, stopping when it stops improving on 15% of them that it does not learn from."""
    torch.manual_seed(seed)
    x, y = torch.tensor(rates, dtype=torch.float32), torch.tensor(spikes, dtype=torch.float32)
    order = torch.randperm(len(x))
    tune, fit = order[: len(x) * 15 // 100], order[len(x) * 15 // 100:]
    model = Emulator()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    best, best_weights, stale = np.inf, None, 0
    for _ in range(400):
        model.train()
        for batch in fit[torch.randperm(len(fit))].split(64):
            optimizer.zero_grad()
            nn.functional.mse_loss(model(x[batch]), y[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = nn.functional.mse_loss(model(x[tune]), y[tune]).item()
        best, best_weights, stale = (loss, copy.deepcopy(model.state_dict()), 0) if loss < best else (best, best_weights, stale + 1)
        if stale == 20:
            break
    model.load_state_dict(best_weights)
    return model.eval()


def stretches(rates):
    """(recording, first bin, end bin, name) for every stretch of steady input with at least one taste on, leaving out
    stretches cut shorter than 100 ms by the end of the recording."""
    found = []
    for r, x in enumerate(rates):
        edges = [0, *(np.flatnonzero((np.diff(x, axis=0) != 0).any(1)) + 1), len(x)]
        found += [(r, a, b, " + ".join(t for t, on in zip(TASTES, x[a] > 0) if on))
                  for a, b in zip(edges[:-1], edges[1:]) if x[a].any() and b - a >= 10]
    return found


def by_taste(response, found):
    """Average MN9 firing in Hz (both neurons, after the first 30 ms) in each kind of stretch."""
    means = {}
    for r, a, b, name in found:
        means.setdefault(name, []).append(response[r, a + SETTLE:b].mean() / (fb.BIN * fb.DT / 1000))
    return {name: np.mean(values) for name, values in means.items()}


def r2(truth, predicted):
    return 1 - ((truth - predicted) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()


def main(folder):
    fb.setup(folder)
    data = simulate(folder)
    torch.set_num_threads(4)
    emulators = {kind: [train(data[f"train_{kind}_rates"], data[f"train_{kind}_spikes"][:, 0], seed) for seed in range(3)]
                 for kind in ("single", "mixed")}

    print(f"\n{'':<28}{'real brain':>12}{'emulator trained on single tastes':>36}{'emulator trained with mixtures':>34}")
    for test in ("single", "mixed"):
        rates, truth = data[f"test_{test}_rates"], data[f"test_{test}_spikes"].mean(1)  # brain's average over runs
        with torch.no_grad():
            predicted = {kind: [m(torch.tensor(rates, dtype=torch.float32)).numpy() for m in models] for kind, models in emulators.items()}
        found = stretches(rates)
        real = by_taste(truth, found)
        guess = {kind: [by_taste(p, found) for p in runs] for kind, runs in predicted.items()}
        print(f"test on new recordings, {'single tastes' if test == 'single' else 'two tastes at once'} ({len(found)} stretches)")
        cells = [f"{np.mean([r2(truth, p) for p in predicted[kind]]):.2f} (R²)" for kind in emulators]
        print(f"{'  share of response predicted':<28}{'':>12}{cells[0]:>36}{cells[1]:>34}")
        for name in real:
            cells = []
            for kind in emulators:
                values = [g[name] for g in guess[kind]]
                cells.append(f"{np.mean(values):5.1f} Hz ({min(values):.0f} to {max(values):.0f})")
            print(f"{'  ' + name:<28}{real[name]:>9.1f} Hz{cells[0]:>36}{cells[1]:>34}")
        if test == "single":
            assert real["sugar"] > 10 > real["bitter"], "the simulated brain does not respond to sugar as expected"
            assert min(r2(truth, p) for runs in predicted.values() for p in runs) > 0.5, "an emulator failed to learn single tastes"


if __name__ == "__main__":
    main(sys.argv[1])
