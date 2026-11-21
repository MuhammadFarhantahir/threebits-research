"""Emulate a recorded brain area: predict 45 neurons' firing from the other 137, as the Neural Latents Benchmark scores it.

Data: MC_Maze (see baseline_decoder.py), 182 neurons in a monkey's motor and premotor cortex. The recording marks 45 of them
as held out. For each of the 574 validation reaches a model gets only the other 137 neurons and must give the held-out
neurons' firing rates in 5 ms steps, from 250 ms before to 450 ms after the hand starts moving. This is the benchmark's
co-smoothing task (Pei et al. 2021, arXiv:2109.04463). For the 1,721 training reaches it gets all 182. Settings are chosen
on reaches set aside from the training ones; the validation and test reaches are used only for the final scores.

The emulator is a recurrent network that reads the given neurons forwards and backwards in time and gives firing rates for
all 182 neurons. While training, each pass hides a random part of the given neurons' spike counts, and the emulator is also
scored on predicting the hidden counts, so it has to learn how the neurons relate to each other ("coordinated dropout",
Keshtkaran and Pandarinath 2019, as in AutoLFADS).

Score: bits per spike, the benchmark's main number: how much better the predicted rates explain each held-out spike than
that neuron's average rate does. 0 means no better. The leaderboard (eval.ai challenge 1256) closed in January 2026 and its
hidden test answers were published (nlb_tools repository, data/eval_data_test.h5), so the test reaches can be scored here
the same way. Test inputs: data/mc_maze_test_inputs.npz, the given neurons of the 574 test reaches from DANDI 000128's test
file, binned by nlb_tools' make_eval_input_tensors (which needs Python 3.10, pandas 1.3.4 and numpy<2). As a check, the
benchmark's simplest baseline, spike smoothing, is rebuilt here; the leaderboard lists it at 0.211 on the test reaches.

Run (about 30 minutes on 14 cores):
    .venv/bin/python neural_emulator.py data/mc_maze_train.nwb data/mc_maze_test_inputs.npz data/nlb_eval_data_test.h5
"""
import copy
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.special import xlogy
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from baseline_decoder import BIN, PAD, load

HIDDEN, DROPOUT = 128, 0.5  # units each way, and dropout on the network's output (chosen 2026-11-18 on the tuning reaches)
# Chosen 2026-11-19 by the score on the tuning reaches, random start 0: hiding 30% or 50% of the counts, with or without
# 64 shared signals, gave 0.312, 0.318, 0.326 (chosen) and 0.324 bits per spike; without predicting the given neurons, 0.285.
HIDE = 0.5                  # share of the given neurons' counts hidden in each training pass
FACTORS = None              # if set, all rates are read out from this many shared signals
PREDICT_GIVEN = True        # also score the emulator on the hidden counts of the given neurons
SEEDS = 5
SMOOTH = 0.050 / BIN        # width of the smoothing baseline's Gaussian, in bins (50 ms)


def bits_per_spike(rates, spikes):
    """The benchmark's co-smoothing score for predicted rates (spikes per bin) against observed spike counts.
    Gives the same values as nlb_tools.evaluation.bits_per_spike (checked to 6 decimals on 2026-11-19)."""
    average = np.broadcast_to(spikes.mean(axis=(0, 1)), spikes.shape)  # each neuron's mean over all reaches and bins
    loglik = lambda r: (xlogy(spikes, r) - r).sum()                     # Poisson log-likelihood, constant term dropped
    return (loglik(rates) - loglik(average)) / spikes.sum() / np.log(2)


class Emulator(nn.Module):
    """Given neurons (reaches, bins, n_in) -> log firing rates of all neurons, the given ones first, then the held-out ones."""

    def __init__(self, n_in, n_out, hidden=HIDDEN, hide=HIDE, factors=FACTORS):
        super().__init__()
        self.hide = hide
        self.gru = nn.GRU(n_in, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(DROPOUT)
        readout = [nn.Linear(2 * hidden, factors), nn.Linear(factors, n_in + n_out)] if factors else [nn.Linear(2 * hidden, n_in + n_out)]
        self.out = nn.Sequential(*readout)

    def forward(self, spikes, shown=None):
        """`shown` marks the input counts visible in a training pass; the visible ones are scaled up to make up for the rest."""
        if shown is not None:
            spikes = spikes * shown / (1 - self.hide)
        return self.out(self.drop(self.gru(spikes)[0]))


def held_out_rates(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)[..., x.shape[-1]:].exp().numpy()


def score(model, x, y):
    return bits_per_spike(held_out_rates(model, x), y.numpy())


def train(model, x, y, fit, tune, predict_given=PREDICT_GIVEN):
    """Fit to the `fit` reaches until the score on the `tune` reaches stops improving; keep the best weights."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_weights, stale = -np.inf, None, 0
    for _ in range(500):
        model.train()
        for batch in fit[torch.randperm(len(fit))].split(64):
            xb, yb = x[batch], y[batch]
            shown = torch.rand_like(xb) >= model.hide
            loss = nn.functional.poisson_nll_loss(model(xb, shown), torch.cat([xb, yb], -1), reduction="none")
            counted = torch.cat([~shown & predict_given, torch.ones_like(yb, dtype=torch.bool)], -1)  # hidden given counts, all held-out ones
            optimizer.zero_grad()
            loss[counted].mean().backward()
            optimizer.step()
        tuned = score(model, x[tune], y[tune])
        best, best_weights, stale = (tuned, copy.deepcopy(model.state_dict()), 0) if tuned > best else (best, best_weights, stale + 1)
        if stale == 20:
            break
    model.load_state_dict(best_weights)
    return model


def prepare(path):
    """Spike counts in the benchmark's window, given neurons (x) and held-out neurons (y); the reaches of each role."""
    counts, _, split = load(path)
    with h5py.File(path) as f:
        held = f["units/heldout"][:]
    window = counts[:, PAD:-PAD].astype(np.float32)  # 140 bins of 5 ms, from 250 ms before to 450 ms after movement onset
    rows = np.random.default_rng(0).permutation(np.flatnonzero(split == "train"))
    tune, fit = rows[: len(rows) * 15 // 100], rows[len(rows) * 15 // 100:]
    return window[..., ~held], window[..., held], fit, tune, np.flatnonzero(split == "val")


def smoothing_baseline(x_fit, y_fit, x_eval):
    """The benchmark's baseline: smooth each given neuron, then one Poisson regression per held-out neuron."""
    features = lambda x: np.log(gaussian_filter1d(x, SMOOTH, axis=1) + 1e-4).reshape(-1, x.shape[-1])  # within the window only
    fx, ex = features(x_fit), features(x_eval)
    fit = lambda k: make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.0, max_iter=3000)).fit(fx, y_fit[..., k].ravel())
    models = Parallel(n_jobs=-1)(delayed(fit)(k) for k in range(y_fit.shape[-1]))
    return np.stack([m.predict(ex) for m in models], -1).reshape(*x_eval.shape[:2], -1)


def one_seed(args):
    """Train one emulator from a random start. Held-out rates for the validation reaches, for the control (another
    reach's neurons) and, if test inputs are given, for the test reaches."""
    path, test, seed = args
    torch.set_num_threads(3)
    x, y, fit, tune, val = prepare(path)
    torch.manual_seed(seed)
    x, y = torch.tensor(x), torch.tensor(y)
    model = train(Emulator(x.shape[-1], y.shape[-1]), x, y, torch.tensor(fit), torch.tensor(tune))
    rates = [held_out_rates(model, x[val]), held_out_rates(model, x[np.roll(val, 1)])]
    if test:
        rates.append(held_out_rates(model, torch.tensor(np.load(test)["test_spikes_given"])))
    return rates


def main(path, test=None, answers=None):
    x, y, fit, tune, val = prepare(path)
    print(f"{x.shape[-1]} neurons given, {y.shape[-1]} held out | {len(fit)} training, {len(tune)} tuning, {len(val)} validation reaches")
    train_reaches = np.concatenate([fit, tune])
    baseline = bits_per_spike(smoothing_baseline(x[train_reaches], y[train_reaches], x[val]), y[val])
    print(f"validation | spike smoothing (the benchmark's simplest baseline): {baseline:.3f} bits per spike", flush=True)

    with Pool(SEEDS) as pool:
        runs = pool.map(one_seed, [(path, test, seed) for seed in range(SEEDS)])
    for seed, run in enumerate(runs):
        print(f"validation | emulator, random start {seed}: {bits_per_spike(run[0], y[val]):.3f} bits per spike | "
              f"control, given another reach's neurons: {bits_per_spike(run[1], y[val]):.3f}")
    ensemble = bits_per_spike(np.mean([run[0] for run in runs], axis=0), y[val])
    print(f"validation | emulator, average of the {SEEDS} starts: {ensemble:.3f} bits per spike", flush=True)
    assert max(bits_per_spike(run[1], y[val]) for run in runs) < 0.05 < baseline < ensemble, "emulator self-check failed"

    if test:
        with h5py.File(answers) as f:
            truth = f["mc_maze/eval_spikes_heldout"][:].astype(np.float32)
        everything = np.concatenate([train_reaches, val])  # the baseline is fit on every reach of the training file, as on the leaderboard
        smoothed = smoothing_baseline(x[everything], y[everything], np.load(test)["test_spikes_given"])
        print(f"test ({len(truth)} reaches, published answers) | spike smoothing: {bits_per_spike(smoothed, truth):.3f} bits per spike (leaderboard: 0.211)")
        for seed, run in enumerate(runs):
            print(f"test | emulator, random start {seed}: {bits_per_spike(run[2], truth):.3f} bits per spike")
        print(f"test | emulator, average of the {SEEDS} starts: {bits_per_spike(np.mean([run[2] for run in runs], axis=0), truth):.3f} bits per spike")


if __name__ == "__main__":
    main(*sys.argv[1:])
