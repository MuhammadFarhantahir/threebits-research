"""Simulate a fruit fly's whole brain from its wiring diagram, and test how much the connection strengths matter.

A re-implementation, without their simulator library, of the model of Shiu et al. 2024 (Nature 634:210; code and
data: github.com/philshiu/Drosophila_brain_model). 127,400 neurons, each a leaky integrate-and-fire unit, wired by
the FlyWire connectome (version 630, as in the paper). The strength of a connection is its number of synapses. Its
sign (exciting or inhibiting) comes from the sending neuron's predicted transmitter. Nothing is trained.

The experiment is the one in the authors' example: pulse the 21 sugar-sensing taste neurons of the right side and
watch MN9, the motor neuron that extends the mouthparts to feed. The run is first checked against the authors' own
published output, then repeated with the strengths stored coarsely, made equal, or swapped at random.

The connectome is FlyWire data, licence CC BY-NC 4.0 (non-commercial use only; flywire.ai/principles).

Setup, once. The data folder must hold only these downloads; -I keeps Python from loading code out of it:
    uv pip install --python .venv/bin/python pyarrow
    mkdir -p data/fly_brain/results/example && cd data/fly_brain
    for f in 2023_03_23_connectivity_630_final.parquet 2023_03_23_completeness_630_final.csv results/example/sugarR.parquet
    do curl -L -o $f https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/$f; done
Run (about 1.5 minutes on 14 cores):
    .venv/bin/python -I fly_brain.py data/fly_brain
"""
import sys
from collections import deque
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

# Constants of the published model (model.py in the repository above), in ms and mV.
DT = 0.1                    # simulation step
T_MBR, TAU = 20.0, 5.0      # time constants of the membrane and of the synaptic drive
THRESHOLD = 7.0             # a neuron fires when it is this far above rest (rest -52 mV, threshold -45 mV)
REFRACTORY = 22             # steps, counting the one in which it fires, that a neuron ignores its input (2.2 ms)
DELAY = 18                  # steps a spike takes to reach the neurons it connects to (1.8 ms)
W_SYN = 0.275               # drive added per synapse
KICK = 250 * W_SYN          # a stimulation pulse, large enough to make the neuron fire
RATE = 200.0                # pulses per second given to each stimulated neuron, as in the authors' example run
STEPS, TRIALS = 10_000, 30  # each trial lasts 1 s
BIN = 100                   # steps per 10 ms bin, for time-varying stimulation and recordings
BITS = (8, 6, 5, 4, 3, 2)

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
         720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
         720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]  # FlyWire IDs, from the authors' example
MN9 = 720575940660219265

BRAIN = None  # (neuron IDs, where each neuron's outgoing connections start, receiving neurons, signed synapse counts)


def setup(folder):
    """Load the wiring diagram. Runs once in the main process and once in every worker."""
    global BRAIN
    ids = np.loadtxt(f"{folder}/2023_03_23_completeness_630_final.csv", delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
    table = pq.read_table(f"{folder}/2023_03_23_connectivity_630_final.parquet",
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    sender, receiver, counts = (column.to_numpy() for column in table.columns)
    assert (np.diff(ids) > 0).all() and (np.diff(sender) >= 0).all(), "expected sorted neuron IDs and connections"
    BRAIN = ids, np.searchsorted(sender, np.arange(len(ids) + 1)), receiver, counts.astype(float)


def rounded(w, bits):
    """Store every strength with `bits` bits: step * {-2^(bits-1), ..., 2^(bits-1) - 1}, with the step size that leaves
    the smallest squared error. The rounding precision_sweep.py applies to the decoder, with a wider search for the
    step because the fly's strengths span three orders of magnitude."""
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    values, freq = np.unique(w, return_counts=True)
    steps = np.abs(values).max() / -lo * np.geomspace(0.001, 1, 400)
    step = min(steps, key=lambda s: (freq * (np.clip(np.round(values / s), lo, hi) * s - values) ** 2).sum())
    return np.clip(np.round(w / step), lo, hi) * step


@lru_cache(maxsize=1)  # a worker runs one condition at a time
def strengths(condition):
    """Every connection's signed strength, in synapses, under a named condition."""
    counts = BRAIN[3]
    if condition.endswith("bits"):
        return rounded(counts, int(condition.split()[0]))
    if condition.startswith("nearest power of"):  # an uneven scale: fine steps for weak connections, wide ones for strong
        base = int(condition.split()[-1])
        return np.sign(counts) * base ** np.round(np.log(np.abs(counts)) / np.log(base))
    if condition == "all equally strong":
        return np.sign(counts) * np.abs(counts).mean()
    if condition == "strengths swapped at random":
        return np.sign(counts) * np.random.default_rng(0).permutation(np.abs(counts))
    return counts


def trial(starts, receivers, weights, stimulated, seed, rates=None, record=()):
    """How many times each neuron fires in one 1 s trial, and how many times each `record` neuron fires in each 10 ms bin.
    `weights` in mV. `stimulated` neurons get random pulses, RATE per second or `rates`: (bins, len(stimulated)) in Hz."""
    rng = np.random.default_rng(seed)
    n = len(starts) - 1
    record = np.asarray(record, int)
    slot = np.full(n, -1)
    slot[record] = np.arange(len(record))
    binned = np.zeros((STEPS // BIN, len(record)), int)
    chance = (np.full((STEPS // BIN, len(stimulated)), RATE) if rates is None else rates) * DT / 1000
    keep_u, keep_g = np.exp(-DT / T_MBR), np.exp(-DT / TAU)
    gain = TAU / (TAU - T_MBR) * (keep_g - keep_u)  # with keep_u and keep_g: the exact solution of the model over one step
    u, g, fired_count = np.zeros(n), np.zeros(n), np.zeros(n, int)  # potential above rest, synaptic drive, spikes so far
    pulsed = np.zeros(n, bool)
    pulsed[stimulated] = True  # the authors give stimulated neurons no refractory period
    recent = deque([np.empty(0, int)] * (REFRACTORY - 1), maxlen=REFRACTORY - 1)  # who fired in each of the last steps
    for step in range(STEPS):
        u *= keep_u
        u += gain * g
        g *= keep_g
        fired = np.flatnonzero(u > THRESHOLD)
        for sender in recent[-DELAY]:  # spikes fired 1.8 ms ago arrive now
            a, b = starts[sender], starts[sender + 1]
            g[receivers[a:b]] += weights[a:b]
        u[stimulated[rng.random(len(stimulated)) < chance[step // BIN]]] += KICK
        resting = np.concatenate(recent)
        g[resting[~pulsed[resting]]] = 0.0  # input that reaches a refractory neuron is lost, as in the authors' simulator
        u[fired] = g[fired] = 0.0
        fired_count[fired] += 1
        mine = slot[fired]
        binned[step // BIN, mine[mine >= 0]] += 1
        recent.append(fired)
    return fired_count, binned


def job(args):
    """One trial of one condition, in a worker process."""
    condition, seed = args
    ids, starts, receivers, _ = BRAIN
    if condition == "random neurons pulsed":
        stimulated = np.random.default_rng(seed).choice(len(ids), len(SUGAR), replace=False)
    else:
        stimulated = np.searchsorted(ids, SUGAR)
    return trial(starts, receivers, strengths(condition) * W_SYN, stimulated, seed)[0]


def describe(ids, starts, receivers, counts):
    n, size = len(ids), np.abs(counts)
    freq = np.unique(size, return_counts=True)[1] / len(size)
    strongest = np.sort(size)[::-1][: len(size) // 100]
    print(f"{n:,} neurons, {len(size):,} connections, {int(size.sum()):,} synapses")
    print(f"connected pairs, out of all possible pairs of neurons: {len(size) / (n * (n - 1)):.2%}")
    print(f"partners per neuron (median): receives from {np.median(np.bincount(receivers, minlength=n)):.0f}, sends to {np.median(np.diff(starts)):.0f}")
    print(f"synapses per connection: median {np.median(size):.0f}, mean {size.mean():.1f}, 99th percentile {np.percentile(size, 99):.0f}, largest {size.max():.0f}")
    print(f"connections made of one synapse: {(size == 1).mean():.0%}; of five or more: {(size >= 5).mean():.0%}")
    print(f"share of all synapses in the strongest 1% of connections: {strongest.sum() / size.sum():.0%}")
    print(f"inhibitory: {(counts < 0).mean():.0%} of connections, {size[counts < 0].sum() / size.sum():.0%} of synapses")
    print(f"bits needed per connection to record its strength (entropy of the synapse counts): {-(freq * np.log2(freq)).sum():.1f}")


def main(folder):
    setup(folder)
    ids, starts, receivers, counts = BRAIN
    describe(*BRAIN)

    conditions = ["intact", "intact, different random pulses", *(f"{b} bits" for b in BITS),
                  *(f"nearest power of {base}" for base in (2, 4, 16)),
                  "all equally strong", "strengths swapped at random", "random neurons pulsed"]
    with Pool(initializer=setup, initargs=(folder,)) as pool:
        rates = {}  # condition -> every neuron's firing rate in Hz (spikes per 1 s trial, averaged over trials)
        for condition in conditions:
            seeds = range(TRIALS, 2 * TRIALS) if "different" in condition else range(TRIALS)  # same pulses otherwise
            rates[condition] = sum(pool.map(job, [(condition, seed) for seed in seeds])) / TRIALS
    published = pq.read_table(f"{folder}/results/example/sugarR.parquet", columns=["flywire_id"]).column(0).to_numpy()
    published = np.bincount(np.searchsorted(ids, published), minlength=len(ids)) / TRIALS
    control = rates.pop("random neurons pulsed")

    mn9 = np.searchsorted(ids, MN9)
    others = np.ones(len(ids), bool)
    others[np.searchsorted(ids, SUGAR)] = False  # the pulsed neurons fire in every condition, so leave them out

    def match(r):  # correlation with the intact brain's firing rates, over all neurons that are not pulsed
        return np.corrcoef(r[others], rates["intact"][others])[0, 1] if r[others].any() else 0.0

    print(f"\n{'':<34}{'strength levels':>16}{'connections kept':>18}{'MN9 fires (Hz)':>16}{'neurons at 1 Hz or more':>25}{'match with intact':>19}")
    print(f"{'published run (authors)':<34}{'':>16}{'':>18}{published[mn9]:>16.1f}{(published >= 1).sum():>25}{match(published):>19.3f}")
    for condition, r in rates.items():
        size = np.abs(strengths(condition))
        print(f"{condition:<34}{len(np.unique(size[size > 0])):>16}{np.count_nonzero(size) / len(size):>18.1%}"
              f"{r[mn9]:>16.1f}{(r >= 1).sum():>25}{match(r):>19.3f}")
    print(f"control, {len(SUGAR)} randomly chosen neurons pulsed instead (a new group in each trial): MN9 fires at {control[mn9]:.1f} Hz")

    assert match(published) > 0.999 and abs(rates["intact"][mn9] / published[mn9] - 1) < 0.05, "does not reproduce the published run"
    assert control[mn9] < 0.1 * rates["intact"][mn9], "MN9 responds to random stimulation"


if __name__ == "__main__":
    main(sys.argv[1])
