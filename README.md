# Threebits Research

**Causally validated human whole-brain simulation, certified-safe closed-loop control, pure-integer sub-8-bit neural implant decoders, and co-adaptive Brain–AI co-processing.**

- **Live Research Overview:** https://threebits.si/
- **Preprints (HTML, PDF, arXiv LaTeX, BibTeX):** https://threebits.si/papers/
- **Live Browser Neural Gym & 4-Arm EEG Replay:** https://threebits.si/gym/
- **Author:** Muhammad Farhan Tahir ([LinkedIn](https://www.linkedin.com/in/muhammadfarhantahir/))

---

## Preprints & Core Findings

### 1. Causal & Multi-Scale Human Whole-Brain Simulation + Certified-Safe Control + 15-Patient Clinical Memory
- **Manuscript:** [paper_causal_brain_sim.md](paper_causal_brain_sim.md) · [PDF](papers/threebits_paper1_causal_brain_sim.pdf) · [arXiv LaTeX](papers/threebits_paper1_causal_brain_sim_arxiv.tar.gz)
- **Observational vs. Causal Trade-off (`validate_simulator_ccep.py`, `causal_brain_simulator.py`):** Across **74 human patients (316,366 electrical stimulation trials; OpenNeuro `ds004080`)**, tuning a 100-region Hopf simulator to resting-state fMRI (`r: 0.304 → 0.889`) degrades causal CCEP prediction (`ρ: +0.616 → +0.176`; distance-controlled partial `+0.311 → +0.049`). Direct CCEP calibration lifts 5-fold unseen-pair accuracy to **`ρ = +0.710, r = +0.727, partial +0.444`**.
- **Multi-Scale Biological Twin (`multiscale_brain_twin.py`):** Aligning 100/100 cortical parcels with **19 real human PET neurotransmitter receptors** (Hansen et al. 2022, 1,238 healthy adults), **6-band HCP MEG**, and **Mesulam laminar hierarchy** jumps completely unseen stimulated regions (`partial ρ: +0.319 → +0.366`, `r: +0.536 → +0.651`; matched-capacity permuted-biology control `+0.321`), lifts **Leave-One-Yeo-7-Network-Out** (`+0.337 → +0.374`), reaches **`ρ = +0.716, r = +0.744, partial +0.454`** on unseen pairs, and automatically recovers cholinergic facilitation (`A4B2 +0.34/+0.37`, `VAChT +0.27/+0.36`) and `GABAa` inhibitory suppression (`−0.22`).
- **Zero-Shot Healthy-Human TMS-EEG (`validate_simulator_tmseeg.py`):** Transfers with zero EEG tuning to **`44/46` clean healthy adults** (`OpenNeuro ds008037`, partial `ρ | d = +0.247–0.278` vs resting-fMRI `+0.131`, rewired `+0.028`, wrong-site `−0.136`).
- **Certified-Safe Closed-Loop Control (`safe_closedloop_controller.py`):** Exact spectral-radius projection + Small-Gain contraction + Discrete-Time Bounded Real Lemma Riccati Lyapunov certification yields **`160/160` verified stability certificates**, **`0/160` energy or hypersynchrony excursions**, and a **3.1× gain over open-loop** (`+0.0643` vs `+0.0204`) on the CCEP brain.
- **15-Patient Intracranial Memory Policy (`human_memory_policy.py`):** Across **4,500 words & 1,142 real closed-loop stimulations (`OpenNeuro ds005557/5558`)**, held-out recall decoding reaches `0.603` AUC (`p = 0.0001`, `14/15` above chance); deep hippocampal/insular targets consistently impair recall (`−8.46 pp`, `p = 0.0035` vs LTC), and a held-out LOPO policy avoids `6/7` strongly harmful sessions.

### 2. Sub-8-Bit Quantized Neural Decoders Under 6-Week Human Intracortical Drift
- **Manuscript:** [paper_falcon_lowbit.md](paper_falcon_lowbit.md) · [PDF](papers/threebits_paper2_falcon_lowbit.pdf) · [arXiv LaTeX](papers/threebits_paper2_falcon_lowbit_arxiv.tar.gz)
- **Quantization × Drift (`precision_sweep.py`, `falcon_precision_drift.py`, `falcon_anchored_recal.py`):** On `FALCON H1` (`DANDI 000954`, 176 channels, 7-DOF, 6 weeks), 3-bit (`44.1 kB`) and 4-bit (`58.8 kB`) LSQ causal GRUs match 32-bit float accuracy (`0.816` vs `0.813`). Unsupervised channel mean-centering + a 48-second L2-anchored 1% frozen-core adapter keeps 99% of quantized weights read-only on chip and beats full fine-tuning across 3–32 bits.
- **Pure-Integer C99 Hardware Runtime & Firmware (`chip/int4_gru.c`, `chip/firmware_rp2350/`, `falcon_submit/`):** Zero floating-point math inside `gru_int4_step()`, matching PyTorch W4A32 (`0.8191` vs `0.8192` minival; `0.3178` vs `0.3134` 6-week recalibration; reverse-time control `−0.491`) in **`66.35 kB` static Flash + SRAM** (`62.94 kB` ROM + `3.40 kB` SRAM) at **`14.3 µs` per 20 ms bin** (`5.88 MMAC/s` at 50 Hz).

### 3. Working-Memory Limits & Co-Adaptive Brain–AI Co-Processing
- **Manuscript:** [paper_prosthetic_cortex.md](paper_prosthetic_cortex.md) · [PDF](papers/threebits_paper3_prosthetic_cortex.pdf) · [arXiv LaTeX](papers/threebits_paper3_prosthetic_cortex_arxiv.tar.gz)
- **Level-1 Healthy-Human Empirical Anchor (`human_intelligence_wm_eeg.py`, `replicate_tmseeg_g.py`):** Across **123 healthy adults** (`OpenNeuro ds008037`, `33,299` trials), empirical Cowan working-memory recall drops `88.6% → 61.5% → 43.8%` (`K = 2 → 4 → 6`); strictly pre-response scalp EEG detects overload at `0.690` AUC (`p = 7.2×10⁻⁵`, `14/15` adults, up to `0.916`). An initial `n = 22` resting TMS amplitude vs. `g` correlation failed unseen `n = 24` replication (`ρ = −0.104`, reported transparently).
- **Level-2 Scaling Law & Phase Transition (`brain_ai_coprocessor.py`):** Sweeping bandwidth × latency at `K = 12` reveals a sharp transition at **`320 bits/s` (`~265 bits/s` MI) and `≤ 100 ms`** (`33% → 91.5%`, `97.4%` at `K = 16`), passing **Strict Outperformance** on a complementary private-rule task (**`87.0%` vs `23.2%` standalone AI, `14.8%` human alone; `+63.7 pp`**; cortical CKA `10.8×`).
- **Triple-Validated 100-Region Functional Human Brain Emulator (`functional_brain_emulator.py`):** Upgrades all 100 cortical parcels into trainable receptor-/MEG-constrained E–I microcircuits anchored to 74-patient CCEP white matter. Even after cognitive training (`K = 2..16` + 4-way executive rotation), it retains **`CCEP ρ = +0.662 (partial +0.378)`** and **`45/46` healthy TMS-EEG (`+0.288`)** vs. rewired controls (`−0.031` / `32/46`), proving that **only clean associative indexing** — an Upgraded Hippocampal Index (`100%` through `K = 8`, `87.2%` at `K = 12`) or a 4-bit **Region 101 Prosthetic Cortex** on the unmodified noisy biological brain (`72.1%` at `K = 12` vs `30.6%` baseline & `35.6%` Sham) — breaks Cowan's wall.
- **Level-3 Live Protocol & App (`level3_human_trial_protocol.md`, `level3_app/`):** Local LSL EEG + real `ds008037` replay server and non-verbal Neural Gym (also live at https://threebits.si/gym/).

---

## Reproducing Benchmarks

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy torch scikit-learn matplotlib

# Compile & benchmark the pure-integer C99 4-bit GRU chip kernel
cd chip && cc -O3 -shared -fPIC int4_gru.c -o libint4_gru.dylib && cd ..
python chip/bench_int4_gru.py

# Run the Level-2 Brain–AI Co-Processor scaling law (zero external downloads needed)
python brain_ai_coprocessor.py

# Run the local Level-3A EEG + Neural Gym app (supports --lsl for live headsets)
python level3_app/server.py --port 8765
```

All human and primate neuroimaging/electrophysiology datasets are open-access public datasets (`OpenNeuro ds004080`, `ds008037`, `ds005557`, `ds005558`; `DANDI 000128`, `000954`; `MICA-MICs`; `ENIGMA HCP`; `Hansen et al. 2022`). Large raw recordings are not stored in this repository; scripts fetch or expect them under `data/`.

## Licence

Code and manuscripts are released under the [MIT License](LICENSE). Source datasets retain their original open licences (CC0, CC-BY-4.0, BSD-3-Clause).
