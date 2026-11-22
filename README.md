# Threebits Research

**Causally validated human whole-brain simulation, certified-safe closed-loop control, pure-integer sub-8-bit neural implant decoders, and co-adaptive Brain–AI co-processing.**

- **Live Research Overview:** https://threebits.si/
- **Interactive 100-Region Human Brain Twin & 50-Patient Stroke Explorer:** https://threebits.si/brain/
- **Preprints (HTML, PDF, arXiv LaTeX, BibTeX):** https://threebits.si/papers/
- **Live Browser Neural Gym, Multi-Modal Bandwidth Lab & Single-Blind Pilot:** https://threebits.si/gym/
- **Physical `< $1,000` Hardware Bench Guide (`RP2350` + LSL EEG/sEMG):** [hardware_bench_guide.md](hardware_bench_guide.md)
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
- **Superhuman Multi-Hop Fluid Reasoning (`superhuman_reasoning_emulator.py`):** Extends the 100-region emulator to compositional relational graph traversal (`N = 10` novel directed cycles + private embodied human goal, trained strictly on `D = 1..4`, tested **zero-shot on `D = 6*` and `D = 8*`**). With an intact biological hippocampus across all brain regimes, a **bias-free 4-bit linear-manifold Region 101 Reasoning Cortex** hits **`100.0% ± 0.0%`** across `D = 1..4` and zero-shot `D = 6*, 8*` vs. `18.0%` biological brain alone, `21.4%` neuromodulatory denoising, `22.6%` Sham 4-bit, and `49.7%` Standalone AI Bayes cap — while preserving post-train `CCEP ρ = +0.661 (+0.380)`, `45/46` healthy TMS-EEG (`+0.2882`), bounded synchrony (`max |Δsync| = 0.0233`), and **`100.0%` frozen-core continual goal-flip recovery** (`90.9%` frozen, `4,528` adapted params, 80 steps).
- **Zero-Replay Continual Multi-Task Learning (`continual_multitask_brain.py`):** Across three output-disjoint cognitive domains learned sequentially with **zero replay** (`T1`: zero-shot `D = 6*` multi-hop reasoning → `T2`: shifted `K = 10` working memory → `T3`: 4-way reflected modular planning; 3 seeds), a task-conditioned 235k **Causal Transformer** suffers **`−84.9 pp` catastrophic forgetting** (`11.6%` final average) and **Unprotected Brain + 4-bit R101** suffers **`−92.1 pp` forgetting** (`35.5%` avg), whereas **Threebits Continual Brain** (`58k` params, `96.3%` frozen on `T2`, `85.1%` on `T3` + PET-receptor-gated prefrontal-motor adapters) achieves **`100.0%` final 3-task average with `0.0 pp` forgetting**, preserving 74-patient CCEP (`+0.661`) and `45/46` healthy TMS-EEG (`+0.2882`).
- **Autonomous Task-Blind Continual Brain (`autonomous_brain_bg_thalamus.py`):** Removes `task_id` completely via a **Basal Ganglia (`D1` Go / `D2` NoGo + Dopamine RPE) + Mediodorsal Thalamus + Hippocampal SWR** upgrade. Autonomously catches **`6/6`** unannounced shifts in 3 steps, discovers Tick 1 & Tick 2 over fixed 6-tick rollouts, and locks onto hidden rules purely from 1-bit reward in **`2.15` trials** (`99.8%` steady-state, `91.2%` overall, `0.0 pp` forgetting, CCEP `+0.661`, `44/46` TMS-EEG `+0.288`) vs `41.2%` matched-weights Sham Dopamine RPE and `10.5%` Causal Transformer.
- **Unified Silicon-in-the-Loop Spiking Brain + Counterfactual Region 101 (`unified_spiking_chip_brain.py`):** Unifies Papers 1 + 2 + 3 by embedding `176` biophysical LIF E–I spiking neurons inside `LH_Cont_PFCl_1` decoded live by our **compiled zero-float C99 binary (`chip/int4_gru.c`, `65.43 kB`, `23.7 µs/bin`, `100.0%` W4A32/W4A8 parity)** driving **81-branch counterfactual planning** under Paper 1 Lyapunov safety bounds across 450 unseen causal trap worlds — achieving **`100.0% ± 0.0%`** Day 1 and Week-6 drift recovery (`1%` adapter, SHA256-verified byte-identical ROM, `0%` seizures vs `126 Hz / 100%` without guard, `5.6%` bio alone, `3.8%` Sham C99, `49.8%` AI Bayes cap).
- **Zero-Backprop Local Brain–Chip Co-Adaptation (`zero_backprop_coadapt.py`, `chip/c99_local_plasticity.c`, `chip/firmware_rp2350/coadapt_bench.c`):** Removes backpropagation, labels, and floating-point arithmetic completely (`176`-LIF Three-Factor STDP ↔ **`3.27 kB` pure-integer C99 local plasticity kernel**, `0.564 µs/trial` bare-metal `-O3`). Across 5 seeds × 1,200 online trials, bidirectional local plasticity learns **`90.0% ± 11.5%`** from random scratch, survives an unannounced Trial-540 `25%` electrode permutation + private rule flip (`2.7%` shock), and autonomously recovers live online to **`93.0% ± 6.7%`** vs `45.8%` 1-way bio STDP, `25.0%` bio alone, `15.4%` 1-way C99, and `11.2%` Sham RPE.
- **Open-Ended ARC-Style `5×5` Visual-Program Synthesis (`arc_program_brain.py`):** Across `110` unique compositional visual programs (`8` primitives, trained on `28` seen `L = 2` pairs, tested strictly on `28` unseen `L = 2*` & `336` unseen deeper `L = 3*` + private human constraint; strict `25/25` exact pixel match, zero raw pixel access by Region 101), Human Brain + 4-bit Region 101 hits **`98.7%` seen `L=2`, `98.0%` unseen `L=2*`, and `98.2%` zero-shot deeper `L=3*` (`98.1%` OOD mean)**, beating even an oracle Standalone AI Symbolic Program Solver (`50.1%` cap, **`+48.0 pp`**) and Direct-Pixel Transformer / Sham / Bio Alone (`0.0%`) while preserving 74-patient CCEP (`+0.661`) and `44/46` TMS-EEG (`+0.290`).
- **50 Personalized Real Human Brain Twins & Focal Stroke Rescue (`personalized_50brain_rescue.py`):** Trains ONE shared 100-region brain + 4-bit Region 101 core on independent HCP + 74-patient CCEP, freezes **`100%`** of the brain & 4-bit Region 101 core (`92.6%` of all params), and evaluates out-of-dataset across **all 50 real `MICA-MICs` structural connectomes (`HC001–HC050`, CC0, zero synthetic rotation)**: transfers across **`50/50`** healthy brains (`9.8%` bio alone / `46.6%` zero-shot → **`99.6% ± 0.4%`**, $p = 3.59 \times 10^{-10}$) and rescues severe **65% Frontoparietal Stroke + 30% Focal Cortical Infarct** across **`50/50`** individuals (`9.7%` bio alone / `15.8%` zero-shot → **`85.1% ± 9.8%`**, `49/50 > 50%` AI cap, $p = 3.74 \times 10^{-10}$; median $|\Delta\text{sync}| = 0.045$, `50/50 < 0.10`).
- **Interactive 100-Region Human Brain Twin & 50-Patient Stroke Explorer (`brain_twin/` → https://threebits.si/brain/):** Browser lab built from the 74-patient CCEP connectome (`ds004080`), 19-receptor PET atlas, HCP MEG, and all **50 real `MICA-MICs` individuals (`HC001–HC050`)** — click any parcel to fire a causal pulse, select any patient `HC001–HC050`, toggle `65% Frontoparietal Stroke` ON/OFF, and explore all 8 verified `v2.0` benchmark tabs.
- **Level-3 Live Protocol, `< $1,000` Hardware Bench & Pilot API (`hardware_bench_guide.md`, `level3_app/`, `pilot_api_server.py`, `analyze_human_gym_pilot.py`):** Multi-modal streaming server (`--lsl` EEG, `--lsl-emg` silent wrist sEMG, `--serial` physical RP2350 UART, `GET /api/hardware_status`), Multi-Modal Sub-Verbal Bandwidth Lab (`?tab=bandwidth`), and **Single-Blind 18-Trial Living-Human Pilot** (`https://threebits.si/gym/?tab=pilot`).

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

# Run the 100-region continual 3-task benchmark vs. Causal Transformer & GRU
python continual_multitask_brain.py

# Run Track A (Autonomous Task-Blind BG/Thalamus) & Track B (Unified 176-LIF Spiking + Real C99 int4 Chip)
python autonomous_brain_bg_thalamus.py
python unified_spiking_chip_brain.py

# Compile & run v2.0 Track E (Zero-Backprop STDP <-> C99 Local Plasticity), Track F (ARC Visual Programs) & Track G (50 MICA-MICs Brains)
cc -O3 -std=c99 -Wall -Wextra -shared -fPIC chip/c99_local_plasticity.c -o chip/libc99_local.dylib
cc -O3 -std=c99 -Wall -Wextra -Ichip chip/firmware_rp2350/coadapt_bench.c chip/c99_local_plasticity.c -o chip/firmware_rp2350/coadapt_bench && ./chip/firmware_rp2350/coadapt_bench
python zero_backprop_coadapt.py
python arc_program_brain.py
python personalized_50brain_rescue.py

# Run the local Level-3A EEG + sEMG + RP2350 UART Neural Gym & Pilot API server
python level3_app/server.py --port 8765 --lsl --lsl-emg --serial /dev/cu.usbmodem*

# Analyze exported living-human single-blind pilot CSV(s) or pull directly from the live Pilot API
python analyze_human_gym_pilot.py path/to/threebits_pilot_*.csv
python analyze_human_gym_pilot.py --from-api https://threebits.si/api/pilot/csv
```

All human and primate neuroimaging/electrophysiology datasets are open-access public datasets (`OpenNeuro ds004080`, `ds008037`, `ds005557`, `ds005558`; `DANDI 000128`, `000954`; `MICA-MICs`; `ENIGMA HCP`; `Hansen et al. 2022`). Large raw recordings are not stored in this repository; scripts fetch or expect them under `data/`.

## Licence

Code and manuscripts are released under the [MIT License](LICENSE). Source datasets retain their original open licences (CC0, CC-BY-4.0, BSD-3-Clause).
