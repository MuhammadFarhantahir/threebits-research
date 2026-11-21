# Level-3 Healthy-Human Non-Invasive Proof Protocol (Zero Surgery)

**Threebits Research** · Experimental Blueprint (November 2026)

This document specifies the two immediate **live healthy-human experiments** (Level 3A & Level 3B) that directly follow our Level-1 healthy-human `ds008037` results (`human_intelligence_wm_eeg.py`) and Level-2 Co-Adaptive Brain–AI Scaling Law (`brain_ai_coprocessor.py`). Neither experiment requires brain surgery.

---

## Experiment 3A — Closed-Loop Non-Invasive Brain–AI Co-Processor (4-Arm Randomized Trial)

### Scientific Hypothesis
A real-time closed-loop neural decoder (`chip/int4_gru.c`, 50 Hz, 14.3 µs/step) reading pre-response working-memory overload and imminent cognitive-failure signatures from scalp EEG can trigger timely, low-friction external AI co-processor assistance *before* human working-memory breakdown — outperforming unassisted humans, manual AI querying, and a **sham closed-loop control** (identical AI assistance triggered by scrambled/delayed EEG).

### Hardware Bill of Materials (Under $10,000)
- **EEG acquisition:** 32- or 64-channel research-grade active wet/gel EEG cap @ 500–1000 Hz with hardware trigger sync (e.g., OpenBCI Galea/Cyton+Daisy or Brain Products LiveAmp / g.tec Unicorn Hybrid Black — $1k–$9k).
- **Real-time inference host:** Laptop or ARM Cortex-M7/RP2350 board running our compiled `chip/int4_gru.c` (`66.35 kB`, `< 1 ms` total pipeline latency from EEG packet to state score).
- **Sync:** Lab Streaming Layer (LSL) timestamping visual stimuli, EEG packets, decoder state scores, and behavioral responses to `< 2 ms`.

### 4-Arm Within-Subject Randomized Crossover Design (`N = 30` Healthy Adults)
Every participant completes 4 counterbalanced blocks of high-load timed working-memory & relational reasoning (`K = 4, 6, 8` continuous visual-spatial + relational binding trials, directly extending OpenNeuro `ds008037`):
1. **Arm 1 — Human Alone:** Standard timed cognitive task, no AI assistance.
2. **Arm 2 — Human + Manual AI UI:** Participant can manually request an on-screen memory recall / hint via keypress (bottlenecked by conscious decision + motor I/O).
3. **Arm 3 — Human + Sham Closed-Loop Co-Processor *(critical placebo control)*:** Identical low-friction adaptive co-processor interface, triggered at the *exact same frequency* as Arm 4, but driven by **trial-permuted (sham) EEG state scores** from prior blocks.
4. **Arm 4 — Human + Real Closed-Loop Neural Co-Processor:** Our causal EEG decoder continuously estimates real-time cognitive load (`set_size` / alpha-beta suppression + parietal slow-wave) and pre-response failure risk during the `[0.0, 1.5 s]` retention delay; when predicted failure risk exceeds threshold, the AI co-processor automatically surfaces a minimal peripheral cue / offloaded memory anchor *before* recall breakdown.

### Pre-Registered Proof Criteria
- **Primary endpoint:** Within-subject recall precision (`< 30°` accuracy & mean angular error) and composite speed-accuracy throughput on overloaded (`K = 6, 8`) trials.
- **Pass criterion:** Arm 4 achieves statistically significant improvement ($p < 0.01$, paired within-subject) over **Arm 1 (Human Alone)**, **Arm 2 (Manual AI)**, **AND Arm 3 (Sham Closed-Loop)** — proving the gain causally requires real-time neural state readout rather than generic AI help or placebo.
- **Secondary endpoint:** Online causal decoder AUC (`> 0.60` pre-response failure detection, `> 0.70` cognitive overload detection) verified prospectively against offline ground truth.

---

## Experiment 3B — Simulator-Guided Closed-Loop Prefrontal TMS-EEG in Healthy Adults (University Partner IRB Pilot)

### Scientific Hypothesis
Using our **invasive-CCEP + non-invasive-TMS-EEG validated whole-brain simulator** (`causal_brain_simulator.py`, `validate_simulator_tmseeg.py`) and **Lyapunov-certified safe closed-loop control framework** (`safe_closedloop_controller.py`), state-timed prefrontal/parietal TMS (or temporal-interference / focused ultrasound) delivered specifically during detected low-encoding / working-memory-drop states improves healthy-human working-memory precision compared to sham stimulation and open-loop fixed-rhythm stimulation.

### Design (`N = 24` Healthy Adults, University TMS-EEG Partner Lab under IRB)
- **Baseline mapping (Session 1):** Collect each subject's resting MRI/EEG + 3-site single-pulse TMS-EEG (`prefrontal [-46, 33, 34]`, `parietal [-26, -69, 63]`, `shoulder sham`, identical to `ds008037`) to calibrate an individual causal propagation model (strictly avoiding resting-fMRI overfitting!).
- **Closed-loop intervention (Session 2, double-blind within-subject crossover):**
  1. **Sham stimulation** (shoulder / sham coil)
  2. **Open-loop rhythmic stimulation** (matched total pulse energy, uncoupled from brain state)
  3. **Simulator-guided closed-loop stimulation** (triggered only during decoder-detected pre-response failure windows at the individually calibrated frontoparietal target)
- **Safety guardrails:** Standard Rossi et al. (2021) international TMS safety limits (pulse energy, inter-train intervals, motor-threshold calibration), hard per-step hardware current/intensity caps, and real-time EEG epileptiform/hypersynchrony interlock stopping stimulation immediately if global synchrony exceeds baseline $+0.03$.

---

## Transition to Level 4 — The Prosthetic Cortex Superintelligence Criterion

Once non-invasive Level-3 trials validate closed-loop cognitive gains in healthy adults, crossing from **+15–30% augmentation** into **superhuman cognition** requires replacing scalp EEG (~10–40 bits/s) with next-generation high-bandwidth minimally invasive interfaces (thin-film micro-ECoG, endovascular arrays, or focused ultrasound/optical read-write) achieving:
1. **Verified bidirectional bandwidth `> 100 → > 1,000 bits/s`** (exceeding the ~39 bits/s human conscious language ceiling) at **`< 50 ms` loop latency** (our Level-2 `brain_ai_coprocessor.py` phase transition).
2. **Strict Outperformance:** Coupled Human + Co-Processor beats **both** unassisted top-0.01% humans **and** standalone AI on complementary out-of-distribution reasoning tasks.
3. **Measurable Cortical Co-Adaptation:** Neural recordings confirm cortical representations reorganize to communicate directly with the ANN's latent space rather than routing through conscious language.
