# From Biological Working-Memory Limits to Co-Adaptive Brain–AI Co-Processing: A Quantitative Proof Ladder

**Muhammad Farhan Tahir**  
Threebits Research · https://threebits.si · https://github.com/MuhammadFarhantahir/threebits-research  
Preprint v1.1 · 21 November 2026

---

## Abstract

Can artificial neural networks raise human intelligence beyond biological limits? Human cognition is constrained by three physical bottlenecks: working-memory capacity (~4 active chunks; Cowan, 2001), noisy neural encoding, and conscious language I/O (~39 bits/s across human languages; Coupé et al., 2019). Here we separate **biological state-steering** (closed-loop brain stimulation alone) from **co-adaptive Brain–AI co-processing ("Prosthetic Cortex")** and evaluate both across a quantitative four-level proof ladder anchored in healthy-human data.

First (**Level 1, healthy adults, zero surgery**, OpenNeuro `ds008037`, CC0), we extract a 10-task general cognitive ability (`g`-factor) across **123 healthy adults** (`90/90` positive pairwise correlations, PC1 explaining `37.2%` of variance) and measure the empirical human working-memory cliff across **33,299 continuous recall trials (`n = 117` adults)**: recall precision within $30^\circ$ drops steeply from **`88.6%` (`15.9°` error, `5.0%` failure) at $K = 2$** to **`61.5%` (`38.6°`) at $K = 4$** and **`43.8%` (`57.3°`, `45.0%` failure) at $K = 6$** (worse at $K = 6$ in `116/117` adults). Using strictly pre-response scalp EEG (`[0.00, 1.50 s]` before response screen onset) across **15 healthy adults (`3,744` trials)**, a compact causal decoder detects cognitive overload ($K = 6$ vs $2$) at **`0.690 ± 0.033` ROC AUC** ($p = 7.22 \times 10^{-5}$, `14/15` above chance, up to `0.916`; null `0.500`) and within-load pre-response failure weakly at **`0.529` GRU AUC** ($p = 0.040$). Critically, an independent unseen replication (`n = 24` new adults beyond `n = 22` discovery) shows that scalar single-pulse resting TMS-EEG amplitude does **not** reliably predict trait `g` ($\rho = +0.581 \to -0.104$; pooled `n = 46`: $\rho = +0.228, p = 0.128$), whereas zero-shot invasive-CCEP → non-invasive TMS-EEG spatial propagation validation scales cleanly across all **`n = 46` healthy adults** (`+0.247` individual mean, positive in **`44/46`** vs resting-fMRI-tuned `+0.131`, rewired `+0.028`, wrong-site `−0.136`).

Second (**Level 2, *in silico* scaling law**), we couple a noise- and capacity-limited biological cortex calibrated to Cowan's wall to a digital KV-cache attention co-processor via a **4-bit quantized read** and **norm-bounded safe write** interface — with **zero direct AI access to raw task stimuli**. Across $K = 2 \dots 16$ items (3 seeds): (1) **optimal memoryless state-steering** lifts biological accuracy by **`+19% to +23%`** at $K = 4\text{–}8$ (`41.6% → 51.1%`, `30.6% → 37.0%`, `29.9% → 35.7%`), then hits the biological wall at $K = 12\text{–}16$ (`21.2% / 19.7%`); (2) **conscious-bandwidth tool use (`40 bits/s`, 200 ms)** starves (`18.1% / 17.0%`), matching **sham-interface controls** (`15.5% / 17.1%`); (3) **co-adaptive 4-bit Brain–AI co-processing (`50 ms`, `537 bits/s` empirical MI)** achieves **`98.7%` at $K = 12$ and `97.4%` at $K = 16$**; (4) sweeping a $7 \times 5$ **Bandwidth × Latency phase diagram** at $K = 12$ reveals a sharp phase transition from `18–33%` ($\le 160\text{ bits/s}$ cap) to **`83.7–91.5%`** at **`320 bits/s` cap (`~265 bits/s` MI, ~7× human speech)** for **`≤ 100 ms` latency**, collapsing to `37.8% / 32.9%` at `200–250 ms`; and (5) on a complementary task combining $K = 12$ items with a private embodied human rule invisible to standalone AI, the co-adapted Prosthetic Cortex passes **Strict Outperformance** (**`87.0%` vs `23.2%` AI alone, `14.8%` Human alone, `17.1%` 40-bps Screen AI, `16.5%` Sham; `+63.7 pp`**) while increasing across-seed cortical representational coherence `10.8×` (Linear CKA `0.059 → 0.635`).

---

## 1. Methods

### 1.1 Healthy-Human Behavioral Battery & Pre-Response EEG Decoding (`ds008037`)
From OpenNeuro `ds008037` (Biabani & Rogasch, CC0), behavioral logs across 123 healthy adults were aggregated over 10 cognitive tasks (`wm_prec2/4/6`, `nback`, `bds`, `sym`, `opn`, `sbs`, `clt`, `ort`), z-scored, and factored via PCA to extract `g` (PC1). Working-memory continuous color recall trials (`K ∈ {2, 4, 6}`, 33,299 trials across 117 complete adults) were aligned 300/300 between `events.tsv` and `beh.tsv`. For 15 adults (`3,744` trials, 62 scalp channels at 1,000 Hz, 50 Hz notch + 1–45 Hz bandpass), features were extracted strictly within `[0.00, 1.50 s]` post-array onset (preceding `response_screen_onset ≥ 1.517 s` on every trial) into 6 × 250 ms bins × 5 spectral bands plus binned ERPs. Out-of-fold decoding used block-level `GroupKFold` cross-validation (L2 logistic regression and causal GRU) verified against 20-permutation block-shuffled nulls.

### 1.2 Co-Adaptive Brain–AI Co-Processor Dynamical Benchmark
A biological recurrent cortex ($H_{\text{bio}} = 24$, Gaussian state noise $\sigma = 0.065$) receives sequential spatial-color pairs $(x_t, c_t)$ across $K \in \{2, 4, 6, 8, 12, 16\}$ items, followed by delay steps and a probe $x_q$. Critically, **the AI co-processor has zero direct access to raw task stimuli**: at each loop step it receives only a straight-through-estimated $b$-bit quantized projection $q_t = Q_b(W_{\text{read}} h_t^{\text{bio}})$ arriving after latency $\tau$, stores latent tokens in an external KV-cache attention memory, and returns a norm-bounded safe write vector $u_t = \text{clip}(W_{\text{write}} a_t, \|u_t\|_2 \le u_{\max})$ to biological cortex. Effective neural read mutual information (bits/s) is estimated directly from quantized channel entropy minus conditional noise entropy divided by step period. Cortical co-adaptation across independent training seeds is quantified using Linear Centered Kernel Alignment (CKA; Kornblith et al., 2019).

---

## 2. Level-1 Healthy-Human Empirical Anchor (`ds008037`, CC0)

### 2.1 General Cognitive Ability (`g`) and Cowan's Working-Memory Cliff (`n = 123` Adults)
Across all 10 behavioral tasks in OpenNeuro `ds008037`, every single pairwise correlation is positive (`90/90`, mean $r = +0.294$), with PC1 (`g`) capturing `37.2%` of total cognitive variance and continuous working-memory precision at load 4 loading highest (`+0.39`). Across `117` complete adults (`33,299` trials):

| Working-Memory Load ($K$) | Mean Angular Error | Recall Precision ($< 30^\circ$) | High-Error Failure ($> 45^\circ$) |
| :---: | :---: | :---: | :---: |
| **$K = 2$** | $15.9^\circ$ | **88.6%** | 5.0% |
| **$K = 4$** | $38.6^\circ$ | **61.5%** | 27.5% |
| **$K = 6$** | $57.3^\circ$ | **43.8%** | **45.0%** (`9×` load 2) |

### 2.2 Non-Invasive Pre-Response Working-Memory EEG Decoding (`n = 15` Adults, `3,744` Trials)
- **Overload detection ($K = 6$ vs $2$):** L2 LogReg **`0.690 ± 0.033` ROC AUC** ($p = 7.22 \times 10^{-5}$, `14/15` above chance; peak `sub-020 = 0.916`, `sub-010 = 0.866`, `sub-008 = 0.851`, `sub-007 = 0.838`); causal GRU **`0.608`** ($p = 0.0042$, `13/15`); 20-perm null `0.500`.
- **Within-load imminent failure prediction (zero load confounding):** Causal GRU **`0.529`** ($p = 0.040$, `9/15`, top `sub-021 = 0.628` / LogReg `0.654`); null `0.497`. Comparing scalp EEG (`0.529`) against intracranial iEEG (`0.603`, $p = 0.0001$, `14/15` in RAM `ds005557`/`ds005558`) quantifies why non-invasive EEG supports macro-load gating (Level 3A) while high-bandwidth Level-4 prosthetic cognition requires closer-to-cortex sensing or multi-modal non-invasive fusion.

### 2.3 Honest Out-of-Sample Replication of TMS-EEG Markers (`n = 46` Clean Healthy Adults)
- **Null trait-IQ replication (reported transparently):** Although sham-subtracted parietal-to-frontoparietal TMS-EEG amplitude correlated with `g` in our initial `n = 22` discovery cohort ($\rho = +0.581, p = 0.0046$; sham $\rho = +0.014$), an immediate independent replication on **`n = 24` completely unseen healthy adults** yielded $\rho = -0.104$ ($p = 0.628$; pooled `n = 46`: $\rho = +0.228, p = 0.128$). Static single-pulse resting TMS amplitude does not reliably predict trait IQ across individuals.
- **Confirmed spatial-propagation transfer (`n = 46` healthy adults):** Conversely, zero-shot prediction of sham-subtracted healthy-human TMS-EEG scalp topography from our invasive-CCEP Hopf simulator replicates across **`44/46` individual healthy adults** (mean partial $\rho \mid \text{coil distance} = +0.247$), nearly doubling resting-fMRI-tuned Hopf (`+0.131`, `34/46`) and far exceeding rewired (`+0.028`, `26/46`) and wrong-site (`−0.136`, `9/46`) controls.

---

## 3. Level-2 Co-Adaptive Brain–AI Scaling Law

### 3.1 Capacity Curve ($K = 2 \dots 16$, 3 Seeds Each, Zero Direct AI Stimulus Access)

| Regime | $K = 2$ | $K = 4$ | $K = 6$ | $K = 8$ | $K = 12$ | $K = 16$ | Empirical MI @ $K=12$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Empirical 117 Healthy Humans (`ds008037`)* | *88.6%* | *61.5%* | *43.8%* | — | — | — | — |
| Biological Cortex Alone | 98.3% | 41.6% | 30.6% | 29.9% | 25.5% | 24.3% | 0 bits/s |
| Optimal State-Steering Only (no ext. mem.) | 78.5% | **51.1%** | **37.0%** | **35.7%** | 21.2% | 19.7% | 0 bits/s |
| 40-bps Screen AI Tool (`200 ms`) | 32.1% | 27.0% | 26.0% | 19.3% | 18.1% | 17.0% | 33 bits/s |
| Sham-Interface Control (permuted read) | 24.6% | 17.0% | 17.4% | 16.6% | 15.5% | 17.1% | 148 bits/s |
| **4-Bit Brain–AI Co-Processor (`50 ms`)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **98.7%** | **97.4%** | **537 bits/s** |

### 3.2 Bandwidth × Latency Phase Transition & Level-4 Strict Outperformance
Crossing from biological/tool performance (`18–33%`) into superhuman working-memory capacity (`84–100%`) at $K = 12$ requires crossing **both** a minimum neural interface bandwidth (**`> 160 → 320 bits/s` cap**, `~265 bits/s` empirical MI — roughly **7× human conscious speech**) **and** a maximum closed-loop latency (**`≤ 100 ms`** at `320 bps`, `≤ 200 ms` at `640–960 bps`, collapsing to `30–33%` at `250 ms`; Figure 1).

On a complementary task combining $K = 12$ items with a 4-way private embodied human goal visible **only** inside cortex:
- **Human Alone:** `14.8%` (`90.5°` MAE) · **State-Steering:** `16.5%` · **40-bps Screen AI:** `17.1%` · **Sham Loop:** `16.5%` · **Standalone AI (public stream only):** `23.2%` (`68.7°` MAE).
- **Co-Adaptive Prosthetic Cortex (`4-bit read + bounded write, 50 ms, 484 bits/s` MI):** **`87.0%` (`18.2°` MAE; seeds `[90.4%, 96.5%, 74.0%]`) — passing Strict Outperformance by `+63.7 pp`**.
- **Cortical Reorganization (Linear CKA):** Unassisted cortex across seeds is incoherent under $K = 12$ overload (`CKA = 0.059`), whereas co-adapted cortex converges across independent seeds onto a reproducible interface representation (`prost-vs-prost CKA = 0.635`, `10.8×` higher; `alone-vs-prost = 0.127`).

![Figure 1: Healthy-human working-memory cliff (n=117) and pre-response EEG overload decoding (n=15), alongside the Level-2 Brain-AI capacity curve, 2D Bandwidth x Latency phase transition (320 bps at <=100 ms), and Level-4 Strict Outperformance (+63.7 pp).](figures/fig5_prosthetic_cortex.png)

---

## 4. Limitations & Next Empirical Gate (Level-3 Human Trials)

- **In-silico vs living human co-adaptation:** Level 2 demonstrates the mathematical scaling law and phase transition in a recurrent dynamical neural model; proving genuine cortical co-adaptation in living humans requires longitudinal human experiments.
- **Interactive browser lab & Level-3A/3B protocols:** An open non-verbal sensory-substitution Neural Gym (`K = 4–16`, monochrome 4-bit polar harmonic glyph + stereo Web Audio chord + private human rule) and 4-arm healthy-human EEG trial replay are live at https://threebits.si/gym/. Full pre-registered protocols for **Experiment 3A** (4-arm randomized EEG Brain–AI Co-Processor trial, zero surgery) and **Experiment 3B** (double-blind university TMS-EEG closed-loop stimulation pilot) constitute the immediate empirical next gate.

---

## Ethics Statement, Data & Code Availability, and Competing Interests

- **Ethics Statement:** Secondary computational analysis of de-identified public human behavioral and EEG/TMS-EEG data (`OpenNeuro ds008037`, CC0) combined with *in silico* dynamical simulations. No new human subjects data were collected for this preprint.
- **Data & Code Availability:** Source human data are publicly available on OpenNeuro (`ds008037`). Reproducible scripts, interactive Neural Gym (including continuous 8-finger sub-verbal burst Shannon throughput), and Level-3 trial protocols are publicly available at https://github.com/MuhammadFarhantahir/threebits-research, https://threebits.si/papers/, and https://threebits.si/gym/.
- **Competing Interests:** Muhammad Farhan Tahir is the founder of Threebits.

---

## References

1. Biabani, M. & Rogasch, N. C. (2025). The neural dynamics of working memory and transcranial magnetic stimulation. *OpenNeuro* `ds008037` (CC0). https://openneuro.org/datasets/ds008037
2. Coupé, C., Oh, Y. M., Dediu, D., & Pellegrino, F. (2019). Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche. *Science Advances*, 5(9):eaaw2594. https://doi.org/10.1126/sciadv.aaw2594
3. Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences*, 24(1):87–114. https://doi.org/10.1017/S0140525X01003922
4. Ezzyat, Y. et al. (2018). Closed-loop stimulation of temporal cortex rescues functional networks and improves memory. *Nature Communications*, 9:365. https://doi.org/10.1038/s41467-017-02753-0
5. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML 2019*. https://arxiv.org/abs/1905.00414
