# Threebits Physical Hardware Bench Guide (< \$1,000, Zero Surgery)

This guide takes Threebits out of pure simulation and onto a **real physical \$5 ARM Cortex-M33 microcontroller (`RP2350`) + live wearable EEG / silent wrist sEMG + eye tracking** running our verified pure-integer C99 `int4` decoders (`66.35 kB` GRU + `3.27 kB` zero-backprop local co-adaptation kernel).

---

## 1. Complete Bill of Materials (< \$1,000)

| Component | Recommended Hardware | Approx. Cost | Role in Closed-Loop Stack |
| :--- | :--- | :---: | :--- |
| **Edge Implant-Class MCU** | **Raspberry Pi Pico 2 (`RP2350`, Dual Cortex-M33 @ 150 MHz, 520 kB SRAM)** | **\$5** | Runs `chip/firmware_rp2350/` (`66.35 kB` packed `int4` GRU + `3.27 kB` zero-backprop C99 local learner) at `< 1%` CPU duty cycle |
| **Power & Latency Bench** | **Nordic Power Profiler Kit II (PPK2)** or USB-C INA219 current meter | **\$99** | Measures true oscilloscope milliwatt power traces (`mW` active vs. sleep) per `20 ms` bin for Paper 2 hardware tables |
| **Non-Invasive EEG** | **OpenBCI Ganglion (4 ch)** / **Cyton (8 ch)** or **Muse S / Unicorn Hybrid Black** (LSL stream) | **\$250–\$550** | Streams live scalp EEG (`type='EEG'`) over Lab Streaming Layer (`--lsl`) for Experiment 3A overload gating |
| **Silent Sub-Verbal sEMG** | **MindRove 8-ch Wristband** or **OpenBCI 8-ch forearm surface electrodes** (LSL stream) | **\$200–\$350** | Unlocks sub-50 ms silent finger/chord readout (`--lsl-emg`) to push past speech's `39 bits/s` ceiling toward our Paper 3 `320 bits/s` phase transition |
| **Gaze / Head Tracking** | Any built-in laptop webcam (already supported in `https://threebits.si/gym/?tab=bandwidth`) | **\$0** | Adds `+2 bits/burst` parallel spatial channel (100% local in browser) |

---

## 2. Build & Flash Pure-Integer C99 Firmware on `RP2350`

### A. Verify Bare-Metal Host Executables & Virtual PTY Loopback Locally (Zero Hardware Needed)
```bash
# 1. Compile unified RP2350 firmware (FALCON H1 4-bit GRU + zero-backprop C99 local plasticity)
cc -O3 -std=c99 -Wall -Wextra -Werror -DHOST_TEST \
   chip/firmware_rp2350/main.c chip/int4_gru.c chip/c99_local_plasticity.c \
   -o /tmp/rp2350_host_emu
/tmp/rp2350_host_emu --bench 2000

# 2. Run automated C99, CRC-16 POSIX PTY loopback, EvalAI & Pilot API verification suite:
.venv/bin/python -m unittest discover -s tests -v
```

### B. Cross-Compile & Flash to a Physical \$5 Raspberry Pi Pico 2 (`RP2350`)
```bash
# Install Pico SDK & ARM toolchain (macOS Homebrew)
brew install cmake arm-none-eabi-gcc picotool
git clone -b master https://github.com/raspberrypi/pico-sdk.git ~/pico-sdk
git -C ~/pico-sdk submodule update --init

# Build UF2 binary from chip/firmware_rp2350/
export PICO_SDK_PATH=~/pico-sdk
cmake -S chip/firmware_rp2350 -B chip/firmware_rp2350/build -DPICO_BOARD=pico2
cmake --build chip/firmware_rp2350/build -j

# Hold BOOTSEL on Pico 2, plug USB, and flash:
picotool load -x chip/firmware_rp2350/build/threebits_int4_gru.uf2
```

### C. Measure Milliwatt Power Trace (PPK2)
1. Power the Pico 2 `VSYS` rail through the Nordic PPK2 in **Source Meter** mode (`3.3 V`).
2. Run 1,000 × `20 ms` bins (`50 Hz`, `5.88 MMAC/s`) with `WFI` sleep between bins.
3. Export CSV from nRF Connect Power Profiler → report active `mW`, sleep `mW`, and `µJ/bin` directly into Paper 2 Section 2.3!

---

## 3. Run Live Multi-Modal Closed-Loop Server (EEG + sEMG + RP2350 Serial)

```bash
# Install optional hardware bridge packages inside .venv
.venv/bin/pip install pylsl pyserial

# Run with live LSL EEG + live LSL silent wrist sEMG + physical RP2350 USB-CDC serial bridge:
# (Speaks CRC-16/CCITT-FALSE line protocol in chip/firmware_rp2350/rp2350_protocol.h;
#  GET /api/hardware_status & POST /api/mcu_step report live hardware microsecond latency)
.venv/bin/python level3_app/server.py --port 8765 --lsl --lsl-emg --serial /dev/cu.usbmodem14101
```
Then open `http://127.0.0.1:8765/?tab=bandwidth` or `http://127.0.0.1:8765/?tab=l3a` to run real-time closed-loop trials with live hardware latency & throughput logging.
