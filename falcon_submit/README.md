# FALCON H1 EvalAI Submission — Threebits 4-Bit Pure-Integer GRU Decoder

Pure-integer C99 packed 4-bit causal GRU (`58.8 kB` weight ROM, `int4` weights × `int8` activations & hidden state) with `1%` L2-anchored per-day input/readout calibration adapters.

## 1. Export model bundle locally (once)
```bash
.venv/bin/python falcon_submit/decoder.py --mode train \
  --data-dir data/falcon_h1 \
  --out falcon_submit/h1_int4_bundle.pkl
```

## 2. Build Docker container (requires Docker Desktop)
```bash
docker build -t threebits-h1-int4 -f falcon_submit/Dockerfile .
```

## 3. Submit to EvalAI (Challenge 2319 — FALCON Benchmark)
1. Create a team on https://eval.ai/web/challenges/challenge-page/2319/overview
2. Install EvalAI CLI (`pip install evalai`) and set your token (`evalai set_token <TOKEN>`).
3. Push to `minival` first, then `test`:
```bash
evalai push threebits-h1-int4:latest --phase few-shot-minival-2319 --private
evalai push threebits-h1-int4:latest --phase few-shot-test-2319 --private
```
