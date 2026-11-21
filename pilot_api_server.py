"""Zero-dependency HTTP API for Threebits Single-Blind Living-Human Pilot Study.

Collects anonymized 18-trial single-blind pilot sessions from https://threebits.si/gym/?tab=pilot
(or local http://127.0.0.1:8765/?tab=pilot), validates every trial row, saves clean CSVs matching
analyze_human_gym_pilot.py, and serves live aggregated cohort statistics + combined CSV export.

Endpoints:
  GET  /api/pilot/health   -> {"ok": true, "n_participants": ..., "n_trials": ...}
  GET  /api/pilot/summary  -> live anonymized cohort statistics (arm precision, MAE, bits/s, perm p)
  GET  /api/pilot/csv      -> combined anonymized cohort CSV (all stored sessions)
  POST /api/pilot/submit   -> validate + persist a completed pilot session, return session + cohort stats

Run standalone (production behind nginx on 127.0.0.1:8791):
    python3 pilot_api_server.py --host 127.0.0.1 --port 8791 --data-dir /var/lib/threebits-pilot
"""
import argparse
import csv
import io
import json
import math
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

FIELDS = [
    "participant_id", "trial_idx", "arm", "set_size", "rule_deg",
    "target_deg", "reported_deg", "err_deg", "rt_ms", "eff_bps", "ts",
]
VALID_ARMS = ("unassisted", "sham_code", "latent_coproc")
PID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_BODY_BYTES = 65536


def _perm_p_two_sided(a, b, n_perm=2000, seed=42):
    if not a or not b:
        return None
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pool = list(a) + list(b)
    na = len(a)
    nb = len(b)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        ma = sum(pool[:na]) / na
        mb = sum(pool[na:]) / nb
        if abs(ma - mb) >= obs - 1e-9:
            ge += 1
    return round((ge + 1) / (n_perm + 1), 4)


def _paired_sign_perm_p(diffs, n_perm=4000, seed=42):
    """Exact/Monte-Carlo paired sign-flip permutation p-value across N participants."""
    n = len(diffs)
    if n < 2:
        return None
    obs = abs(sum(diffs) / n)
    if n <= 12:
        total = 1 << n
        ge = 0
        for mask in range(total):
            s = 0.0
            for i in range(n):
                s += diffs[i] if ((mask >> i) & 1) else -diffs[i]
            if abs(s / n) >= obs - 1e-9:
                ge += 1
        return round(ge / total, 4)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        s = sum(d if rng.getrandbits(1) else -d for d in diffs)
        if abs(s / n) >= obs - 1e-9:
            ge += 1
    return round((ge + 1) / (n_perm + 1), 4)


def validate_pilot_submission(payload):
    """Validate incoming pilot payload strictly; returns (clean_pid, clean_rows) or raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not (6 <= len(raw_rows) <= 100):
        raise ValueError("Expected 6 to 100 completed trial rows")

    raw_pid = str(payload.get("participant_id") or raw_rows[0].get("participant_id") or "P-ANON").strip()
    clean_pid = re.sub(r"[^A-Za-z0-9_-]", "", raw_pid)[:32] or "P-ANON"
    if not PID_RE.match(clean_pid):
        raise ValueError("Invalid participant_id")

    clean_rows = []
    seen_arms = set()
    for idx, r in enumerate(raw_rows, start=1):
        if not isinstance(r, dict):
            raise ValueError(f"Row {idx} must be an object")
        arm = str(r.get("arm", "")).strip()
        if arm not in VALID_ARMS:
            raise ValueError(f"Row {idx}: invalid arm '{arm}'")
        seen_arms.add(arm)

        try:
            trial_idx = int(r.get("trial_idx", idx))
            set_size = int(r.get("set_size", 12))
            rule_deg = int(round(float(r.get("rule_deg", 0))))
            target_deg = float(r["target_deg"])
            reported_deg = float(r["reported_deg"])
            err_deg = float(r["err_deg"])
            rt_ms = int(round(float(r["rt_ms"])))
            eff_bps = float(r.get("eff_bps", 0.0))
            ts = float(r.get("ts", time.time()))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Row {idx}: malformed numeric field ({exc})")

        if not (1 <= trial_idx <= 200):
            raise ValueError(f"Row {idx}: trial_idx out of range")
        if not (2 <= set_size <= 32):
            raise ValueError(f"Row {idx}: set_size out of range")
        if rule_deg not in (0, 90, 180, 270):
            raise ValueError(f"Row {idx}: invalid rule_deg")
        if not (0.0 <= err_deg <= 180.5) or not math.isfinite(err_deg):
            raise ValueError(f"Row {idx}: err_deg out of range")
        if not (0 <= rt_ms <= 3600000):
            raise ValueError(f"Row {idx}: rt_ms out of range")
        if not (0.0 <= eff_bps <= 200.0) or not math.isfinite(eff_bps):
            raise ValueError(f"Row {idx}: eff_bps out of range")

        clean_rows.append({
            "participant_id": clean_pid,
            "trial_idx": trial_idx,
            "arm": arm,
            "set_size": set_size,
            "rule_deg": rule_deg,
            "target_deg": round(target_deg % 360.0, 2),
            "reported_deg": round(reported_deg % 360.0, 2),
            "err_deg": round(err_deg, 2),
            "rt_ms": rt_ms,
            "eff_bps": round(eff_bps, 2),
            "ts": round(ts, 2),
        })

    if len(seen_arms) < 2:
        raise ValueError("Submission must span at least 2 pilot arms")
    return clean_pid, clean_rows


def _summarize_rows(rows):
    arms_out = {}
    by_arm = {a: [] for a in VALID_ARMS}
    for r in rows:
        arm = r.get("arm")
        if arm in by_arm:
            by_arm[arm].append(r)

    for a in VALID_ARMS:
        sub = by_arm[a]
        if not sub:
            arms_out[a] = {"n": 0, "acc30": None, "mae_deg": None, "rt_ms": None, "eff_bps": None}
        else:
            errs = [float(r["err_deg"]) for r in sub]
            rts = [float(r["rt_ms"]) for r in sub]
            bps = [float(r["eff_bps"]) for r in sub]
            arms_out[a] = {
                "n": len(sub),
                "acc30": round(sum(1 for e in errs if e < 30.0) / len(errs) * 100.0, 1),
                "mae_deg": round(sum(errs) / len(errs), 1),
                "rt_ms": round(sum(rts) / len(rts)),
                "eff_bps": round(sum(bps) / len(bps), 2),
            }

    e_cop = [float(r["err_deg"]) for r in by_arm["latent_coproc"]]
    e_una = [float(r["err_deg"]) for r in by_arm["unassisted"]]
    e_sha = [float(r["err_deg"]) for r in by_arm["sham_code"]]
    tests = {}
    if e_cop and e_una:
        tests["coproc_vs_unassisted_mae_drop_deg"] = round(sum(e_una) / len(e_una) - sum(e_cop) / len(e_cop), 1)
        tests["coproc_vs_unassisted_perm_p"] = _perm_p_two_sided(e_cop, e_una)
    if e_cop and e_sha:
        tests["coproc_vs_sham_mae_drop_deg"] = round(sum(e_sha) / len(e_sha) - sum(e_cop) / len(e_cop), 1)
        tests["coproc_vs_sham_perm_p"] = _perm_p_two_sided(e_cop, e_sha)
    if len(e_cop) >= 2:
        half = len(e_cop) // 2
        tests["coadapt_first_half_mae_deg"] = round(sum(e_cop[:half]) / half, 1)
        tests["coadapt_second_half_mae_deg"] = round(sum(e_cop[half:]) / len(e_cop[half:]), 1)
    return arms_out, tests


def save_pilot_csv(data_dir: Path, pid: str, rows: list):
    data_dir.mkdir(parents=True, exist_ok=True)
    # Use clean filename; append short timestamp suffix if same participant submits a second session
    base = data_dir / f"threebits_pilot_{pid}.csv"
    out_path = base if not base.exists() else data_dir / f"threebits_pilot_{pid}_{int(time.time())}.csv"
    tmp_path = out_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    tmp_path.replace(out_path)
    return out_path


def load_all_pilot_rows(data_dir: Path):
    all_rows = []
    sessions = []
    if not data_dir.exists():
        return all_rows, sessions
    for p in sorted(data_dir.glob("threebits_pilot_*.csv")):
        try:
            with p.open("r", newline="") as f:
                s_rows = []
                for row in csv.DictReader(f):
                    if row.get("arm") not in VALID_ARMS:
                        continue
                    s_rows.append({
                        "participant_id": row.get("participant_id", p.stem),
                        "trial_idx": int(float(row.get("trial_idx", 0))),
                        "arm": row["arm"],
                        "set_size": int(float(row.get("set_size", 12))),
                        "rule_deg": int(float(row.get("rule_deg", 0))),
                        "target_deg": float(row.get("target_deg", 0)),
                        "reported_deg": float(row.get("reported_deg", 0)),
                        "err_deg": float(row["err_deg"]),
                        "rt_ms": int(float(row.get("rt_ms", 0))),
                        "eff_bps": float(row.get("eff_bps", 0)),
                        "ts": float(row.get("ts", 0)),
                    })
            if s_rows:
                all_rows.extend(s_rows)
                s_arms, _ = _summarize_rows(s_rows)
                sessions.append({
                    "session_file": p.name,
                    "participant_id": s_rows[0]["participant_id"],
                    "n_trials": len(s_rows),
                    "arms": s_arms,
                })
        except Exception:
            continue
    return all_rows, sessions


def compute_pilot_cohort_summary(data_dir: Path):
    all_rows, sessions = load_all_pilot_rows(data_dir)
    arms_pooled, pooled_tests = _summarize_rows(all_rows)
    n_sessions = len(sessions)

    # Participant-level paired summaries when N >= 2
    paired = {}
    if n_sessions >= 2:
        d_una, d_sha = [], []
        for s in sessions:
            a = s["arms"]
            if a["latent_coproc"]["mae_deg"] is not None and a["unassisted"]["mae_deg"] is not None:
                d_una.append(a["unassisted"]["mae_deg"] - a["latent_coproc"]["mae_deg"])
            if a["latent_coproc"]["mae_deg"] is not None and a["sham_code"]["mae_deg"] is not None:
                d_sha.append(a["sham_code"]["mae_deg"] - a["latent_coproc"]["mae_deg"])
        if len(d_una) >= 2:
            paired["n_paired_unassisted"] = len(d_una)
            paired["mean_mae_drop_vs_unassisted_deg"] = round(sum(d_una) / len(d_una), 1)
            paired["paired_perm_p_vs_unassisted"] = _paired_sign_perm_p(d_una)
        if len(d_sha) >= 2:
            paired["n_paired_sham"] = len(d_sha)
            paired["mean_mae_drop_vs_sham_deg"] = round(sum(d_sha) / len(d_sha), 1)
            paired["paired_perm_p_vs_sham"] = _paired_sign_perm_p(d_sha)

    return {
        "ok": True,
        "n_sessions": n_sessions,
        "n_trials": len(all_rows),
        "arms_pooled": arms_pooled,
        "pooled_tests": pooled_tests,
        "paired_cohort": paired,
        "sessions": [
            {
                "participant_id": s["participant_id"],
                "n_trials": s["n_trials"],
                "unassisted_acc30": s["arms"]["unassisted"]["acc30"],
                "sham_acc30": s["arms"]["sham_code"]["acc30"],
                "coproc_acc30": s["arms"]["latent_coproc"]["acc30"],
                "unassisted_mae": s["arms"]["unassisted"]["mae_deg"],
                "sham_mae": s["arms"]["sham_code"]["mae_deg"],
                "coproc_mae": s["arms"]["latent_coproc"]["mae_deg"],
            }
            for s in sessions[-50:]
        ],
    }


def export_combined_pilot_csv(data_dir: Path) -> bytes:
    all_rows, _ = load_all_pilot_rows(data_dir)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    for r in all_rows:
        w.writerow({k: r[k] for k in FIELDS})
    return buf.getvalue().encode("utf-8")


def make_pilot_handler(data_dir: Path):
    rate_lock = threading.Lock()
    write_hits = {}  # ip -> [timestamps]

    def allow_write(ip: str) -> bool:
        now = time.time()
        with rate_lock:
            hits = [t for t in write_hits.get(ip, []) if now - t < 3600.0]
            if len(hits) >= 30:
                write_hits[ip] = hits
                return False
            hits.append(now)
            write_hits[ip] = hits
            return True

    class PilotHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _client_ip(self):
            xff = self.headers.get("X-Forwarded-For", "")
            return xff.split(",")[0].strip() if xff else (self.client_address[0] if self.client_address else "local")

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path.rstrip("/")
            if route == "/api/pilot/health":
                summ = compute_pilot_cohort_summary(data_dir)
                self._send_json({"ok": True, "n_sessions": summ["n_sessions"], "n_trials": summ["n_trials"]})
            elif route == "/api/pilot/summary":
                self._send_json(compute_pilot_cohort_summary(data_dir))
            elif route == "/api/pilot/csv":
                data = export_combined_pilot_csv(data_dir)
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=threebits_pilot_cohort.csv")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"ok": False, "error": "not_found"}, code=404)

        def do_POST(self):
            route = urlsplit(self.path).path.rstrip("/")
            if route != "/api/pilot/submit":
                self._send_json({"ok": False, "error": "not_found"}, code=404)
                return
            if not allow_write(self._client_ip()):
                self._send_json({"ok": False, "error": "rate_limited"}, code=429)
                return
            try:
                ln = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._send_json({"ok": False, "error": "bad_content_length"}, code=400)
                return
            if ln <= 0 or ln > MAX_BODY_BYTES:
                self._send_json({"ok": False, "error": "payload_size_invalid"}, code=413)
                return
            try:
                payload = json.loads(self.rfile.read(ln).decode("utf-8"))
                pid, clean_rows = validate_pilot_submission(payload)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, code=400)
                return

            saved_path = save_pilot_csv(data_dir, pid, clean_rows)
            s_arms, s_tests = _summarize_rows(clean_rows)
            cohort = compute_pilot_cohort_summary(data_dir)
            self._send_json({
                "ok": True,
                "saved_file": saved_path.name,
                "participant_id": pid,
                "n_trials": len(clean_rows),
                "session_arms": s_arms,
                "session_tests": s_tests,
                "cohort": cohort,
            }, code=201)

    return PilotHandler


def main():
    p = argparse.ArgumentParser(description="Threebits Living-Human Pilot Study HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--data-dir", default="/var/lib/threebits-pilot")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), make_pilot_handler(data_dir))
    print(f"Threebits Pilot API listening on http://{args.host}:{args.port} (data={data_dir})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
