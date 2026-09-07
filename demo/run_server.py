"""Real networked Flower server for the live panel demo.

Run this FIRST, in its own terminal, then run demo/run_hospital.py once per
hospital in separate terminals. Uses real gRPC sockets -- this is not a
simulation, hospitals are separate OS processes (optionally on separate
machines on the same network) that never share raw data, only model
weights.

Same FedAvg aggregation as experiments/run_fl_sweep.py -- only the
transport differs (real network sockets here, in-process simulation
there), so this demo is faithful to the actual research code, not a
separate toy reimplementation.

Also starts a small local HTTP server (default port 8090) serving
demo/dashboard/index.html plus a live-updating dashboard_state.json --
open http://localhost:8090 in a browser to watch the run visually instead
of reading terminal logs. The dashboard is purely a read-only view of the
real state below; it never influences training.
"""
import argparse
import functools
import http.server
import json
import socketserver
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl

from fl.strategies import weighted_average

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
STATE_PATH = DASHBOARD_DIR / "dashboard_state.json"

_state_lock = threading.Lock()
_state = {
    "status": "connecting",  # connecting -> training -> done
    "round": 0,
    "total_rounds": 0,
    "hospitals": {},  # id(str) -> {id, n_train, n_val, accuracy, auc, phase}
    "global_accuracy_history": [],  # [{round, accuracy}]
    "events": [],  # rolling narration log, most recent last
}


def _init_state(num_hospitals: int, total_rounds: int) -> None:
    with _state_lock:
        _state["total_rounds"] = total_rounds
        _state["hospitals"] = {
            str(i): {"id": i, "n_train": None, "n_val": None, "accuracy": None, "auc": None, "phase": "waiting"}
            for i in range(num_hospitals)
        }
    _write_state()


def _add_event(msg: str) -> None:
    with _state_lock:
        _state["events"].append(msg)
        _state["events"] = _state["events"][-40:]


def _write_state() -> None:
    DASHBOARD_DIR.mkdir(exist_ok=True)
    with _state_lock:
        payload = dict(_state)
        payload["hospitals"] = sorted(_state["hospitals"].values(), key=lambda h: h["id"])
    tmp_path = STATE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(STATE_PATH)  # atomic, so the dashboard never reads a half-written file


def start_dashboard_server(port: int) -> None:
    DASHBOARD_DIR.mkdir(exist_ok=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DASHBOARD_DIR))
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), handler)
    httpd.allow_reuse_address = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


def make_narrated_strategy(num_hospitals: int) -> fl.server.strategy.FedAvg:
    def fit_aggregation(metrics):
        with _state_lock:
            _state["status"] = "training"
        for n, m in metrics:
            cid = m.get("client_id")
            if cid is None:
                continue
            with _state_lock:
                h = _state["hospitals"].setdefault(str(cid), {"id": cid})
                h["n_train"] = n
                h["phase"] = "sent"
            _add_event(f"Hospital {cid} finished local training on {n} records and sent updated weights to the server.")
        _write_state()
        return {}

    def evaluate_aggregation(metrics):
        agg = weighted_average(metrics)

        print(f"\n{'=' * 70}")
        print("CENTRAL SERVER -- aggregating this round's hospital updates")
        for n, m in sorted(metrics, key=lambda x: x[1].get("client_id", -1)):
            cid = m.get("client_id", "?")
            print(f"  Hospital {cid}: local accuracy={m['accuracy']:.4f}  (n={n} patient records -- never left that hospital)")
        print(f"  -> New global model accuracy (weighted average across hospitals): {agg['accuracy']:.4f}")
        print(f"{'=' * 70}")

        with _state_lock:
            _state["round"] += 1
            round_num = _state["round"]
            total_rounds = _state["total_rounds"]
        for n, m in metrics:
            cid = m.get("client_id")
            if cid is None:
                continue
            with _state_lock:
                h = _state["hospitals"].setdefault(str(cid), {"id": cid})
                h.update({"n_val": n, "accuracy": m["accuracy"], "auc": m.get("auc"), "phase": "evaluated"})
        with _state_lock:
            _state["global_accuracy_history"].append({"round": round_num, "accuracy": agg["accuracy"]})
            if round_num >= total_rounds:
                _state["status"] = "done"
        _add_event(f"Round {round_num} complete -- new global model accuracy: {agg['accuracy'] * 100:.2f}%")
        _write_state()
        return agg

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_hospitals,
        min_evaluate_clients=num_hospitals,
        min_available_clients=num_hospitals,
        fit_metrics_aggregation_fn=fit_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_aggregation,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        default="0.0.0.0:8080",
        help="listen address:port. Use 0.0.0.0 to accept hospital connections from other machines on the LAN.",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--num-hospitals", type=int, default=5)
    parser.add_argument("--dashboard-port", type=int, default=8090)
    parser.add_argument("--no-dashboard", action="store_true", help="disable the local live dashboard web server")
    args = parser.parse_args()

    port = args.address.split(":")[-1]
    print(f"Central server starting on {args.address} -- waiting for {args.num_hospitals} hospitals to connect...")
    print(f"In {args.num_hospitals} other terminals (same machine or others on the LAN), run:")
    for i in range(args.num_hospitals):
        print(f"  python demo/run_hospital.py --hospital-id {i} --server <this-machine-ip>:{port}")
    print()

    _init_state(args.num_hospitals, args.rounds)
    if not args.no_dashboard:
        start_dashboard_server(args.dashboard_port)
        print(f"Live dashboard: http://localhost:{args.dashboard_port}\n")

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=make_narrated_strategy(args.num_hospitals),
    )
    print("\nServer finished. The final global model is the product of every hospital's contribution -- the server never saw a single row of raw patient data from any of them.")

    if not args.no_dashboard:
        # The dashboard polls every 800ms; without this pause the process
        # (and its dashboard web server) can exit before the browser's next
        # poll, so the very last round never gets displayed. Keep the
        # dashboard reachable for a few seconds so the final "done" state
        # is guaranteed to be seen.
        print("Keeping the dashboard up for 5 more seconds so the final result is visible...")
        time.sleep(5)


if __name__ == "__main__":
    main()
