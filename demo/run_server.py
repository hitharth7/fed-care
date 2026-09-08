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

This same server also answers the dashboard's "Ask a Specialist" section
(GET /api/specialties, GET /api/specialty/<name>, POST /api/query) --
one unified server rather than a separate process, at the cost of the
specialist-query feature only being reachable while this live-training
server is up (it exits after --rounds rounds finish).
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

from fl.drift_controller import DriftAwareFedAvg
from fl.strategies import weighted_average
from router.specialist_router import answer_query, get_specialty_schema, list_specialties

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
STATE_PATH = DASHBOARD_DIR / "dashboard_state.json"

_state_lock = threading.Lock()
_state = {
    "status": "connecting",  # connecting -> training -> done
    "round": 0,
    "total_rounds": 0,
    "adaptive": False,  # whether the drift-aware controller is active this run
    "hospitals": {},  # id(str) -> {id, n_train, n_val, accuracy, auc, phase, controller:{...}}
    "global_accuracy_history": [],  # [{round, accuracy}]
    "events": [],  # rolling narration log, most recent last
}

_DEFAULT_CONTROLLER = {
    "mu": None, "target_epsilon": None, "cumulative_epsilon": None,
    "drift": None, "overfit_gap": None, "action": None,
}


def _init_state(num_hospitals: int, total_rounds: int, adaptive: bool) -> None:
    with _state_lock:
        _state["total_rounds"] = total_rounds
        _state["adaptive"] = adaptive
        _state["hospitals"] = {
            str(i): {
                "id": i, "n_train": None, "n_val": None, "accuracy": None, "auc": None, "phase": "waiting",
                "controller": dict(_DEFAULT_CONTROLLER),
            }
            for i in range(num_hospitals)
        }
    _write_state()


def _on_controller_decision(state) -> None:
    """Callback wired into DriftAwareFedAvg -- fires once per hospital per
    round, right after the controller updates that hospital's settings for
    next round. Mirrors the real controller state into the dashboard.

    target_epsilon and cumulative_epsilon are deliberately two different
    numbers, both shown: target_epsilon is what the controller wants THIS
    round to cost (it can go up or down round to round); cumulative_epsilon
    is the true composed privacy cost across every round so far, which
    only ever grows. Showing only the first would repeat exactly the kind
    of misleading-epsilon mistake Step 7's privacy/dp.py was built to
    avoid -- so it isn't shortcut here either.
    """
    cumulative_epsilon = state.cumulative_epsilon() if state.dp_history else None
    with _state_lock:
        h = _state["hospitals"].setdefault(str(state.hospital_id), {"id": state.hospital_id})
        h["controller"] = {
            "mu": state.mu,
            "target_epsilon": state.target_epsilon,
            "cumulative_epsilon": cumulative_epsilon,
            "drift": state.last_drift,
            "overfit_gap": state.last_overfit_gap,
            "action": state.last_action,
        }
    _add_event(f"Controller (Hospital {state.hospital_id}): {state.last_action}")
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
    # tmp_path.replace(STATE_PATH) is atomic on POSIX but NOT lock-safe on
    # Windows: if the dashboard's GET handler (a separate thread) has
    # STATE_PATH open for reading at this exact instant, Windows refuses the
    # replace with PermissionError (WinError 5) -- this actually happened,
    # repeatedly, crashing the training round unrecovered and looking
    # exactly like a hang. The reader holds the file open for microseconds,
    # so a short retry clears it almost every time; this is the standard
    # fix for this well-known Windows file-replace race, not a workaround
    # for a one-off fluke.
    for attempt in range(5):
        try:
            tmp_path.replace(STATE_PATH)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the dashboard's static files (default behavior) plus the
    "Ask a Specialist" JSON routes, wrapping router/specialist_router.py
    directly -- no query logic lives here, this is a thin HTTP adapter."""

    def log_message(self, fmt, *args):
        print(f"[dashboard] {self.address_string()} - {fmt % args}")

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/specialties":
            self._json({"specialties": list_specialties()})
            return
        if self.path.startswith("/api/specialty/"):
            name = self.path[len("/api/specialty/") :]
            schema = get_specialty_schema(name)
            if schema is None:
                self._json({"error": f"no trained specialty '{name}'"}, status=404)
                return
            self._json(schema)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/query":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                specialty = body["specialty"]
                hospital_id = int(body["hospital_id"])
                patient_features = body["patient_features"]
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._json({"status": "error", "message": f"bad request: {e}"}, status=400)
                return
            result = answer_query(specialty, hospital_id, patient_features)
            self._json(result)
            return
        self._json({"error": "not found"}, status=404)


def start_dashboard_server(port: int) -> None:
    DASHBOARD_DIR.mkdir(exist_ok=True)
    handler = functools.partial(DashboardHandler, directory=str(DASHBOARD_DIR))
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), handler)
    httpd.allow_reuse_address = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


def make_narrated_strategy(num_hospitals: int, adaptive: bool, base_mu: float, base_epsilon: float) -> fl.server.strategy.FedAvg:
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

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_hospitals,
        min_evaluate_clients=num_hospitals,
        min_available_clients=num_hospitals,
        fit_metrics_aggregation_fn=fit_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_aggregation,
    )
    if adaptive:
        return DriftAwareFedAvg(
            base_mu=base_mu, base_epsilon=base_epsilon, on_decision=_on_controller_decision, **common_kwargs
        )
    return fl.server.strategy.FedAvg(**common_kwargs)


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
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="disable the drift-aware controller and run plain FedAvg instead (for a before/after comparison)",
    )
    parser.add_argument("--base-mu", type=float, default=0.01, help="starting FedProx mu the controller adjusts from")
    parser.add_argument("--base-epsilon", type=float, default=8.0, help="starting DP target epsilon the controller adjusts from")
    args = parser.parse_args()
    adaptive = not args.no_adaptive

    port = args.address.split(":")[-1]
    print(f"Central server starting on {args.address} -- waiting for {args.num_hospitals} hospitals to connect...")
    print(f"Drift-aware controller: {'ON' if adaptive else 'OFF (plain FedAvg)'}")
    print(f"In {args.num_hospitals} other terminals (same machine or others on the LAN), run:")
    for i in range(args.num_hospitals):
        adaptive_flag = "" if adaptive else " --no-adaptive"
        print(f"  python demo/run_hospital.py --hospital-id {i} --server <this-machine-ip>:{port}{adaptive_flag}")
    print()

    _init_state(args.num_hospitals, args.rounds, adaptive)
    if not args.no_dashboard:
        start_dashboard_server(args.dashboard_port)
        print(f"Live dashboard: http://localhost:{args.dashboard_port}")
        print("(same page also serves \"Ask a Specialist\" -- available for as long as this server is up)\n")

    fl.server.start_server(
        server_address=args.address,
        # round_timeout: without this, a single unresponsive hospital (dead
        # connection, or just starved of CPU by something else on the same
        # machine) makes Flower's client manager wait up to its own default
        # of 24h with zero error printed -- this is what actually happened
        # once already in this project's history. 180s is generous for any
        # real round in this pipeline; a round that still isn't done by then
        # fails loudly instead of hanging silently.
        config=fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=180),
        strategy=make_narrated_strategy(args.num_hospitals, adaptive, args.base_mu, args.base_epsilon),
    )
    print("\nServer finished. The final global model is the product of every hospital's contribution -- the server never saw a single row of raw patient data from any of them.")

    if not args.no_dashboard:
        # Keep this process (and its dashboard/router HTTP server) running
        # indefinitely after training -- "Ask a Specialist" lives on this
        # same page/process, so it should stay usable after the 10 rounds
        # finish, not just for a few seconds. Ctrl+C to stop.
        print("Training done -- dashboard and \"Ask a Specialist\" stay up. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
