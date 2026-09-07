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
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl

from fl.strategies import weighted_average


def make_narrated_strategy(num_hospitals: int) -> fl.server.strategy.FedAvg:
    def narrated_aggregation(metrics):
        agg = weighted_average(metrics)
        print(f"\n{'=' * 70}")
        print("CENTRAL SERVER -- aggregating this round's hospital updates")
        for n, m in sorted(metrics, key=lambda x: x[1].get("client_id", -1)):
            cid = m.get("client_id", "?")
            print(f"  Hospital {cid}: local accuracy={m['accuracy']:.4f}  (n={n} patient records -- never left that hospital)")
        print(f"  -> New global model accuracy (weighted average across hospitals): {agg['accuracy']:.4f}")
        print(f"{'=' * 70}")
        return agg

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_hospitals,
        min_evaluate_clients=num_hospitals,
        min_available_clients=num_hospitals,
        evaluate_metrics_aggregation_fn=narrated_aggregation,
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
    args = parser.parse_args()

    port = args.address.split(":")[-1]
    print(f"Central server starting on {args.address} -- waiting for {args.num_hospitals} hospitals to connect...")
    print(f"In {args.num_hospitals} other terminals (same machine or others on the LAN), run:")
    for i in range(args.num_hospitals):
        print(f"  python demo/run_hospital.py --hospital-id {i} --server <this-machine-ip>:{port}")
    print()

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=make_narrated_strategy(args.num_hospitals),
    )
    print("\nServer finished. The final global model is the product of every hospital's contribution -- the server never saw a single row of raw patient data from any of them.")


if __name__ == "__main__":
    main()
