"""Run ONE hospital as a real network client connecting to demo/run_server.py.

Run demo/prepare_demo_data.py once first. Then run this once per hospital
(e.g. --hospital-id 0 through --hospital-id 4, in 5 separate terminals --
or on 5 separate machines on the same network) after the server is already
running.

This process loads ONLY its own hospital_{id}_{train,val}.npz file -- it
never reads another hospital's data file, and only ever sends model
weights (not data) over the network, via the exact same DiabetesFlowerClient
/ DiabetesMLP code used in experiments/run_fl_sweep.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl
import numpy as np
import torch

from fl.client import DiabetesFlowerClient
from models.mlp import DiabetesMLP

DATA_DIR = Path(__file__).resolve().parent / "hospital_data"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hospital_loaders(hospital_id: int, batch_size: int = 128):
    train_path = DATA_DIR / f"hospital_{hospital_id}_train.npz"
    val_path = DATA_DIR / f"hospital_{hospital_id}_val.npz"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found -- run `python demo/prepare_demo_data.py` once before starting hospitals."
        )
    train_npz, val_npz = np.load(train_path), np.load(val_path)

    def to_loader(npz, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(npz["X"]), torch.from_numpy(npz["y"]))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(train_npz, shuffle=True)
    val_loader = to_loader(val_npz, shuffle=False)
    return train_loader, val_loader, train_npz["X"].shape[1]


class NarratedHospitalClient(DiabetesFlowerClient):
    """Identical training logic to DiabetesFlowerClient -- these prints are
    purely so a live audience can watch each hospital's own training and
    weight-sharing happen round by round."""

    def __init__(self, hospital_id: int, n_local_train: int, n_local_val: int, *args, **kwargs):
        super().__init__(*args, client_id=hospital_id, **kwargs)
        self.hospital_id = hospital_id
        self.n_local_train = n_local_train
        self.n_local_val = n_local_val
        self.round_num = 0

    def fit(self, parameters, config):
        self.round_num += 1
        print(f"\n[Hospital {self.hospital_id}] Round {self.round_num}: received the current global model from the server.")
        if "mu" in config or "target_epsilon" in config:
            print(f"[Hospital {self.hospital_id}] Controller-assigned settings this round: mu={config.get('mu', 0):.4f}, target_epsilon={config.get('target_epsilon', 'inf')}")
        print(f"[Hospital {self.hospital_id}] Training locally on my {self.n_local_train} patient records -- this data never leaves this process.")
        result = super().fit(parameters, config)
        params, n, metrics = result
        if "drift" in metrics:
            print(f"[Hospital {self.hospital_id}] Local update drift from global model: {metrics['drift']:.1%}  |  train-vs-val overfit gap: {metrics['overfit_gap']:.1%}")
        print(f"[Hospital {self.hospital_id}] Local training done -- sending updated model WEIGHTS (not data) back to the server.")
        return result

    def evaluate(self, parameters, config):
        loss, n, metrics = super().evaluate(parameters, config)
        print(f"[Hospital {self.hospital_id}] Evaluated the current global model on my own {n} held-out records: accuracy={metrics['accuracy']:.4f}")
        return loss, n, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital-id", type=int, required=True)
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="don't report drift/overfit-gap metrics -- use only if the server was also started with --no-adaptive",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help=(
            "disable the pos_weight + per-client-threshold label-shift correction "
            "(see PROJECT_GUIDE.md Sec 8.2b) and reproduce the original raw collapse "
            "-- on by default since the correction is strictly a more honest evaluation, "
            "not a tuning choice"
        ),
    )
    args = parser.parse_args()

    device = get_device()
    print(f"[Hospital {args.hospital_id}] device={device}")
    print(f"[Hospital {args.hospital_id}] Loading MY OWN data file only: hospital_data/hospital_{args.hospital_id}_train.npz")

    train_loader, val_loader, input_dim = load_hospital_loaders(args.hospital_id)
    print(f"[Hospital {args.hospital_id}] I have {len(train_loader.dataset)} local training records and {len(val_loader.dataset)} local validation records.")
    print(f"[Hospital {args.hospital_id}] Connecting to central server at {args.server}...")

    model = DiabetesMLP(input_dim=input_dim)
    client = NarratedHospitalClient(
        args.hospital_id,
        len(train_loader.dataset),
        len(val_loader.dataset),
        model,
        train_loader,
        val_loader,
        device,
        local_epochs=args.local_epochs,
        lr=args.lr,
        track_controller_metrics=not args.no_adaptive,
        use_pos_weight=not args.no_correction,
        per_client_threshold=not args.no_correction,
    )

    fl.client.start_client(server_address=args.server, client=client.to_client())
    print(f"[Hospital {args.hospital_id}] Disconnected -- training complete.")


if __name__ == "__main__":
    main()
