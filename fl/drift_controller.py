"""Rule-based drift-aware controller (the "Selected New Idea" in
fedcare_master_project_guide.md section 6).

After each round, for every hospital, this looks at two signals computed
from that hospital's own fit() call and decides that hospital's settings
for the NEXT round:

- **Drift**: how far that hospital's locally-trained weights moved from the
  shared model it started the round with (relative L2 distance). A
  hospital whose patient population looks very different from the rest
  will tend to drift further.
- **Overfit gap**: that hospital's own train accuracy minus its own val
  accuracy, right after local training. A cheap, always-available proxy
  for membership-inference leakage risk -- running an actual attack every
  round (like Step 8 does, properly, after training) would be far too
  slow to do live every round. A model that fits its own training records
  much better than it generalizes to held-out ones is the classic
  precondition for the kind of memorization Step 8's attack measures
  directly.

Rules (deliberately simple thresholds, not learned -- the project's own
risk register calls for keeping this lightweight, not a black box):

- drift rises  -> raise mu (pull that hospital's local training back
  toward the shared model more strongly next round -- protects the group
  from one hospital's update dominating aggregation) and down-weight its
  contribution to *this* round's aggregation.
- drift falls  -> relax mu (let a well-aligned hospital personalize more
  freely).
- overfit gap rises -> tighten privacy for that hospital (lower target
  epsilon -> more DP noise next round).
- overfit gap falls -> relax privacy back toward the baseline epsilon.

This module has no dependency on Flower's simulation vs real-networked
transport -- DriftAwareFedAvg below works identically in both (used by
experiments/run_drift_controller.py in simulation mode and demo/run_server.py
over real sockets).
"""
from dataclasses import dataclass, field

import flwr as fl
from flwr.common import FitIns, FitRes
from opacus.accountants import RDPAccountant

from privacy.dp import DEFAULT_DELTA

DRIFT_HIGH = 0.20
DRIFT_LOW = 0.05
OVERFIT_HIGH = 0.10
OVERFIT_LOW = 0.02

MU_STEP_UP = 1.6
MU_STEP_DOWN = 0.75
MU_MIN, MU_MAX = 0.001, 2.0

EPSILON_STEP_DOWN = 0.6  # tighten (more noise) when leakage risk looks high
EPSILON_STEP_UP = 1.15  # relax back toward baseline when it doesn't
EPSILON_MIN = 0.5


@dataclass
class ClientControllerState:
    hospital_id: int
    mu: float
    target_epsilon: float
    base_epsilon: float
    last_drift: float | None = None
    last_overfit_gap: float | None = None
    weight_multiplier: float = 1.0
    last_action: str = "initializing"
    dp_history: list = field(default_factory=list)  # [(noise_multiplier, sample_rate, steps), ...]

    def cumulative_epsilon(self, delta: float = DEFAULT_DELTA) -> float:
        """True composed epsilon across every round so far -- NOT just the
        current target. Feeds every round's (possibly different) noise
        level into one accountant sequentially, the same discipline
        Step 7's privacy/dp.py uses for a fixed noise level, extended here
        to a noise level that can change round to round."""
        if not self.dp_history:
            return float("inf")
        accountant = RDPAccountant()
        for noise_multiplier, sample_rate, steps in self.dp_history:
            for _ in range(steps):
                accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        return accountant.get_epsilon(delta=delta)


def update_after_round(state: ClientControllerState, drift: float, overfit_gap: float) -> None:
    """Mutates `state` in place: this round's drift/overfit signal decides
    mu / target_epsilon / aggregation weight for the NEXT round."""
    state.last_drift = drift
    state.last_overfit_gap = overfit_gap
    actions = []

    if drift > DRIFT_HIGH:
        state.mu = min(MU_MAX, state.mu * MU_STEP_UP)
        state.weight_multiplier = 0.5
        actions.append(f"high drift ({drift:.1%}) -> raised stability pull (mu) to {state.mu:.3f}, down-weighted this round's contribution")
    elif drift < DRIFT_LOW:
        state.mu = max(MU_MIN, state.mu * MU_STEP_DOWN)
        state.weight_multiplier = 1.0
        actions.append(f"low drift ({drift:.1%}) -> relaxed stability pull (mu) to {state.mu:.3f}")
    else:
        state.weight_multiplier = 1.0

    if overfit_gap > OVERFIT_HIGH:
        state.target_epsilon = max(EPSILON_MIN, state.target_epsilon * EPSILON_STEP_DOWN)
        actions.append(f"high overfit gap ({overfit_gap:.1%}) -> tightened privacy, target epsilon now {state.target_epsilon:.2f}")
    elif overfit_gap < OVERFIT_LOW:
        state.target_epsilon = min(state.base_epsilon, state.target_epsilon * EPSILON_STEP_UP)
        actions.append(f"low overfit gap ({overfit_gap:.1%}) -> relaxed privacy toward baseline, target epsilon now {state.target_epsilon:.2f}")

    state.last_action = "; ".join(actions) if actions else "no change -- drift and overfit gap both within normal range"


class DriftAwareFedAvg(fl.server.strategy.FedAvg):
    """FedAvg, but each round: (1) injects the controller's current
    mu/target_epsilon into each client's fit() config, (2) reads back that
    client's drift/overfit_gap from the fit results to decide next round's
    settings, and (3) down-weights high-drift clients in this round's
    aggregation. Aggregation math itself is still plain FedAvg -- only the
    per-client fit() behavior and per-client weighting are adaptive.
    """

    def __init__(self, *args, base_mu: float = 0.01, base_epsilon: float = 8.0, on_decision=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_mu = base_mu
        self.base_epsilon = base_epsilon
        self.on_decision = on_decision  # optional callback(ClientControllerState) -- e.g. dashboard hook
        self.client_states: dict[str, ClientControllerState] = {}  # keyed by Flower's own client.cid

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []
        for client, fit_ins in instructions:
            state = self.client_states.get(client.cid)
            config = dict(fit_ins.config)
            config["mu"] = state.mu if state else self.base_mu
            config["target_epsilon"] = state.target_epsilon if state else self.base_epsilon
            new_instructions.append((client, FitIns(fit_ins.parameters, config)))
        return new_instructions

    def aggregate_fit(self, server_round, results, failures):
        adjusted_results = []
        for client, fit_res in results:
            m = fit_res.metrics
            hid = m.get("client_id")
            state = self.client_states.get(client.cid)
            if state is None:
                state = ClientControllerState(
                    hospital_id=hid if hid is not None else -1,
                    mu=self.base_mu,
                    target_epsilon=self.base_epsilon,
                    base_epsilon=self.base_epsilon,
                )
                self.client_states[client.cid] = state
            elif hid is not None:
                state.hospital_id = hid

            drift, overfit_gap = m.get("drift"), m.get("overfit_gap")
            if drift is not None and overfit_gap is not None:
                update_after_round(state, drift, overfit_gap)

            noise_multiplier, sample_rate, steps = m.get("noise_multiplier"), m.get("sample_rate"), m.get("steps")
            if noise_multiplier is not None and sample_rate is not None and steps is not None and noise_multiplier > 0:
                state.dp_history.append((noise_multiplier, sample_rate, int(steps)))

            if self.on_decision is not None:
                self.on_decision(state)

            damped_n = max(1, int(fit_res.num_examples * state.weight_multiplier))
            adjusted_results.append((
                client,
                FitRes(status=fit_res.status, parameters=fit_res.parameters, num_examples=damped_n, metrics=fit_res.metrics),
            ))

        return super().aggregate_fit(server_round, adjusted_results, failures)
