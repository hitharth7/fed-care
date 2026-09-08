"""FedAvg strategy config. FedProx's mu-controlled strategy is added in Step 6."""
import flwr as fl
from flwr.common import Metrics


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    out = {
        "accuracy": sum(n * m["accuracy"] for n, m in metrics) / total,
        "f1": sum(n * m["f1"] for n, m in metrics) / total,
        "auc": sum(n * m["auc"] for n, m in metrics) / total,
    }
    if "balanced_accuracy" in metrics[0][1]:
        out["balanced_accuracy"] = sum(n * m["balanced_accuracy"] for n, m in metrics) / total
    return out


def make_strategy(
    strategy_cls,
    num_clients: int,
    per_round_sink: dict | None = None,
    param_sink: dict | None = None,
    **strategy_kwargs,
):
    """Same sinks/participation config as make_fedavg_strategy, but for ANY
    Flower strategy class -- this is what lets the benchmark compare
    DriftAwareFedAvg against FedAvg/FedProx/QFedAvg/FedAdam/... on an
    identical harness, rather than each baseline being run differently.

    Server-optimizer strategies (FedAdam/FedYogi/FedAdagrad/FedOpt) and
    QFedAvg require `initial_parameters`; pass it through strategy_kwargs.
    """

    def evaluate_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
        if per_round_sink is not None:
            per_round_sink["latest"] = metrics
        return weighted_average(metrics)

    def evaluate_fn(server_round, parameters, config):
        if param_sink is not None:
            param_sink["params"] = parameters
        return None

    return strategy_cls(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        evaluate_fn=evaluate_fn if param_sink is not None else None,
        **strategy_kwargs,
    )


def make_fedavg_strategy(
    num_clients: int, per_round_sink: dict | None = None, param_sink: dict | None = None
) -> fl.server.strategy.FedAvg:
    """All `num_clients` participate in every round -- with only 5 simulated
    hospitals there's no reason to subsample.

    If `per_round_sink` is given, the raw (num_examples, metrics) list from
    the most recent evaluate round is stashed at per_round_sink["latest"].
    Flower's own History only keeps the aggregated average, but the
    project's key metric is per-client (worst-served-hospital) accuracy.

    If `param_sink` is given, the final global model's raw parameters
    (ndarrays) are stashed at param_sink["params"] via a no-op centralized
    `evaluate_fn` hook -- needed later (Step 8) to run the membership-
    inference attack against the actual trained model, not just its metrics.
    """

    def evaluate_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
        if per_round_sink is not None:
            per_round_sink["latest"] = metrics
        return weighted_average(metrics)

    def evaluate_fn(server_round, parameters, config):
        if param_sink is not None:
            param_sink["params"] = parameters
        return None

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        evaluate_fn=evaluate_fn if param_sink is not None else None,
    )
