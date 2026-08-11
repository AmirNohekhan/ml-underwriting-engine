from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from underwriting_engine.config import ensure_dirs, load_config
from underwriting_engine.data import generate_synthetic_portfolio, split_train_holdout, write_dataset
from underwriting_engine.diagnostics import fairness_report, stability_report
from underwriting_engine.explain import global_explanations
from underwriting_engine.features import FEATURES
from underwriting_engine.modeling import load_model, train_and_select
from underwriting_engine.monitoring import drift_report
from underwriting_engine.pricing import price_policies


def generate_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_dirs(config)
    df = generate_synthetic_portfolio(config["data"]["n_rows"], config["data"]["n_geographies"], config["seed"])
    train, holdout = split_train_holdout(df, config["seed"])
    write_dataset(train, holdout, config["data"]["train_path"], config["data"]["holdout_path"])
    print(f"Wrote {len(train):,} train rows and {len(holdout):,} holdout rows.")


def train_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_dirs(config)
    train = pd.read_csv(config["data"]["train_path"])
    holdout = pd.read_csv(config["data"]["holdout_path"])
    result = train_and_select(train, holdout, config)
    print(f"Selected {result.model_name}: {result.metrics}")


def score_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_dirs(config)
    holdout = pd.read_csv(config["data"]["holdout_path"])
    model, calibrator = load_model(config["artifacts"]["model_dir"])
    pred = calibrator.predict(model.predict(holdout[FEATURES]))
    scored = price_policies(holdout, pred, **config["pricing"])
    out_dir = Path(config["artifacts"]["report_dir"])
    scored.to_csv(out_dir / "scored_holdout.csv", index=False)
    fairness_report(scored).to_csv(out_dir / "fairness_report.csv", index=False)
    stability_report(scored).to_csv(out_dir / "stability_report.csv", index=False)
    drift_report(
        pd.read_csv(config["data"]["train_path"]),
        holdout,
        FEATURES,
        config["monitoring"]["psi_warning"],
        config["monitoring"]["psi_alert"],
    ).to_csv(out_dir / "drift_report.csv", index=False)
    global_explanations(model, holdout.sample(min(1000, len(holdout)), random_state=config["seed"]), out_dir / "global_explanations.csv")
    print(f"Wrote reports to {out_dir}.")


def all_cmd(args: argparse.Namespace) -> None:
    generate_cmd(args)
    train_cmd(args)
    score_cmd(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML underwriting engine workflows")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(required=True)
    for name, fn in [("generate", generate_cmd), ("train", train_cmd), ("score", score_cmd), ("all", all_cmd)]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
