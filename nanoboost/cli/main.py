import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.rule import Rule
from rich.text import Text

_console = Console()

app = typer.Typer(help="NanoBoost: wavelet-enhanced ML pipeline for nanopore sensing.")

_MODEL_MAP = {"xgboost": "XG", "rf": "RF", "svm": "SVM", "dt": "DT"}


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _run_preprocess(abf_path: Path, output: Path, resistive: bool = False) -> int:
    from nanoboost.scripts.discrete_wavelet_transform.novel_DWT import load_to_event_data_nofeatures

    event_time, event_data, _, sd_threshold, sd_threshold_lower, mean_noise = \
        load_to_event_data_nofeatures(str(abf_path), resistive=resistive)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump({
            "event_time": event_time,
            "event_data": event_data,
            "sd_threshold": sd_threshold,
            "sd_threshold_lower": sd_threshold_lower,
            "mean_noise": mean_noise,
        }, f)
    return len(event_data)


def _run_transform(events_pkl: Path, output: Path, label: str,
                   wavelet: str, threshold: float) -> int:
    from nanoboost.scripts.discrete_wavelet_transform.novel_DWT import DWT_and_features_thresh_trace

    label_val = int(label) if label.isdigit() else label

    with open(events_pkl, "rb") as f:
        ev = pickle.load(f)

    _, _, _, features_df, features_list, labels, _ = DWT_and_features_thresh_trace(
        ev["event_time"], ev["event_data"], ev["mean_noise"],
        ev["sd_threshold"], ev["sd_threshold_lower"],
        label_val, wavelet, threshold,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump({"features_df": features_df, "features_list": features_list, "labels": labels}, f)
    return len(features_list)


def _run_train(feature_pkls: List[Path], output: Path,
               model_key: str, search_type: str) -> dict:
    from nanoboost.scripts.ml_model_building.supervised_learning.training_and_hyperparameter_op import hyperparam_op

    all_X, all_y = [], []
    for p in feature_pkls:
        with open(p, "rb") as f:
            d = pickle.load(f)
        all_X.extend(d["features_list"])
        all_y.extend(d["labels"])

    _, _, _, _, _, search_obj, best_params = hyperparam_op(
        _MODEL_MAP[model_key], search_type, np.array(all_X), np.array(all_y)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(search_obj.best_estimator_, f)
    return best_params


@app.command()
def preprocess(
    input: Path = typer.Argument(..., help="Path to .abf recording"),
    output: Path = typer.Argument(..., help="Output events pickle path"),
    resistive: bool = typer.Option(False, "--resistive/--no-resistive",
                                   help="Include resistive (trough) peaks"),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML config file"),
):
    """Baseline-correct an ABF file and isolate translocation events."""
    cfg = _load_config(config).get("preprocessing", {}) if config else {}
    n = _run_preprocess(input, output, resistive=resistive or cfg.get("resistive", False))
    typer.echo(f"Isolated {n} events → {output}")


@app.command()
def transform(
    input: Path = typer.Argument(..., help="Events pickle from preprocess"),
    output: Path = typer.Argument(..., help="Output features pickle path"),
    label: str = typer.Argument(..., help="NP label: 5 or 10 (nm), NR or NS (shape)"),
    wavelet: Optional[str] = typer.Option(None, "--wavelet", help="Mother wavelet (default: bior3.3)"),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="DWT coefficient threshold (default: 0.2)"),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML config file"),
):
    """Apply per-event DWT and extract ~25 features per event."""
    cfg = _load_config(config).get("transform", {}) if config else {}
    n = _run_transform(
        input, output, label,
        wavelet=wavelet or cfg.get("wavelet", "bior3.3"),
        threshold=threshold if threshold is not None else cfg.get("threshold", 0.2),
    )
    typer.echo(f"Extracted features for {n} events → {output}")


@app.command()
def train(
    inputs: List[Path] = typer.Argument(..., help="One or more feature pickle files"),
    output: Path = typer.Option(Path("model.pkl"), "--output", "-o", help="Output model path"),
    model: Optional[str] = typer.Option(None, "--model", help="xgboost | rf | svm | dt"),
    search: Optional[str] = typer.Option(None, "--search", help="random | bayes | sobol | grid"),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML config file"),
):
    """Train a classifier with hyperparameter optimisation across one or more feature sets."""
    cfg = _load_config(config).get("training", {}) if config else {}
    model_key = (model or cfg.get("model", "xgboost")).lower()
    search_type = search or cfg.get("search", "random")
    best_params = _run_train(inputs, output, model_key, search_type)
    typer.echo(f"Model saved → {output}")
    typer.echo(f"Best params: {best_params}")


@app.command()
def run(
    config: Path = typer.Argument(Path("config.yaml"), help="YAML config file"),
):
    """Run the full pipeline end-to-end: preprocess → transform → train."""
    cfg = _load_config(config)
    output_dir = Path(cfg["data"]["output_dir"])
    pre_cfg = cfg.get("preprocessing", {})
    t_cfg = cfg.get("transform", {})
    tr_cfg = cfg.get("training", {})

    _console.print(Rule("[bold cyan]NanoBoost Pipeline[/bold cyan]", style="cyan"))

    feature_pkls: List[Path] = []
    for entry in cfg["data"]["files"]:
        stem = Path(entry["path"]).stem
        events_pkl = output_dir / f"{stem}_events.pkl"
        features_pkl = output_dir / f"{stem}_features.pkl"

        _console.print(Rule(f"[bold]preprocess[/bold]  {entry['path']}", align="left", style="dim"))
        n_events = _run_preprocess(Path(entry["path"]), events_pkl,
                                   resistive=pre_cfg.get("resistive", False))
        _console.print(f"  [green]✓[/green] {n_events} events isolated → [dim]{events_pkl}[/dim]")

        _console.print(Rule(f"[bold]transform[/bold]   {stem}  label={entry['label']}", align="left", style="dim"))
        n_feats = _run_transform(events_pkl, features_pkl, str(entry["label"]),
                                 wavelet=t_cfg.get("wavelet", "bior3.3"),
                                 threshold=t_cfg.get("threshold", 0.2))
        _console.print(f"  [green]✓[/green] {n_feats} feature vectors → [dim]{features_pkl}[/dim]")
        feature_pkls.append(features_pkl)

    model_path = output_dir / "model.pkl"
    _console.print(Rule(f"[bold]train[/bold]   {len(feature_pkls)} dataset(s)  model={tr_cfg.get('model', 'xgboost')}", align="left", style="dim"))
    best_params = _run_train(
        feature_pkls, model_path,
        (tr_cfg.get("model", "xgboost")).lower(),
        tr_cfg.get("search", "random"),
    )
    _console.print(f"  [green]✓[/green] model saved → [dim]{model_path}[/dim]")
    _console.print(f"  [dim]best params: {best_params}[/dim]")
    _console.print(Rule(style="cyan"))


if __name__ == "__main__":
    app()
