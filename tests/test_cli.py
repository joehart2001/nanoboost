import pickle
import numpy as np
import pytest
from typer.testing import CliRunner
from nanoboost.cli.main import app

runner = CliRunner()


# ── help smoke tests ─────────────────────────────────────────────────────────

def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "preprocess" in result.output
    assert "transform" in result.output
    assert "train" in result.output
    assert "run" in result.output


@pytest.mark.parametrize("cmd", ["preprocess", "transform", "train", "run"])
def test_command_help(cmd):
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0


# ── transform integration ─────────────────────────────────────────────────────

def _make_events_pkl(path, n_events=4):
    """Create a synthetic events pickle with Gaussian-shaped events."""
    rng = np.random.default_rng(42)
    event_data, event_time = [], []
    for _ in range(n_events):
        t = np.linspace(0, 2e-3, 200)
        sig = np.exp(-((t - 1e-3) / 1.5e-4) ** 2) * 60.0  # amplitude 60 pA
        sig += rng.normal(0, 1.0, len(t))
        event_data.append(sig)
        event_time.append(t)
    with open(path, "wb") as f:
        pickle.dump({
            "event_time": event_time,
            "event_data": event_data,
            "sd_threshold": 10.0,
            "sd_threshold_lower": -np.inf,  # non-resistive (avoids None comparison bug)
            "mean_noise": 0.0,
        }, f)


def test_transform_creates_features_pkl(tmp_path):
    events_pkl = str(tmp_path / "events.pkl")
    features_pkl = str(tmp_path / "features.pkl")
    _make_events_pkl(events_pkl)

    result = runner.invoke(app, ["transform", events_pkl, features_pkl, "5"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "features.pkl").exists()

    with open(features_pkl, "rb") as f:
        d = pickle.load(f)
    assert "features_list" in d
    assert "labels" in d
    assert len(d["features_list"]) == 4
    assert all(label == 0 for label in d["labels"])  # label 5 → 0


def test_transform_label_10_gives_ones(tmp_path):
    events_pkl = str(tmp_path / "events.pkl")
    features_pkl = str(tmp_path / "features.pkl")
    _make_events_pkl(events_pkl)
    runner.invoke(app, ["transform", events_pkl, features_pkl, "10"])
    with open(features_pkl, "rb") as f:
        d = pickle.load(f)
    assert all(label == 1 for label in d["labels"])  # label 10 → 1


# ── train integration ─────────────────────────────────────────────────────────

def _make_features_pkl(path, n_per_class=25, n_features=25, label_0=True):
    rng = np.random.default_rng(0 if label_0 else 1)
    mean = 0.0 if label_0 else 1.0
    X = rng.normal(mean, 1.0, (n_per_class, n_features)).tolist()
    y = [0] * n_per_class if label_0 else [1] * n_per_class
    with open(path, "wb") as f:
        pickle.dump({"features_list": X, "labels": y, "features_df": None}, f)


def test_train_produces_model(tmp_path):
    pkl_0 = str(tmp_path / "f0.pkl")
    pkl_1 = str(tmp_path / "f1.pkl")
    _make_features_pkl(pkl_0, label_0=True)
    _make_features_pkl(pkl_1, label_0=False)
    model_path = str(tmp_path / "model.pkl")

    result = runner.invoke(app, [
        "train", pkl_0, pkl_1,
        "--output", model_path,
        "--model", "dt",
        "--search", "random",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "model.pkl").exists()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict")


def test_train_model_predicts(tmp_path):
    pkl_0 = str(tmp_path / "f0.pkl")
    pkl_1 = str(tmp_path / "f1.pkl")
    _make_features_pkl(pkl_0, label_0=True)
    _make_features_pkl(pkl_1, label_0=False)
    model_path = str(tmp_path / "model.pkl")

    runner.invoke(app, [
        "train", pkl_0, pkl_1,
        "--output", model_path,
        "--model", "dt",
        "--search", "random",
    ])
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    rng = np.random.default_rng(99)
    X_new = rng.normal(0, 1, (5, 25))
    preds = model.predict(X_new)
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1})
