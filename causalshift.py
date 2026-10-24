"""CausalShift Lab: reproducible concept-drift experiment."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["x1", "x2", "x3", "x4"]


@dataclass
class ExperimentConfig:
    n_samples: int = 3000
    shift_point: int = 1500
    seed: int = 42
    train_fraction: float = 0.6
    window_size: int = 100
    confidence_threshold: float = 0.70
    alpha: float = 0.01

    def validate(self) -> None:
        if self.n_samples < 500:
            raise ValueError("n_samples must be at least 500.")
        if not 100 <= self.shift_point < self.n_samples - 100:
            raise ValueError("shift_point must leave at least 100 observations on each side.")
        if not 0.2 <= self.train_fraction <= 0.8:
            raise ValueError("train_fraction must be between 0.2 and 0.8.")
        if self.window_size < 20:
            raise ValueError("window_size must be at least 20.")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0 < self.alpha < 0.5:
            raise ValueError("alpha must be between 0 and 0.5.")


def generate_stream(config: ExperimentConfig) -> pd.DataFrame:
    """Generate a labelled time stream with a known abrupt concept shift."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    n = config.n_samples
    t = np.arange(n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # A covariate shift in x3 occurs at the same known change point.
    x3 = rng.normal(loc=(t >= config.shift_point) * 0.65, size=n)
    x4 = rng.normal(size=n)

    pre_logit = 1.7 * x1 - 1.1 * x2 + 0.35 * x3 - 0.2 * x4
    post_logit = -1.25 * x1 + 1.35 * x2 + 0.15 * x3 + 0.65 * x4
    logits = np.where(t < config.shift_point, pre_logit, post_logit)
    probability = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probability)

    return pd.DataFrame({
        "time": t,
        "x1": x1, "x2": x2, "x3": x3, "x4": x4,
        "y": y,
        "true_shift": (t >= config.shift_point).astype(int),
        "true_probability": probability,
    })


def two_proportion_pvalue(errors_a: np.ndarray, errors_b: np.ndarray) -> float:
    """Two-sided pooled z-test for a change in two binary error-rate windows."""
    a = np.asarray(errors_a, dtype=int)
    b = np.asarray(errors_b, dtype=int)
    if a.size == 0 or b.size == 0:
        return 1.0
    pooled = (a.sum() + b.sum()) / (a.size + b.size)
    se = np.sqrt(pooled * (1 - pooled) * (1 / a.size + 1 / b.size))
    if se == 0:
        return 1.0
    z = (a.mean() - b.mean()) / se
    return float(2 * norm.sf(abs(z)))


def windowed_metrics(predictions: pd.DataFrame, window_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(predictions), window_size):
        chunk = predictions.iloc[start:start + window_size]
        if len(chunk) == 0:
            continue
        rows.append({
            "start_time": int(chunk["time"].iloc[0]),
            "end_time": int(chunk["time"].iloc[-1]),
            "n": int(len(chunk)),
            "true_shift_fraction": float(chunk["true_shift"].mean()),
            "accuracy": float(accuracy_score(chunk["y"], chunk["prediction"])),
            "f1": float(f1_score(chunk["y"], chunk["prediction"], zero_division=0)),
            "error_rate": float((chunk["y"].to_numpy() != chunk["prediction"].to_numpy()).mean()),
            "mean_confidence": float(chunk["confidence"].mean()),
            "alarm_count": int(chunk["drift_alarm"].sum()),
        })
    return pd.DataFrame(rows)


def selective_curve(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    confidence = np.maximum(probability, 1 - probability)
    prediction = (probability >= 0.5).astype(int)
    rows = []
    for threshold in np.linspace(0.50, 0.95, 19):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        risk = float((prediction[accepted] != y[accepted]).mean()) if accepted.any() else float("nan")
        rows.append({"threshold": float(threshold), "coverage": coverage, "selective_risk": risk})
    return pd.DataFrame(rows)


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> dict[str, Any]:
    config.validate()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stream = generate_stream(config)

    train_end = int(config.n_samples * config.train_fraction)
    # Ensure the model's initial training data comes from before the true shift.
    if train_end >= config.shift_point:
        train_end = config.shift_point
    train = stream.iloc[:train_end]
    test = stream.iloc[train_end:].copy()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.seed),
    )
    model.fit(train[FEATURES], train["y"])
    prob = model.predict_proba(test[FEATURES])[:, 1]
    pred = (prob >= 0.5).astype(int)
    confidence = np.maximum(prob, 1 - prob)
    errors = (pred != test["y"].to_numpy()).astype(int)

    # Compare two adjacent rolling windows. A flag is attached to the later window.
    alarms = np.zeros(len(test), dtype=int)
    w = config.window_size
    for i in range(2 * w, len(test) + 1, w):
        prior = errors[i - 2 * w:i - w]
        recent = errors[i - w:i]
        p_value = two_proportion_pvalue(prior, recent)
        if p_value < config.alpha:
            alarms[i - w:i] = 1

    test["prediction"] = pred
    test["probability"] = prob
    test["confidence"] = confidence
    test["error"] = errors
    test["drift_alarm"] = alarms
    test.to_csv(out / "predictions.csv", index=False)

    windows = windowed_metrics(test, w)
    windows.to_csv(out / "window_metrics.csv", index=False)
    curve = selective_curve(test["y"].to_numpy(), prob)
    curve.to_csv(out / "selective_curve.csv", index=False)

    pre = test[test["time"] < config.shift_point]
    post = test[test["time"] >= config.shift_point]
    first_alarm_time = int(test.loc[test["drift_alarm"] == 1, "time"].min()) if alarms.any() else None
    detection_delay = (
        max(0, first_alarm_time - config.shift_point)
        if first_alarm_time is not None and first_alarm_time >= config.shift_point
        else None
    )
    accepted = confidence >= config.confidence_threshold
    overall = {
        "accuracy": float(accuracy_score(test["y"], pred)),
        "f1": float(f1_score(test["y"], pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(test["y"], prob)),
        "brier_score": float(brier_score_loss(test["y"], prob)),
        "selective_coverage": float(accepted.mean()),
        "selective_risk": float((pred[accepted] != test["y"].to_numpy()[accepted]).mean()) if accepted.any() else None,
    }
    summary = {
        "project": "CausalShift Lab",
        "config": asdict(config),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "known_shift_point": int(config.shift_point),
        "pre_shift_test_n": int(len(pre)),
        "post_shift_test_n": int(len(post)),
        "pre_shift_error_rate": float(pre["error"].mean()) if len(pre) else None,
        "post_shift_error_rate": float(post["error"].mean()) if len(post) else None,
        "first_alarm_time": first_alarm_time,
        "detection_delay_observations": detection_delay,
        "total_alarm_observations": int(alarms.sum()),
        "overall_metrics": overall,
        "note": "Synthetic benchmark only; results are not evidence of real-world performance.",
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot 1: windowed performance.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(windows["start_time"], windows["f1"], marker="o", label="Window F1")
    ax.plot(windows["start_time"], windows["error_rate"], marker="s", label="Window error rate")
    ax.axvline(config.shift_point, linestyle="--", label="True shift")
    ax.set(title="Predictive performance over the stream", xlabel="Time index", ylabel="Score / rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "performance_over_time.png", dpi=160)
    plt.close(fig)

    # Plot 2: selective prediction risk vs coverage.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["coverage"], curve["selective_risk"], marker="o")
    ax.set(title="Selective prediction: coverage vs risk", xlabel="Coverage (fraction accepted)", ylabel="Error among accepted predictions")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "confidence_coverage.png", dpi=160)
    plt.close(fig)

    # Plot 3: alarms over time.
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(test["time"], test["drift_alarm"], s=8, label="Alarm flag")
    ax.axvline(config.shift_point, linestyle="--", label="True shift")
    ax.set(title="Known concept shift and detector alarms", xlabel="Time index", ylabel="Alarm")
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "drift_timeline.png", dpi=160)
    plt.close(fig)

    return summary
