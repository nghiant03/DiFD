"""Tests for evaluation module: EvalResult save/load and Evaluator predictions."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from DiFD.evaluation.evaluator import EvalResult, Evaluator
from DiFD.models.lstm import LSTMClassifier
from DiFD.schema import EvaluateConfig
from DiFD.schema.types import FaultType
from DiFD.training.callbacks import ClassMetrics

NUM_CLASSES = FaultType.count()
SEQ_LEN = 10
FEATURES = 4


def _small_model() -> LSTMClassifier:
    return LSTMClassifier(
        input_size=FEATURES,
        num_classes=NUM_CLASSES,
        hidden_size=8,
        num_layers=1,
        bidirectional=False,
    )


def _make_data(
    n: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, SEQ_LEN, FEATURES)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, (n, SEQ_LEN)).astype(np.int32)
    return X, y


class TestEvalResultSaveLoad:
    def test_save_creates_files(self, tmp_path: Path) -> None:
        result = EvalResult(
            loss=0.5,
            accuracy=0.9,
            macro_f1=0.85,
            class_metrics=ClassMetrics(
                precision=[0.9, 0.8, 0.7, 0.6],
                recall=[0.85, 0.75, 0.65, 0.55],
                f1=[0.87, 0.77, 0.67, 0.57],
                support=[100, 50, 30, 20],
            ),
            y_true=np.array([0, 1, 2, 3], dtype=np.int32),
            y_pred=np.array([0, 1, 2, 2], dtype=np.int32),
            y_prob=np.array([[0.9, 0.05, 0.03, 0.02],
                             [0.1, 0.8, 0.05, 0.05],
                             [0.1, 0.1, 0.7, 0.1],
                             [0.1, 0.1, 0.5, 0.3]], dtype=np.float32),
        )
        result.save(tmp_path / "eval")
        assert (tmp_path / "eval" / "eval_metrics.json").exists()
        assert (tmp_path / "eval" / "predictions.csv").exists()

        import csv
        with (tmp_path / "eval" / "predictions.csv").open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header[:2] == ["y_true", "y_pred"]
            rows = list(reader)
            assert len(rows) == 4
            assert rows[0][0] == "NORMAL"
            assert rows[1][0] == "SPIKE"

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        y_true = np.array([0, 1, 2, 3, 0, 1], dtype=np.int32)
        y_pred = np.array([0, 1, 2, 2, 0, 0], dtype=np.int32)
        y_prob = np.random.default_rng(0).random((6, NUM_CLASSES)).astype(np.float32)

        original = EvalResult(
            loss=0.42,
            accuracy=0.88,
            macro_f1=0.82,
            class_metrics=ClassMetrics(
                precision=[0.9, 0.8, 0.7, 0.6],
                recall=[0.85, 0.75, 0.65, 0.55],
                f1=[0.87, 0.77, 0.67, 0.57],
                support=[100, 50, 30, 20],
            ),
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        save_dir = tmp_path / "eval_roundtrip"
        original.save(save_dir)
        loaded = EvalResult.load(save_dir)

        assert abs(loaded.loss - original.loss) < 1e-6
        assert abs(loaded.accuracy - original.accuracy) < 1e-6
        assert abs(loaded.macro_f1 - original.macro_f1) < 1e-6
        np.testing.assert_array_equal(loaded.y_true, original.y_true)
        np.testing.assert_array_equal(loaded.y_pred, original.y_pred)
        np.testing.assert_array_almost_equal(loaded.y_prob, original.y_prob, decimal=5)

    def test_save_with_configs(self, tmp_path: Path) -> None:
        result = EvalResult(
            loss=0.5,
            accuracy=0.9,
            macro_f1=0.85,
            class_metrics=ClassMetrics(
                precision=[0.9], recall=[0.85], f1=[0.87], support=[100],
            ),
        )
        train_cfg = {"model": "lstm", "epochs": 10}
        inject_cfg = {"seed": 42, "resample_freq": "5min"}
        result.save(tmp_path / "eval", train_config=train_cfg, injection_config=inject_cfg)

        meta = json.loads((tmp_path / "eval" / "eval_metrics.json").read_text())
        assert meta["train_config"] == train_cfg
        assert meta["injection_config"] == inject_cfg


class TestEvaluatorPredictions:
    def test_evaluate_returns_predictions(self) -> None:
        model = _small_model()
        X, y = _make_data(n=16)

        evaluator = Evaluator(config=EvaluateConfig(batch_size=8), device="cpu")
        result = evaluator.evaluate(model, X, y)

        total_timesteps = 16 * SEQ_LEN
        assert result.y_true.shape == (total_timesteps,)
        assert result.y_pred.shape == (total_timesteps,)
        assert result.y_prob.shape == (total_timesteps, NUM_CLASSES)
        assert result.y_prob.dtype == np.float32
        np.testing.assert_allclose(result.y_prob.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate_predictions_match_accuracy(self) -> None:
        model = _small_model()
        X, y = _make_data(n=16)

        evaluator = Evaluator(device="cpu")
        result = evaluator.evaluate(model, X, y)

        manual_acc = (result.y_pred == result.y_true).mean()
        assert abs(manual_acc - result.accuracy) < 1e-6
