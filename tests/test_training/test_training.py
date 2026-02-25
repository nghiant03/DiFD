"""Tests for the training module: FocalLoss, oversampling, callbacks, and Trainer."""

import numpy as np
import pytest
import torch

from DiFD.models.lstm import LSTMClassifier
from DiFD.schema import TrainConfig
from DiFD.schema.types import FaultType
from DiFD.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    TrainMetrics,
)
from DiFD.training.loss import FocalLoss
from DiFD.training.oversampling import oversample_minority
from DiFD.training.trainer import Trainer, TrainResult, _build_loss, _prepare_data

NUM_CLASSES = FaultType.count()
SEQ_LEN = 10
FEATURES = 4
BATCH = 20


def _small_model() -> LSTMClassifier:
    return LSTMClassifier(
        input_size=FEATURES, num_classes=NUM_CLASSES,
        hidden_size=8, num_layers=1, bidirectional=False,
    )


def _make_data(
    n: int = BATCH,
    seq_len: int = SEQ_LEN,
    features: int = FEATURES,
    num_classes: int = NUM_CLASSES,
    minority_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic X, y arrays for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, seq_len, features)).astype(np.float32)
    y = np.zeros((n, seq_len), dtype=np.int32)
    n_minority = int(n * minority_frac)
    for i in range(n_minority):
        fault = rng.integers(1, num_classes)
        y[i, :] = fault
    return X, y


class TestFocalLoss:
    def test_output_shape_mean(self) -> None:
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_output_shape_none(self) -> None:
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (BATCH,)

    def test_output_shape_sum(self) -> None:
        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_gamma_zero_matches_ce(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        focal = FocalLoss(gamma=0.0)(logits, targets)
        ce = torch.nn.CrossEntropyLoss()(logits, targets)
        torch.testing.assert_close(focal, ce, atol=1e-5, rtol=1e-5)

    def test_with_alpha(self) -> None:
        alpha = torch.ones(NUM_CLASSES)
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_loss_positive(self) -> None:
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_higher_gamma_lower_loss_for_easy(self) -> None:
        torch.manual_seed(0)
        logits = torch.zeros(1, NUM_CLASSES)
        logits[0, 0] = 10.0
        targets = torch.tensor([0])
        low_gamma = FocalLoss(gamma=0.0)(logits, targets)
        high_gamma = FocalLoss(gamma=5.0)(logits, targets)
        assert high_gamma.item() <= low_gamma.item()


class TestOversampling:
    def test_no_oversampling_when_disabled(self) -> None:
        X, y = _make_data()
        X_out, y_out = oversample_minority(X, y, ratio=1.0, seed=0)
        assert len(X_out) >= len(X)

    def test_all_normal_unchanged(self) -> None:
        X, _ = _make_data()
        y = np.zeros((BATCH, SEQ_LEN), dtype=np.int32)
        X_out, y_out = oversample_minority(X, y, ratio=1.0, seed=0)
        assert len(X_out) == len(X)
        np.testing.assert_array_equal(y_out, y)

    def test_oversampling_increases_count(self) -> None:
        X, y = _make_data(n=100, minority_frac=0.1, seed=42)
        X_out, y_out = oversample_minority(X, y, ratio=1.0, seed=42)
        assert len(X_out) > len(X)

    def test_oversampling_preserves_shapes(self) -> None:
        X, y = _make_data(n=100, minority_frac=0.1, seed=42)
        X_out, y_out = oversample_minority(X, y, ratio=1.0, seed=42)
        assert X_out.shape[1:] == X.shape[1:]
        assert y_out.shape[1:] == y.shape[1:]

    def test_oversampling_deterministic(self) -> None:
        X, y = _make_data(n=50, minority_frac=0.2, seed=0)
        X1, y1 = oversample_minority(X, y, ratio=1.0, seed=99)
        X2, y2 = oversample_minority(X, y, ratio=1.0, seed=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_ratio_half(self) -> None:
        X, y = _make_data(n=100, minority_frac=0.05, seed=7)
        X_out, _ = oversample_minority(X, y, ratio=0.5, seed=7)
        assert len(X_out) > len(X)
        assert len(X_out) < len(X) * 2


class TestCallbacks:
    def test_logging_callback_returns_true(self) -> None:
        cb = LoggingCallback()
        metrics = TrainMetrics(epoch=1, train_loss=0.5, train_acc=0.9)
        model = _small_model()
        assert cb.on_epoch_end(metrics, model) is True

    def test_early_stopping_patience(self) -> None:
        cb = EarlyStoppingCallback(patience=3)
        model = _small_model()
        for i in range(3):
            result = cb.on_epoch_end(
                TrainMetrics(epoch=i + 1, train_loss=1.0, val_loss=1.0), model
            )
            assert result is True
        result = cb.on_epoch_end(
            TrainMetrics(epoch=4, train_loss=1.0, val_loss=1.0), model
        )
        assert result is False

    def test_early_stopping_resets_on_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2)
        model = _small_model()
        cb.on_epoch_end(TrainMetrics(epoch=1, train_loss=1.0, val_loss=1.0), model)
        cb.on_epoch_end(TrainMetrics(epoch=2, train_loss=1.0, val_loss=1.0), model)
        result = cb.on_epoch_end(TrainMetrics(epoch=3, train_loss=1.0, val_loss=0.5), model)
        assert result is True

    def test_early_stopping_no_val(self) -> None:
        cb = EarlyStoppingCallback(patience=2)
        model = _small_model()
        for i in range(10):
            assert cb.on_epoch_end(
                TrainMetrics(epoch=i + 1, train_loss=1.0), model
            ) is True

    def test_checkpoint_callback(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "ckpt.pt"  # type: ignore[operator]
        cb = CheckpointCallback(save_path=str(path))
        model = _small_model()
        cb.on_epoch_end(TrainMetrics(epoch=1, train_loss=0.5), model)
        assert path.exists()  # type: ignore[union-attr]

    def test_checkpoint_callback_saves_config(self, tmp_path: pytest.TempPathFactory) -> None:
        import json

        path = tmp_path / "ckpt_cfg.pt"  # type: ignore[operator]
        config = TrainConfig(epochs=5, learning_rate=0.01)
        cb = CheckpointCallback(save_path=str(path), config_dict=config.to_dict())
        model = _small_model()
        cb.on_epoch_end(TrainMetrics(epoch=1, train_loss=0.5), model)
        assert path.exists()  # type: ignore[union-attr]
        checkpoint = torch.load(str(path), weights_only=True)
        assert "config" in checkpoint
        restored = json.loads(checkpoint["config"])
        assert restored["epochs"] == 5
        assert restored["learning_rate"] == 0.01

    def test_checkpoint_callback_without_config(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "ckpt_no_cfg.pt"  # type: ignore[operator]
        cb = CheckpointCallback(save_path=str(path))
        model = _small_model()
        cb.on_epoch_end(TrainMetrics(epoch=1, train_loss=0.5), model)
        checkpoint = torch.load(str(path), weights_only=True)
        assert "config" not in checkpoint


class TestBuildLoss:
    def test_cross_entropy_default(self) -> None:
        config = TrainConfig()
        loss_fn = _build_loss(config, torch.device("cpu"))
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)

    def test_focal_loss_when_enabled(self) -> None:
        config = TrainConfig(use_focal_loss=True, focal_gamma=3.0)
        loss_fn = _build_loss(config, torch.device("cpu"))
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.gamma == 3.0

    def test_focal_loss_with_alpha(self) -> None:
        config = TrainConfig(
            use_focal_loss=True,
            focal_alpha=[1.0, 2.0, 2.0, 2.0],
        )
        loss_fn = _build_loss(config, torch.device("cpu"))
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha is not None


class TestPrepareData:
    def test_no_oversampling(self) -> None:
        config = TrainConfig(oversample=False)
        X, y = _make_data()
        X_out, y_out = _prepare_data(X, y, config)
        assert X_out is X
        assert y_out is y

    def test_with_oversampling(self) -> None:
        config = TrainConfig(oversample=True, oversample_ratio=1.0)
        X, y = _make_data(n=100, minority_frac=0.1)
        X_out, y_out = _prepare_data(X, y, config)
        assert len(X_out) >= len(X)


class TestTrainer:
    def test_fit_basic(self) -> None:
        config = TrainConfig(epochs=2, batch_size=16, learning_rate=0.01, val_ratio=0.0)
        model = _small_model()
        X, y = _make_data(n=32)

        trainer = Trainer(config=config, callbacks=[], device="cpu")
        result = trainer.fit(model, X, y)

        assert isinstance(result, TrainResult)
        assert len(result.history) == 2
        assert result.stopped_epoch == 2

    def test_fit_with_validation(self) -> None:
        config = TrainConfig(epochs=2, batch_size=16, val_ratio=0.0)
        model = _small_model()
        X_train, y_train = _make_data(n=24, seed=0)
        X_val, y_val = _make_data(n=8, seed=1)

        trainer = Trainer(config=config, callbacks=[], device="cpu")
        result = trainer.fit(model, X_train, y_train, X_val, y_val)

        assert len(result.history) == 2
        assert result.history[0].val_loss is not None
        assert result.history[0].val_acc is not None
        assert result.best_val_loss is not None

    def test_fit_with_focal_loss(self) -> None:
        config = TrainConfig(
            epochs=2, batch_size=16, use_focal_loss=True, focal_gamma=2.0, val_ratio=0.0
        )
        model = _small_model()
        X, y = _make_data(n=32)

        trainer = Trainer(config=config, callbacks=[], device="cpu")
        result = trainer.fit(model, X, y)
        assert len(result.history) == 2

    def test_fit_with_oversampling(self) -> None:
        config = TrainConfig(
            epochs=2, batch_size=16, oversample=True, oversample_ratio=1.0, val_ratio=0.0
        )
        model = _small_model()
        X, y = _make_data(n=32, minority_frac=0.1)

        trainer = Trainer(config=config, callbacks=[], device="cpu")
        result = trainer.fit(model, X, y)
        assert len(result.history) == 2

    def test_fit_with_focal_and_oversampling(self) -> None:
        config = TrainConfig(
            epochs=2,
            batch_size=16,
            use_focal_loss=True,
            focal_gamma=2.0,
            focal_alpha=[0.25, 0.75, 0.75, 0.75],
            oversample=True,
            oversample_ratio=0.5,
            val_ratio=0.0,
        )
        model = _small_model()
        X, y = _make_data(n=32, minority_frac=0.1)

        trainer = Trainer(config=config, callbacks=[], device="cpu")
        result = trainer.fit(model, X, y)
        assert len(result.history) == 2

    def test_early_stopping_integration(self) -> None:
        config = TrainConfig(epochs=20, batch_size=16, learning_rate=0.1, val_ratio=0.0)
        model = _small_model()
        X_train, y_train = _make_data(n=24, seed=0)
        X_val, y_val = _make_data(n=8, seed=1)

        callbacks = [EarlyStoppingCallback(patience=3)]
        trainer = Trainer(config=config, callbacks=callbacks, device="cpu")
        result = trainer.fit(model, X_train, y_train, X_val, y_val)

        assert result.stopped_epoch <= 20

    def test_train_loss_decreases(self) -> None:
        config = TrainConfig(epochs=5, batch_size=16, learning_rate=0.01, val_ratio=0.0)
        model = _small_model()
        X, y = _make_data(n=32, seed=42)

        trainer = Trainer(config=config, callbacks=[], device="cpu")
        result = trainer.fit(model, X, y)

        first_loss = result.history[0].train_loss
        last_loss = result.history[-1].train_loss
        assert last_loss < first_loss


class TestTrainConfig:
    def test_defaults(self) -> None:
        config = TrainConfig()
        assert config.use_focal_loss is False
        assert config.focal_gamma == 2.0
        assert config.focal_alpha is None
        assert config.oversample is False
        assert config.oversample_ratio == 1.0

    def test_to_dict_roundtrip(self) -> None:
        config = TrainConfig(
            use_focal_loss=True,
            focal_gamma=3.0,
            focal_alpha=[0.25, 0.75, 0.75, 0.75],
            oversample=True,
            oversample_ratio=0.5,
        )
        d = config.to_dict()
        restored = TrainConfig.from_dict(d)
        assert restored.use_focal_loss is True
        assert restored.focal_gamma == 3.0
        assert restored.focal_alpha == [0.25, 0.75, 0.75, 0.75]
        assert restored.oversample is True
        assert restored.oversample_ratio == 0.5

    def test_from_dict_defaults(self) -> None:
        config = TrainConfig.from_dict({})
        assert config.use_focal_loss is False
        assert config.oversample is False


class TestModelSaveWithConfig:
    def test_save_with_config(self, tmp_path: pytest.TempPathFactory) -> None:
        import json

        path = str(tmp_path / "model.pt")  # type: ignore[operator]
        model = _small_model()
        config = TrainConfig(epochs=5, learning_rate=0.01)
        model.save(path, config_dict=config.to_dict())
        checkpoint = torch.load(path, weights_only=True)
        assert "config" in checkpoint
        restored = json.loads(checkpoint["config"])
        assert restored["epochs"] == 5
        assert restored["learning_rate"] == 0.01

    def test_save_without_config(self, tmp_path: pytest.TempPathFactory) -> None:
        path = str(tmp_path / "model.pt")  # type: ignore[operator]
        model = _small_model()
        model.save(path)
        checkpoint = torch.load(path, weights_only=True)
        assert "config" not in checkpoint

    def test_load_config(self, tmp_path: pytest.TempPathFactory) -> None:
        from DiFD.models.base import BaseModel

        path = str(tmp_path / "model.pt")  # type: ignore[operator]
        model = _small_model()
        config = TrainConfig(model="lstm", epochs=20, batch_size=64)
        model.save(path, config_dict=config.to_dict())
        loaded_config = BaseModel.load_config(path)
        assert loaded_config is not None
        restored = TrainConfig.from_dict(loaded_config)
        assert restored.epochs == 20
        assert restored.batch_size == 64

    def test_load_config_returns_none_when_absent(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from DiFD.models.base import BaseModel

        path = str(tmp_path / "model.pt")  # type: ignore[operator]
        model = _small_model()
        model.save(path)
        assert BaseModel.load_config(path) is None
