
import numpy as np
import pytest

from src.models.lstm import LSTMConfig, LSTMVolatilityForecaster


def make_data():
    rng = np.random.default_rng(42)

    training_sequences = rng.normal(
        size=(64, 8, 4)
    ).astype(np.float32)

    validation_sequences = rng.normal(
        size=(16, 8, 4)
    ).astype(np.float32)

    training_targets = np.exp(
        -4.5 + 0.1 * rng.normal(size=64)
    )

    validation_targets = np.exp(
        -4.5 + 0.1 * rng.normal(size=16)
    )

    return (
        training_sequences,
        training_targets,
        validation_sequences,
        validation_targets,
    )


def test_lstm_fits_and_produces_positive_forecasts():
    config = LSTMConfig(
        hidden_size=8,
        batch_size=16,
        max_epochs=4,
        patience=2,
        random_state=42,
    )

    model = LSTMVolatilityForecaster(config)

    x_train, y_train, x_validation, y_validation = make_data()

    model.fit(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    forecasts = model.predict(x_validation)

    assert forecasts.shape == (len(x_validation),)
    assert np.isfinite(forecasts).all()
    assert (forecasts > 0).all()
    assert 1 <= model.best_epoch_ <= model.n_epochs_trained_
    assert np.isfinite(model.best_validation_loss_)


def test_lstm_rejects_nonpositive_targets():
    config = LSTMConfig(
        hidden_size=8,
        batch_size=16,
        max_epochs=2,
        patience=1,
    )

    model = LSTMVolatilityForecaster(config)

    x_train, y_train, x_validation, y_validation = make_data()
    y_train[0] = 0.0

    with pytest.raises(ValueError, match="positive"):
        model.fit(
            x_train,
            y_train,
            x_validation,
            y_validation,
        )


def test_lstm_rejects_mismatched_sequence_dimensions():
    config = LSTMConfig(
        hidden_size=8,
        batch_size=16,
        max_epochs=2,
        patience=1,
    )

    model = LSTMVolatilityForecaster(config)

    x_train, y_train, x_validation, y_validation = make_data()
    x_validation = x_validation[:, :, :2]

    with pytest.raises(ValueError, match="dimensions must match"):
        model.fit(
            x_train,
            y_train,
            x_validation,
            y_validation,
        )


def test_lstm_requires_training_before_prediction():
    model = LSTMVolatilityForecaster()

    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.ones((2, 8, 4), dtype=np.float32))


def test_lstm_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="hidden_size"):
        LSTMConfig(hidden_size=0)

    with pytest.raises(ValueError, match="dropout"):
        LSTMConfig(dropout=1.0)
