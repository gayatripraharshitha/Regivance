import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class MarketRegimeDetector:
    """Fit market regimes on training features and classify later dates."""

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state

    @staticmethod
    def _validate(features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")

        if features.empty or features.shape[1] == 0:
            raise ValueError("features must contain rows and columns.")

        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("features must have a DatetimeIndex.")

        if features.index.hasnans or features.index.has_duplicates:
            raise ValueError("Dates must be non-missing and unique.")

        if not features.index.is_monotonic_increasing:
            raise ValueError("Dates must be in ascending order.")

        if features.columns.has_duplicates:
            raise ValueError("Feature names must be unique.")

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in features.dtypes
        ):
            raise TypeError("Features must be numeric.")

        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Features must be finite and non-missing.")

    def fit(self, training_features: pd.DataFrame):
        """Learn scaling parameters and cluster centers from training data."""
        self._validate(training_features)

        if (
            not isinstance(self.n_regimes, int)
            or isinstance(self.n_regimes, bool)
            or self.n_regimes < 2
        ):
            raise ValueError("n_regimes must be an integer of at least 2.")

        if len(training_features.drop_duplicates()) < self.n_regimes:
            raise ValueError("Not enough distinct rows for the regimes.")

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(training_features)

        model = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=20,
        )
        model.fit(scaled_features)

        self.scaler_ = scaler
        self.model_ = model
        self.feature_names_ = training_features.columns.tolist()
        self.training_end_ = training_features.index[-1]

        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Assign regimes without changing the fitted scaler or clusters."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before predict().")

        self._validate(features)

        if features.columns.tolist() != self.feature_names_:
            raise ValueError(
                "Feature names and order must match the training data."
            )

        scaled_features = self.scaler_.transform(features)
        labels = self.model_.predict(scaled_features)

        return pd.Series(labels, index=features.index, name="regime")

    def cluster_centers(self) -> pd.DataFrame:
        """Return cluster centers in the original feature units."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before cluster_centers().")

        centers = self.scaler_.inverse_transform(
            self.model_.cluster_centers_
        )

        return pd.DataFrame(
            centers,
            columns=self.feature_names_,
            index=pd.Index(range(self.n_regimes), name="regime"),
        )

