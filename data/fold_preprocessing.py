"""Raw outer splits and explicitly fold-local preprocessing for new experiments."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from data.datasets import _load_raw


def raw_split(name, seed, max_train=40, max_test=20, test_size=0.3):
    X, y = _load_raw(name, seed)
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size,
                             random_state=seed, stratify=y)
    if len(tr) > max_train:
        tr, _ = train_test_split(tr, train_size=max_train, random_state=seed, stratify=y[tr])
    if len(te) > max_test:
        te, _ = train_test_split(te, train_size=max_test, random_state=seed, stratify=y[te])
    return X[tr], X[te], y[tr].astype(float), y[te].astype(float), tr, te


class AnglePreprocessor:
    def __init__(self, n_qubits, seed=0):
        self.n_qubits, self.seed = n_qubits, seed

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.pca = PCA(n_components=self.n_qubits, random_state=self.seed).fit(Z) if Z.shape[1] > self.n_qubits else None
        self.angles = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(self._reduce(Z))
        return self

    def _reduce(self, Z):
        if self.pca is not None:
            return self.pca.transform(Z)
        return np.pad(Z, ((0, 0), (0, self.n_qubits-Z.shape[1])))

    def transform(self, X):
        # No clipping: preserve historical transformation; test values can exceed train range.
        return self.angles.transform(self._reduce(self.scaler.transform(X)))

    def state(self):
        return {'mean': self.scaler.mean_.tolist(), 'scale': self.scaler.scale_.tolist(),
                'pca_components': None if self.pca is None else self.pca.components_.tolist(),
                'pca_mean': None if self.pca is None else self.pca.mean_.tolist(),
                'angle_scale': self.angles.scale_.tolist(), 'angle_min': self.angles.min_.tolist()}
