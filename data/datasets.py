"""
data/datasets.py

Phase 1 datasets (small, so we can compute full kernel matrices):
    - XOR        : synthetic sanity-check
    - iris       : binary subset (classes 0 vs 1)
    - wine       : binary subset (classes 0 vs 1)
    - breast_cancer : already binary

Added while testing the "kernel diversity predicts adaptive benefit"
hypothesis (see README.md) -- more datasets, more data points to check the
hypothesis against:
    - digits     : binary subset (digit 3 vs digit 8, a genuinely harder
                   pairwise-confusable pair than 0 vs 1)
    - moons      : synthetic two-interleaving-crescents (sklearn make_moons)
    - circles    : synthetic concentric circles (sklearn make_circles) --
                   NOT linearly separable, unlike most binary subsets above

All datasets are:
    1. train/test split
    2. standardized (zero mean / unit variance) on train, applied to test
    3. PCA-reduced to `n_qubits` features if the raw dimensionality is higher
    4. scaled into [-pi, pi] so they can be fed into angle-encoding circuits
    5. labels mapped to {-1, +1} (needed for kernel-target alignment)
"""

import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_moons, make_circles, make_classification, make_gaussian_quantiles,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def _make_xor(n_samples=200, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)
    X = X + rng.normal(0, noise, X.shape)
    return X, y


def _digit_pair(a, b):
    d = load_digits()
    mask = (d.target == a) | (d.target == b)
    X, y = d.data[mask], d.target[mask]
    y = np.where(y == a, -1, 1)
    return X, y


def _load_raw(name, seed=0):
    if name == "xor":
        return _make_xor(seed=seed)
    if name == "iris":
        d = load_iris()
        mask = d.target < 2  # classes 0 vs 1 -> binary, linearly-ish ok
        X, y = d.data[mask], d.target[mask]
        y = np.where(y == 0, -1, 1)
        return X, y
    if name == "wine":
        d = load_wine()
        mask = d.target < 2
        X, y = d.data[mask], d.target[mask]
        y = np.where(y == 0, -1, 1)
        return X, y
    if name == "wine_12":
        d = load_wine()
        mask = d.target > 0  # classes 1 vs 2 -- a different pair than "wine" (0 vs 1)
        X, y = d.data[mask], d.target[mask]
        y = np.where(y == 1, -1, 1)
        return X, y
    if name == "breast_cancer":
        d = load_breast_cancer()
        X, y = d.data, d.target
        y = np.where(y == 0, -1, 1)
        return X, y
    if name == "digits":
        return _digit_pair(3, 8)  # a genuinely confusable pair
    if name == "digits_01":
        return _digit_pair(0, 1)  # an easy, well-separated pair
    if name == "digits_69":
        return _digit_pair(6, 9)  # another confusable pair
    if name == "digits_45":
        return _digit_pair(4, 5)  # medium difficulty
    if name == "moons":
        X, y = make_moons(n_samples=200, noise=0.2, random_state=seed)
        y = np.where(y == 0, -1, 1)
        return X, y
    if name == "circles":
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=seed)
        y = np.where(y == 0, -1, 1)
        return X, y
    if name == "synth_easy":
        X, y = make_classification(
            n_samples=200, n_features=6, n_informative=4, n_redundant=0,
            class_sep=2.0, random_state=seed,
        )
        y = np.where(y == 0, -1, 1)
        return X, y
    if name == "synth_medium":
        X, y = make_classification(
            n_samples=200, n_features=6, n_informative=4, n_redundant=1,
            class_sep=1.0, random_state=seed,
        )
        y = np.where(y == 0, -1, 1)
        return X, y
    if name == "synth_hard":
        X, y = make_classification(
            n_samples=200, n_features=6, n_informative=3, n_redundant=2,
            class_sep=0.5, flip_y=0.05, random_state=seed,
        )
        y = np.where(y == 0, -1, 1)
        return X, y
    if name == "gaussian_quantiles":
        X, y = make_gaussian_quantiles(n_samples=200, n_features=4, n_classes=2, random_state=seed)
        y = np.where(y == 0, -1, 1)
        return X, y
    raise ValueError(f"Unknown dataset: {name}")


def load_dataset(name, n_qubits=4, test_size=0.3, seed=0, max_train=80, max_test=40):
    """
    Returns X_train, X_test, y_train, y_test  (y in {-1, +1})
    Features are angle-scaled to [-pi, pi], dimensionality == n_qubits
    (via PCA if the raw data has more features than n_qubits).

    max_train / max_test cap the size so O(n^2) kernel matrix computation
    with a quantum simulator stays fast during prototyping.
    """
    X, y = _load_raw(name, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Cap sizes for fast prototyping (kernel matrices are O(n^2) simulator calls)
    if len(X_train) > max_train:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_train, random_state=seed, stratify=y_train
        )
    if len(X_test) > max_test:
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=max_test, random_state=seed, stratify=y_test
        )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    if n_features > n_qubits:
        pca = PCA(n_components=n_qubits, random_state=seed).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    elif n_features < n_qubits:
        # pad with zeros so every dataset uses the same circuit width
        pad_tr = np.zeros((X_train.shape[0], n_qubits - n_features))
        pad_te = np.zeros((X_test.shape[0], n_qubits - n_features))
        X_train = np.hstack([X_train, pad_tr])
        X_test = np.hstack([X_test, pad_te])

    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(X_train)
    X_train = angle_scaler.transform(X_train)
    X_test = angle_scaler.transform(X_test)

    return X_train, X_test, y_train.astype(float), y_test.astype(float)


DATASET_NAMES = [
    "xor", "iris", "wine", "wine_12", "breast_cancer",
    "digits", "digits_01", "digits_69", "digits_45",
    "moons", "circles", "synth_easy", "synth_medium", "synth_hard",
    "gaussian_quantiles",
]

