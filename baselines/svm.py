"""baselines/svm.py -- classical baselines (Phase 12 of the plan)."""

from sklearn.svm import SVC


def classical_baselines(X_train, y_train, X_test, y_test):
    results = {}
    for name, kernel in [("linear_svm", "linear"), ("rbf_svm", "rbf"), ("poly_svm", "poly")]:
        clf = SVC(kernel=kernel)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results[name] = acc
    return results


def precomputed_kernel_svm_accuracy(K_train, y_train, K_test, y_test):
    """K_train: (n_train, n_train) train-train kernel. K_test: (n_test, n_train) test-train kernel."""
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    return clf.score(K_test, y_test)
