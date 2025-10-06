import numpy as np

def add_intercept(X):
    X = np.asarray(X)
    intercept = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([intercept, X])

def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]

    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def batch_iterator(X, y, batch_size=32, shuffle=True, seed=None):
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]

def shuffle_arrays(X, y, seed=None):
    X = np.asarray(X)
    y = np.asarray(y)
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def standardize(X, mean=None, std=None):
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std_corrected = np.where(std == 0, 1, std)
    X_scaled = (X - mean) / std_corrected
    return X_scaled, mean, std_corrected
