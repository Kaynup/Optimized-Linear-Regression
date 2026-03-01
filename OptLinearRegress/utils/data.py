import random
import math

def add_intercept(X):
    # X is expected to be a list of lists (2D array)
    return [[1.0] + list(row) for row in X]

def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    n_samples = len(X)
    
    indices = list(range(n_samples))
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        
    split_idx = int(n_samples * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    
    return X_train, y_train, X_test, y_test

def batch_iterator(X, y, batch_size=32, shuffle=True, seed=None):
    n_samples = len(X)
    indices = list(range(n_samples))
    
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        
    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        X_batch = [X[i] for i in batch_idx]
        y_batch = [y[i] for i in batch_idx]
        yield X_batch, y_batch

def shuffle_arrays(X, y, seed=None):
    n_samples = len(X)
    indices = list(range(n_samples))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)
    
    X_shuffled = [X[i] for i in indices]
    y_shuffled = [y[i] for i in indices]
    return X_shuffled, y_shuffled

def standardize(X, mean=None, std=None):
    n_samples = len(X)
    if n_samples == 0:
        return X, mean, std
        
    n_features = len(X[0])
    
    if mean is None:
        mean = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        
    if std is None:
        std = []
        for j in range(n_features):
            variance = sum((X[i][j] - mean[j]) ** 2 for i in range(n_samples)) / n_samples
            std.append(math.sqrt(variance))
            
    std_corrected = [s if s != 0 else 1.0 for s in std]
    
    X_scaled = []
    for i in range(n_samples):
        row = [(X[i][j] - mean[j]) / std_corrected[j] for j in range(n_features)]
        X_scaled.append(row)
        
    return X_scaled, mean, std_corrected
