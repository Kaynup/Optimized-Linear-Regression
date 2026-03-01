import math

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n if n > 0 else 0.0

def mean_absolute_error(y_true, y_pred):
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n if n > 0 else 0.0

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    n = len(y_true)
    if n == 0: return 0.0
    mean_y_true = sum(y_true) / n
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((y_true[i] - mean_y_true) ** 2 for i in range(n))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def explained_variance_score(y_true, y_pred):
    n = len(y_true)
    if n == 0: return 0.0
    diff = [y_true[i] - y_pred[i] for i in range(n)]
    mean_diff = sum(diff) / n
    var_res = sum((d - mean_diff) ** 2 for d in diff) / n
    
    mean_y_true = sum(y_true) / n
    var_true = sum((y - mean_y_true) ** 2 for y in y_true) / n
    
    return 1 - var_res / var_true if var_true > 0 else 0.0
