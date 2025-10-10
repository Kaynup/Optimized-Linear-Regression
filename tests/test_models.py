from OptLinearRegress.models import LinearRegressor

def test_fit_predict():
    X = [[1,2],[3,4],[5,6]]
    y = [1,2,3]

    model = LinearRegressor(alpha=1e-8)
    coef = model.fit(X, y)
    assert len(coef) == len(X[0])+1  # includes intercept

    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    for yp, yt in zip(y_pred, y):
        assert abs(yp-yt) < 1e-6

def test_coefficients():
    X = [[1,2],[3,4],[5,6]]
    y = [1,2,3]

    model = LinearRegressor()
    model.fit(X, y)
    coef = model.coefficients()
    assert isinstance(coef, list)
    assert len(coef) == len(X[0])+1
