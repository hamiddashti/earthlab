def plsr(X, y, max_comp):
    import numpy as np 
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict
    mse = np.zeros(max_comp)
    idx_selected = []
    for i in range(max_comp):
        pls1 = PLSRegression(n_components=i + 1)
        pls1.fit(X, y)
        ths = np.mean(np.abs(pls1.coef_)) + (np.std(np.abs(pls1.coef_)))
        idx = np.argwhere(np.abs(np.squeeze(pls1.coef_)) > ths)
        X_selected = np.squeeze(X[:, idx])
        pls2 = PLSRegression(n_components=i + 1)
        pls2.fit(X_selected, y)
        y_cv = cross_val_predict(pls2, X_selected, y, cv=10)
        mse[i] = mean_squared_error(y, y_cv)
        idx_selected.append(idx)
    mse_min = np.min(mse)
    opt_comp = np.argmin(mse) + 1
    I = idx_selected[np.argmin(mse)]
    X_opt = np.squeeze(X[:, I])
    pls_opt = PLSRegression(n_components=opt_comp)
    pls_opt.fit(X_opt, y)
    y_c = pls_opt.predict(X_opt)
    y_cv = cross_val_predict(pls_opt, X_opt, y, cv=10)
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    return (pls_opt, I, y_c, y_cv, score_c, score_cv, mse_c, mse_cv)


def _predict(X_m, model):
    import numpy as np 
    X_m = X_m - model._x_mean
    X_m = X_m / model._x_std
    Ypred = np.dot(X_m, model.coef_)
    return Ypred + model._y_mean


def predict(xdr, model, dim):
    import xarray as xr 
    pred = xr.apply_ufunc(_predict,
                          xdr,
                          input_core_dims=[[dim]],
                          kwargs={"model": model},
                          vectorize=True)
    return pred


def gdown_file(url, outname):
    import gdown
    gdown.download(url, outname, quiet=False)
