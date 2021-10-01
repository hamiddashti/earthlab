import numpy as np
import matplotlib.pylab as plt
from numpy.core.numeric import outer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import xarray as xr
import rioxarray as rxr
import pandas as pd


class hyper_plsr:
    def __init__(self):
        self.model = None
        self.scores = None

    def simple_pls_cv(self, X, y, n_comp, scores=False):
        # Run PLS with suggested number of components
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X, y)
        if scores:
            y_c = pls.predict(X)
            # Cross-validation
            y_cv = cross_val_predict(pls, X, y, cv=10)
            # Calculate scores for calibration and cross-validation
            r2_c = r2_score(y, y_c)
            r2_cv = r2_score(y, y_cv)
            # Calculate mean square error for calibration and cross validation
            mse_c = mean_squared_error(y, y_c)
            mse_cv = mean_squared_error(y, y_cv)
            return pls, r2_c, r2_cv, mse_c, mse_cv, y_cv
        return pls

    def vip(self, model):
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array(
                [(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)]
            )
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        return vips

    def plsr_vip(self, X, y, max_comp):
        mse = np.zeros(max_comp)
        idx_selected = []
        vip_values = np.zeros((max_comp, X.shape[1]))
        for i in range(max_comp):
            pls1 = self.simple_pls_cv(X, y, n_comp=i + 1, scores=False)
            vip_values[i, :] = self.vip(pls1)
            idx = np.where(vip_values[i, :] > 1)
            X_selected = np.squeeze(X[:, idx])
            pls_sel = self.simple_pls_cv(X_selected, y, n_comp=i + 1, scores=False)
            # pls2 = PLSRegression(n_components=i + 1)
            # pls2.fit(X_selected, y)
            y_cv = cross_val_predict(pls_sel, X_selected, y, cv=10)
            mse[i] = mean_squared_error(y, y_cv)
            idx_selected.append(idx)
        opt_comp = np.argmin(mse) + 1
        I = idx_selected[np.argmin(mse)]
        X_opt = np.squeeze(X[:, I])
        pls_opt, r2_c, r2_cv, mse_c, mse_cv, y_cv = self.simple_pls_cv(
            X_opt, y, n_comp=opt_comp, scores=True
        )
        # pls_opt = PLSRegression(n_components=opt_comp)
        # pls_opt.fit(X_opt, y)
        # y_c = pls_opt.predict(X_opt)
        # y_cv = cross_val_predict(pls_opt, X_opt, y, cv=10)
        # # Calculate scores for calibration and cross-validation
        # r2_c = r2_score(y, y_c)
        # r2_cv = r2_score(y, y_cv)
        # mse_c = mean_squared_error(y, y_c)
        # mse_cv = mean_squared_error(y, y_cv)
        self.model = pls_opt
        self.scores = [r2_c, r2_cv, mse_c, mse_cv]
        self.selected_index = I
        self.y_cv = y_cv

    def plsr_vs(self, X, y, max_comp):
        # Define MSE array to be populated
        mse = np.zeros((max_comp, X.shape[1]))
        # Loop over the number of PLS components
        for i in range(max_comp):
            # Regression with specified number of components, using full spectrum
            pls1 = PLSRegression(n_components=i + 1)
            pls1.fit(X, y)
            # Indices of sort spectra according to ascending absolute value of PLS coefficients
            sorted_ind = np.argsort(np.abs(pls1.coef_[:, 0]))
            # Sort spectra accordingly
            Xc = X[:, sorted_ind]
            # Discard one wavelength at a time of the sorted spectra,
            # regress, and calculate the MSE cross-validation
            for j in range(Xc.shape[1] - (i + 1)):
                pls2 = PLSRegression(n_components=i + 1)
                pls2.fit(Xc[:, j:], y)
                y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=5)
                mse[i, j] = mean_squared_error(y, y_cv)

        # # Calculate and print the position of minimum in MSE
        mseminx, mseminy = np.where(mse == np.min(mse[np.nonzero(mse)]))

        # Calculate PLS with optimal components and export values
        pls = PLSRegression(n_components=mseminx[0] + 1)
        pls.fit(X, y)
        sorted_ind = np.argsort(np.abs(pls.coef_[:, 0]))
        Xc = X[:, sorted_ind]
        opt_Xc = Xc[:, mseminy[0] :]
        opt_ncomp = mseminx[0] + 1
        wav = mseminy[0]
        sorted_ind = sorted_ind
        # selected_index = sorted_ind[wav:]
        pls_opt, r2_c, r2_cv, mse_calib, mse_cv, y_cv = self.simple_pls_cv(
            opt_Xc, y, opt_ncomp, scores=True
        )
        self.model = pls_opt
        self.scores = [r2_c, r2_cv, mse_calib, mse_cv]
        self.y_cv = y_cv
        self.sorted_ind = sorted_ind
        self.wav = wav

    @staticmethod
    def predict(X_m, model):
        X_m = X_m - model._x_mean
        X_m = X_m / model._x_std
        Ypred = np.dot(X_m, model.coef_)
        return Ypred + model._y_mean

    @staticmethod
    def xr_predict(xdr, model, dim):
        pred = xr.apply_ufunc(
            hyper_plsr.predict,
            xdr,
            input_core_dims=[[dim]],
            kwargs={"model": model},
            vectorize=True,
        )
        return pred
