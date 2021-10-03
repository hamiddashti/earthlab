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
    """PLSR regression with two different featuer selection

    - The first feature selection method (vip-model) is based on variable importance
    projection (VIP). The VIP function calculates the influence of each PLSR
    regression coefficient. VIP values greater than 1 are usally considered as
    important.
    (https://www.sciencedirect.com/science/article/pii/S0169743912001542)

    - The second feature selection method (vs-model) is based on discarding features
    that have small coefficients iteratively. The idea is to optimize for both
    numbers of components and features simultaneously.
    https://nirpyresearch.com/variable-selection-method-pls-python/


    """

    def __init__(self):
        self.model = None
        self.scores = None

    def simple_pls_cv(self, X, y, n_comp, scores=False):
        # Run PLS with suggested number of components
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X, y)
        # Calculate R2 and mean square error
        if scores:
            y_c = pls.predict(X)
            # Cross-validation
            y_cv = cross_val_predict(pls, X, y, cv=10)
            # Calculate r2 for calibration and cross-validation
            r2_c = r2_score(y, y_c)
            r2_cv = r2_score(y, y_cv)
            # Calculate mean square error for calibration and cross validation
            mse_c = mean_squared_error(y, y_c)
            mse_cv = mean_squared_error(y, y_cv)
            return pls, r2_c, r2_cv, mse_c, mse_cv, y_cv
        return pls

    def vip(self, model):
        """Calculate Variable Importance Projection

        There is a dicsuccion to infclude this into skit-learn
        https://github.com/scikit-learn/scikit-learn/issues/7050

        :argument sklearn-object model: The fitted plsr model
        """
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
        """fit PLSR based on vip feature selection

        :param np.array X: the feature matrix
        :param np.array y: the target matrix
        :param int max_comp: maximum components allowed for PLSR

        """
        # Initiate matrices for later propogation
        # Mean sqaure error
        mse = np.zeros(max_comp)
        # list of selected index for each component
        idx_selected = []
        # list of caculated VIPs for each component
        vip_values = np.zeros((max_comp, X.shape[1]))
        for i in range(max_comp):
            pls1 = self.simple_pls_cv(X, y, n_comp=i + 1, scores=False)
            vip_values[i, :] = self.vip(pls1)
            # Select VIP values greater than 1
            idx = np.where(vip_values[i, :] > 1)
            # select spectra accordingly
            X_selected = np.squeeze(X[:, idx])
            # fit plsr with selected bands
            pls_sel = self.simple_pls_cv(X_selected, y, n_comp=i + 1, scores=False)
            y_cv = cross_val_predict(pls_sel, X_selected, y, cv=10)
            mse[i] = mean_squared_error(y, y_cv)
            idx_selected.append(idx)
        # get the components and spectra that led to the minimum MSE
        opt_comp = np.argmin(mse) + 1
        I = idx_selected[np.argmin(mse)]
        X_opt = np.squeeze(X[:, I])
        pls_opt, r2_c, r2_cv, mse_c, mse_cv, y_cv = self.simple_pls_cv(
            X_opt, y, n_comp=opt_comp, scores=True
        )
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
        selected_index = sorted_ind[wav:]
        pls_opt, r2_c, r2_cv, mse_calib, mse_cv, y_cv = self.simple_pls_cv(
            opt_Xc, y, opt_ncomp, scores=True
        )
        self.model = pls_opt
        self.scores = [r2_c, r2_cv, mse_calib, mse_cv]
        self.y_cv = y_cv
        self.selected_index = selected_index

    @staticmethod
    def predict(X_m, model):
        """Apply plsr model to new observation

        :param np.array X_m: new features
        :returns: predicted target (e.g. nitrogen)
        """
        X_m = X_m - model._x_mean
        X_m = X_m / model._x_std
        Ypred = np.dot(X_m, model.coef_)
        return Ypred + model._y_mean

    @staticmethod
    def xr_predict(xdr, model, dim):
        """Wrapper around predict function to apply it to xarray objects

        :param xr.dataarray x_dr: the xarray image
        :param sklearn_obj model: the final plsr model
        :param str dim: dimension along which model should be applied (e.g. "band")
        """
        pred = xr.apply_ufunc(
            hyper_plsr.predict,
            xdr,
            input_core_dims=[[dim]],
            kwargs={"model": model},
            vectorize=True,
        )
        return pred
