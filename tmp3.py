import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import xarray as xr
import rioxarray as rxr


def vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips


def simple_pls_cv(X, y, n_comp, scores=False):
    from sklearn.model_selection import cross_val_predict

    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X, y)
    if scores:
        y_c = pls.predict(X)
        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=10)
        # Calculate scores for calibration and cross-validation
        score_c = r2_score(y, y_c)
        score_cv = r2_score(y, y_cv)
        # Calculate mean square error for calibration and cross validation
        mse_c = mean_squared_error(y, y_c)
        mse_cv = mean_squared_error(y, y_cv)
        return pls, score_c, score_cv, mse_c, mse_cv, y_cv
    return pls


def pls_variable_selection(X, y, max_comp):
    from sklearn.model_selection import cross_val_predict
    from sys import stdout

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

        comp = 100 * (i + 1) / (max_comp)
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # # Calculate and print the position of minimum in MSE
    mseminx, mseminy = np.where(mse == np.min(mse[np.nonzero(mse)]))
    print("Optimised number of PLS components: ", mseminx[0] + 1)
    print("Wavelengths to be discarded ", mseminy[0])
    print("Optimised MSEP ", mse[mseminx, mseminy][0])
    stdout.write("\n")
    # plt.imshow(mse, interpolation=None)
    # plt.show()
    # Calculate PLS with optimal components and export values
    pls = PLSRegression(n_components=mseminx[0] + 1)
    pls.fit(X, y)

    sorted_ind = np.argsort(np.abs(pls.coef_[:, 0]))

    Xc = X[:, sorted_ind]
    return (Xc[:, mseminy[0] :], mseminx[0] + 1, mseminy[0], sorted_ind)


def plsr_vip(X, y, max_comp):
    mse = np.zeros(max_comp)
    idx_selected = []
    vip_values = np.zeros((max_comp, X.shape[1]))
    for i in range(max_comp):
        # pls1 = PLSRegression(n_components=i + 1)
        # pls1.fit(X, y)
        pls1 = simple_pls_cv(X, y, n_comp=i + 1, scores=False)
        vip_values[i, :] = vip(pls1)
        idx = np.where(vip_values[i, :] > 1)
        X_selected = np.squeeze(X[:, idx])
        pls_sel = simple_pls_cv(X_selected, y, n_comp=i + 1, scores=False)
        # pls2 = PLSRegression(n_components=i + 1)
        # pls2.fit(X_selected, y)
        y_cv = cross_val_predict(pls_sel, X_selected, y, cv=10)
        mse[i] = mean_squared_error(y, y_cv)
        idx_selected.append(idx)

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

    return (pls_opt, y_c, y_cv, score_c, score_cv, mse_c, mse_cv)


pls_vip, y_vip_c, y_vip_cv, r2_vip_c, r2_vip_cv, mse_vip_c, mse_vip_cv = plsr_vip(
    X, y, 15
)

opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X, y, max_comp=15)
pls_vs, r2_vs_c, r2_vs_cv, mse_vs_calib, mse_vs_cv, y_vs_cv = simple_pls_cv(
    opt_Xc, y, ncomp, scores=True
)

img = rxr.open_rasterio(in_dir + "data/raw/hyper_image.tif")
# img = rxr.open_rasterio(in_dir + "hyper_flightline.tif")
img["band"] = img["band"] - 1
img = img.isel(band=not_nan)
img = img.drop(bad_bands, dim="band")
img_sorted = img[sorted_ind, :, :]
img_selected = img_sorted[wav:, :, :]

nitrogen_map = predict(img_selected, pls_vs, "band")
plt.close()
nitrogen_map.plot()
plt.savefig(out_dir + "test.png")
# -------- Reading the data --------------
in_dir = "/data/home/hamiddashti/mnt/nasa_above/earthlab/"
out_dir = "/data/home/hamiddashti/mnt/nasa_above/earthlab/reports/figures/"
data = pd.read_csv(in_dir + "tmp.csv")
print(data)
# -------- Data cleaning -----------------
# Remove samples with no N measurments
data = data[~np.isnan(data["nitrogen"])]
# Select bands that there is no nan in spectral measurments
y = data["nitrogen"].values
spec = data.iloc[:, 1:]
spec = spec[spec > 0]
not_nan = np.where(~np.any(np.isnan(spec), axis=0))[0]
spec = spec.iloc[:, not_nan]

# Remove the water absorption bands
wl = np.array(spec.columns.values)
f = np.vectorize(float)
wl = f(wl)
water_abs1 = np.where((wl >= 1300) & (wl <= 1450))
water_abs2 = np.where((wl >= 1750) & (wl <= 2000))
bad_bands = np.concatenate((water_abs1, water_abs2), axis=1)[0]
wl = np.delete(wl, bad_bands, 0)
X = np.delete(spec.values, bad_bands, 1)

pls_opt, I, y_c, y_cv, score_c, score_cv, mse_c, mse_cv = plsr(X, y, 15)

img = rxr.open_rasterio(in_dir + "data/raw/hyper_image.tif")
# img = rxr.open_rasterio(in_dir + "hyper_flightline.tif")
img["band"] = img["band"] - 1
img = img.isel(band=not_nan)
img = img.drop(bad_bands, dim="band")
img = img.isel(band=np.squeeze(I))
# img = img.chunk(chunks={'x': 100, 'y': 200})
print(img)


def predict(X_m, model):
    X_m = X_m - model._x_mean
    X_m = X_m / model._x_std
    Ypred = np.dot(X_m, model.coef_)
    return Ypred + model._y_mean


def xr_predict(xdr, model, dim):
    pred = xr.apply_ufunc(
        predict, xdr, input_core_dims=[[dim]], kwargs={"model": model}, vectorize=True
    )
    return pred


nitrogen_map = predict(img, pls_opt, "band")

plt.close()
fig, ax = plt.subplots(figsize=(8, 9))
ax.plot(wl, X.T, color="gray")
[plt.axvline(x=wl[i], color="r", alpha=0.3, label="axvline - full height") for i in I]
plt.savefig(out_dir + "selected_bands.png")

plt.close()
nitrogen_map.plot.imshow()
plt.savefig(out_dir + "nitrogen_map.png")


def gdown_file(url, outname):
    import gdown

    gdown.download(url, outname, quiet=False)


url_tif = "https://drive.google.com/uc?id=1UOEeyzHW-h0el2Qzk1o7BiSsqT8f8ax2"
url_tfw = "https://drive.google.com/uc?id=1I3Ns7sQ4ETFVsYD6sEQXzR65xHtyFIfD"
gdown_file(url_tif, "./data/raw/hyper_image.tif")
gdown_file(url_tfw, "./data/raw/hyper_image.tfw")
