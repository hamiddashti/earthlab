import numpy as np

import matplotlib.pylab as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import xarray as xr
import rioxarray as rxr

import main_funs
import pandas as pd

# -------- Reading the data --------------
data = pd.read_csv("./tmp.csv")
print(f"\n\n {data.shape[0]-1} observations and {data.shape[1]-1} spectral bands.")

class hyper_plsr:

    def __init__(self):
        self.model = None 
        self.sel_index = None
        self.scores = None 


    @staticmethod
    def clean_data(data):
        # -------- Data cleaning -----------------
        # Remove samples with no N measurments
        data = data[~np.isnan(data["nitrogen"])]
        # Select bands that there is no nan in spectral measurments
        spec = data.iloc[:, 1:]
        spec = spec[spec > 0]
        not_nan = np.where(~np.any(np.isnan(spec), axis=0))[0]
        cleaned_data = spec.iloc[:, not_nan]
        wl = np.array(cleaned_data.columns.values)
        f = np.vectorize(float)
        wl = f(wl)
        water_abs1 = np.where((wl >= 1300) & (wl <= 1450))
        water_abs2 = np.where((wl >= 1750) & (wl <= 2000))
        bad_bands = np.concatenate((water_abs1, water_abs2), axis=1)[0]
        wl = np.delete(wl, bad_bands, 0)
        X = np.delete(cleaned_data.values, bad_bands, 1)
        return (X)
    
   
    def plsr_fit(self,X, y, max_comp):
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
        opt_comp = np.argmin(mse) + 1
        sel_index = idx_selected[np.argmin(mse)]
        X_opt = np.squeeze(X[:, sel_index])
        pls_opt = PLSRegression(n_components=opt_comp)
        pls_opt.fit(X_opt, y)
        y_c = pls_opt.predict(X_opt)
        y_cv = cross_val_predict(pls_opt, X_opt, y, cv=10)
        # Calculate scores for calibration and cross-validation
        r2_c = r2_score(y, y_c)
        r2_cv = r2_score(y, y_cv)
        mse_c = mean_squared_error(y, y_c)
        mse_cv = mean_squared_error(y, y_cv)
        self.model = pls_opt 
        self.scores = [r2_c,r2_cv,mse_c,mse_cv]
        self.sel_index = sel_index 


ml = hyper_plsr()
ml.plsr_fit(X,y,max_comp=15)

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
y = data["nitrogen"].values

pls_opt, I, y_c, y_cv, score_c, score_cv, mse_c, mse_cv = plsr(X, y, 15)

img = rxr.open_rasterio("./data/hyper_image.tif")
# img = rxr.open_rasterio(in_dir + "hyper_flightline.tif")
img["band"] = img["band"] - 1
img = img.isel(band=not_nan)
img = img.drop(bad_bands, dim="band")
img = img.isel(band=np.squeeze(I))
# img = img.chunk(chunks={'x': 100, 'y': 200})
print(img)


nitrogen_map = predict(img, pls_opt, "band")

plt.close()
fig, ax = plt.subplots(figsize=(8, 9))
ax.plot(wl, X.T, color="gray")
[plt.axvline(x=wl[i], color="r", alpha=0.3, label="axvline - full height") for i in I]
plt.savefig("./reports/figures/selected_bands.png")

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


def plsr(X, y, max_comp):
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
    X_m = X_m - model._x_mean
    X_m = X_m / model._x_std
    Ypred = np.dot(X_m, model.coef_)
    return Ypred + model._y_mean


def predict(xdr, model, dim):
    pred = xr.apply_ufunc(
        _predict, xdr, input_core_dims=[[dim]], kwargs={"model": model}, vectorize=True
    )
    return pred
