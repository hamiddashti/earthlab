import numpy as np


def clean_data(data):
    # -------- Data cleaning -----------------
    # Remove samples with no N measurments
    data = data[~np.isnan(data["nitrogen"])]
    y = data["nitrogen"].values
    # Select bands that there is no nan in spectral measurments
    spec = data.iloc[:, 1:]
    spec = spec[spec > 0]
    non_negative_columns = np.where(~np.any(np.isnan(spec), axis=0))[0]
    spec = spec.iloc[:, non_negative_columns]
    wl = np.array(spec.columns.values)
    f = np.vectorize(float)
    wl = f(wl)
    water_abs1 = np.where((wl >= 1300) & (wl <= 1450))
    water_abs2 = np.where((wl >= 1750) & (wl <= 2000))
    water_bands = np.concatenate((water_abs1, water_abs2), axis=1)[0]
    wl = np.delete(wl, water_bands, 0)
    X = np.delete(spec.values, water_bands, 1)
    return (X, y, non_negative_columns, water_bands, wl)
