# Cleaning the training data
import numpy as np


def clean_data(data):
    """
    Cleaning training data for negative numbers and
    bad values including water absorption regions.


    :param csv data: A N*M pandas data frame (pd.read_csv()) where N is observation and M is features
    :returns: X: The cleaned X (features) matrix
    :returns: y: The target values
    :returns: non_negative_columns&water_bands: Index of bad bands (removed)
    :returns: np.array wl: Cleaned wavelength
    """
    # Remove samples with no N measurments
    data = data[~np.isnan(data["nitrogen"])]
    y = data["nitrogen"].values
    # Select bands that there is no nan in spectral measurments
    spec = data.iloc[:, 1:]
    spec = spec[spec > 0]
    non_negative_columns = np.where(~np.any(np.isnan(spec), axis=0))[0]
    spec = spec.iloc[:, non_negative_columns]
    # Get the wavelength values
    wl = np.array(spec.columns.values)
    f = np.vectorize(float)
    wl = f(wl)
    water_abs1 = np.where((wl >= 1300) & (wl <= 1450))
    water_abs2 = np.where((wl >= 1750) & (wl <= 2000))
    water_bands = np.concatenate((water_abs1, water_abs2), axis=1)[0]
    # Remove bad bands
    wl = np.delete(wl, water_bands, 0)
    X = np.delete(spec.values, water_bands, 1)
    return (X, y, non_negative_columns, water_bands, wl)
