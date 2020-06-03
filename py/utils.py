"""Utils for Peru soil moisture analysis.

Todo
----
- replace pickle, e.g. by csv files

"""
import datetime
import glob
import itertools
import os
import pickle

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import sklearn.metrics as me
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm


def getHyperspectralBands(preprocessed=False):
    """Get list of hyperspectral bands.

    Parameters
    ----------
    preprocessed : bool, optional (default=False)
        If False, all bands are included.
        If True, only bands which are included after pre-processing are
        included.

    Returns
    -------
    bands : np.array of float
        List of bands.

    """
    bands = np.array([
        888.219, 897.796, 907.373, 916.95, 926.527, 936.104, 945.681, 955.257,
        964.834, 974.411, 983.988, 993.565, 1003.14, 1012.72, 1022.3, 1031.87,
        1041.45, 1051.03, 1060.6, 1070.18, 1079.76, 1089.33, 1098.91, 1108.49,
        1118.06, 1127.64, 1137.22, 1146.8, 1156.37, 1165.95, 1175.53, 1185.1,
        1194.68, 1204.26, 1213.83, 1223.41, 1232.99, 1242.56, 1252.14, 1261.72,
        1271.29, 1280.87, 1290.45, 1300.03, 1309.6, 1319.18, 1328.76, 1338.33,
        1347.91, 1357.49, 1367.06, 1376.64, 1386.22, 1395.79, 1405.37, 1414.95,
        1424.53, 1434.1, 1443.68, 1453.26, 1462.83, 1472.41, 1481.99, 1491.56,
        1501.14, 1510.72, 1520.29, 1529.87, 1539.45, 1549.02, 1558.6, 1568.18,
        1577.76, 1587.33, 1596.91, 1606.49, 1616.06, 1625.64, 1635.22, 1644.79,
        1654.37, 1663.95, 1673.52, 1683.1, 1692.68, 1702.25, 1711.83, 1721.41,
        1730.99, 1740.56, 1750.14, 1759.72, 1769.29, 1778.87, 1788.45, 1798.02,
        1807.6, 1817.18, 1826.75, 1836.33, 1845.91, 1855.49, 1865.06, 1874.64,
        1884.22, 1893.79, 1903.37, 1912.95, 1922.52, 1932.1, 1941.68, 1951.25,
        1960.83, 1970.41, 1979.98, 1989.56, 1999.14, 2008.72, 2018.29, 2027.87,
        2037.45, 2047.02, 2056.6, 2066.18, 2075.75, 2085.33, 2094.91, 2104.48,
        2114.06, 2123.64, 2133.21, 2142.79, 2152.37, 2161.95, 2171.52, 2181.1,
        2190.68, 2200.25, 2209.83, 2219.41, 2228.98, 2238.56, 2248.14, 2257.71,
        2267.29, 2276.87, 2286.45, 2296.02, 2305.6, 2315.18, 2324.75, 2334.33,
        2343.91, 2353.48, 2363.06, 2372.64, 2382.21, 2391.79, 2401.37, 2410.94,
        2420.52, 2430.1, 2439.68, 2449.25, 2458.83, 2468.41, 2477.98, 2487.56,
        2497.14, 2506.71])

    if preprocessed:
        ignored_bands = getIgnoredBands()
        return bands[~np.isin(np.arange(len(bands)), ignored_bands)]

    return bands


def getIgnoredBands():
    """Get list of indices of all ignored bands.

    Returns
    -------
    np.array of int
        List of indices of the ignored hyperspectral bands.

    """
    return np.array([0, 1, 49, 50, 51, 52, 53, 98, 99, 100, 101, 102, 103, 104,
                     105, 106, 107, 108, 109, 110, 111, 153, 154, 155, 156,
                     157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
                     168, 169])


def processHeadwallData(data, verbose=0):
    """Pre-process Headwall hyperspectral images (170 to 133 bands).

    Parameters
    ----------
    data : np.array
        Raw hyperspectral image with 170 bands. Shape (pixel, bands)
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    data : np.array
        Pre-processed hyperspectral image with 146 bands.

    """
    if data.shape[1] < 170:
        print("Error: Array already pre-processed.")
        return data

    # remove bands
    if verbose:
        print("| Shape with all bands:", data.shape)
    ignore_bands = getIgnoredBands()
    mask = np.ones(data.shape[1], np.bool)
    mask[ignore_bands] = 0
    data = data[:, mask]
    if verbose:
        print("| Shape after band removal:", data.shape)

    # cap reflectance data to 1
    data[data > 1.0] = 1.0

    return data


def trainAutoencoder(data_pre_gen, epochs=20,
                     bottleneck=5, title="", verbose=0):
    """Train autoencoder on Headwall hyperspectral data.

    Parameters
    ----------
    data_pre_gen : generator
        Data generator with pre-processed data
    epochs : int
        Number of epochs for the Autoencoder training
    bottleneck : int
        Number of neurons at the bottleneck layer of the Autoencoder
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    autoencoder : Keras model
        Trained model of the Autoencoder
    encoder : Keras model
        Trained model of the encoder
    ae_weights_path : str
        Path to the trained Autoencoder weights
    en_weights_path : str
        Path to the trained encoder weights

    """
    run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    n_features = 132    # TODO refactor

    inp = Input(shape=(n_features))
    encoded = Dense(bottleneck, activation="relu")(inp)
    decoded = Dense(n_features, activation="linear")(encoded)

    autoencoder = Model(input, decoded)
    encoder = Model(input, encoded)

    earlystopping = EarlyStopping(
        monitor="loss", mode="min", patience=20)

    autoencoder.compile(optimizer='nadam', loss='mean_squared_error')
    autoencoder.fit_generator(data_pre_gen, epochs=epochs,
                              steps_per_epoch=100, shuffle=True,
                              verbose=verbose, callbacks=[earlystopping])

    ae_weights_path = "data/models/autoencoder_"+str(title)+"_"+run+".h5"
    autoencoder.save(ae_weights_path)
    en_weights_path = "data/models/encoder_"+str(title)+"_"+run+".h5"
    encoder.save(en_weights_path)

    return autoencoder, encoder, ae_weights_path, en_weights_path


def getPixelsAroundMeasurement(x, y, n_pixels):
    """Get `n_pixels` pixel indices around one center pixels.

    Parameters
    ----------
    x : int
        X-coordinate from 0 to width-1.
    y : int
        Y-coordinate from 0 to height-1.
    n_pixels : int
        Number of pixels around center pixel.

    Returns
    -------
    pixel_list : list of tuples (int, int)
        List of pixels around center pixel.

    """
    neighborhood = itertools.product(
        range(-n_pixels, n_pixels+1),
        range(-n_pixels, n_pixels+1))

    pixel_list = [(x+z[0], y+z[1]) for z in neighborhood]

    return pixel_list


def getTrainingData(gdf, tif_file, n_pixels_nbh=4, verbose=0):
    """Get training data.

    Remember: We expect to have missing points per tif_file, since we sometimes
    have several tif_images for one field.

    Parameters
    ----------
    gdf : Geopandas DataFrame
        Soil moisture data
    tif_file : tif image
        Hyperspectral image
    n_pixels_nbh : int, optional (default=4)
        Number of pixels in neighborhood of soil moisture measurement. For
        example, `n_pixels_nbh=4` means that 9 x 9 pixels are used.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    X
    y
    n_missing_points :

    """
    if verbose:
        print("| GDF before transformation", gdf.crs)
        print(gdf.head())

    # transform to similar crs
    gdf = gdf.to_crs(tif_file.crs.to_dict())
    if verbose:
        print("| GDF after transformation", gdf.crs)
        print("| tif-file", list(tif_file.xy(tif_file.height // 2,
                                             tif_file.width // 2)))

    X = []
    y = []
    n_missing_points = 0
    for i, row in gdf.iterrows():
        point_x = row["geometry"].x
        point_y = row["geometry"].y
        index_x, index_y = tif_file.index(point_x, point_y)

        if verbose > 1:
            print("| Point", i, ":", point_x, point_y, "|", index_x, index_y)

        pixel_list = getPixelsAroundMeasurement(index_x, index_y, n_pixels_nbh)
        spectra_temp = []
        for p in pixel_list:
            p_x, p_y = tif_file.xy(p[0], p[1])
            spectrum_temp = list(tif_file.sample([(p_x, p_y)]))[0]
            if np.sum(spectrum_temp) > 0.:
                spectra_temp.append(spectrum_temp)

        if not spectra_temp:
            n_missing_points += 1
            continue

        # get median spectrum
        spectrum = np.median(spectra_temp, axis=0)

        soilmoisture = row["soilmoistu"]
        if verbose > 1:
            print("| Soilmoisture:", soilmoisture)
        X.append(spectrum)
        y.append(soilmoisture)

    X = np.array(X)
    y = np.array(y)

    return X, y, n_missing_points


def preprocessingPixelHeadwall(x, area=None):
    """Preprocess Headwall pixel with 146 bands.

    Calculate `mean` and `std` with `calcPreprocessingStats()`

    Parameters
    ----------
    x : list or array
        Original input pixel(s) with 146 spectral bands
    area : str or None, optional (default=None)
        Name of measurement area as str. If `None`, all areas are used.

    """
    # get mean and std
    file_path = "data/scaling/"
    if area is None:
        file_path += "area_all"
    else:
        file_path += "area_"+str(area)
    mean = pickle.load(open(file_path+"_mean.p", "rb"))
    std = pickle.load(open(file_path+"_std.p", "rb"))

    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x


def calcPreprocessingStats(area=None):
    """Calculate mean and std for preprocessingPixelHeadwall().

    Parameters
    ----------
    area : str or None, optional (default=None)
        Name of measurement area as str. If `None`, all areas are used.

    """
    if area is None:
        files = glob.glob("data/hyp_data_pre_area*.p")
    elif isinstance(area, str):
        files = glob.glob("data/hyp_data_pre_area"+str(area)+"*.p")

    X = []
    for f in files:
        a = pickle.load(open(f, "rb"))
        X.append(a)
    X = np.concatenate(X)

    np.set_printoptions(suppress=True)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.   # fix for 4 bands in area "5a"

    # save
    output_path = "data/scaling/"
    os.makedirs(output_path, exist_ok=True)
    if area is None:
        output_path += "area_all"
    else:
        output_path += "area_"+str(area)

    pickle.dump(list(mean), open(output_path+"_mean.p", "wb"))
    pickle.dump(list(std), open(output_path+"_std.p", "wb"))
    # return list(mean), list(std)


def processImages(hyp_path: str,
                  ref_path: str,
                  areas: list = ["1", "2_1", "2_2", "3", "4", "5"],
                  return_mode: str = "pickle",
                  verbose=0):
    """Process hyperspectral images into training data and full image data.

    For every area in `areas`, the following routine is performed:
        1. Load tif file from that area.
        2. Load reference data from that area
        3. Extract training data with `getTrainingData()`
        4. Mask full image.
        5. Pre-processing with `processHeadwallData()`

    In `pickle` mode:
        6. Save training data for each area and save full image split up into
        chunks for each area.

    In `return` mode:
        6. Return training data and full image.

    Parameters
    ----------
    hyp_path : str
        Path to the hyperspectral images.
    ref_path : str
        Path to the shape files.
    areas : list of str, optional
        List of areas. The full list is `["1", "2_1", "2_2", "3", "4", "5"]`.
    return_mode : str, optional (default="pickle")
        Defines what to do with the data.
            - "pickle": Saves the data in pickle binary files.
            - "return": Returns the files.
            - "csv": Saves csv file.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    hyp_data_pre_list : list
        Full hyperspectral data
    X_list : list
        Training data (features).
    y_list : list
        Training data (labels).

    """
    output_path = "data/processed/"
    hyp_data_pre_list = []
    X_list = []
    y_list = []
    df_list = []
    hypbands = getHyperspectralBands(True)

    for area in areas:
        f = hyp_path + "Area" + area + "_multi_or_rf"
        print("-"*40)
        print("Area:", area)

        if verbose:
            print(f)

        # load hyperspectral data
        tif_file = rasterio.open(f)
        hyp_data = tif_file.read()
        if verbose:
            print("| Hyp data:", tif_file.shape, tif_file.crs)

        # load reference data
        gdf = geopandas.read_file(
            ref_path+"peru_soilmoisture_area"+str(area[0])+".shp")

        # get training dataset
        X, y, n_missing_points = getTrainingData(
            gdf, tif_file, verbose=verbose)
        print("| Missing points: {0} of {1}".format(
            n_missing_points, gdf.shape[0]))
        if n_missing_points == gdf.shape[0]:
            print("| No data.")
            continue

        # calculate image mask & create list of all (!) pixels
        if hyp_data.shape[0] == 170:
            hyp_data = hyp_data.transpose(1, 2, 0)
        width, height, n_features = hyp_data.shape
        hyp_data_list = hyp_data.reshape((width*height, n_features))
        image_mask = np.argwhere(np.mean(hyp_data_list, axis=1) != 0).flatten()
        # print("There are {0:.2f} % non-zero pixels.".format(
        #     np.count_nonzero(image_mask)/hyp_data_list.shape[0]*100))

        # pre-processing of training data X
        X = processHeadwallData(X, verbose=verbose)

        # pre-processing of full (masked) dataset hyp_data_list
        hyp_data_pre = processHeadwallData(
            hyp_data_list[image_mask], verbose=verbose)

        X_list.append(X)
        y_list.append(y)
        if return_mode == "return":
            hyp_data_pre_list.append(hyp_data_pre)

        elif return_mode == "pickle":
            hyp_data_pre_chunks = np.array_split(
                hyp_data_pre, hyp_data_pre.shape[0]//1e5)
            for i, c in enumerate(tqdm(hyp_data_pre_chunks)):
                pickle.dump(c, open(
                    output_path+"hyp_data_pre_area"+str(area)+"_"+str(i)+".p",
                    "wb"))
            pickle.dump(X, open(output_path+"area"+str(area)+"_X.p", "wb"))
            pickle.dump(y, open(output_path+"area"+str(area)+"_y.p", "wb"))
            print("| Saved all files for area", area)

        elif return_mode == "csv":
            df_temp = pd.DataFrame(X, columns=hypbands)
            df_temp["soilmoisture"] = y
            df_temp["area"] = area
            df_list.append(df_temp)

    # return
    if return_mode == "return":
        return hyp_data_pre_list, X_list, y_list

    # save training data in pickle files
    if return_mode == "pickle":
        X_list = np.concatenate(X_list)
        y_list = np.concatenate(y_list)
        pickle.dump(X_list, open(output_path+"X_list.p", "wb"))
        pickle.dump(y_list, open(output_path+"y_list.p", "wb"))

    # save data to csv
    elif return_mode == "csv":
        df = pd.concat(df_list, axis=0, ignore_index=True)
        df.to_csv(output_path+"peru_data.csv")


def genHypDataPre(area=None, scaling=False):
    """Generate hyperspectral data with optional pre-processing.

    Parameters
    ----------
    area : str or None, optional (default=None)
        Name of measurement area as str. If `None`, all areas are used.
    scaling : bool, optional (default=False)
        If True, the data is scaled (mean 0, std 1).

    Yields
    -------
    a : np.array
        Hyperspectral training data.

    """
    if area is None:
        files = glob.glob("data/processed/hyp_data_pre_area*.p")
    else:
        files = glob.glob("data/processed/hyp_data_pre_area"+str(area)+"*.p")

    while True:
        f = np.random.choice(files, size=1, replace=False)[0]
        a = pickle.load(open(f, "rb"))
        if scaling:
            a = preprocessingPixelHeadwall(a)
        yield a, a


def getHypDataPre(area=None, scaling=False):
    """Get hyperspectral data with optional pre-processing.

    Parameters
    ----------
    area : str or None, optional (default=None)
        Name of measurement area as str. If `None`, all areas are used.
    scaling : bool, optional (default=False)
        If True, the data is scaled (mean 0, std 1).

    Returns
    -------
    output : np.array
        Hyperspectral training data.

    """
    if area is None:
        files = glob.glob("data/processed/hyp_data_pre_area*.p")
    else:
        files = glob.glob("data/processed/hyp_data_pre_area"+str(area)+"*.p")

    output = []
    for f in files:
        a = pickle.load(open(f, "rb"))
        if scaling:
            a = preprocessingPixelHeadwall(a)
        output.append(a)
    output = np.stack(a)
    return output


def getEncoder(area, dim_red_mode="autoencoder"):
    """Load encoder by `area` and `mode`.

    Parameters
    ----------
    area : str
        Measurement area.
    dim_red_mode : str, optional (default="autoencoder")
        Type of dimensionality reduction, from {None, "PCA", "autoencoder"}.

    Returns
    -------
    keras model or pca model
        Encoder model, either PCA or Autoencoder.

    """
    if dim_red_mode == "autoencoder":
        encoders = {
            "1": "encoder_120191017_071814.h5",
            "2_1": "encoder_2_120191017_073022.h5",
            "2_2": "encoder_2_220191017_074235.h5",
            "3_2": "encoder_3_220191017_075443.h5",
            "4n": "encoder_4n20191017_080652.h5",
            "5a": "encoder_5a20191017_081906.h5",
            "5b": "encoder_5b20191017_083129.h5",
        }
        encoder = load_model("data/models/"+encoders[area], compile=False)

    elif dim_red_mode == "pca":
        encoders = {
            '1': 'pca_1_20191115_020050.p',
            '2_1': 'pca_2_1_20191115_020100.p',
            '2_2': 'pca_2_2_20191115_020116.p',
            '3_2': 'pca_3_2_20191115_020121.p',
            '4n': 'pca_4n_20191115_020130.p',
            '5a': 'pca_5a_20191115_020136.p',
            '5b': 'pca_5b_20191115_020142.p',
        }
        encoder = pickle.load(open("data/models/"+encoders[area], "rb"))

    return encoder


def trainAutoencoderPerArea(areas, bottleneck=5):
    """Train autoencoder for every area.

    Parameters
    ----------
    areas : list of str
        List of measurement areas.
    bottleneck : int, optional (default=5)
        Number of components of the bottleneck layer.

    """
    for area in areas:
        print("Area", area)
        _, _, _, en_path = trainAutoencoder(
            data_pre_gen=genHypDataPre(area=area), verbose=0,
            epochs=30, bottleneck=bottleneck, title=area)

        print("Trained encoder at", en_path)


def trainPCAPerArea(areas, n_components=10):
    """Train principal component analysis (PCA) for every area.

    Parameters
    ----------
    areas : list of str
        List of measurement areas.
    n_components : int, optional (default=10)
        Number of principal components to be included.

    """
    for area in areas:
        pca_path = trainPCA(
            data=getHypDataPre(area=area),
            n_components=n_components,
            title=area)
        print("'{area}': '{filename}',".format(
            area=area, filename=pca_path.split("/")[1]))


def trainPCA(data, n_components, title, verbose=0):
    """Train principal component analysis (PCA).

    Parameters
    ----------
    data : np.array
        Hyperspectral training data in the shape (n_datapoints, n_bands).
    n_components : int
        Number of principal components to be included.
    title : str
        Title of the saved model file.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    pca_path : str
        Path to trained PCA model.

    """
    run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # train PCA
    pca = PCA(n_components=n_components)
    pca.fit(data)

    # save PCA
    pca_path = "data/models/pca_"+str(title)+"_"+run+".p"
    pickle.dump(pca, open(pca_path, "wb"))

    if verbose:
        print("PCA path:", pca_path)

    return pca_path


def loadDataForSemiEstimation(area, scaling=False, max_sm=25., verbose=0):
    """Load data for estimation of semi-supervised learning.

    Parameters
    ----------
    area : str or list
        Area(s) to be loaded for estimation.
    scaling : bool, optional (default=False)
        If True, the data is scaled (mean 0, std 1).
    max_sm : float, optional (default=25)
        Maximum soil moisture value.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    X_train_semi : np.array
        Hyperspectral data with unlabeled and labeled data.
    X_test : np.array
        Hyperspectral data with labeled data.
    y_train : np.array
        Soil moisture data with real labels between 0-1 and dummy labels -1.
    y_test : np.array
        Soil moisture data with real labels between 0-1.

    """
    X, _, y, _ = loadDataForEstimation(
        area=area, dim_red_mode=None, scaling=scaling, max_sm=max_sm,
        verbose=verbose)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=0)

    X_full = getHypDataPre(area=area, scaling=scaling)

    X_train_semi = np.copy(X_full)
    y_train_semi = np.full(shape=(X_full.shape[0], ), fill_value=-1)

    X_train_semi = np.concatenate([X_train_semi, X_train], axis=0)
    y_train_semi = np.concatenate([y_train_semi, y_train], axis=0)

    return X_train_semi, X_test, y_train_semi, y_test


def loadDataForEstimation(area, dim_red_mode, scaling=False, max_sm=25.,
                          verbose=0):
    """Load data for supervised estimation.

    Parameters
    ----------
    area : str or list
        Area(s) to be loaded for estimation.
    dim_red_mode : str
        Type of dimensionality reduction, from {None, "PCA", "autoencoder"}.
    scaling : bool, optional (default=False)
        If True, the data is scaled (mean 0, std 1).
    max_sm : float, optional (default=25)
        Maximum soil moisture value.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    X_train, X_test : np.array
        Hyperspectral data with a shape (n_datapoints, n_bands).
    y_train, y_test : np.array
        Soil moisture data with a shape (n_datapoints, ).

    """
    # get data
    if isinstance(area, list):
        X_list = []
        y_list = []
        for a in area:
            X_list.append(pickle.load(
                open("data/processed/area"+str(a)+"_X.p", "rb")))
            y_list.append(pickle.load(
                open("data/processed/area"+str(a)+"_y.p", "rb")))
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

    else:
        X = pickle.load(open("data/processed/area"+str(area)+"_X.p", "rb"))
        y = pickle.load(open("data/processed/area"+str(area)+"_y.p", "rb"))

    # pre-processing
    if scaling:
        X = preprocessingPixelHeadwall(X)
    if verbose:
        print("| X, y:", X.shape, y.shape)

    # removal of outliers
    valid_indices = np.argwhere(y < max_sm)[:, 0]
    X = X[valid_indices]
    y = y[valid_indices]
    if verbose:
        print("| After outlier removal:", X.shape, y.shape)

    # encoding with encoder
    if dim_red_mode != None:
        encoder = getEncoder(area, dim_red_mode=dim_red_mode)
        if dim_red_mode == "autoencoder":
            X = encoder.predict(X)
        elif dim_red_mode == "pca":
            X = encoder.transform(X)

    # "split" data
    X_train = X
    X_test = X
    y_train = y
    y_test = y
    if verbose:
        print("| Split:", X_train.shape, X_test.shape, y_train.shape,
              y_test.shape)

    return X_train, X_test, y_train, y_test


def trainSoilmoistureEstimator(area, dim_red_mode="autoencoder",
                               verbose=0):
    """Train estimator for soil moisture.

    Parameters
    ----------
    area : str
        Name of measurement area.
    dim_red_mode : [type], optional (default="autoencoder")
        Type of dimensionality reduction, from {None, "PCA", "autoencoder"}.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    model : model
        Trained regression model.

    """
    X_train, X_test, y_train, y_test = loadDataForEstimation(
        area=area, dim_red_mode=dim_red_mode, verbose=verbose)

    # training
    model = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # evaluation
    rmse = np.sqrt(me.mean_squared_error(y_test, y_pred))
    r2 = me.r2_score(y_test, y_pred)
    if verbose:
        print("| R2: {0:.1f} %".format(r2*100))
        print("| RMSE: {0:.2f}".format(rmse))

    return model


def predictSoilmoistureMap(area, model, dim_red_mode=None, sm_range=None,
                           postfix="", verbose=0):
    """Predict soil moisture map for measurement `area`.

    Parameters
    ----------
    area : str
        Name of measurement area.
    model : model
        Regression model.
    dim_red_mode : [type], optional (default=None)
        Type of dimensionality reduction, from {None, "PCA", "autoencoder"}.
    sm_range : tuple of float, optional (default=None)
        If not None, this is a tuple of (a, b) with a<b which defines the
        range of soil moisture that should be considered in the prediction.
    postfix : str, optional (default="")
        Postfix for output file.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    hyp_data_map : np.array
        Hyperspectral data map

    """
    hyp_path = "/home/felix/data/"
    tif_file = hyp_path + "Area" + str(area) + "_multi_or_rf"

    # read in data and transform
    if verbose:
        print("| Reading in data ...")
    tif_file = rasterio.open(tif_file)
    width = tif_file.width
    height = tif_file.height
    n_features = 170
    hyp_data = tif_file.read()
    if hyp_data.shape[0] == 170:
        hyp_data = hyp_data.transpose(1, 2, 0)
    hyp_data_list = hyp_data.reshape((width*height, n_features))

    # mask and preprocess
    if verbose:
        print("| Masking data ...")
    image_mask = np.argwhere(np.mean(hyp_data_list, axis=1) != 0).flatten()
    hyp_data_pre = processHeadwallData(
        hyp_data_list[image_mask], verbose=True)

    # predict
    if verbose:
        print("| Predicting ...")
    if not dim_red_mode:
        hyp_data_pred = model.predict(
            preprocessingPixelHeadwall(hyp_data_pre))
    else:
        encoder = getEncoder(area, dim_red_mode=dim_red_mode)
        if dim_red_mode == "autoencoder":
            hyp_data_pred = model.predict(encoder.predict(
                preprocessingPixelHeadwall(hyp_data_pre)))
        elif dim_red_mode == "pca":
            hyp_data_pred = model.predict(encoder.transform(
                preprocessingPixelHeadwall(hyp_data_pre)))

    # transform
    hyp_data_map = np.zeros(shape=(width*height, ))
    hyp_data_map[image_mask] = hyp_data_pred
    hyp_data_map = hyp_data_map.reshape((height, width))

    # save
    if verbose:
        print("| Saving data ...")
    new_dataset = rasterio.open(
        "estimations/area_"+str(area)+"_"+str(dim_red_mode)+postfix+".tif",
        'w', driver='GTiff', height=height, width=width, count=1,
        dtype=hyp_data_map.dtype, crs=tif_file.crs,
        transform=tif_file.transform)
    new_dataset.write(hyp_data_map, 1)
    new_dataset.close()

    # plotSoilMoistureMap(hyp_data_map=hyp_data_map, area=area,
    #                     sm_range=sm_range, postfix=postfix)
    return hyp_data_map


def plotSoilMoistureMap(hyp_data_map, area, sm_range=None, postfix=""):
    """Plot soil moisture map

    Parameters
    ----------
    hyp_data_map : np.array
        Hyperspectral data map
    area : str
        Name of measurement area.
    sm_range : tuple of float, optional (default=None)
        If not None, this is a tuple of (a, b) with a<b which defines the
        range of soil moisture that should be considered in the prediction.
    postfix : str, optional (default="")
        Postfix for output file.

    """
    fontsize = 12

    _, ax = plt.subplots(1, 1, figsize=(6, 6))

    if sm_range:
        plt.imshow(np.ma.masked_where(hyp_data_map == 0, hyp_data_map),
                   cmap="viridis_r", vmin=sm_range[0], vmax=sm_range[1])
    else:
        plt.imshow(np.ma.masked_where(hyp_data_map == 0, hyp_data_map),
                   cmap="viridis_r")

    # plt.title("Area "+str(area))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_ylabel('Soil moisture in %', fontsize=fontsize, labelpad=10)

    ax.set_xlabel("X in a.u.", fontsize=fontsize)
    ax.set_ylabel("Y in a.u.", fontsize=fontsize)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    plt.savefig("plots/estimation_map_area_" + str(area)+"_" + postfix +
                ".pdf", bbox_inches="tight")


def loadCSVData(areas=None, max_sm=40.):
    """Load data from CSV

    Parameters
    ----------
    areas : list of str, optional
        List of areas. The full list is `["1", "2_1", "2_2", "3", "4", "5"]`.
    max_sm : float, optional (default=40)
        Maximum soil moisture value.

    Returns
    -------
    X : np.array
        Hyperspectral data
    y : np.array
        Soil moisture labels

    """
    # load data
    df = pd.read_csv("data/processed/peru_data.csv", index_col=0)

    # remove areas which are not used
    if areas:
        df = df[df["area"].isin(list(areas))]

    # remove too large soil moisture values
    df = df[df["soilmoisture"] <= max_sm]

    # define hyperspectral bands
    hypbands = getHyperspectralBands(True)

    # create arrays
    X = df[hypbands.astype("str")].values
    y = df["soilmoisture"].values

    return X, y


def getMCSoilMoistureData(area, test_size=0.5, std=1, n_new=5,
                          max_sm=25., random_state=None, verbose=0):
    """Get Monte Carlo (MC) augmented soil moisture data, incl. split.

    Parameters
    ----------
    area : str
        Name of measurement area.
    test_size : float, optional (default=0.5)
        Size of the test dataset, between 0 and 1.
    std : float
        Standard deviation of the MC augmentation
    n_new : int
        Number of new datapoints per input datapoint
    max_sm : [type], optional
        [description], by default 25.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    X_train_new : np.array
        Hyperspectral data of the MC augmented dataset for the training.
    X_test_new : np.array
        Hyperspectral data of the MC augmented dataset for the evaluation.
    y_train_new : np.array
        Soil moisture data of the MC augmented dataset for the training.
    y_test_new : np.array
        Soil moisture data of the MC augmented dataset for the evaluation.

    """
    # load data X, y
    X, y = loadCSVData(areas=area, max_sm=max_sm)
    if verbose:
        print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    X_train_new, y_train_new = generateMCData(
        X=X_train, y=y_train, std=std, n_new=n_new, random_state=random_state+1,
        verbose=verbose)
    X_test_new, y_test_new = generateMCData(
        X=X_test, y=y_test, std=std, n_new=n_new, random_state=random_state+2,
        verbose=verbose)

    return X_train_new, X_test_new, y_train_new, y_test_new


def generateMCData(X, y, std, n_new, random_state=None, verbose=0):
    """Generate Monte Carlo (MC) augmented dataset

    Parameters
    ----------
    X : np.array
        Hyperspectral data
    y : np.array
        Soil moisture labels
    std : float
        Standard deviation of the MC augmentation
    n_new : int
        Number of new datapoints per input datapoint
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    np.array
        Hyperspectral data of augmented dataset
    np.array
        New soil moisture labels of augmented dataset

    """
    np.random.seed(random_state)
    X_new = []
    y_new = []
    for i, _ in enumerate(X):
        # --- X
        for _ in range(n_new+1):
            X_new.append(X[i])

        # --- y
        y_new.append(y[i])
        y_mc_temp = np.random.normal(loc=y[i], scale=std, size=n_new)
        for y_mc in y_mc_temp:
            y_new.append(y_mc)

    return np.array(X_new), np.array(y_new)
