import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

def save_predictions_to_csv(model, test_data, features, output_path):
    
    distances = cdist(test_data[['lat', 'lon']], features[['lat', 'lon']], metric='euclidean')
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = np.min(distances, axis=1)

    merged_data_test = test_data.copy()
    merged_data_test['nearest_index'] = nearest_indices
    merged_data_test['nearest_distance'] = nearest_distances
    merged_data_test = merged_data_test[merged_data_test['nearest_distance'] < 0.5]
    merged_data_test = pd.merge(merged_data_test, features, left_on='nearest_index', right_index=True)
    merged_data_test = merged_data_test.drop(['nearest_index', 'nearest_distance'], axis=1)

    predictions = model.predict(merged_data_test)
    res = pd.DataFrame(predictions)
    res["id"] = merged_data_test["id"]
    res = res[['id', 0]]
    res = res.rename(columns={0: "score"})

    res.to_csv(output_path, index=False)
