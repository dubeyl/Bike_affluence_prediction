import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000 # to have the distance in meters

class GWR:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def _gaussian_kernel(self, distances):
        weights = np.exp(-(distances ** 2) / (2 * self.bandwidth ** 2))
        return weights / np.sum(weights, axis=1, keepdims=True)

    def fit(self, X, y, coordinates):
        self.X = X
        self.y = y
        self.coordinates = coordinates

    def predict(self, X_pred):
        y_pred = np.zeros(X_pred.shape[0])
        for i, x in enumerate(X_pred):
            distances = cdist([x], self.coordinates)
            weights = self._gaussian_kernel(distances)
            local_X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)
            local_X_weighted = local_X * weights
            local_y_weighted = self.y * weights.squeeze()
            beta = np.linalg.lstsq(local_X_weighted, local_y_weighted, rcond=None)[0]
            y_pred[i] = np.dot(np.concatenate(([1], x)), beta)
        return y_pred

if __name__ == "__main__":
    # We will be training two models: one for the departure and one for the arrivals.

    with open('../Data/2023_edges_extended_header.csv', 'r') as file:
        columns = file.readline().strip().split(',') 
    df_edges = pd.read_csv('../Data/2023_edges_extended.csv', header=None)
    df_edges.columns = columns
    
    feature_columns = [':START_ID', 'month', 'day', 'hour', 'conditions', 'temperature', 'wind_speed'] # features
    target_column = 'n'  # target
    coordinate_columns = ['start_latitude', 'start_longitude']  # coordinates

    X = df_edges[feature_columns].values
    y = df_edges[target_column].values
    coordinates = df_edges[coordinate_columns].values

    # Split the dataset 
    X_train, X_test, y_train, y_test, coord_train, coord_test = train_test_split(
        X, y, coordinates, test_size=0.2, random_state=42
    )

    # Create and train the model
    model = GWR(bandwidth=2.0)
    model.fit(X_train, y_train, coord_train)

    # Make predictions on the test set
    predictions = model.predict(X_test, coord_test)
    print("Predictions on the test set:", predictions)
