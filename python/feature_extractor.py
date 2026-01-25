import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict
from typing import List, Dict
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER = [
    "current_speed",
    "mean_speed",
    "speed_std",
    "gyro_z_mean",
    "gyro_z_max",
    "acc_y_mean",
    "acc_y_max",
    "curve_radius",
    "severity_proxy"
]

def compute_severity(speed, radius):
    """
    Computes theoretical lateral acceleration (v^2 / r)
    """
    if radius == 0 or np.isinf(radius):
        return 0.0
    return (speed ** 2) / radius

def extract_window_features(window_df: pd.DataFrame, curve_radius: float) -> OrderedDict:
    """
    Extracts features from a standard dataframe window.
    Assumes dataframe has columns: speed, gyro_z, acc_y
    curve_radius is passed in from map-matching logic.
    """
    features = OrderedDict()
    
    # Speed features
    features["current_speed"] = window_df['speed'].iloc[-1]
    features["mean_speed"] = window_df['speed'].mean()
    features["speed_std"] = window_df['speed'].std() if len(window_df) > 1 else 0.0
    
    # Gyro features (Yaw)
    features["gyro_z_mean"] = window_df['gyro_z'].abs().mean()
    features["gyro_z_max"] = window_df['gyro_z'].abs().max()
    
    # Accel features (Lateral)
    features["acc_y_mean"] = window_df['acc_y'].abs().mean()
    features["acc_y_max"] = window_df['acc_y'].abs().max()
    
    # Map features
    features["curve_radius"] = curve_radius if curve_radius > 0 else 9999.0
    
    # Synthetic Proxy
    features["severity_proxy"] = compute_severity(features["mean_speed"], features["curve_radius"])
    
    return features

def fit_scaler(df: pd.DataFrame, output_path: str):
    """
    Fits a StandardScaler on the features and saves it.
    df should strictly contain the feature columns.
    """
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_ORDER])
    joblib.dump(scaler, output_path)
    print(f"Scaler saved to {output_path}")
    return scaler

def apply_scaler(X: pd.DataFrame, scaler_path: str) -> np.ndarray:
    """
    Loads scaler and transforms X.
    """
    scaler = joblib.load(scaler_path)
    return scaler.transform(X[FEATURE_ORDER])
