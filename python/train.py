import pandas as pd
import numpy as np
import argparse
import os
import json
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from .geo_utils import curvature_from_polyline
from .feature_extractor import extract_window_features, fit_scaler, apply_scaler, FEATURE_ORDER

# Label Config
LABEL_MAP = {0: "safe", 1: "mild", 2: "urgent", 3: "hectic"}

def label_window(severity):
    """
    Auto-label based on lateral acceleration severity (m/s^2)
    """
    if severity < 3.0: return 0
    if severity < 5.0: return 1
    if severity < 8.0: return 2
    return 3

def process_data(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Compute Curvature
    coords = list(zip(df.lat, df.lon))
    df['curvature'] = curvature_from_polyline(coords)
    df['radius'] = df['curvature'].apply(lambda k: 1/k if k > 1e-4 else 9999.0)
    
    # 2. Sliding Windows (3s window, 1s step? or 0.5s step)
    # 20hz data -> 60 samples = 3s
    window_size = 60
    step_size = 10
    
    X_rows = []
    y_labels = []
    
    print("Extracting features (this may take a moment)...")
    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i : i+window_size]
        
        # Use radius from middle or end? 
        # For prediction we usually key off the 'current' state (end of window).
        # We need a 'lookahead' curvature ideally, but let's use current position curvature
        # derived from the polyline logic.
        curr_radius = window['radius'].iloc[-1]
        
        feats = extract_window_features(window, curr_radius)
        
        # Determine Label
        severity = feats['severity_proxy'] # v^2 / r
        label = label_window(severity)
        
        X_rows.append(list(feats.values()))
        y_labels.append(label)
        
    X_df = pd.DataFrame(X_rows, columns=FEATURE_ORDER)
    y = np.array(y_labels)
    
    return X_df, y

def train_stats(y):
    classes, counts = np.unique(y, return_counts=True)
    print("Label distribution:", dict(zip(classes, counts)))

def train_and_export(input_csv, out_model_path, quick=False):
    # 1. Prepare Data
    X_df, y = process_data(input_csv)
    train_stats(y)
    
    # 2. Scale
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    scaler_path = os.path.join(os.path.dirname(out_model_path), "scaler.pkl")
    fit_scaler(X_df, scaler_path)
    X_scaled = apply_scaler(X_df, scaler_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Model
    # Keras MLP
    model = keras.Sequential([
        keras.layers.Input(shape=(len(FEATURE_ORDER),)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    epochs = 2 if quick else 10
    print(f"Training for {epochs} epochs...")
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
    
    # Eval
    y_pred = list(np.argmax(model.predict(X_test), axis=1))
    print("\nClassification Report (MLP):")
    print(classification_report(y_test, y_pred))
    
    # 4. Export TFLite
    print(f"Exporting TFLite to {out_model_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(out_model_path, 'wb') as f:
        f.write(tflite_model)
        
    # 5. Schema
    schema_path = os.path.join(os.path.dirname(out_model_path), "feature_schema.json")
    schema = {
        "features": FEATURE_ORDER,
        "scaler": "scaler.pkl",
        "label_map": LABEL_MAP
    }
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input CSV")
    parser.add_argument('--out-model', type=str, default='models/curve_detector.tflite', help="Output TFLite path")
    parser.add_argument('--quick', action='store_true', help="Run fast for CI")
    
    args = parser.parse_args()
    
    train_and_export(args.input, args.out_model, args.quick)
