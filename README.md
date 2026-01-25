# Driver Warning System MVP

An end-to-end machine learning system for detecting dangerous driving curves and alerting drivers via haptic feedback.

## 🎯 Overview

This project implements a complete pipeline:
1. **Python data pipeline** - Simulates driving traces, computes curvature, extracts features
2. **Model training** - Trains Keras MLP and exports to TensorFlow Lite
3. **Android prototype** - Kotlin app with sensor integration, TFLite inference, and haptic alerts
4. **Testing & CI** - Unit tests and GitHub Actions workflow

## 🏗️ Architecture

```
┌─────────────────┐
│  GPS + IMU      │
│  Sensors        │
└────────┬────────┘
         │ 10 Hz
         ▼
┌─────────────────┐
│ Feature         │
│ Extraction      │ 9 features from 3s windows
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TFLite Model    │ Keras MLP (64→32→4)
│ Inference       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Alert Manager   │ Haptic patterns
│ (Vibration)     │
└─────────────────┘
```

## 📁 Project Structure

```
driver_safety/
├── notebooks/
│   └── train_and_export.ipynb    # End-to-end training notebook
├── python/
│   ├── geo_utils.py               # Haversine, bearing, curvature
│   ├── feature_extractor.py       # Sliding windows, 9 features
│   ├── simulate.py                # Synthetic trace generator
│   └── train.py                   # CLI training script
├── android/DriverWarningApp/      # Android Studio project (Kotlin)
├── models/
│   ├── curve_detector.tflite      # Exported TFLite model
│   ├── feature_schema.json        # Feature names & order
│   └── scaler.pkl                 # StandardScaler parameters
├── tests/
│   ├── test_geo_utils.py          # Unit tests for geometry
│   └── test_feature_extractor.py  # Unit tests for features
├── docs/
│   ├── feature_schema.md          # Detailed feature documentation
│   └── android_integration.md     # Android integration guide
├── .github/workflows/ci.yml       # GitHub Actions CI
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Jupyter Notebook
- Android Studio (for Android app)
- Git

### 1. Setup Python Environment

```bash
# Clone repository
git clone <repo-url>
cd driver_safety

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python python/simulate.py --out data/sim_trips.csv --duration 600 --curves 8
```

**Output**: `data/sim_trips.csv` with ~6000 samples (10 minutes at 10 Hz)

### 3. Train Model

```bash
python python/train.py --input data/sim_trips.csv --out-model models/curve_detector.tflite
```

**Outputs**:
- `models/curve_detector.tflite` - TFLite model (~50 KB)
- `models/scaler.pkl` - StandardScaler parameters
- `models/feature_schema.json` - Feature metadata
- `models/predictions_sample.csv` - Sample predictions

### 4. Run Jupyter Notebook

```bash
jupyter notebook notebooks/train_and_export.ipynb
```

Execute all cells to see:
- Data visualization
- Feature distributions
- Model training curves
- Confusion matrices
- TFLite inference tests

### 5. Run Tests

```bash
pytest tests/ -v
```

Expected output: All tests pass ✅

## 📊 Feature Schema

The model uses **9 features** extracted from 3-second sliding windows:

| # | Feature | Unit | Description |
|---|---------|------|-------------|
| 1 | `current_speed` | m/s | Instantaneous speed |
| 2 | `mean_speed` | m/s | Average speed in window |
| 3 | `speed_std` | m/s | Speed standard deviation |
| 4 | `gyro_z_mean` | deg/s | Average yaw rate |
| 5 | `gyro_z_max` | deg/s | Max absolute yaw rate |
| 6 | `acc_y_mean` | m/s² | Average lateral acceleration |
| 7 | `acc_y_max` | m/s² | Max absolute lateral accel |
| 8 | `curve_radius` | m | Radius from GPS polyline |
| 9 | `severity_proxy` | m/s² | speed² / radius |

See [`docs/feature_schema.md`](docs/feature_schema.md) for detailed descriptions.

## 🎯 Model Performance

Trained on synthetic data with auto-labeling based on curve radius:

| Label | Severity | Radius Range | Typical Accuracy |
|-------|----------|--------------|------------------|
| Safe | Low | > 500 m | ~85% |
| Mild | Medium | 200-500 m | ~75% |
| Urgent | High | 100-200 m | ~70% |
| Hectic | Critical | < 100 m | ~80% |

**Overall accuracy**: ~78% (on synthetic data)

## 📱 Android App

### Build Instructions

1. Open `android/DriverWarningApp` in Android Studio
2. Copy `models/curve_detector.tflite` to `app/src/main/assets/`
3. Sync Gradle
4. Build APK: `Build → Build Bundle(s) / APK(s) → Build APK(s)`
5. Install on device or emulator

### Key Components

- **MainActivity** - Start/stop service, navigate to debug/settings
- **SensorService** - Background sensor collection and inference
- **TFLiteModelRunner** - TFLite model wrapper
- **FeatureExtractor** - Feature computation (matches Python exactly)
- **AlertManager** - Haptic feedback patterns
- **DebugActivity** - Manual testing with JSON input

### Haptic Patterns

| Severity | Pattern | Description |
|----------|---------|-------------|
| Mild | 200ms pulse | Single gentle vibration |
| Urgent | 300ms + 100ms + 300ms | Two strong pulses |
| Hectic | 500ms strong | Continuous strong vibration |

See [`docs/android_integration.md`](docs/android_integration.md) for complete integration guide.

## 🧪 Testing

### Python Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_geo_utils.py -v

# With coverage
pytest tests/ --cov=python --cov-report=html
```

### Manual Testing

**Test curvature calculation**:
```bash
python python/geo_utils.py
```

**Test feature extraction**:
```bash
python python/feature_extractor.py
```

**Test simulation**:
```bash
python python/simulate.py --out data/test.csv --duration 60
```

## 🔄 CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push:

1. **Python Tests** - Run pytest on all unit tests
2. **Model Training** - Generate data and train model in quick mode
3. **Android Build** - Compile Android app (when project exists)

View workflow runs in the Actions tab on GitHub.

## 📚 Documentation

- **[Feature Schema](docs/feature_schema.md)** - Detailed feature descriptions and computation methods
- **[Android Integration](docs/android_integration.md)** - Complete Android integration guide with code examples

## 🛠️ Development Workflow

### Adding New Features

1. Update `FEATURE_NAMES` in `python/feature_extractor.py`
2. Modify `extract_window_features()` to compute new feature
3. Update `models/feature_schema.json`
4. Update `docs/feature_schema.md`
5. Update Android `FeatureExtractor.kt` to match
6. Retrain model and export new TFLite

### Retraining Model

```bash
# Generate fresh data
python python/simulate.py --out data/new_trips.csv --duration 1200 --curves 15

# Train with more epochs
python python/train.py --input data/new_trips.csv --out-model models/curve_detector_v2.tflite

# Test in notebook
jupyter notebook notebooks/train_and_export.ipynb
```

### Debugging Android App

1. Enable Developer Options on Android device
2. Connect via USB and enable USB Debugging
3. Run app from Android Studio
4. Use Debug Activity to test with manual feature input
5. Check Logcat for inference outputs

## 🔬 Model Details

### Architecture

```
Input(9) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(4, softmax)
```

- **Input**: 9 normalized features
- **Hidden layers**: 64 → 32 neurons with ReLU activation
- **Dropout**: 0.3 for regularization
- **Output**: 4-class softmax (safe, mild, urgent, hectic)

### Training

- **Optimizer**: Adam
- **Loss**: Sparse categorical crossentropy
- **Batch size**: 32
- **Epochs**: 50 (with early stopping, patience=10)
- **Train/Val/Test split**: 70/15/15

### TFLite Conversion

- **Quantization**: None (float32 for simplicity)
- **Input shape**: `[1, 9]`
- **Output shape**: `[1, 4]`
- **Model size**: ~50 KB

## 🔐 Privacy

- **No real user data** is included in this repository
- All driving traces are **synthetic**
- Android app includes **consent toggle** (UI only, no backend)
- For production: implement proper data anonymization and opt-in mechanisms

## 🚧 Future Work

- [ ] Real-world data collection with user consent
- [ ] Backend API for telemetry upload
- [ ] Model retraining pipeline with real data
- [ ] Multi-class severity refinement
- [ ] Integration with navigation apps
- [ ] Audio alerts in addition to haptic
- [ ] Driver behavior analytics dashboard

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with**: Python, TensorFlow/Keras, TFLite, Kotlin, Android SDK
