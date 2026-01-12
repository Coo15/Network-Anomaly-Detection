import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models


def ids_detect(sample_scaled, model, threshold):
    recon = model.predict(sample_scaled.reshape(1, -1), verbose=0)
    error = np.mean((sample_scaled - recon) ** 2)
    return 1 if error > threshold else 0, error


#
PROC_DIR = Path("../data/processed")

X_train_normal = np.load(PROC_DIR / "X_train_normal.npy")
X_test  = np.load(PROC_DIR / "X_test.npy")
y_test  = np.load(PROC_DIR / "y_test.npy")

#
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

input_dim = X_train_scaled.shape[1]

#
autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(input_dim, activation="linear")
])

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

#
history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled,
    epochs=20,
    batch_size=512,
    validation_split=0.1,
    verbose=1
)

#
train_recon = autoencoder.predict(X_train_scaled)
train_error = np.mean((X_train_scaled - train_recon) ** 2, axis=1)

print(pd.Series(train_error).describe())

#
threshold = np.percentile(train_error, 95)

print("Anomaly threshold:", threshold)

#
y_pred = []
scores = []

for i in range(len(X_test_scaled)):
    pred, score = ids_detect(X_test_scaled[i], autoencoder, threshold)
    y_pred.append(pred)
    scores.append(score)

y_pred = np.array(y_pred)
scores = np.array(scores)

#
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, scores))

#
print("=== IDS LIVE DEMO ===")

for i in range(10):
    pred, score = ids_detect(X_test_scaled[i], autoencoder, threshold)
    label = "ATTACK" if pred == 1 else "NORMAL"
    
    print(f"[Flow {i}] Score={score:.4f} → {label}")

