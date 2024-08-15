# ALPLA demo- Implementation with a small scale dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import joblib

# import sys
# print("Python version")
# print(sys.version)


np.set_printoptions(threshold=np.inf)

# Load measurement file:
measurement = np.load('dataset/meas_symm_1.npz', allow_pickle=False)
header, data = measurement['header'], measurement['data']

dataset_slice = data['cirs']

# Split data into training and test sets
X_train, X_test = train_test_split(dataset_slice, test_size=0.2, random_state=42)

# Extract real and imaginary parts for Alice (channel 3 is the legitimate channel between Alice and Bob)
alice_real = X_train[:3, 3, :, 0]
alice_imag = X_train[:3, 3, :, 1]

# Compute magnitudes
alice_complex = alice_real + 1j * alice_imag
alice_magnitude = np.abs(alice_complex)

# Scaling Real, Imaginary and Magnitude parts
scaler = MinMaxScaler()

# Fit and transform the features
alice_real_scaled = scaler.fit_transform(alice_real)
alice_imag_scaled = scaler.fit_transform(alice_imag)
alice_mag_scaled = scaler.fit_transform(alice_magnitude)

# Concatenate the scaled features along the third axis
alice_features = np.column_stack((alice_real_scaled, alice_imag_scaled, alice_mag_scaled))


# Train the One-Class SVM
ocsvm = OneClassSVM(kernel='linear', gamma='auto', nu=0.01)
ocsvm.fit(alice_features)


