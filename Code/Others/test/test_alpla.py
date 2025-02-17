# ALPLA demo- Implementation with a small scale dataset

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

np.set_printoptions(threshold=np.inf)

# Load measurement file:
measurement = np.load('dataset/meas_symm_1.npz', allow_pickle=False)
header, data = measurement['header'], measurement['data']

dataset_slice = data['cirs']
print(data['cirs'].shape)
# Split data into training and test sets
X_train, X_test = train_test_split(dataset_slice, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

# Extract real and imaginary parts for Alice (channel 3 is the legitimate channel between Alice and Bob)
alice_real = X_train[:, 3, :, 0]
alice_imag = X_train[:, 3, :, 1]

# Compute magnitudes
alice_complex = alice_real + 1j * alice_imag
alice_magnitude = np.abs(alice_complex)

# Scaling Real, Imaginary and Magnitude parts
scaler = MinMaxScaler()

# Fit and transform the features
alice_real_scaled = scaler.fit_transform(alice_real)
alice_imag_scaled = scaler.fit_transform(alice_imag.reshape(-1, 1)).reshape(alice_imag.shape + (1,))
alice_mag_scaled = scaler.fit_transform(alice_magnitude.reshape(-1, 1)).reshape(alice_magnitude.shape + (1,))

# Concatenate the scaled features along the third axis
alice_features = np.concatenate((alice_real_scaled, alice_imag_scaled, alice_mag_scaled), axis=2)
# Reshape the concatenated features to match the training data shape
alice_features = alice_features.reshape(X_train.shape[0], -1)

ocsvm = OneClassSVM(kernel='linear', gamma='auto', nu=0.01)

# Train the One-Class SVM
ocsvm.fit(alice_features)

# Extract features for testing
incoming_real = X_test[:, 6, :, 0]
incoming_imag = X_test[:, 6, :, 1]
incoming_mag = np.abs(incoming_real + 1j * incoming_imag)

# Scale the test data
test_real_scaled = scaler.transform(incoming_real.reshape(-1, 1)).reshape(incoming_real.shape  + (1,))
test_imag_scaled = scaler.transform(incoming_imag.reshape(-1, 1)).reshape(incoming_imag.shape  + (1,))
test_mag_scaled = scaler.transform(incoming_mag.reshape(-1, 1)).reshape(incoming_mag.shape  + (1,))

test_features = np.concatenate((test_real_scaled, test_imag_scaled, test_mag_scaled), axis=2).reshape(X_test.shape[0], -1)

predictions = ocsvm.predict(test_features)

legitimate = predictions == 1
illegitimate = predictions == -1
print("Legitimate signals:", legitimate.sum())
print("Illegitimate signals:", illegitimate.sum())