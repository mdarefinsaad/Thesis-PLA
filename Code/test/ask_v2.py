# ALPLA demo- Implementation with a small scale dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM


# Function to update the data vector
def update_features(features, new_cir):
    # Remove the oldest CIR (first row)
    updated_features = np.delete(features, 0, axis=0)
    # Append the new CIR to the end
    updated_features = np.vstack([updated_features, new_cir])
    return updated_features


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

ocsvm = OneClassSVM(kernel='linear', gamma='auto', nu=0.01)
# Train the One-Class SVM
ocsvm.fit(alice_features)

# Extract features for testing
incoming_real = X_test[:, 6, :, 0]
incoming_imag = X_test[:, 6, :, 1]
incoming_mag = np.abs(incoming_real + 1j * incoming_imag)

# Scale the test data
test_real_scaled = scaler.transform(incoming_real)
test_imag_scaled = scaler.transform(incoming_imag)
test_mag_scaled = scaler.transform(incoming_mag)

test_features = np.column_stack((test_real_scaled, test_imag_scaled, test_mag_scaled))

for i in range(X_test.shape[0]):
# for i in range(2):
    # Extract the current test CIR
    incoming_real = X_train[i, 2, :, 0]
    incoming_imag = X_train[i, 2, :, 1]
    incoming_mag = np.abs(incoming_real + 1j * incoming_imag)

    # Scale the test CIR
    test_real_scaled = scaler.transform(incoming_real.reshape(1, -1))
    test_imag_scaled = scaler.transform(incoming_imag.reshape(1, -1))
    test_mag_scaled = scaler.transform(incoming_mag.reshape(1, -1))
 
    # Create a feature vector for the test CIR
    test_features = np.column_stack((test_real_scaled, test_imag_scaled, test_mag_scaled))

    # Predict using the OCC-SVM
    prediction = ocsvm.predict(test_features)

    # If the CIR is accepted, update the data vector
    if prediction == 1:
        alice_features = update_features(alice_features, test_features)
        print(f"CIR {i} accepted, updated features:")
        # print(alice_features)


print(alice_features.shape)
# The updated alice_features will contain the most recent CIRs