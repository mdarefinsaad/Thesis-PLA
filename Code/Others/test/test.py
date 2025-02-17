# ALPLA demo- Implementation with a small scale dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import joblib


# Understanding Array
# Our testing CIR dataset
new_data = {
    'cirs': np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                    [5, 6]
                ],
                [
                    [9, 10],
                    [11, 12],
                    [13, 14]
                ],
                [
                    [17, 18],
                    [19, 20],
                    [21, 22]
                ]
            ],
            [
                [
                    [25, 26],
                    [27, 28],
                    [29, 30]
                ],
                [
                    [33, 34],
                    [35, 36],
                    [37, 38]
                ],
                [
                    [41, 42],
                    [43, 44],
                    [45, 46]
                ]
            ],
            [
                [
                    [49, 50],
                    [51, 52],
                    [53, 54]
                ],
                [
                    [57, 58],
                    [59, 60],
                    [61, 62]
                ],
                [
                    [65, 66],
                    [67, 68],
                    [69, 70]
                ]
            ]
        ]
    )
}

# Step 1.1 - Extract real and imaginary parts for Alice and Eve (assuming channels 0 and 1)
alice_real = new_data['cirs'][:, 0, :, 0]
alice_imag = new_data['cirs'][:, 0, :, 1]
eve_real = new_data['cirs'][:, 1, :, 0]
eve_imag = new_data['cirs'][:, 1, :, 1]

# Step 1.2 - Compute magnitudes
alice_mag = np.sqrt(alice_real**2 + alice_imag**2)
eve_mag = np.sqrt(eve_real**2 + eve_imag**2)

# Step 1.3 - Flatten the arrays
# Flatten features for Alice
alice_real_flat = alice_real.flatten()
alice_imag_flat = alice_imag.flatten()
alice_mag_flat = alice_mag.flatten()
# Flatten features for Eve
eve_real_flat = eve_real.flatten()
eve_imag_flat = eve_imag.flatten()
eve_mag_flat = eve_mag.flatten()
# We flatten the arrays to make them suitable for the MinMaxScaler
# We flat all three(real, imaginary and magnitude) parts of the signal

# Step 1.4 - Combine features (making a feature vector)
alice_features = np.column_stack((alice_real_flat, alice_imag_flat, alice_mag_flat))
eve_features = np.column_stack((eve_real_flat, eve_imag_flat, eve_mag_flat))
# By doing np.column_stack, we are making a feature vector of the signal (combining all three parts of the signal)

# Step 1.5 - Normalize the data (to train the model)
scaler = MinMaxScaler()
alice_features_scaled = scaler.fit_transform(alice_features)
eve_features_scaled = scaler.transform(eve_features) # Used the samne way as Alice's data was scaled


# Step 2.1 - Train OCC-SVM for Alice
occ_svm = OneClassSVM(kernel='linear', nu=0.01)  # You can experiment with other kernels like 'rbf', 'poly', 'sigmoid'
occ_svm.fit(alice_features_scaled)
joblib.dump(occ_svm, 'occ_svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 3.1 - Test Incoming Signals
occ_svm = joblib.load('occ_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
# Example incoming signal data for testing (using the same dataset here for simplicity)
incoming_data = new_data['cirs']
# Extract features for testing
incoming_real = incoming_data[:, 0, :, 0]
incoming_imag = incoming_data[:, 0, :, 1]
incoming_mag = np.sqrt(incoming_real**2 + incoming_imag**2)
# Combine and normalize features
incoming_features_vector = np.column_stack((incoming_real.flatten(), incoming_imag.flatten(), incoming_mag.flatten()))
# Scaled the incoming features using the same scaler used for Alice
incoming_features_scaled = scaler.transform(incoming_features_vector)

# Predict using OCC-SVM
predictions = occ_svm.predict(incoming_features_scaled)
# Check predictions (1 for inliers/legitimate, -1 for outliers/illegitimate)
print(predictions)
legitimate = predictions == 1
illegitimate = predictions == -1
print("Legitimate signals:", legitimate.sum())
print("Illegitimate signals:", illegitimate.sum())

# Output
# [-1 -1 -1 -1 -1 -1 -1 -1 -1]
# Legitimate signals: 0
# Illegitimate signals: 9