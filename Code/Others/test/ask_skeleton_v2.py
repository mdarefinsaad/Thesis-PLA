# ALPLA demo- Implementation with a small scale dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def update_features(features, new_cir):
    updated_features = np.delete(features, 0, axis=0)
    updated_features = np.vstack([updated_features, new_cir])
    return updated_features

def servesImportant64Samples(real, imag):

    # Number of signals
    num_signals = real.shape[0]  # 3 in this case
    
    # Initialize lists to store the focused samples
    imp_real_parts = []
    imp_imag_parts = []
    img_mag_parts = []
    
    # print(real)
    # print(imag.shape)
    # print(num_signals)
    
    for i in range(num_signals):
        # Calculate the magnitude
        magnitude = np.abs(real[i] + 1j * imag[i])    
        
        # find the peak index
        peak_index = np.argmax(magnitude)
        
        # Calculate the start and end indices for the focused part
        start_index = max(0, peak_index - 32)
        end_index = min(magnitude.shape[0], peak_index + 32)
        
        # Extract the part of the signal around the peak
        real_part_focus = real[i, start_index:end_index]
        imag_part_focus = imag[i, start_index:end_index]
        mag_part_focus = magnitude[start_index:end_index]
        
        imp_real_parts.append(real_part_focus)
        imp_imag_parts.append(imag_part_focus)
        img_mag_parts.append(mag_part_focus)
        
    # Convert lists back to arrays for further processing if needed
    imp_real_parts = np.array(imp_real_parts)
    imp_imag_parts = np.array(imp_imag_parts)
    img_mag_parts = np.array(img_mag_parts)

    return imp_real_parts, imp_imag_parts, img_mag_parts

def giveMefeatureVector(real_f, imag_f, mag_f): 
    interleaved_trainig_data = np.empty((real_f.shape[0], real_f.shape[1] * 3), dtype=float)
    interleaved_trainig_data[:, 0::3] = real_f
    interleaved_trainig_data[:, 1::3] = imag_f
    interleaved_trainig_data[:, 2::3] = mag_f 
    return interleaved_trainig_data

# Load measurement file:
measurement = np.load('dataset/meas_symm_1.npz', allow_pickle=False)
header, data = measurement['header'], measurement['data']

# Initialize the scaler
scaler = MinMaxScaler()

# Initialize the One-Class SVM
ocsvm = OneClassSVM(kernel='poly', gamma='auto', nu=0.05)

#------------------ Spliting the data into training and test sets ------------------
dataset_slice = data['cirs']

# Split data into training and test sets
X_train, X_test = train_test_split(dataset_slice, test_size=0.2, random_state=42)

# ---------------------------------------- Training ----------------------------------------
# Initial training with 10 samples/CIRs
initial_training_real_251 = X_train[:500, 3, :, 0]
initial_training_imag_251 = X_train[:500, 3, :, 1]

init_train_real_part, init_train_imag_part, init_train_mag_part = servesImportant64Samples(initial_training_real_251, initial_training_imag_251)

init_train_real = scaler.fit_transform(init_train_real_part)
init_train_imag = scaler.fit_transform(init_train_imag_part)
init_train_mag = scaler.fit_transform(init_train_mag_part)

# Receive the feature vector as [R1 I1 M1 R2 I2 M2 R3 I3 M3...]
init_train_vector = giveMefeatureVector(init_train_real, init_train_imag, init_train_mag)

ocsvm.fit(init_train_vector)

# creating feature vector
alice_real_251 = X_train[501:504, 3, :, 0]
alice_imag_251 = X_train[501:504, 3, :, 1]

# serves the important 64 samples
alice_real, alice_imag, alice_mag = servesImportant64Samples(alice_real_251, alice_imag_251)

alice_real_scaled = scaler.fit_transform(alice_real)
alice_imag_scaled = scaler.fit_transform(alice_imag)
alice_mag_scaled = scaler.fit_transform(alice_mag)

# Gets the feature vector (New)
alice_features = giveMefeatureVector(alice_real_scaled, alice_imag_scaled, alice_mag_scaled)

# Train the model
ocsvm.fit(alice_features)

# ---------------------------------------- Testing ----------------------------------------
# Initialize lists to hold true labels and predictions
true_labels = []
predictions = []
selected_channel = [3,6]

count = 0
neg_count = 0

for cir in range(X_test.shape[0]):
    for channel in selected_channel:
        
        # Determine the true label based on the channel
        if channel == 3:
            true_label = 1  # Legitimate user (Alice)
            count += 1
        else:
            neg_count += 1
            true_label = -1  # Illegitimate user (Not Alice)
        
        true_labels.append(true_label)
        
        # Extract the current test CIR
        incoming_real_251 = X_test[cir, channel, :, 0].reshape(1, -1)
        incoming_imag_251 = X_test[cir, channel, :, 1].reshape(1, -1)
        # incoming_mag_251 = np.abs(incoming_real + 1j * incoming_imag).reshape(1, -1)

        incoming_real, incoming_imag, incoming_mag = servesImportant64Samples(incoming_real_251, incoming_imag_251)
        
        # Scale the test CIR
        test_real_scaled = scaler.transform(incoming_real)
        test_imag_scaled = scaler.transform(incoming_imag)
        test_mag_scaled = scaler.transform(incoming_mag)
    
        # # Create a feature vector for the test CIR
        test_features = giveMefeatureVector(test_real_scaled, test_imag_scaled, test_mag_scaled)

        # Predict using the OCC-SVM
        prediction = ocsvm.predict(test_features)
        predictions.append(prediction[0])
        # If the CIR is accepted, update the data vector
        if prediction == 1:
            alice_features = update_features(alice_features, test_features)
            ocsvm.fit(alice_features)
            # print(alice_features.shape)
            print(f"CIR {cir} channel {channel} accepted, model is retrained")


# ---------------------------------------- Evaluation ----------------------------------------
# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[-1, 1]).ravel()

# # Missed Detection Rate (MDR)
MDR = fp / (fp + tn)

# # False Alarm Rate (FAR)
FAR = fn / (fn + tp)

# # Gamma calculation
gamma = (tp + fn) / (tn + fp)

# # Authentication Rate (AR)
AR = (tp + gamma * tn) / ((tp + fn) + gamma * (tn + fp))

print(f"MDR: {MDR}")
print(f"FAR: {FAR}")
print(f"AR: {AR}")