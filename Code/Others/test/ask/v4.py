# ALPLA demo- Implementation with a small scale dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def servesImportant64Samples(real, imag):

    # Number of signals
    num_signals = real.shape[0]  # 3 in this case
    
    # Initialize lists to store the focused samples
    imp_real_parts = []
    imp_imag_parts = []
    img_mag_parts = []
    
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

def update_features(features, new_cir, max_window_size=100):

    # If the number of rows (samples) exceeds the max window size, remove the oldest sample
    if features.shape[0] >= max_window_size:
        # Remove the oldest CIR (first row)
        features = np.delete(features, 0, axis=0)
    
    # Append the new CIR to the end
    updated_features = np.vstack([features , new_cir])
    return updated_features


# Load measurement file:
measurement = np.load('/../dataset/meas_symm_1.npz', allow_pickle=False)

header, data = measurement['header'], measurement['data']

# Initialize the scaler
scaler = MinMaxScaler()

# Initialize the One-Class SVM
ocsvm = OneClassSVM(kernel='linear', gamma='auto', nu=0.001)

#------------------ Spliting the data into training and test sets ------------------
dataset_slice = data['cirs']

# Split data into training and test sets
X_train, X_test = train_test_split(dataset_slice, test_size=0.2, random_state=42)

alice_real_251 = X_train[:50, 3, :, 0]
alice_imag_251 = X_train[:50, 3, :, 1]

# took important 64 samples
alice_real, alice_imag, alice_mag = servesImportant64Samples(alice_real_251, alice_imag_251)

# feature set
combined_train_features = np.column_stack((alice_real, alice_imag, alice_mag))

# fit the scaler to data
scaler.fit(combined_train_features)
# transform the features using the scaler
scaled_train_features = scaler.transform(combined_train_features)


alice_features = scaled_train_features


ocsvm.fit(scaled_train_features)



true_labels = []
predictions = []
# Buffer to hold positive samples before retraining
positive_sample_buffer = []
batch_size = 200  # Retrain after collecting 10 new positive samples

selected_channel = [3,6]


# for cir in range(150):
for cir in range(X_test.shape[0]):
    for channel in selected_channel:
        
        # Determine the true label based on the channel
        if channel == 3:
            true_label = 1  # Legitimate user (Alice)
        else:
            true_label = -1  # Illegitimate user (Not Alice)
        
        true_labels.append(true_label)
        
        # Extract the current test CIR
        incoming_real_251 = X_test[cir, channel, :, 0].reshape(1, -1)
        incoming_imag_251 = X_test[cir, channel, :, 1].reshape(1, -1)

        incoming_real, incoming_imag, incoming_mag = servesImportant64Samples(incoming_real_251, incoming_imag_251)

        combine_test_features = np.column_stack((incoming_real, incoming_imag, incoming_mag))
        
        combine_test_features_scaled = scaler.transform(combine_test_features)
        
        # Predict using the OCC-SVM
        prediction = ocsvm.predict(combine_test_features_scaled)
        predictions.append(prediction[0])

        
        if prediction == 1:
            # Add new positive sample to buffer
            positive_sample_buffer.append(combine_test_features_scaled)
            
            # Retrain the model in batches
            if len(positive_sample_buffer) >= batch_size:
                # np.vstack() stacks arrays in sequence vertically (row wise)
                alice_features = update_features(alice_features, np.vstack(positive_sample_buffer), max_window_size=200)   
                
                # print(alice_features.shape)
                ocsvm.fit(alice_features)
                positive_sample_buffer.clear()  # Reset buffer
            
            

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[-1, 1]).ravel()

print(f"tn: {tn}")
print(f"fp: {fp}")
print(f"fn: {fn}")
print(f"tp: {tp}")

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