# ALPLA demo- Implementation with a small scale dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

# Load measurement file:
measurement = np.load('dataset/meas_symm_1.npz', allow_pickle=False)
header, data = measurement['header'], measurement['data']

def get64Samples(real, imag):

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


def update_features(features, new_cir, max_window_size=100, remove_count = 10):
    
    # print(features.shape[0])
    # If the number of rows (samples) exceeds the max window size, remove the oldest sample
    if features.shape[0] >= max_window_size:
        # Remove the oldest CIR (first row)
        features = np.delete(features, np.s_[:remove_count], axis=0)
    
    # Append the new CIR to the end
    updated_features = np.vstack([features , new_cir])
    print(updated_features.shape)
    print()
    
    return updated_features


# Initialize the scaler
scaler = MinMaxScaler()
# Initialize the One-Class SVM
ocsvm = OneClassSVM(kernel='linear', gamma='scale', nu=0.01)

#------------------ Spliting the data into training and test sets ------------------
dataset_slice = data['cirs']

# Split data into training and test sets
X_train, X_test = train_test_split(dataset_slice, test_size=0.2, random_state=42)

# ------------------ Training the model ------------------
# Extract features
alice_real_251 = X_train[:150, 3, :, 0]
alice_imag_251 = X_train[:150, 3, :, 1]

alice_real, alice_imag, alice_mag = get64Samples(alice_real_251, alice_imag_251)
combined_train_features = np.column_stack((alice_real, alice_imag, alice_mag))

# Scaling
scaler.fit(combined_train_features)
scaled_train_features = scaler.transform(combined_train_features)

# Trainning
ocsvm.fit(scaled_train_features)

alice_features = scaled_train_features

# ------------------ Testing the model ------------------
true_labels = []
predictions = []
# Buffer to hold positive samples before retraining
positive_sample_buffer = []
# batch_size = 800  # Retrain after collecting 10 new positive samples

# Channels
selected_channel = [3,6]

for cir in range(X_test.shape[0]):
    for channel in selected_channel:
        
        # Setting True label
        if channel == 3:
            # Legitimate user (Alice)
            true_label = 1  
        else:
            # Illegitimate user (Eve)
            true_label = -1
        
        true_labels.append(true_label)
        
        # Extract feature   
        incoming_real_251 = X_test[cir, channel, :, 0].reshape(1, -1)
        incoming_imag_251 = X_test[cir, channel, :, 1].reshape(1, -1)

        # Extract important 64 samples 
        incoming_real, incoming_imag, incoming_mag = get64Samples(incoming_real_251, incoming_imag_251)

        # Feature set
        combine_test_features = np.column_stack((incoming_real, incoming_imag, incoming_mag))
        
        # Scaling
        combine_test_features_scaled = scaler.transform(combine_test_features)
        
        # Prediction
        prediction = ocsvm.predict(combine_test_features_scaled)
        predictions.append(prediction[0])
        
        
        if prediction == 1:
            # Add new positive sample to buffer
            positive_sample_buffer.append(combine_test_features_scaled)
            # ocsvm.fit(combine_test_features_scaled)
            
            # Retrain the model in batches
            # if len(positive_sample_buffer) >= batch_size:
                
                # Update the features set
                # alice_features = update_features(alice_features, np.vstack(positive_sample_buffer), 800, 700)
                
                # print(np.vstack(positive_sample_buffer).shape)
                # print(alice_features.shape)
                # Retrain the model
                # ocsvm.fit(alice_features)
                
                # Reset buffer
                # positive_sample_buffer.clear()


vector = np.vstack(positive_sample_buffer)
ocsvm.fit(vector)