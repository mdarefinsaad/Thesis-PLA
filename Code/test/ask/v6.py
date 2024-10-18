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

# Initialize the scaler
scaler = MinMaxScaler()

# Initialize the One-Class SVM
ocsvm = OneClassSVM(kernel='linear', gamma='scale', nu=0.01)

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

def update_features(features, new_cir):
    
    # Remove the oldest CIR (first row)
    updated_features = np.delete(features, 0, axis=0)
    
    # Append the new CIR to the end
    updated_features = np.vstack([updated_features, new_cir])
    return updated_features

#------------------ Spliting the data into training and test sets ------------------
dataset_slice = data['cirs']

# Split data into training and test sets
X_train, X_test = train_test_split(dataset_slice, test_size=0.2, random_state=42)

print('Training set size: ', X_train.shape)
print('Test set size: ', X_test.shape)

# ------------------ Training the model ------------------
# Extract features
alice_real_251 = X_train[:500, 3, :, 0]
alice_imag_251 = X_train[:500, 3, :, 1]

# Extract important 64 samples
alice_real, alice_imag, alice_mag = get64Samples(alice_real_251, alice_imag_251)

# feature set
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
            alice_features = update_features(alice_features, combine_test_features_scaled)
            ocsvm.fit(alice_features)


# Calculate confusion matrix
print(f"\nTotal testing channel: {X_test.shape[0]*2}")

tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[-1, 1]).ravel()

print(f"tp: {tp}")
print(f"tn: {tn}")
print(f"fp: {fp}")
print(f"fn: {fn}")

# Missed Detection Rate (MDR) - how often the model incorrectly predicted illegitimate users as legitimate.
MDR = fp / (fp + tn)

# False Alarm Rate (FAR)
FAR = fn / (fn + tp)

# Gamma calculation
gamma = (tp + fn) / (tn + fp)

# Authentication Rate (AR)
AR = (tp + gamma * tn) / ((tp + fn) + gamma * (tn + fp))



# Accuracy - (how many predictions (both legitimate and illegitimate) were correct out of the total predictions made)
AC = (tp + tn) / (tp + tn + fp + fn)

# PRECISION - Of all the predicted positives (legitimate users), how many were actually correct?
# If Precision is 80%, it means that 80% of the times the model predicted a positive, it was correct, 
# but 20% of the positive predictions were wrong (false positives).
PR = tp / (tp + fp)

# Recall- Of all the actual positives, how many did the model correctly identify?
# If Recall is 90%, it means the model correctly identified 90% of all actual positives, but it missed 10% (false negatives).
RC = tp / (tp + fn)

print(f"Missed detection rate: {MDR}")
print(f"False alarm rate: {FAR}")
print(f"Authenticate rate: {AR}\n")

print(f"Accuracy: {AC}")
print(f"Precision: {PR}")
print(f"Recall: {RC}")
    
