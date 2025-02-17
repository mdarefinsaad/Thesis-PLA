# ALPLA demo- Implementation with a small scale dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
ocsvm = OneClassSVM(kernel='linear', gamma='auto', nu=0.01)

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

#------------------ Spliting the data into training and test sets ------------------
dataset_slice = data['cirs']

# Split data into training and test sets
X_train, X_test = train_test_split(dataset_slice, test_size=0.2, random_state=42)

alice_real_251 = X_train[:10, 3, :, 0]
alice_imag_251 = X_train[:10, 3, :, 1]

alice_real, alice_imag, alice_mag = servesImportant64Samples(alice_real_251, alice_imag_251)

alice_real_scaled = scaler.fit_transform(alice_real).flatten()
alice_imag_scaled = scaler.fit_transform(alice_imag).flatten()
alice_mag_scaled = scaler.fit_transform(alice_mag).flatten()

alice_features = np.column_stack((alice_real_scaled, alice_imag_scaled, alice_mag_scaled)).reshape(10, -1)
ocsvm.fit(alice_features)

print(ocsvm.predict(alice_features))