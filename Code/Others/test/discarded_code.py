
# print(data['cirs'].shape)
# result = data['cirs'][0:2]
# print(result)
# result = data['cirs'][1, :1, :1, :1]


print()
print('Second CIR - First Channel (Eve):')
print(data['cirs'][1][0])



############################################
print('Alice Real:')
print(alice_real)
print('Alice Imaginary:')
print(alice_imag)

print()

print('Eve Real:')
print(eve_real)
print('Eve Imaginary:')
print(eve_imag)


############################################
print('Alice Magnitude:')
print(alice_mag)

print()

print('Eve Magnitude:')
print(eve_mag)

############################################
print('Alice Real Flat:')
print(alice_real_flat)

print()

print('Alice Imaginary Flat:')
print(alice_imag_flat)

print()

print('Alice Magnitude Flat:')
print(alice_mag_flat)


############################################
print('Alice Combined Features:')
print(alice_features)

############################################
print('Alice Scaled Features:')
print(alice_features_scaled)





new_header_channels = {
    'channels': np.array([
        [
            ('t0', 'a0', 0), ('t0', 'a1', 0), ('t0', 'a2', 0), ('a0', 't0', 0),('a0', 'a1', 0), ('a0', 'a2', 0), ('a1', 't0', 0), ('a1', 'a0', 0),('a1', 'a2', 0), ('a2', 't0', 0), ('a2', 'a0', 0), ('a2', 'a1', 0),('t0', 'a0', 1), ('t0', 'a1', 1), ('t0', 'a2', 1)
        ]
    ], dtype=[('transmitter', '<U2'), ('receiver', '<U2'), ('index', 'u1')])
}



# # Define the data for anchor_positions
# anchor_positions_data = np.array([
#     [
#         [204., 2859., 1150.],
#         [5658., 2856., 1150.],
#         [2900., 184., 1150.],
#     ]
# ], dtype='<f8')  # dtype for float64

# # Define the data for channels
# channels_data = np.array([
#     [
#         ('t0', 'a0', 0), ('t0', 'a1', 0), ('t0', 'a2', 0), ('a0', 't0', 0),
#         ('a0', 'a1', 0), ('a0', 'a2', 0), ('a1', 't0', 0), ('a1', 'a0', 0),
#         ('a1', 'a2', 0), ('a2', 't0', 0), ('a2', 'a0', 0), ('a2', 'a1', 0),
#         ('t0', 'a0', 1), ('t0', 'a1', 1), ('t0', 'a2', 1)
#     ]
# ], dtype=[('transmitter', '<U2'), ('receiver', '<U2'), ('index', 'u1')])


real_pl = X_train[0][3][:, 0]
imag_pl = X_train[0][3][:, 1]
mag_pl = np.abs(real_pl + 1j * imag_pl)

# Find the index of the peak value in the magnitude
peak_index = np.argmax(mag_pl)

# Define the range around the peak (20 samples before and 20 after)
start_index = max(0, peak_index - 20)  # Ensuring we don't go below index 0
end_index = min(251, peak_index + 21)  # Ensuring we don't go beyond index 250

# Extract the part of the signal around the peak
real_part_focus = real_pl[start_index:end_index]
imaginary_part_focus = imag_pl[start_index:end_index]

plt.figure(figsize=(10, 6))
plt.title('Channel Impulse Response (CIR) for Channel 3')
plt.plot(real_pl, label='Real focus Part')
plt.plot(imag_pl, label='Imaginary focus Part')
plt.plot(real_part_focus, label='Real focus Part')
plt.plot(imaginary_part_focus, label='Imaginary focus Part')
# plt.plot(mag_pl, label='Magnitude')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()


#------------------ Spliting the data into training and test sets ------------------
dataset_slice = data['cirs']
X_train, X_test = train_test_split(dataset_slice, test_size=0.2, random_state=42)

alice_real = X_train[0, 3, :, 0]
alice_imag = X_train[0, 3, :, 1]
alice_mag = np.abs(alice_real + 1j * alice_imag)

scaler = MinMaxScaler()

alice_real_scaled = scaler.fit_transform([alice_real])
alice_imag_scaled = scaler.fit_transform([alice_imag])
alice_mag_scaled = scaler.fit_transform([alice_mag])

print(alice_real_scaled)



# # Fit the scaler to data
# scaler.fit(combined_train_features)
# # Transform the features using the scaler
# scaled_train_features = scaler.transform(combined_train_features)

# alice_features = scaled_train_features

# # Create labels for your training data (all ones since they're normal)
# y_train = np.ones(alice_features.shape[0])

# # Define custom scoring function
# def custom_scorer(y_true, y_pred):
#     y_pred = np.where(y_pred == 1, 1, 0)
#     y_true = np.where(y_true == 1, 1, 0)
#     return f1_score(y_true, y_pred)

# # Set up parameter grid
# param_grid = {
#     'nu': [0.001, 0.01, 0.05, 0.1],
#     'gamma': ['scale', 'auto', 0.01, 0.1, 1],
#     'kernel': ['rbf', 'linear', 'sigmoid']
# }

# # Set up cross-validation strategy
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Initialize GridSearchCV
# grid_search = GridSearchCV(
#     estimator=OneClassSVM(),
#     param_grid=param_grid,
#     scoring=make_scorer(custom_scorer),
#     cv=cv,
#     n_jobs=-1
# )

# # Fit GridSearchCV
# grid_search.fit(alice_features, y_train)

# # Get the best model
# best_ocsvm = grid_search.best_estimator_

# # Use the best model
# ocsvm = best_ocsvm




def estimate_cfo(signal_real, signal_imag):
    """
    Estimate Carrier Frequency Offset (CFO) from real and imaginary parts of the signal.
    Assumes that the signal is in complex baseband format.
    :param signal_real: Real part of the signal
    :param signal_imag: Imaginary part of the signal
    :return: CFO estimate
    """
    # Create a complex signal from real and imaginary parts
    complex_signal = signal_real + 1j * signal_imag
    
    # Calculate phase difference between consecutive symbols
    phase_diff = np.angle(complex_signal[1:] * np.conj(complex_signal[:-1]))
    
    # Estimate the average phase difference
    avg_phase_diff = np.mean(phase_diff)
    
    # Convert phase difference to CFO (normalized frequency offset)
    cfo = avg_phase_diff / (2 * np.pi)
    return cfo



    # Use decision_function and apply a custom threshold
    # decision_scores = ocsvm.decision_function(combine_test_features_scaled)  # Get decision scores
    # threshold = -0.2  # Custom threshold, adjust this value to tune FP/TP trade-off
    # prediction = (decision_scores >= threshold).astype(int)  # Apply custom threshold
    # predictions.append(prediction[0])
    
    
    
    # init_real_251 = X_train[:7000, 3, :, 0]
    # init_imag_251 = X_train[:7000, 3, :, 1]
    # init_features = np.column_stack((init_real_scaled, init_imag_scaled, init_mag_scaled)).reshape(7000, 192)
    
    
    
    # Calculate performance metrics
    # MDR = fp / (fp + tn) if (fp + tn) > 0 else 0
    # FAR = fn / (fn + tp) if (fn + tp) > 0 else 0
    # gamma = (tp + fn) / (tn + fp) if (tn + fp) > 0 else 0
    # AR = (tp + gamma * tn) / ((tp + fn) + gamma * (tn + fp)) if ((tp + fn) + gamma * (tn + fp)) > 0 else 0
    
    
    # # Initialize the Lasso model with alpha as a regularization parameter
# lasso = Lasso(alpha=0.1)  # You can adjust alpha for desired sparsity

# # Fit the Lasso model to find sparse coefficients
# lasso.fit(dictionary, y)

# # Get the coefficients (sparse representation)
# coefficients = lasso.coef_

# # print(f"Sparse Coefficients: {coefficients}")#


        # pca_attributes.append({
        #     'components': pca.components_,
        #     'explained_variance': pca.explained_variance_
        # })