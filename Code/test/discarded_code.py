
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

