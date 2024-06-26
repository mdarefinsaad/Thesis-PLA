
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
