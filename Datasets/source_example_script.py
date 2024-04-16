#!/usr/bin/env python3
"""
Example script that shows how to use the data. It calculates the normalized
peak cross-correlation between the CIRs of different channels and shows the
average values.
"""
import sys
import itertools
import numpy
from matplotlib import pyplot


# Define function that calculates the magnitude of a complex CIR:
def cir_magnitude(cir):
    """Calculates the magnitude of the given complex CIR."""
    cir_squared = numpy.array(cir, dtype=numpy.float64) ** 2
    cir_sum = numpy.sum(cir_squared, axis=1)
    return numpy.sqrt(cir_sum)


# Define function that calculates the correlation between two complex CIRs:
def cir_correlation(cir1, cir2):
    """Calculates the peak cross-correlation of the two given complex CIRs."""
    # Calculate CIR magnitudes:
    cir1_mag = cir_magnitude(cir1)
    cir2_mag = cir_magnitude(cir2)

    # Calculate CIR energies:
    energy1 = numpy.sum(numpy.square(cir1_mag))
    energy2 = numpy.sum(numpy.square(cir2_mag))

    # Calculate peak cross-correlation:
    cc = numpy.max(numpy.correlate(cir1_mag, cir2_mag, mode='full'))

    # Normalize it using the energies:
    return cc / numpy.sqrt(energy1 * energy2)


# Define function that returns the label of a given channel:
def channel_label(channel):
    """Returns a label for the given channel."""
    math_labels = {'t0': 'T_0', 'a0': 'A_0', 'a1': 'A_1', 'a2': 'A_2'}

    tx_label = math_labels[channel['transmitter']]
    rx_label = math_labels[channel['receiver']]

    return f'${tx_label} \\Rightarrow {rx_label}, {channel["index"]}$'


# Check command line arguments:
if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} measurement.npz')
    sys.exit(1)


# Load measurement file:
measurement = numpy.load(sys.argv[1], allow_pickle=False)
header, data = measurement['header'], measurement['data']


# Print header information:
print('Anchor positions (coordinates in mm):')
for i, anchor in enumerate(header['anchor_positions'].squeeze()):
    print(f'  A{i}: ({anchor[0]:4.0f}, {anchor[1]:4.0f}, {anchor[2]:4.0f})')

print('Channels:')
channels = header['channels'].squeeze()
for i, c in enumerate(channels):
    print(f'  {i:2d}: {c["transmitter"]} -> {c["receiver"]}'
          f', msg #{c["index"]}')


# Calculate cross-correlation of all CIR combinations:
print('Average CIR cross-correlations:')
average_correlations = numpy.zeros((len(channels), len(channels)))
for i, j in itertools.combinations_with_replacement(range(len(channels)), 2):
    # Create array of correlations:
    ccs = numpy.fromiter(
        (cir_correlation(row['cirs'][i], row['cirs'][j]) for row in data),
        dtype=numpy.float64, count=len(data))

    # Calculate and print average:
    average_correlations[i, j] = numpy.average(ccs)
    average_correlations[j, i] = average_correlations[i, j]
    print(f'  {i:2d} <-> {j:2d}: {average_correlations[i, j]:.2f}')


# Plot it as a matrix:
fig, ax = pyplot.subplots(tight_layout=True)
mat = ax.matshow(average_correlations)

# Add a colorbar:
fig.colorbar(mat)

# Set title and axis labels:
ax.set_title('Average normalized peak cross-correlation between the CIRs')
ax.set_xlabel('Channel A')
ax.set_ylabel('Channel B')

# Set ticks:
channel_labels = [channel_label(c) for c in channels]
tick_positions = list(range(len(channels)))
ax.set_xticks(tick_positions)
ax.set_xticklabels(channel_labels, rotation=90)
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks(tick_positions)
ax.set_yticklabels(channel_labels)

pyplot.show()
