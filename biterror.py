import numpy as np
from itertools import product

# Step 1: Define the channel matrix
H = np.array([
    [1, 2],
    [3, 4]
], dtype=float)

# Step 2: Generate all possible BPSK candidates
candidates = np.array(
    list(product([-1, 1], repeat=2)),
    dtype=float
)

# Step 3: Set simulation parameters
snr_db = 10
num_trials = 1000

# Step 4: Initialize error counters
bit_errors = 0
total_bits = 0

# Step 5: Repeat the communication process
for trial in range(num_trials):

    # Generate a random transmitted BPSK signal
    x = np.random.choice([-1, 1], size=2).astype(float)

    # Signal after passing through the channel
    clean_y = H @ x

    # Calculate signal power
    signal_power = np.mean(clean_y ** 2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise strength
    noise_variance = signal_power / snr_linear
    noise_std = np.sqrt(noise_variance)

    # Generate Gaussian noise
    n = noise_std * np.random.randn(2)

    # Received signal
    y_noisy = clean_y + n

    # ML detection
    best_candidate = None
    best_error = np.inf

    for candidate in candidates:

        predicted_y = H @ candidate

        error = np.linalg.norm(y_noisy - predicted_y) ** 2

        if error < best_error:
            best_error = error
            best_candidate = candidate

    # Count the detected bit errors
    bit_errors += np.sum(x != best_candidate)

    # Count total transmitted bits
    total_bits += len(x)

# Step 6: Calculate BER
ber = bit_errors / total_bits

print("SNR:", snr_db, "dB")
print("Number of trials:", num_trials)
print("Bit errors:", bit_errors)
print("Total bits:", total_bits)
print("BER:", ber)
