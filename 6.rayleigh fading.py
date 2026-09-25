import numpy as np

# Step 1: Generate Rayleigh fading coefficient

h_I = np.random.randn()
h_Q = np.random.randn()

h = h_I + 1j * h_Q

print("In-phase component:", h_I)
print("Quadrature component:", h_Q)
print("Complex fading coefficient:", h)
# Step 2: Calculate fading amplitude

amplitude = np.abs(h)

print("Fading amplitude:", amplitude)
# Generate a new Rayleigh fading channel
H_I = np.random.randn(2, 2)
H_Q = np.random.randn(2, 2)

H_rayleigh = H_I + 1j * H_Q
# Normalize Rayleigh fading channel
channel_power = np.mean(np.abs(H_rayleigh) ** 2)

H_rayleigh = H_rayleigh / np.sqrt(channel_power)

print("\nRayleigh fading MIMO channel matrix:")
print(H_rayleigh)
# Step 4: Transmit BPSK symbols through Rayleigh channel

x = np.array([1, -1], dtype=complex)

# Received signal without noise
y_rayleigh = H_rayleigh @ x

print("\nTransmitted BPSK symbols:")
print(x)

print("\nReceived signal through Rayleigh fading:")
print(y_rayleigh)
# Step 5: Add complex Gaussian noise

snr_db = 0
snr_linear = 10 ** (snr_db / 10)

# Signal power
signal_power = np.mean(np.abs(y_rayleigh) ** 2)

# Complex Gaussian noise standard deviation
noise_std = np.sqrt(signal_power / (2 * snr_linear))

# Generate complex Gaussian noise
noise = noise_std * (
    np.random.randn(2) + 1j * np.random.randn(2)
)

# Received signal with noise
y_noisy_rayleigh = y_rayleigh + noise

print("\nGaussian noise:")
print(noise)

print("\nReceived signal with Rayleigh fading and noise:")
print(y_noisy_rayleigh)
# Step 6: Generate candidate transmitted signals

from itertools import product

candidates = list(product([-1, 1], repeat=2))

print("\nCandidate predictions:")

for candidate in candidates:
    candidate_x = np.array(candidate, dtype=complex)

    predicted_y = H_rayleigh @ candidate_x

    print("\nCandidate:", candidate_x)
    print("Predicted received signal:", predicted_y)
# Step 7: Calculate error for each candidate

print("\nErrors for each candidate:")

for candidate in candidates:
    candidate_x = np.array(candidate, dtype=complex)

    predicted_y = H_rayleigh @ candidate_x

    error = np.linalg.norm(
        y_noisy_rayleigh - predicted_y
    ) ** 2

    print("\nCandidate:", candidate_x)
    print("Error:", error)
# Step 8: ML Detection

min_error = float("inf")
detected_signal = None

for candidate in candidates:
    candidate_x = np.array(candidate, dtype=complex)

    predicted_y = H_rayleigh @ candidate_x

    error = np.linalg.norm(
        y_noisy_rayleigh - predicted_y
    ) ** 2

    if error < min_error:
        min_error = error
        detected_signal = candidate_x

print("\nFinal ML Detection:")
print("Detected signal:", detected_signal)
print("Minimum error:", min_error)
# Step 9: Check detection

if np.array_equal(x, detected_signal):
    print("Detection Status: CORRECT")
else:
    print("Detection Status: INCORRECT")
# Step 10A: BER simulation settings

num_trials = 1000
bit_errors = 0
total_bits = 0

print("\nBER Simulation Settings:")
print("Number of trials:", num_trials)
print("Initial bit errors:", bit_errors)
print("Initial total bits:", total_bits)
# Step 10B: Start BER simulation loop

for trial in range(num_trials):

    # Generate a new Rayleigh fading channel
    H_I = np.random.randn(2, 2)
    H_Q = np.random.randn(2, 2)

    H_rayleigh = H_I + 1j * H_Q

    # Generate random BPSK bits
    bits = np.random.randint(0, 2, size=2)

    x = 2 * bits - 1
    x = x.astype(complex)

    # Transmit BPSK signal through Rayleigh channel
    y_rayleigh = H_rayleigh @ x
    # Add complex Gaussian noise

    snr_db = 5
    snr_linear = 10 ** (snr_db / 10)
    # Use fixed reference signal power

    signal_power = 1.0

    noise_std = np.sqrt(
    signal_power / (2 * snr_linear)
    )

    

    noise = noise_std * (
        np.random.randn(2) + 1j * np.random.randn(2)
    )

    y_noisy_rayleigh = y_rayleigh + noise
    # ML Detection for current trial

    min_error = float("inf")
    detected_signal = None

    for candidate in candidates:
        candidate_x = np.array(candidate, dtype=complex)

        predicted_y = H_rayleigh @ candidate_x

        error = np.linalg.norm(
            y_noisy_rayleigh - predicted_y
        ) ** 2

        if error < min_error:
            min_error = error
            detected_signal = candidate_x
        # Convert detected BPSK symbols to bits

    detected_bits = (
        (detected_signal.real + 1) / 2
    ).astype(int)

    # Count bit errors

    bit_errors += np.sum(bits != detected_bits)

    total_bits += len(bits)

    #print("Trial:", trial + 1)
    #print("Bits:", bits)
    #print("BPSK symbols:", x)

print("\nBER simulation loop completed.")
# Step 10I: Calculate final BER

ber = bit_errors / total_bits

print("\nFinal BER Results:")
print("SNR (dB):", snr_db)
print("Number of trials:", num_trials)
print("Bit errors:", bit_errors)
print("Total bits:", total_bits)
print("BER:", ber)

