import numpy as np

# Number of transmitted bits
N = 10000

# SNR
snr_db = 15

# Generate random bits
bits = np.random.randint(0, 2, N)

# BPSK mapping
# 0 -> -1
# 1 -> +1
x = 2 * bits - 1

print("First 10 bits:", bits[:10])
print("First 10 BPSK symbols:", x[:10])
1# Step 2: Generate Rayleigh fading channel

h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

print("First 5 channel coefficients:", h[:5])

print("Average channel power:", np.mean(np.abs(h)**2))
# Step 3: Convert SNR from dB to linear

snr_linear = 10 ** (snr_db / 10)

print("SNR (dB):", snr_db)
print("SNR (linear):", snr_linear)
# Step 4: Calculate noise power spectral density

N0 = 1 / snr_linear

print("Noise power spectral density (N0):", N0)
# Step 5: Generate complex AWGN noise

noise = np.sqrt(N0 / 2) * (
    np.random.randn(N) + 1j * np.random.randn(N)
)

print("First 5 noise samples:", noise[:5])
# Step 6: Received signal

y = h * x + noise

print("First 5 received signals:", y[:5])
# Step 7: Equalization

y_eq = y / h

print("First 5 equalized signals:", y_eq[:5])
# Step 8: BPSK detection

detected_symbols = np.where(
    np.real(y_eq) >= 0, 1, -1
)

print("First 5 detected symbols:", detected_symbols[:5])
# Step 9: Convert detected symbols back into bits

detected_bits = (detected_symbols + 1) // 2

print("First 10 detected bits:", detected_bits[:10])
# Step 10: Calculate BER

bit_errors = np.sum(bits != detected_bits)

ber = bit_errors / N

print("SNR (dB):", snr_db)
print("Total bits:", N)
print("Bit errors:", bit_errors)
print("BER:", ber)
# Step 11: Theoretical Rayleigh BER

theoretical_ber = 0.5 * (
    1 - np.sqrt(snr_linear / (1 + snr_linear))
)

print("Theoretical BER:", theoretical_ber)

