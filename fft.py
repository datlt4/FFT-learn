import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

# ---------------------------
# Utility function: reverse bits
# Used for bit-reversal permutation in FFT
# ---------------------------
def reverse(x, N):
    y = 0
    for i in range(N):
        y = (y << 1) + (x & 1)
        x >>= 1
    return y

# ---------------------------
# Manual FFT implementation (Cooley-Tukey, iterative)
# Version 1: Explicit twiddle factor computation inside loop
# ---------------------------
def fft(x):
    N = len(x)
    log2_N = int(np.log2(N))
    X = np.zeros_like(x, dtype=complex)

    # Step 1: Apply bit-reversal permutation
    for i in range(N):
        X[i] = x[reverse(i, log2_N)]

    # Step 2: FFT stages
    for i in range(log2_N):
        _2_i = 1 << i
        _N = _2_i << 1
        for k in range(N):
            if k & _2_i:
                continue
            _k = k & (_2_i - 1)
            # Compute twiddle factor for this stage
            _cos = math.cos(2 * math.pi * _k / _N)
            _sin = math.sin(2 * math.pi * _k / _N)
            W_k_N = _cos - 1j * _sin
            # Butterfly computation
            A = X[k] + W_k_N * X[k + _2_i]
            B = X[k] - W_k_N * X[k + _2_i]
            X[k] = A
            X[k + _2_i] = B

    return X

# ---------------------------
# Manual FFT implementation (Cooley-Tukey, iterative)
# Version 2: Standard butterfly with precomputed W_m increment
# ---------------------------
def fft2(x):
    N = len(x)
    log2_N = int(math.log2(N))
    X = np.zeros_like(x, dtype=complex)

    # Step 1: Bit-reversal permutation
    for i in range(N):
        X[i] = x[reverse(i, log2_N)]

    # Step 2: Butterfly computation
    size = 2
    while size <= N:
        half = size // 2
        theta = -2j * math.pi / size
        w_m = np.exp(theta)  # Twiddle factor increment for this stage
        for k in range(0, N, size):
            w = 1
            for j in range(half):
                t = w * X[k + j + half]  # Weighted second half
                u = X[k + j]             # First half
                X[k + j] = u + t
                X[k + j + half] = u - t
                w *= w_m
        size *= 2

    return X

if __name__ == "__main__":
    # ---------------------------
    # Parameters
    # ---------------------------
    fs = 128       # Sampling frequency (Hz)
    T = 8          # Signal duration (seconds)
    N = int(fs * T)  # Total number of samples

    # Time vector
    t = np.linspace(0, T, N, endpoint=False)

    # Create test signal: sum of two sinusoids (0.7 Hz & 1.3 Hz)
    f1, f2 = 0.7, 1.3
    x = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)

    # Add Gaussian noise
    noise = 0.1 * np.random.normal(size=N)
    x += noise

    # Optionally replace with signal from C++ binary file
    with open("signals.bin", "rb") as f:
        x = np.fromfile(f, dtype=np.float32)

    # ---------------------------
    # NumPy FFT for reference
    # ---------------------------
    X_np = np.fft.fft(x)
    print("NumPy FFT time:", timeit.timeit(lambda: np.fft.fft(x), number=1))

    magnitude_np = np.abs(X_np)
    frequencies_np = np.fft.fftfreq(N, 1/fs)  # Frequency bins

    # ---------------------------
    # Manual FFT #1
    # ---------------------------
    my_X = fft(x)
    print("Manual FFT #1 time:", timeit.timeit(lambda: fft(x), number=1))
    df = fs / N
    my_frequencies = np.arange(N) * df
    my_magnitude = np.abs(my_X)

    # ---------------------------
    # Manual FFT #2
    # ---------------------------
    my_X2 = fft2(x)
    print("Manual FFT #2 time:", timeit.timeit(lambda: fft2(x), number=1))
    my_frequencies2 = np.arange(N) * df
    my_magnitude2 = np.abs(my_X2)

    # ---------------------------
    # Load C++ FFT results
    # ---------------------------
    with open("signals.bin", "rb") as f:
        x_cpp = np.fromfile(f, dtype=np.float32)

    with open("fft_result.bin", "rb") as f:
        X_cpp = np.fromfile(f, dtype=np.float32)

    with open("ifft_result.bin", "rb") as f:
        x_cpp_inv = np.fromfile(f, dtype=np.float32)

    # ---------------------------
    # Plot results
    # ---------------------------
    plt.figure(figsize=(12, 5))

    # Original signal (Python)
    plt.subplot(7, 1, 1)
    plt.plot(t, x, label='Signal (Python)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    # NumPy FFT spectrum
    plt.subplot(7, 1, 2)
    plt.plot(frequencies_np, magnitude_np, label='NumPy FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.legend()

    # Manual FFT #1 spectrum
    plt.subplot(7, 1, 3)
    plt.plot(my_frequencies, my_magnitude, label='Manual FFT #1 (Python)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.legend()

    # Manual FFT #2 spectrum
    plt.subplot(7, 1, 4)
    plt.plot(my_frequencies2, my_magnitude2, label='Manual FFT #2 (Python)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.legend()

    # Original signal from C++
    plt.subplot(7, 1, 5)
    plt.plot(t, x_cpp, label='Signal (C++)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    # FFT spectrum from C++
    plt.subplot(7, 1, 6)
    plt.plot(my_frequencies, X_cpp, label='FFT (C++)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.legend()
    
    # IFFT spectrum from C++
    plt.subplot(7, 1, 7)
    plt.plot(my_frequencies, x_cpp_inv, label='IFFT (C++)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.5)
    plt.show()
