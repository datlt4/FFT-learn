import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

def reverse(x, N):
    y = 0
    for i in range(N):
        y = (y << 1) + (x & 1)
        x >>= 1
    return y

def fft(x):
    N = len(x)
    log2_N = int(np.log2(N))
    X = np.zeros_like(x, dtype=complex)
    for i in range(N):
        X[i] = x[reverse(i, log2_N)]
    

    for i in range(log2_N):
        _2_i = 1 << i
        _N = _2_i << 1

        for k in range(N):
            if k & _2_i:
                continue
            _k = k & (_2_i - 1)
            _cos = math.cos(2 * math.pi * _k / _N)
            _sin = math.sin(2 * math.pi * _k / _N)
            W_k_N = _cos - 1j * _sin
            A = X[k] + W_k_N * X[k + _2_i]
            B = X[k] - W_k_N * X[k + _2_i]
            X[k] = A
            X[k + _2_i] = B

    return X

def fft2(x):
    N = len(x)
    log2_N = int(math.log2(N))
    X = np.zeros_like(x, dtype=complex)

    # Step 1: Bit-reversal permutation
    for i in range(N):
        X[i] = x[reverse(i, log2_N)]

    # Step 2: Cooley-Tukey butterfly
    size = 2
    while size <= N:
        half = size // 2
        theta = -2j * math.pi / size
        w_m = np.exp(theta)
        for k in range(0, N, size):
            w = 1
            for j in range(half):
                t = w * X[k + j + half]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + half] = u - t
                w *= w_m
        size *= 2

    return X

if __name__ == "__main__":
    # Tham số
    fs = 128  # Tần số lấy mẫu (Hz)
    # T = 8     # Thời gian (giây)
    T = 8
    N = int(fs * T)  # Số mẫu

    # Tạo thời gian
    t = np.linspace(0, T, N, endpoint=False)

    # Tạo tín hiệu: 5 Hz và 10 Hz
    f1, f2 = 0.7, 1.3
    x = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)

    # Thêm nhiễu Gaussian
    noise = 0.1 * np.random.normal(size=N)
    x += noise

    # Tính FFT
    X_np = np.fft.fft(x)
    # fft(x)
    print(timeit.timeit(lambda: np.fft.fft(x), number=1))

    # Tính độ lớn
    magnitude_np = np.abs(X_np)

    # Tính tần số (chỉ lấy f > 0)
    frequencies_np = np.fft.fftfreq(N, 1/fs)

    # FFT thủ công
    my_X = fft(x)
    print(timeit.timeit(lambda: fft(x), number=1))
    df = fs / N
    my_frequencies = np.arange(N) * df
    my_magnitude = np.abs(my_X)
    
    # FFT thủ công
    my_X2 = fft2(x)
    print(timeit.timeit(lambda: fft2(x), number=1))
    df = fs / N
    my_frequencies2 = np.arange(N) * df
    my_magnitude2 = np.abs(my_X2)

    # Load binary files
    with open("fft.bin", "rb") as f:
        X_cpp = np.fromfile(f, dtype=np.float32)

    # Vẽ tín hiệu trong miền thời gian
    plt.figure(figsize=(12, 5))
    plt.subplot(5, 1, 1)
    plt.plot(t, x, label='Tín hiệu')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Biên độ')
    # plt.title('Tín hiệu trong miền thời gian')
    plt.grid()
    plt.legend()
    
    # Vẽ phổ tần số (f>0)
    plt.subplot(5, 1, 2)
    plt.plot(frequencies_np, magnitude_np, label='Phổ tần số')
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Độ lớn')
    # plt.title('Phổ tần số')
    plt.grid()
    plt.legend()

    # Vẽ tín hiệu trong miền tần số
    plt.subplot(5, 1, 3)
    plt.plot(my_frequencies, my_magnitude, label='FFT thủ công python')
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Độ lớn')
    # plt.title('FFT thủ công python')
    plt.grid()
    plt.legend()

    # Vẽ tín hiệu trong miền tần số
    plt.subplot(5, 1, 4)
    plt.plot(my_frequencies, my_magnitude, label='FFT thủ công chatgpt')
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Độ lớn')
    # plt.title('FFT thủ công chatgpt')
    plt.grid()
    plt.legend()

    # Vẽ phổ tần số (f>0)
    plt.subplot(5, 1, 5)
    plt.plot(my_frequencies, X_cpp, label='FFT thủ công cpp')
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Độ lớn')
    # plt.title('FFT thủ công cpp')
    plt.grid()
    plt.legend()

    # Hiển thị
    plt.subplots_adjust(hspace=0.5)  # Adjust this value (e.g., 0.1 for tighter spacing)
    # plt.tight_layout()
    plt.show()
