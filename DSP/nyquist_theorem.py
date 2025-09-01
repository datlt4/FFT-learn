import numpy as np
import matplotlib.pyplot as plt

# 1. Tạo tín hiệu sóng sin liên tục (tín hiệu analog)
# Giả sử chúng ta có một tín hiệu sin với tần số là 5 Hz.
# Tần số tín hiệu (f_max)
f_signal = 5  # Hz

# Thời gian và số điểm cho tín hiệu liên tục để có đồ thị mượt mà
t_continuous = np.linspace(0, 1, 1000, endpoint=False)
x_continuous = np.sin(2 * np.pi * f_signal * t_continuous)

# 2. Lấy mẫu tín hiệu với các tốc độ khác nhau

# Định lý Nyquist nói rằng f_s > 2 * f_max
# f_max của chúng ta là 5 Hz, vậy tốc độ Nyquist là 10 Hz.
# Chúng ta sẽ lấy mẫu với tốc độ lớn hơn 10 Hz (ví dụ 15 Hz).

# a) Tốc độ lấy mẫu TUÂN THỦ định lý Nyquist
f_s_good = 15  # Hz, fs > 2 * f_max (10 Hz)
T_s_good = 1 / f_s_good
n_good = np.arange(0, 1 / T_s_good, 1) * T_s_good
x_sampled_good = np.sin(2 * np.pi * f_signal * n_good)

# b) Tốc độ lấy mẫu VI PHẠM định lý Nyquist
# Chúng ta sẽ lấy mẫu với tốc độ nhỏ hơn 10 Hz (ví dụ 8 Hz).
f_s_bad = 8  # Hz, fs < 2 * f_max (10 Hz)
T_s_bad = 1 / f_s_bad
n_bad = np.arange(0, 1 / T_s_bad, 1) * T_s_bad
x_sampled_bad = np.sin(2 * np.pi * f_signal * n_bad)

# 3. Vẽ đồ thị để minh họa
plt.figure(figsize=(12, 8))

# Đồ thị 1: Tốc độ lấy mẫu TỐT
plt.subplot(2, 1, 1)
plt.plot(t_continuous, x_continuous, label='Tín hiệu Analog (5 Hz)', color='blue', alpha=0.7)
plt.stem(n_good, x_sampled_good, label=f'Lấy mẫu (fs = {f_s_good} Hz)', basefmt=' ', linefmt='red')
plt.title(f'Tuân thủ Định lý Nyquist: fs = {f_s_good} Hz > 2 * f_max (10 Hz)')
plt.xlabel('Thời gian (s)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.legend()

# Đồ thị 2: Tốc độ lấy mẫu XẤU (Aliasing)
plt.subplot(2, 1, 2)
plt.plot(t_continuous, x_continuous, label='Tín hiệu Analog (5 Hz)', color='blue', alpha=0.7)
plt.stem(n_bad, x_sampled_bad, label=f'Lấy mẫu (fs = {f_s_bad} Hz)', basefmt=' ', linefmt='red')
# Vẽ sóng sin "ảo" mà tín hiệu số thực sự biểu diễn
f_alias = np.abs(f_signal - f_s_bad)  # Tính tần số aliasing
x_alias = np.sin(2 * np.pi * f_alias * t_continuous)
plt.plot(t_continuous, x_alias, '--', color='green', label=f'Tín hiệu Aliasing (Tần số = {f_alias} Hz)')
plt.title(f'Vi phạm Định lý Nyquist: fs = {f_s_bad} Hz < 2 * f_max (10 Hz) -> Aliasing')
plt.xlabel('Thời gian (s)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()