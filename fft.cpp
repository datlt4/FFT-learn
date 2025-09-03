#include <iostream>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>

#define N_samples 1024       // Number of samples in the FFT
#define log2_N 10            // log2(N_samples) = 10 for 1024 points
#define frequence_sample 128 // Sampling frequency in Hz

// Return codes for error handling
enum ret_code
{
    ret_success = 0,
    ret_fail_memory_generate_order = -1,
    ret_fail_memory_generate_signals = -2,
    ret_fail_memory_generate_W_k_N = -3,
    ret_fail_memory_generate_W_k_N_inv = -4,
    ret_fail_memory_fft = -5,
    ret_fail_memory_ifft = -6,
    ret_fail_memory_abs_fft = -7,
    ret_save_file = -8
};

// Utility: reallocates pointer array memory
template <typename T>
bool reallocate_ptr(T **x, int N)
{
    if (*x != nullptr)
    {
        delete[] *x;
    }
    *x = new T[N];
    if (*x == nullptr)
    {
        return false;
    }
    return true;
}

// Reverse bits for FFT bit-reversal ordering
int reverse(int x, int _N)
{
    int y = 0;
    for (int i = 0; i < _N; ++i)
    {
        y = (y << 1) + (x & 1);
        x >>= 1;
    }
    return y;
}

// Generate bit-reversal order lookup table
bool generate_order(unsigned int **order, int N)
{
    if (*order != nullptr)
    {
        delete[] *order;
    }
    *order = new unsigned int[N];
    if (*order == nullptr)
    {
        return false;
    }
    for (int i = 0; i < N; ++i)
    {
        (*order)[i] = reverse(i, log2_N);
    }
    return true;
}

/** Generate parabolic signals
 * @param A Amplitude
 * @param f Wave frequency in Hz
 * @param t time to get the value (ms)
 * @return value of signal in timestamp
*/
float generate_parabolic_signals(float A, float f, float t)
{
    return A * pow(t * f, 2) * cos(2 * M_PI * f * t);
}

/** Generate sine signals
 * @param A Amplitude
 * @param f Wave frequency in Hz
 * @param t time to get the value (ms)
 * @return value of signal in timestamp
*/
float generate_sine_signals(float A, float f, float t)
{
    return A * sin(2 * M_PI * f * t);
}   

/** Generate cosine signals
 * @param A Amplitude
 * @param f Wave frequency in Hz
 * @param t time to get the value (ms)
 * @return value of signal in timestamp
*/
float generate_cosine_signals(float A, float f, float t)
{
    return A * cos(2 * M_PI * f * t);
}

/** Generate sawtooth signals
 * @param A Amplitude
 * @param f Wave frequency in Hz
 * @param t time to get the value (ms)
 * @return value of signal in timestamp
*/
float generate_sawtooth_signals(float A, float f, float t)
{
    return A * (2 * (t * f - floor(t * f + 0.5)));
}

/** Generate square signals
 * @param A Amplitude
 * @param f Wave frequency in Hz
 * @param t time to get the value (ms)
 * @return value of signal in timestamp
*/
float generate_square_signals(float A, float f, float t)
{
    return A * (fmod(t * f, 1) < 0.5 ? 1 : -1);
}

/**
 * Generate triangle signals
 * @param A Amplitude
 * @param f Wave frequency in Hz
 * @param t time to get the value (ms)
 * @return value of signal in timestamp
 */
float generate_triangle_signals(float A, float f, float t)
{
    return A * (1 - fabs(fmod(t * f, 1) - 0.5) * 2);
}

/** Generate white noise
 * @param A Amplitude
 * @return value of white noise
 */
float generate_white_noise(float A)
{
    return A * (rand() / (float)RAND_MAX * 2 - 1);
}

/** Generate pink noise
 * @param A Amplitude
 * @return value of pink noise
 */
float generate_pink_noise(float A)
{
    static float pink[16] = {0};
    static int index = 0;
    float white = generate_white_noise(A);
    pink[index] = (pink[index] + white) / 2;
    index = (index + 1) % 16;
    return pink[index];
}

// Generate a test signal + noise
bool generate_signals(float **x_RE, float **x_IM, int N, float fs)
{
    if (!reallocate_ptr<float>(x_RE, N))
        return false;
    if (!reallocate_ptr<float>(x_IM, N))
        return false;

    float f_0 = 0.1, f1 = 0.7, f2 = 1.3, f3 = 2.5, f4 = 3.7, f5 = 5.1, f6 = 0.01; // Frequencies in Hz
    float delta_t = 1.0 / fs;
    float t = 0;

    for (int i = 0; i < N; ++i)
    {
        // Two sine waves
        (*x_RE)[i] = generate_sine_signals(0.5, f1, t)
                    + generate_sine_signals(0.3, f2, t)
                    + generate_sine_signals(0.2, f3, t)
                    + generate_sine_signals(0.1, f4, t)
                    + generate_sine_signals(0.05, f5, t)
                    + generate_sine_signals(0.025, f6, t)
                    + generate_pink_noise(0.2);
        (*x_IM)[i] = 0;
        t += delta_t;
    }
    return true;
}

// Precompute twiddle factors W_k_N for FFT
// Stored in a flat 1D array of size N-1, indexed per FFT stage
bool generate_W_k_N(float **W_k_N_RE, float **W_k_N_IM, int N, bool inv = false)
{
    if (!reallocate_ptr<float>(W_k_N_RE, N - 1))
        return false;
    if (!reallocate_ptr<float>(W_k_N_IM, N - 1))
        return false;
    for (int _N = 0; _N < log2_N; ++_N)
    {
        for (int k = 0; k < (1 << _N); ++k)
        {
            int _k = k & ((1 << _N) - 1);
            int w_idx = (1 << _N) - 1 + _k; // Index into flattened twiddle array
            double angle = 2 * M_PI * _k / (1 << (_N + 1));
            (*W_k_N_RE)[w_idx] = cos(angle);
            (*W_k_N_IM)[w_idx] = inv ? sin(angle) : -sin(angle);
        }
    }
    return true;
}

// Perform iterative Cooley-Tukey FFT using precomputed twiddle factors
bool fft(float **x_RE, float **x_IM, float **X_RE, float **X_IM, float **freq,
         float fs, int N, unsigned int **order,
         float **W_k_N_RE, float **W_k_N_IM, float factor = 1.0f)
{
    if (!reallocate_ptr<float>(X_RE, N))
        return false;
    if (!reallocate_ptr<float>(X_IM, N))
        return false;
    if (!reallocate_ptr<float>(freq, N))
        return false;

    // Bit-reversal reordering of input signal
    for (int i = 0; i < N; ++i)
    {
        (*X_RE)[i] = (*x_RE)[(*order)[i]];
        (*X_IM)[i] = (*x_IM)[(*order)[i]];
    }

    // FFT stages
    for (int i = 0; i < log2_N; ++i)
    {
        unsigned int _2_i = 1 << i;  // Half size of current FFT stage
        unsigned int _N = _2_i << 1; // Full size of current FFT stage

        for (int k = 0; k < N; ++k)
        {
            if (k & _2_i)
                continue; // Skip already processed indices

            unsigned int _k = k & (_2_i - 1);
            unsigned int W_idx = _k + (_2_i - 1);

            // Butterfly computation
            float A_RE = (*X_RE)[k] + (*W_k_N_RE)[W_idx] * (*X_RE)[k + _2_i] - (*W_k_N_IM)[W_idx] * (*X_IM)[k + _2_i];
            float A_IM = (*X_IM)[k] + (*W_k_N_RE)[W_idx] * (*X_IM)[k + _2_i] + (*W_k_N_IM)[W_idx] * (*X_RE)[k + _2_i];
            float B_RE = (*X_RE)[k] - (*W_k_N_RE)[W_idx] * (*X_RE)[k + _2_i] + (*W_k_N_IM)[W_idx] * (*X_IM)[k + _2_i];
            float B_IM = (*X_IM)[k] - (*W_k_N_RE)[W_idx] * (*X_IM)[k + _2_i] - (*W_k_N_IM)[W_idx] * (*X_RE)[k + _2_i];

            (*X_RE)[k] = A_RE;
            (*X_IM)[k] = A_IM;
            (*X_RE)[k + _2_i] = B_RE;
            (*X_IM)[k + _2_i] = B_IM;

            // Frequency axis mapping
            (*freq)[k] = k * fs / N;
        }
    }

    if (factor != 1.0f)
    {
        for (int i = 0; i < N; ++i)
        {
            (*X_RE)[i] *= factor;
            (*X_IM)[i] *= factor;
        }
    }
    return true;
}

// Compute magnitude of FFT result
bool abs_fft(float **X_RE, float **X_IM, float **X_abs, int N)
{
    if (!reallocate_ptr<float>(X_abs, N))
        return false;
    for (int i = 0; i < N; ++i)
    {
        (*X_abs)[i] = sqrt((*X_RE)[i] * (*X_RE)[i] + (*X_IM)[i] * (*X_IM)[i]);
    }
    return true;
}

// Butterworth Filter - Low-pass filter
// Butterworth Filter - High-pass filter
// Butterworth Filter - Band-pass filter
// Butterworth Filter - Band-stop filter

// Chebyshev Filter - Low-pass filter
bool chebyshev_lowpass(float **x_RE, float **x_IM, float **X_RE, float **X_IM, float cutoff_freq, float ripple, int N)
{
    // Design and apply Chebyshev low-pass filter
    return true;
}

// Chebyshev Filter - High-pass filter
// Chebyshev Filter - Band-pass filter
// Chebyshev Filter - Band-stop filter

// Save binary data to file
template <typename T>
bool save_to_file(const char *filename, T *data, int N)
{
    std::ofstream bin_file(filename, std::ios::binary);
    if (!bin_file.is_open())
        return false;
    bin_file.write(reinterpret_cast<const char *>(data), N * sizeof(T));
    bin_file.close();
    return true;
}

int main(int argc, char const *argv[])
{
    unsigned int *order = nullptr;
    float *x_RE = nullptr;
    float *x_IM = nullptr;
    float *X_RE = nullptr;
    float *X_IM = nullptr;
    float *X_abs = nullptr;
    float *X_RE_inv = nullptr;
    float *X_IM_inv = nullptr;
    // float *x_abs_inv = nullptr;
    float *W_k_N_RE = nullptr;
    float *W_k_N_IM = nullptr;
    float *W_k_N_inv_RE = nullptr;
    float *W_k_N_inv_IM = nullptr;
    float *freq = nullptr;

    // Generate FFT bit-reversal order
    if (!generate_order(&order, N_samples))
        return ret_code::ret_fail_memory_generate_order;

    // Generate test input signal
    if (!generate_signals(&x_RE, &x_IM, N_samples, frequence_sample))
        return ret_code::ret_fail_memory_generate_signals;

    // Save time-domain signal to file
    if (!save_to_file("signals.bin", x_RE, N_samples))
    {
        std::cerr << "Failed to save signals to file." << std::endl;
        return ret_code::ret_save_file;
    }

    // Precompute FFT twiddle factors
    if (!generate_W_k_N(&W_k_N_RE, &W_k_N_IM, N_samples))
    {
        std::cerr << "Failed to generate FFT twiddle factors." << std::endl;
        return ret_code::ret_fail_memory_generate_W_k_N;
    }

    // Measure performance
    auto start_fft = std::chrono::high_resolution_clock::now();
    unsigned int n_iter = 1000; // Number of iterations for timing
    for (int i = 0; i < n_iter; ++i)
    {
        if (!fft(&x_RE, &x_IM, &X_RE, &X_IM, &freq, frequence_sample, N_samples, &order, &W_k_N_RE, &W_k_N_IM))
        {
            std::cerr << "Failed to perform FFT." << std::endl;
            return ret_code::ret_fail_memory_fft;
        }

        if (!abs_fft(&X_RE, &X_IM, &X_abs, N_samples))
        {
            std::cerr << "Failed to compute FFT magnitude." << std::endl;
            return ret_code::ret_fail_memory_abs_fft;
        }
    }
    auto end_fft = std::chrono::high_resolution_clock::now();

    // Print performance results
    auto duration_fft_us = std::chrono::duration_cast<std::chrono::microseconds>(end_fft - start_fft);
    unsigned int duration_fft = duration_fft_us.count() / n_iter;
    std::cout << "Time elapsed: " << duration_fft / 1000. << " ms"
              << "\tfps: " << 1. / duration_fft * 1000000. << std::endl;

    // Save FFT magnitude to file
    if (!save_to_file("fft_result.bin", X_abs, N_samples))
    {
        std::cerr << "Failed to save FFT result to file." << std::endl;
        return ret_code::ret_save_file;
    }

    // Precompute IFFT twiddle factors
    if (!generate_W_k_N(&W_k_N_inv_RE, &W_k_N_inv_IM, N_samples, true))
    {
        std::cerr << "Failed to generate IFFT twiddle factors." << std::endl;
        return ret_code::ret_fail_memory_generate_W_k_N_inv;
    }

    // Measure performance
    auto start_ifft = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; ++i)
    {
        float factor = 1.0f / N_samples;
        if (!fft(&X_RE, &X_IM, &X_RE_inv, &X_IM_inv, &freq, frequence_sample, N_samples, &order, &W_k_N_inv_RE, &W_k_N_inv_IM, factor))
        {
            std::cerr << "Failed to perform IFFT." << std::endl;
            return ret_code::ret_fail_memory_ifft;
        }

        // if (!abs_fft(&X_RE_inv, &X_IM_inv, &x_abs_inv, N_samples))
        // {
        //     std::cerr << "Failed to compute IFFT magnitude." << std::endl;
        //     return ret_code::ret_fail_memory_abs_fft;
        // }
    }
    auto end_ifft = std::chrono::high_resolution_clock::now();

    // Print performance results
    auto duration_ifft_us = std::chrono::duration_cast<std::chrono::microseconds>(end_ifft - start_ifft);
    unsigned int duration_ifft = duration_ifft_us.count() / n_iter;
    std::cout << "Time elapsed: " << duration_ifft / 1000. << " ms"
              << "\tfps: " << 1. / duration_ifft * 1000000. << std::endl;

    // Save IFFT result to file
    if (!save_to_file("ifft_result.bin", X_RE_inv, N_samples))
    {
        std::cerr << "Failed to save IFFT result to file." << std::endl;
        return ret_code::ret_save_file;
    }

    // Free memory
    delete[] X_RE;
    delete[] X_IM;
    delete[] x_RE;
    delete[] x_IM;
    delete[] X_abs;
    delete[] X_RE_inv;
    delete[] X_IM_inv;
    // delete[] x_abs_inv;
    delete[] order;
    delete[] W_k_N_RE;
    delete[] W_k_N_IM;
    delete[] W_k_N_inv_RE;
    delete[] W_k_N_inv_IM;
    delete[] freq;

    return ret_code::ret_success;
}

// Compile & run:
// g++ -std=c++17 fft.cpp -o fft -lm && ./fft
