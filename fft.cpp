#include <iostream>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>

#define N_samples 1024
#define log2_N 10 // log2(1024) = 10
#define frequence_sample 128

enum ret_code
{
    ret_success=0,
    ret_fail_memory_generate_order=-1,
    ret_fail_memory_generate_signals=-2,
    ret_fail_memory_generate_W_k_N=-3,
    ret_fail_memory_fft=-4,
    ret_fail_memory_abs_fft=-5

};

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

int reverse(int x, int _N)
{
    int y = 0;
    for (int i =0; i< _N; ++i)
    {
        y = (y << 1) + (x& 1);
        x >>= 1;
    }
    return y;
}

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

bool generate_signals(float **x, int N, float fs)
{
    if (!reallocate_ptr<float>(x, N))
    {
        return false;
    }
    // Tạo tín hiệu: 0.7 Hz và 1.3 Hz
    float f1 = 0.7, f2 = 1.3;
    float delta_t = 1.0 / fs;
    float t = 0;

    for (int i = 0; i < N; ++i)
    {
        (*x)[i] = 0.5 * sin(2 * M_PI * f1 * t) + 0.3 * sin(2 * M_PI * f2 * t);
        t += delta_t;
        // Thêm nhiễu Gaussian
        float noise = 0.1 * rand() / RAND_MAX - 0.05; // Giả sử nhiễu Gaussian có trung bình 0 và phương sai 0.01
        (*x)[i] += noise;
    }
    return true;
}

bool generate_W_k_N(float **W_k_N_RE, float **W_k_N_IM, int N)
{
    if (!reallocate_ptr<float>(W_k_N_RE, N-1))
    {
        return false;
    }
    if (!reallocate_ptr<float>(W_k_N_IM, N-1))
    {
        return false;
    }
    for (int _N = 0; _N < log2_N; ++_N)
    {
        for (int k = 0; k < (1 << _N); ++k)
        {
            int _k = k & ((1 << _N) - 1);
            int w_idx = (1 << _N) - 1 + _k; // Reverse index for W_k_N
            double angle = 2 * M_PI * _k / (1 << (_N + 1));
            (*W_k_N_RE)[w_idx] = cos(angle);
            (*W_k_N_IM)[w_idx] = -sin(angle);
        }
    }
    return true;
}

bool fft(float **x, float** X_RE, float** X_IM, float** freq, float fs, int N, unsigned int **order, float **W_k_N_RE, float **W_k_N_IM)
{
    if (!reallocate_ptr<float>(X_RE, N))
    {
        return false;
    }
    if (!reallocate_ptr<float>(X_IM, N))
    {
        return false;
    }
    if (!reallocate_ptr<float>(freq, N))
    {
        return false;
    }
    for (int i = 0; i < N; ++i)
    {
        (*X_RE)[i] = (*x)[(*order)[i]];
        (*X_IM)[i] = 0;
    }

    for (int i = 0; i < log2_N; ++i)
    {
        unsigned int _2_i = 1 << i;
        unsigned int _N = _2_i << 1;
        for (int k = 0; k < N; ++k)
        {
            if (k & _2_i)
            {
                continue;
            }
            unsigned int _k = k & (_2_i - 1);
            unsigned int W_idx = _k + (_2_i - 1);
            float A_RE = (*X_RE)[k] + (*W_k_N_RE)[W_idx] * (*X_RE)[k + _2_i] - (*W_k_N_IM)[W_idx] * (*X_IM)[k + _2_i];
            float A_IM = (*X_IM)[k] + (*W_k_N_RE)[W_idx] * (*X_IM)[k + _2_i] + (*W_k_N_IM)[W_idx] * (*X_RE)[k + _2_i];
            float B_RE = (*X_RE)[k] - (*W_k_N_RE)[W_idx] * (*X_RE)[k + _2_i] + (*W_k_N_IM)[W_idx] * (*X_IM)[k + _2_i];
            float B_IM = (*X_IM)[k] - (*W_k_N_RE)[W_idx] * (*X_IM)[k + _2_i] - (*W_k_N_IM)[W_idx] * (*X_RE)[k + _2_i];
            (*X_RE)[k] = A_RE;
            (*X_IM)[k] = A_IM;
            (*X_RE)[k + _2_i] = B_RE;
            (*X_IM)[k + _2_i] = B_IM;
            (*freq)[k] = k * fs / N;
        }
    }
    return true;
}

bool abs_fft(float **X_RE, float **X_IM, float **X_abs, int N)
{
    if (!reallocate_ptr<float>(X_abs, N))
    {
        return false;
    }
    for (int i = 0; i < N; ++i)
    {
        (*X_abs)[i] = sqrt((*X_RE)[i] * (*X_RE)[i] + (*X_IM)[i] * (*X_IM)[i]);
    }
    return true;
}

int main(int argc, char const *argv[])
{
    unsigned int *order = nullptr;
    float *x =  nullptr;
    float *W_k_N_RE = nullptr;
    float *W_k_N_IM = nullptr;
    float *X_RE = nullptr;
    float *X_IM = nullptr;
    float *freq = nullptr;
    float *X = nullptr;

    if (!generate_order(&order, N_samples))
    {
        return ret_fail_memory_generate_order;
    }

    if (!generate_signals(&x, N_samples, frequence_sample))
    {
        return ret_fail_memory_generate_signals;
    }

    if (!generate_W_k_N(&W_k_N_RE, &W_k_N_IM, N_samples))
    {
        return ret_fail_memory_generate_W_k_N;
    }

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int n_iter = 1000;
    for (int i = 0; i < n_iter; ++i)
    {
        if (!fft(&x, &X_RE, &X_IM, &freq, frequence_sample, N_samples, &order, &W_k_N_RE, &W_k_N_IM))
        {
            return ret_fail_memory_fft;
        }
        if (!abs_fft(&X_RE, &X_IM, &X, N_samples))
        {
            return ret_fail_memory_abs_fft;
        }
    }

    std::ofstream bin_file("fft.bin", std::ios::binary);
    if (bin_file.is_open())
    {
        bin_file.write(reinterpret_cast<const char*>(X), N_samples * sizeof(float));
        bin_file.close();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    unsigned int duration = duration_us.count() / n_iter;
    std::cout << "Time elapsed: " << duration/1000. << " ms" << "\tfps: " << 1./duration * 1000000. << std::endl;

    delete[] x;
    delete[] order;
    delete[] W_k_N_RE;
    delete[] W_k_N_IM;
    delete[] X_RE;
    delete[] X_IM;
    delete[] freq;
    return 0;
}

// g++ -std=c++17 fft.cpp -o fft -lm && ./fft
