import numpy as np
import matplotlib.pyplot as plt
import math
import cmath


# Naive Discrete Fourier Transform Application
def dft(audio):
    num_samples = audio.shape[0]
    dft_vector = np.zeros(num_samples, dtype=complex)
    for k in range(num_samples):
        for n in range(num_samples):
            phase = (2 * math.pi * n * k) / num_samples
            dft_vector[k] += audio[n] * cmath.exp(-1j * phase)
    return dft_vector


# Fast Fourier Transform Algorithm
def fft(audio):
    num_samples = audio.shape[0]

    # Recursion base case
    if (num_samples == 1):
        return audio

    # Split vector into even and odd indecies
    half_samples = num_samples // 2

    # Recursive call on each split
    even_vector = fft(audio[::2])
    odd_vector = fft(audio[1::2])

    freq_bins = np.zeros(num_samples, dtype=complex)
    for k in range(num_samples // 2):
        scaled_odd = cmath.exp(-2j * math.pi * k / num_samples) * odd_vector[k]
        freq_bins[k] = even_vector[k] + scaled_odd
        freq_bins[k + half_samples] = even_vector[k] - scaled_odd
    return freq_bins


if __name__ == "__main__":
    sin_vector = np.sin(np.array(range(1024)))
    dft_vector = fft(sin_vector)
    dft_mag = np.abs(dft_vector)
    dft_phase = np.angle(dft_vector)
    plt.plot(range(dft_vector.shape[0]), dft_phase)
    plt.show()
