import numpy as np
import matplotlib.pyplot as plt
import math
import cmath

def sign_change(num1 , num2):
    if (num1 < 0 and num2 >= 0):
        return 1
    if (num1 >= 0 and num2 < 0):
        return 1
    return 0

def zero_crossing_rate(audio):
    num_cross = 0
    num_samples = audio.shape[0]
    for i in range(1, num_samples):
        num_cross += sign_change(audio[i - 1], audio[i])
    return num_cross / (num_samples - 1)

def root_mean_square(audio):
    num_samples = audio.shape[0]
    square_mean = np.square(audio).sum(axis=0) / num_samples
    return np.sqrt(square_mean)