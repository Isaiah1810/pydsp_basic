import numpy as np
import scipy
import matplotlib.pyplot as plt
import math

FRAME_LENGTH = 1024
HOP_LENGTH = 512  # 50% Frame Overlap
 


def hann_window(audio):
    num_samples = audio.shape[0]
    hann_vector = np.array([0.5 * (1 - math.cos(
        2 * math.pi * i / num_samples)) for i in range(num_samples)])
    return audio * hann_vector

def hamming_window(audio):
    num_samples = audio.shape[0]
    hamm_vector = np.array([0.54 - 0.46 * (1 - math.cos(
        2 * math.pi * i / num_samples)) for i in range(num_samples)])
    return audio * hamm_vector

def split_audio(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
                window=None):
    num_samples = audio.shape[0]
    num_frames = ((num_samples - frame_length) // hop_length) + 1
    frames = []
    
    #Split up frames
    for i in range(num_frames):
        frame = audio[i * hop_length: (i + 1) * hop_length]
        if (window is not None):
            if (window == "hamming"):
                frame = hamming_window(frame)
            elif (window == "hann"):
                frame = hann_window(frame)
        frames.append(frame)

    #Handle last frame
    start = (num_frames - 1) * hop_length
    if (start + frame_length > num_samples):
        last_frame = np.zeros(frame_length, dtype=audio.dtype)
        last_frame[:num_samples - start] = audio[start:]
        if (window is not None):
            if (window == "hamming"):
                last_frame = hamming_window(last_frame)
            elif (window == "hann"):
                last_frame = hann_window(last_frame)
        frames.append(last_frame)

    return frames
    
