# src/counter.py
import numpy as np
from scipy.signal import find_peaks

def count_jumps(pose_data, fps=30, min_interval=0.4):
    hip_y = (pose_data[:,11,1] + pose_data[:,12,1]) / 2
    signal = -hip_y
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-6)
    peaks, _ = find_peaks(signal, distance=int(fps*min_interval), prominence=0.1)
    return len(peaks), peaks, signal
