# src/optimized_counter.py
import numpy as np
from scipy.signal import find_peaks

def optimized_count_jumps(pose_data, fps=30, min_interval=0.4):
    """
    多特征融合计数：髋部 + 脚踝 + 节奏
    """
    T, V, C = pose_data.shape
    if T < 10: return 0, [], np.zeros(T)

    # 特征1：髋部垂直运动
    hip_y = (pose_data[:, 11, 1] + pose_data[:, 12, 1]) / 2
    # 特征2：脚踝垂直运动
    ankle_y = (pose_data[:, 15, 1] + pose_data[:, 16, 1]) / 2
    # 特征3：脚踝与髋部相对位移
    rel_y = ankle_y - hip_y

    # 归一化
    hip_sig = (hip_y - hip_y.min()) / (hip_y.max() - hip_y.min() + 1e-6)
    ankle_sig = (ankle_y - ankle_y.min()) / (ankle_y.max() - ankle_y.min() + 1e-6)
    rel_sig = (rel_y - rel_y.min()) / (rel_y.max() - rel_y.min() + 1e-6)

    # 加权融合信号
    signal = 0.4 * (1 - hip_sig) + 0.4 * (1 - ankle_sig) + 0.2 * rel_sig

    # 自适应阈值
    threshold = np.mean(signal) + 0.5 * np.std(signal)
    distance = max(5, int(fps * min_interval))

    peaks, _ = find_peaks(signal, height=threshold, distance=distance, prominence=0.1)

    return len(peaks), peaks.tolist(), signal.tolist()
