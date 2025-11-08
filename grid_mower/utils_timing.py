# utils_timing.py
import numpy as np

def estimate_robot_timeline(waypoints, v_robot_cells_per_step=1.0):
    """
    waypoints: (N,2) with (x, y) in *grid cells*
    v_robot_cells_per_step: 로봇 속도(셀/스텝)

    return: t_robot (N,) 각 웨이포인트 예상 도달 시각(스텝)
    """
    wps = np.asarray(waypoints, dtype=float)
    if len(wps) == 0:
        return np.array([], dtype=float)
    d = np.zeros(len(wps), dtype=float)
    for i in range(1, len(wps)):
        d[i] = np.linalg.norm(wps[i] - wps[i-1], ord=2)
    t = np.cumsum(d) / max(v_robot_cells_per_step, 1e-6)
    return t