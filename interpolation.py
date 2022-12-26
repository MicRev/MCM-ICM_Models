import numpy as np
from scipy import interpolate

def spline(xs: np.array, ys: np.array, new_x):
    """三次样条插值

    Parameters
    ----------
    xs : np.array
        已知点的横坐标
    ys : np.array
        已知点的纵坐标
    new_x : float | np.array
        插值点的横坐标

    Returns
    -------
    float | np.array
        插值点的纵坐标
    """
    t = interpolate.splrep(xs, ys)
    return interpolate.splev(new_x, t)

def pchip(xs: np.array, ys: np.array, new_x):
    """Hermit插值

    Parameters
    ----------
    xs : np.array
        已知点的横坐标
    ys : np.array
        已知点的纵坐标
    new_x : float | np.array
        插值点的横坐标

    Returns
    -------
    float | np.array
        插值点的纵坐标
    """
    return interpolate.krogh_interpolate(xs, ys, new_x)
    
