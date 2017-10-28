import numpy as np

_seed = 0
_alpha = 1
_beta = 10

def get_clean_data(x):
    return np.sin(x)

def get_points(n=1, x=None):
    if x is None:
        x = np.random.uniform(low=0, high=2*np.pi, size=n)
    x.shape = (n, 1)
    clear_data = get_clean_data(x)
    noise = np.random.normal(size=n, loc=0, scale=np.sqrt(1/_beta))
    noise.shape = (n, 1)
    return x, noise + clear_data