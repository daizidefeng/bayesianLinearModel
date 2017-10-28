import data_source as ds

import numpy as np
from numpy.linalg import inv

from scipy import stats as st


class LinearModel:
    def __init__(self,  features_func, features_len, 
                     alpha=ds._alpha, beta=ds._beta):
        self.Features = features_func
        self.alpha = alpha
        self.beta = beta
        self.features_len = features_len
        self.mean_W = np.zeros((features_len, 1))
        self.cov_W = 1 / alpha * np.identity(features_len) 
        
        
    def observe(self, X, T):
        F = self.Features(X) 
        cow_W_0 = self.cov_W 
        self.cov_W = inv(inv(self.cov_W) + self.beta * F.T @ F)

        self.mean_W = self.cov_W @ (inv(cow_W_0) @ self.mean_W + 
                                    self.beta * F.T @ T)
    def variance_t_for_x(self, x):
        f = self.Features(x)
        return 1 / self.beta + np.array(
                [[f[i] @ self.cov_W @ f[i].T] for i in range(len(f))])
    
    def model_value(self, x, W):
        return self.Features(x) @ W
    
    def mean_t_for_x(self, x):
        return self.model_value(x, self.mean_W)
    
    def probability(self, X, T, e=None):
        if e is None:
            try:
                iter(T)
            except:
                raise Exception("X is not iterabe and e is npt specified")    
            d = np.diff(T, axis=0)
            e = np.mean(d)

            
        m_t = self.mean_t_for_x(X)
        s_t = np.sqrt(self.variance_t_for_x(X))

        return (st.norm.cdf(T+e/2, loc=m_t, scale=s_t) - 
                    st.norm.cdf(T-e/2, loc=m_t, scale=s_t))
    
    def maximum_likelihood(self, x):
        return self.mean_t_for_x(x)

    def draw_W(self):
        return st.multivariate_normal.rvs(
                self.mean_W.reshape(self.features_len), self.cov_W)
    
def gaussian_features(X, s=1/4, n=24):
    f = np.concatenate(
                [np.array([gaussian_feature(x[0], m, s) 
                            for m in np.linspace(0, 2*np.pi, num=n)])
                    for x in X])
    f.shape = (len(X), n)
    return f
    
def gaussian_feature(x, m, s):
    return np.exp(-(x-m)**2/(2*s**2))
