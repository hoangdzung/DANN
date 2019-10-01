import numpy as np
a1 = 0.3
a2 = 0.4 
p1 = 3
p2 = 4
k1 = 1
k2 = 2
K = 0.7 

def get_appro_syn_data(n_samples):
    x1 = np.random.randn(n_samples)+5
    x2 = np.random.randn(n_samples)+20

    y = k1*(x1**2) + k2*(x2**2) \
        + a1*np.cos(p1*x1*np.pi) + a2*np.cos(p2*x2*np.pi) + K

    return np.vstack([x1,x2]).T, y 

def get_mono_syn_data(n_samples):
    x1 = np.random.randn(n_samples)+5
    x1_ = 6*x1
    x1__ = 12*x1
    x2 = np.random.randn(n_samples)+20

    y = k1*(x1**2) + k2*(x2**2) \
        + a1*np.cos(p1*x1*np.pi) + a2*np.cos(p2*x2*np.pi) + K
    y_ = k1*(x1_**2) + k2*(x2**2) \
        + a1*np.cos(p1*x1_*np.pi) + a2*np.cos(p2*x2*np.pi) + K
    y__ = k1*(x1__**2) + k2*(x2**2) \
        + a1*np.cos(p1*x1__*np.pi) + a2*np.cos(p2*x2*np.pi) + K

    return (np.vstack([x1,x2]).T, y), (np.vstack([x1_,x2]).T, y_) , (np.vstack([x1__,x2]).T, y__)  