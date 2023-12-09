import numpy as np

def auto_correlation(p,q,sigma, delta,order=1):
    return ((p+q-1)**(order))/(1+((sigma))/(((delta)**2)*(1-p)*(1-q)/((2-p-q)**2)))

def get_sigma_from_variance(Var,p,q,delta):
    return Var - (delta**2)*((1-p)*(1-q)/(2-p-q)**2)



assert get_sigma_from_variance((0.20**2)/12,1,0,0.05) == (0.20**2)/12
assert get_sigma_from_variance((0.20**2)/12,0,1,0.05) == (0.20**2)/12


def make_pannel_A():
    T = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            p = (i+1)/10
            q = (j+1)/10
            sigma2 = get_sigma_from_variance((0.20**2)/12,p,q, 0.05)
            T[i,j] = round(100*auto_correlation(p,q, sigma2, 0.05, 1),1)
    return T

def make_pannel_B():
    T = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            p = (i+1)/10
            q = (j+1)/10
            sigma2 = get_sigma_from_variance((0.50**2)/12,p,q, 0.05)
            T[i,j] = round(100*auto_correlation(p,q, sigma2, 0.05, 1),1)
    return T

def make_pannel_C():
    T = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            p = (i+1)/10
            q = (j+1)/10
            sigma2 = get_sigma_from_variance((0.50**2)/12,p,q, 0.20)
            T[i,j] = round(100*auto_correlation(p,q, sigma2, 0.20, 1),1)
    return T



print(make_pannel_A())
print(make_pannel_B())
print(make_pannel_C())
