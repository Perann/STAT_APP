import numpy as np

def auto_correlation(p,q,sigma, delta,order=1):
    return ((p+q-1)**(order))/(1+((sigma**2))/((delta)**2)*(1-p)*(1-q)/(2-p-q)**2)

def get_sigma_from_variance(Var,p,q,delta):
    return Var - (delta**2)*((1-p)*(1-q)/(2-p-q)**2)

def make_Table1():
    T = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            sigma2 = get_sigma_from_variance((0.20**2)/12,(i+1)/10,(j+1)/10, 0.05)
            T[i,j] = auto_correlation((i+1)/10, (j+1)/10, sigma2, 0.05)
    return T

print(make_Table1())