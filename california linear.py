import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
def cost(w,X,Y) :
    m=X.shape[0]
    errors=np.dot(X,w.T)-Y
    return np.dot(errors,errors.T)/(2*m)
def linear_regression(alpha,w,X,Y,max_step=1000,min_value=1e-6) :
    step=0
    L=cost(w,X,Y)
    m=X.shape[0]
    for i in range(max_step) :
        step=i+1
        errors=np.dot(X,w.T)-Y
        gd_w=(np.dot(errors.T,X))/m
        w=w-alpha*gd_w
        L=cost(w,X,Y)
        if L < min_value :
            break
    return w,step,L
if __name__ == "__main__" :
    cali=fetch_california_housing()
    X=cali.data
    Y=cali.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    w=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
    w=np.array(w,dtype=float)
    W,step,loss=linear_regression(0.7,w,X,Y)
    print(W)
    print(step)
    print(loss)