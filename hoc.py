import numpy as np
import string
def step_decay(epoch, alpha_original=0.1, drop_rate=0.5, step_size=10):
    return alpha_original * (drop_rate ** (epoch // step_size))

def MSE(w):
    return 0.5 * np.dot(w,w.T)

def gradient_descent(w, max_epoch = 500, min_value = 1e-5):
    step = 0
    for epoch in range(max_epoch):
        step += 1
        alpha = step_decay(epoch)
        grad_w = w
        w = w - alpha * grad_w
        J = MSE(w)
        if J < min_value:
            break
    return J, w, step, alpha

if __name__ == "__main__":
    w = list(map(float,input().strip().split()))
    w = np.array(w, dtype=float)
    J, new_w, step, alpha = gradient_descent(w)
    print(str(J) + ' ' + str(new_w) + ' ' + str(step) + ' ' + str(alpha))