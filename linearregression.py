import numpy as np
from sklearn.datasets import fetch_california_housing


def loss(w, X, Y):
    m = X.shape[0]
    errors = np.dot(X, w) - Y
    return np.dot(errors, errors.T) / (2 * m)


def decay_alpha(alpha, epoch, decay_rate=0.1, lower_bound=1e-4):
    return alpha * np.exp(-decay_rate * epoch) if alpha > lower_bound else lower_bound


def linear_regression(alpha, w, X, Y, epochs=1000, min_value=1e-6):
    ''' 
        Describe about what the function is doing, as well as description about all the input variables.
        Also add the example on how to run it properly.
    '''
    step = 0
    losses = []
    L = loss(w, X, Y)
    m = X.shape[0]

    for epoch in range(epochs):
        step = epoch + 1

        errors = np.dot(X, w) - Y

        gd_w = np.dot(X.T, errors) / m

        alpha = decay_alpha(alpha, epoch, decay_rate=0.2, lower_bound=1e-3)

        w = w - alpha * gd_w

        L = loss(w, X, Y)
        
        losses.append(L)

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, alpha={alpha}, Loss is {L}")

        if L < min_value:
            break

    return w, step, L


def predict_house_price(features, w):
    return features @ w


def preprocess_output(Y):
    return Y * 1.0 / 10.0


def reverse_preprocess_output(Y):
    return Y * 10.0


if __name__ == "__main__":
    cali = fetch_california_housing()
    X = cali.data
    Y = cali.target
    Y = preprocess_output(Y)

    n_data, n_features = X.shape

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    bias = np.ones((n_data, 1))
    X = np.concatenate((bias, X), axis=1)     # we add one more column where all the values is 1 because it is for the bias term.

    print(X.shape)

    # we add 1 for the features because we have the bias term that we appended to X.
    w = np.zeros(n_features + 1)
    alpha = 0.05
    learned_w, step, loss_value = linear_regression(alpha, w, X, Y)
    print(learned_w)
    print(f"step={step}, loss={loss_value}")

    print("Now predicting a house price using the below random features")
    features = X[0].copy() 
    print("features:", features)
    # apply all the normalization that we did for X as before.
    # feautres = scaler.fit_transform(features)
    # features = (features - mean) / std
    predicted_normalized_price = predict_house_price(features, learned_w) * 1e6
    print(f"predicted normalized price = {predicted_normalized_price}")
    print(f"final price: {reverse_preprocess_output(predicted_normalized_price)}")