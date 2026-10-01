import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# =========================================================
# 1. L2 LOSS FUNCTION
# =========================================================
def L2(w, X, Y, beta):
    n = X.shape[0]

    # Prediction error
    error = X @ w - Y

    # MSE part:
    # J = 1/(2N) * sum(error^2)
    mse_part = (error.T @ error) / (2 * n)

    # L2 regularization
    # Không regularize bias w[0]
    w_reg = np.copy(w)
    w_reg[0] = 0

    # beta/2 * ||w||^2
    l2_penalty = (beta / 2) * (w_reg.T @ w_reg)

    # Total loss
    loss = mse_part + l2_penalty

    return np.squeeze(loss)


# =========================================================
# 2. LEARNING RATE DECAY
# =========================================================
def decay_alpha(alpha, epoch, decay_rate=0.001, lower_bound=1e-4):

    new_alpha = alpha * np.exp(-decay_rate * epoch)

    if new_alpha < lower_bound:
        new_alpha = lower_bound

    return new_alpha


# =========================================================
# 3. GRADIENT
# =========================================================
def gradient(X, Y, w, beta):
    n = X.shape[0]

    # Prediction
    predictions = X @ w

    # Error
    error = predictions - Y

    # Gradient of MSE
    gradient_mse = (X.T @ error) / n

    # Gradient of L2
    w_reg = np.copy(w)

    # Không regularize bias
    w_reg[0] = 0

    gradient_l2 = beta * w_reg

    # Total gradient
    gradient_total = gradient_mse + gradient_l2

    return gradient_total


# =========================================================
# 4. LINEAR REGRESSION + L2
# =========================================================
def linear_regression(
    w,
    X,
    Y,
    beta,
    alpha,
    max_epoch=1000,
    stop=1e-6
):

    step = 0

    # Initial loss
    loss_value = L2(w, X, Y, beta)

    for epoch in range(max_epoch):

        step += 1

        # Calculate gradient
        grad_w = gradient(X, Y, w, beta)

        # Learning rate decay
        alpha_now = decay_alpha(alpha, epoch)

        # Update weight
        w = w - alpha_now * grad_w

        # Calculate new loss
        loss_value = L2(w, X, Y, beta)

        # Early stopping
        if loss_value < stop:
            break

    return step, w, loss_value


# =========================================================
# 5. PREDICT
# =========================================================
def predict_house_price(X, w):
    return X @ w


# =========================================================
# 6. PREPROCESS OUTPUT
# =========================================================
def preprocess_output(Y):
    return Y / 10.0


# =========================================================
# 7. REVERSE PREPROCESS OUTPUT
# =========================================================
def reverse_preprocess_output(Y):
    return Y * 10.0


# =========================================================
# 8. EVALUATE USING MSE
# =========================================================
def evaluate(w, X, Y):

    # Prediction
    y_pred = X @ w

    # Error
    errors = y_pred - Y

    # MSE
    MSE = np.mean(errors ** 2)

    return MSE


# =========================================================
# 9. MAIN PROGRAM
# =========================================================
if __name__ == "__main__":

    # -----------------------------------------------------
    # Load California Housing dataset
    # -----------------------------------------------------
    cali = fetch_california_housing()

    X = cali.data
    Y = cali.target

    print("Original data shape:")
    print("X =", X.shape)
    print("Y =", Y.shape)

    # -----------------------------------------------------
    # Preprocess Y
    # -----------------------------------------------------
    Y = preprocess_output(Y)

    # -----------------------------------------------------
    # STEP 1: SPLIT DATA
    #
    # 70% Train
    # 30% Temp
    # -----------------------------------------------------
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X,
        Y,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    # -----------------------------------------------------
    # STEP 2: SPLIT TEMP
    #
    # 1/3 of 30% = 10% total -> Validation
    # 2/3 of 30% = 20% total -> Test
    # -----------------------------------------------------
    X_validation, X_test, Y_validation, Y_test = train_test_split(
        X_temp,
        Y_temp,
        test_size=2 / 3,
        random_state=42,
        shuffle=True
    )

    print("\nData split:")
    print("Train      :", X_train.shape[0])
    print("Validation :", X_validation.shape[0])
    print("Test       :", X_test.shape[0])

    # -----------------------------------------------------
    # STEP 3: STANDARDIZATION
    #
    # mean and std are calculated ONLY from training data
    # -----------------------------------------------------
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    # Standardize train
    X_train = (X_train - mean) / std

    # Use the SAME mean and std for validation
    X_validation = (X_validation - mean) / std

    # Use the SAME mean and std for test
    X_test = (X_test - mean) / std

    # -----------------------------------------------------
    # STEP 4: ADD BIAS
    #
    # X = [bias, x1, x2, ..., x8]
    # -----------------------------------------------------

    bias_train = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((bias_train, X_train), axis=1)

    bias_validation = np.ones((X_validation.shape[0], 1))
    X_validation = np.concatenate(
        (bias_validation, X_validation),
        axis=1
    )

    bias_test = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate(
        (bias_test, X_test),
        axis=1
    )

    print("\nFeature shape after adding bias:")
    print("X_train      :", X_train.shape)
    print("X_validation :", X_validation.shape)
    print("X_test       :", X_test.shape)

    # -----------------------------------------------------
    # STEP 5: INITIALIZE WEIGHTS
    # -----------------------------------------------------
    n_features = X.shape[1]

    # 8 features + 1 bias
    w = np.zeros(n_features + 1)

    # -----------------------------------------------------
    # STEP 6: CHOOSE BETA VALUES
    # -----------------------------------------------------
    choose_beta = [
        0.0,
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0
    ]

    # -----------------------------------------------------
    # STEP 7: TRAIN WITH DIFFERENT BETA
    # -----------------------------------------------------
    min_loss = 1e9
    W_use = None
    beta_use = None
    step_use = None

    alpha = 0.05

    print("\nTraining:")

    for beta in choose_beta:

        # Start from the same initial weights
        w_initial = w.copy()

        # Train
        step, W, loss_value = linear_regression(
            w_initial,
            X_train,
            Y_train,
            beta,
            alpha
        )

        # Evaluate on validation set
        validation_MSE = evaluate(
            W,
            X_validation,
            Y_validation
        )

        print(
            "beta =",
            beta,
            "| validation MSE =",
            validation_MSE,
            "| steps =",
            step
        )

        # Select the best beta
        if validation_MSE < min_loss:

            min_loss = validation_MSE
            W_use = W.copy()
            beta_use = beta
            step_use = step

    # -----------------------------------------------------
    # STEP 8: FINAL TEST
    # -----------------------------------------------------
    test_MSE = evaluate(
        W_use,
        X_test,
        Y_test
    )

    # -----------------------------------------------------
    # STEP 9: PRINT RESULT
    # -----------------------------------------------------
    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")

    print("Best beta       :", beta_use)
    print("Training steps  :", step_use)
    print("Validation MSE  :", min_loss)
    print("Test MSE        :", test_MSE)

    # -----------------------------------------------------
    # STEP 10: PREDICT FIRST 5 TEST SAMPLES
    # -----------------------------------------------------
    Y_pred = predict_house_price(
        X_test,
        W_use
    )

    # Convert prediction back to original scale
    Y_pred = reverse_preprocess_output(Y_pred)

    # Convert test Y back to original scale
    Y_test_original = reverse_preprocess_output(Y_test)

    print("\nFirst 5 predictions:")
    print(Y_pred[:5])

    print("\nFirst 5 real values:")
    print(Y_test_original[:5])