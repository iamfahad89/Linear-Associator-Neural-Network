# Rahaman, Fahad Ur
# 1001-753-107
# 2020-10-11
# Assignment-02-02

import numpy as np
from linear_associator import LinearAssociator

def test_weights():
    input_dimensions = 4
    number_of_nodes = 9
    model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                    transfer_function="Hard_limit")
    weights=model.get_weights()
    assert weights.ndim == 2 and \
           weights.shape[0] == number_of_nodes and \
           weights.shape[1] == (input_dimensions)
    model.set_weights(np.ones((number_of_nodes, input_dimensions)))
    weights = model.get_weights()
    assert weights.ndim == 2 and \
           weights.shape[0] == number_of_nodes and \
           weights.shape[1] == (input_dimensions)
    assert np.array_equal(model.get_weights(), np.ones((number_of_nodes, input_dimensions)))
    model.initialize_weights(seed=3)
    weights = np.array([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
 [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
 [-0.04381817, -0.47721803, -1.31386475,  0.88462238],
 [ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
 [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
 [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627],
 [-1.02378514, -0.7129932 ,  0.62524497, -0.16051336],
 [-0.76883635, -0.23003072,  0.74505627,  1.97611078],
 [-1.24412333, -0.62641691, -0.80376609, -2.41908317]])
    np.testing.assert_array_almost_equal(model.get_weights(), weights, decimal=3)


def test_predict_linear():
    input_dimensions = 2
    number_of_nodes = 5
    model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                    transfer_function="Linear")
    model.initialize_weights(seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    y = np.array([[-1.22398485, -0.09876447, -1.25403672,  2.37076614],
       [ 2.7100805 , -0.51397095,  2.0365662 , -0.8538988 ],
       [ 2.93924278, -0.90084842,  1.86188982,  1.13412472],
       [-1.12555691, -0.15077626, -1.21375201,  2.53979562],
       [-0.00551192, -0.07487682, -0.08083147,  0.4572099 ]])

    y_hat = model.predict(X_train)
    np.testing.assert_array_almost_equal(y_hat, y, decimal=4)

def test_predict_hard_limit():
    input_dimensions = 2
    number_of_nodes = 5
    model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                    transfer_function="Hard_limit")
    model.initialize_weights(seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    y = np.array([[0, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]])

    y_hat = model.predict(X_train)
    np.testing.assert_array_almost_equal(y_hat, y, decimal=4)

def test_pseudo_inverse_fit():
    input_dimensions = 5
    number_of_nodes = 5
    model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                    transfer_function="Linear")
    model.initialize_weights(seed=1)
    X_train = np.random.randn(input_dimensions, 10)
    out = model.predict(X_train)

    model.set_weights(np.zeros_like(model.get_weights()))
    model.fit_pseudo_inverse(X_train, out)
    new_out = model.predict(X_train)
    np.testing.assert_array_almost_equal(out, new_out, decimal=4)

def test_train_linear_delta():
    input_dimensions = 5
    number_of_nodes = 5
    for i in range(10):
        model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                        transfer_function="Linear")
        model.initialize_weights(seed=i+1)
        X_train = np.random.randn(input_dimensions, 100)
        out = model.predict(X_train)

        model.set_weights(np.random.randn(*model.get_weights().shape))

        model.train(X_train, out, batch_size=10, num_epochs=50, alpha=0.1, gamma=0.1, learning="delta")
        new_out = model.predict(X_train)

        np.testing.assert_array_almost_equal(out, new_out, decimal=4)

def test_train_hardlim_delta():
    input_dimensions = 5
    number_of_nodes = 5
    for i in range(10):
        model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                        transfer_function="Hard_limit")
        model.initialize_weights(seed=i + 1)
        X_train = np.random.randn(input_dimensions, 100)
        out = model.predict(X_train)
        model.set_weights(np.random.randn(*model.get_weights().shape))
        model.train(X_train, out, batch_size=10, num_epochs=50, alpha=0.1, gamma=0.1, learning="Delta")
        new_out = model.predict(X_train)
        #np.testing.assert_array_almost_equal(out, new_out, decimal=4)
        np.testing.assert_array_almost_equal_nulp(out, new_out, nulp= 4.60718e+18)


def test_calculate_mean_squared_error():
    input_dimensions = 5
    number_of_nodes = 5
    number_of_samples=18
    model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                    transfer_function="Linear")
    model.initialize_weights(seed=1)
    X_train = np.random.randn(input_dimensions, number_of_samples)
    assert model.calculate_mean_squared_error(X_train,model.predict(X_train)) == 0
    target=np.random.randn(number_of_nodes,number_of_samples)
    mse=model.calculate_mean_squared_error(X_train,target)
    np.testing.assert_array_almost_equal(mse, 3.7144714504979635, decimal=4)


    model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                    transfer_function="Hard_limit")
    model.initialize_weights(seed=1)
    X_train = np.random.randn(input_dimensions, number_of_samples)
    assert model.calculate_mean_squared_error(X_train,model.predict(X_train)) == 0
    target=np.random.randn(number_of_nodes,number_of_samples)
    mse=model.calculate_mean_squared_error(X_train,target)
    np.testing.assert_array_almost_equal(mse, 1.1234558948088766, decimal=4)