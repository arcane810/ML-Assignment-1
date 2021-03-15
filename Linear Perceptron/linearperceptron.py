import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

df = pd.read_csv("./dataset_LP_1.csv", header=None)


def train_test_split(dataframe, split=0.70):
    train_size = int(split * len(dataframe))
    test_size = len(dataframe) - train_size
    dataframe = dataframe.sample(frac=1, random_state=69)
    dataframe.insert(0, 0, np.ones(dataframe.shape[0]), True)
    train = dataframe[:train_size].to_numpy()
    test = dataframe[-test_size:].to_numpy()
    x_train = train[:, :-1]
    y_train = 2 * train[:, -1] - 1
    x_test = test[:, :-1]
    y_test = 2 * test[:, -1] - 1
    return x_train, y_train, x_test, y_test


def perceptron(x_train, y_train, learning_rate=1, epochs=1000000):
    np.random.seed(1)
    weights = np.random.rand(x_train.shape[1])
    cost = 0
    for iter in range(epochs):
        if iter % 25000 == 0:
            print("Currently at : ", iter)
        y_predict = 2 * (x_train.dot(weights) > 0) - 1
        misclassified = y_predict != y_train
        cost = misclassified.sum()
        if cost == 0:
            return weights, cost
        weights = weights + learning_rate * (
            x_train[misclassified, :][0] * y_train[misclassified][0]
        )
    return weights, cost


def plot(x_train, y_train):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x_train[y_train == 1, 1],
        x_train[y_train == 1, 2],
        x_train[y_train == 1, 3],
        color="red",
    )
    ax.scatter(
        x_train[y_train == -1, 1],
        x_train[y_train == -1, 2],
        x_train[y_train == -1, 3],
        color="blue",
    )
    xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
    zz = (-weights[1] * xx - weights[2] * yy - weights[0]) * 1.0 / weights[3]
    ax.plot_surface(xx, yy, zz, color="green")
    plt.show()


def predict(x_test, y_test, weights):
    y_predict = 2 * (x_test.dot(weights) > 0) - 1
    misclassified = y_predict != y_test
    cost = misclassified.sum()
    accuracy = 1 - cost / y_test.shape[0]
    testing_accuracy = accuracy * 100
    print("Testing Accuracy: ", testing_accuracy)


x_train, y_train, x_test, y_test = train_test_split(df, 0.70)
weights, cost = perceptron(x_train, y_train, learning_rate=1, epochs=1000000)
weights = weights / np.linalg.norm(weights)
print(weights)
print("Training Accuracy : ", (1 - cost / x_train.shape[0]) * 100)
predict(x_test, y_test, weights)
plot(x_train, y_train)
plot(x_test, y_test)
