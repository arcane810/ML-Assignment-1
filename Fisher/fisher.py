import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fisher(dataset):
    class0_data = dataset[dataset[:, -1] == 0, :-1]
    class1_data = dataset[dataset[:, -1] == 1, :-1]
    M1 = class0_data.mean(axis=0)
    M2 = class1_data.mean(axis=0)

    class0_diff = class0_data - M1
    class1_diff = class0_data - M2

    S1 = np.matmul(class0_diff.transpose(), class0_diff)
    S2 = np.matmul(class1_diff.transpose(), class1_diff)
    Sw = S1 + S2
    return np.matmul(np.linalg.inv(Sw), M1 - M2)


def normpdf(x, mu=0, sigma=1):
    u = (x - mu) / abs(sigma)
    y = np.exp(-u * u / 2) / (np.sqrt(2 * np.pi) * abs(sigma))
    return y


def getLineOffset(dataset, w):
    class0_data = dataset[dataset[:, -1] == 0, :-1]
    class1_data = dataset[dataset[:, -1] == 1, :-1]
    class0_projection = np.dot(class0_data, w.reshape(3, 1))
    class1_projection = np.dot(class1_data, w.reshape(3, 1))
    fig = plt.figure()
    plt.scatter(class0_projection, np.zeros(class0_projection.shape), color="red")
    plt.scatter(class1_projection, np.zeros(class1_projection.shape), color="blue")
    m1 = class0_projection.mean()
    std1 = class0_projection.std()
    m2 = class1_projection.mean()
    std2 = class1_projection.std()
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    roots = np.roots([a, b, c])
    norm_x = np.linspace(-2.5, 2.5, 1000)
    plt.plot(norm_x, normpdf(norm_x, m1, std1), color="red")
    plt.plot(norm_x, normpdf(norm_x, m2, std2), color="blue")
    for i in roots:
        if i < max(m1, m2) and i > min(m1, m2):
            plt.scatter(i, 0, color="green")
            plt.plot([i, i], [-1, 3], color="green")
            plt.show()
            return i


if __name__ == "__main__":
    DATA_PATH = "./dataset_FLD.csv"
    dataset = pd.read_csv(
        DATA_PATH, header=None, names=["Feature1", "Feature2", "Feature3", "Class"]
    ).to_numpy()

    w = fisher(dataset)
    w = w / np.linalg.norm(w)
    print(w)
    offset = getLineOffset(dataset, w)
    print(offset)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        dataset[dataset[:, -1] == 0, 0],
        dataset[dataset[:, -1] == 0, 1],
        dataset[dataset[:, -1] == 0, 2],
        color="red",
    )
    ax.scatter(
        dataset[dataset[:, -1] == 1, 0],
        dataset[dataset[:, -1] == 1, 1],
        dataset[dataset[:, -1] == 1, 2],
        color="blue",
    )
    xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
    zz = (-w[0] * xx - w[1] * yy + offset) * 1.0 / w[2]
    ax.plot_surface(xx, yy, zz, color="green")
    plt.show()
