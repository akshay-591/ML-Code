# This file contains the method for boundary visualization

import numpy as mat
from matplotlib import pyplot as plot

from SupportVectorMachine import Prediction


def visualize_linear(model):
    X = mat.c_[mat.ones(model.X.shape[0]), model.X]

    positive_examples = model.X[mat.ix_(mat.where(model.Y == 1)[0])]
    negative_examples = model.X[mat.ix_(mat.where(model.Y == -1)[0])]

    plot_x = mat.linspace(mat.min(X[:, 1]), mat.max(X[:, 1]), 100)

    plot_y = mat.multiply(mat.divide(-1, model.W[1]),
                          mat.subtract(mat.multiply(model.W[0], plot_x), model.b))

    plot.title("Support Vector Machine")
    plot.xlabel("X1")
    plot.ylabel("X2")
    plot.plot(positive_examples[:, 0], positive_examples[:, 1], "+",
              negative_examples[:, 0], negative_examples[:, 1], "o",
              plot_x.transpose(), plot_y.transpose())
    plot.legend(["Positive", "Negative"])
    plot.show()


def visualize_nonLinear(model):
    plot_x1 = mat.linspace(mat.min(model.X[:, 0]), mat.max(model.X[:, 0]), 100)
    plot_x2 = mat.linspace(mat.min(model.X[:, 1]), mat.max(model.X[:, 1]), 100)
    X1, Y1 = mat.meshgrid(plot_x1.transpose(), plot_x2.transpose())
    values = mat.zeros(X1.shape)
    for i in range(X1.shape[1]):
        x = mat.c_[X1[:, i], Y1[:, i]]
        values[:, i] = Prediction.predict(model, x, "gaussian")
    positive_examples = model.X[mat.ix_(mat.where(model.Y == 1)[0])]
    negative_examples = model.X[mat.ix_(mat.where(model.Y == -1)[0])]

    plot.title("Support Vector Machine")
    plot.xlabel("X1")
    plot.ylabel("X2")
    plot.plot(positive_examples[:, 0], positive_examples[:, 1], "+",
              negative_examples[:, 0], negative_examples[:, 1], "o")
    plot.legend(["Positive", "Negative"])
    plot.contourf(X1, Y1, values, levels=mat.linspace(0.4, 0.5, 2))
    plot.show()
