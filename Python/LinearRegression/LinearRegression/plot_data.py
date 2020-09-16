# this method is used for plotting the data

from matplotlib import pyplot


# this method is used for plotting data
def plot(x, y, title, x_label, y_label, plot_symbol):
    pyplot.title(title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.plot(x, y, plot_symbol)
    print("Plotting data...")
    pyplot.show()


def linoverdata_plot(x, pred_y, y, title, x_label, y_label, plot_symbol):
    pyplot.title(title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    print("Plotting data...")
    pyplot.plot(x, y, "", x, pred_y, plot_symbol)
    pyplot.legend(["Linear Regression", "Training Data"])
    pyplot.show()
