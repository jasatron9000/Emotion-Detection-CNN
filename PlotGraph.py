# PLOT GRAPH
# ============================================================
# The task of this file is to make a class that makes it easier
# to use the python library for drawing plots
# ============================================================

# Imports needed for the following code to work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np

# genLinePlot -> a class that holds the information that a single plot needs
# Params:
#   -colour             -> what colour is the line graph
#   -title              -> title of the graph
#   -xlabel and ylabel  -> x-axis and y-axis labels
#   -average            -> the amount of iterations it will go through the graph and averages it
#   -x and y            -> pre-defined x and y data
class genLinePlot:
    def __init__(self, title="", xlabel="", ylabel="", x=None, y=None, numOfLines=1, legendList=None):
        self.numOfLines = numOfLines
        self.legendList = legendList

        if y is None:
            self.y = []

            if numOfLines > 1:
                for i in range(numOfLines):
                    self.y.append([])
        else:
            self.y = y

        if x is None:
            self.x = []

            if numOfLines > 1:
                for i in range(numOfLines):
                    self.x.append([])
        else:
            self.x = x

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel


# plot_confusion_matrix  -> This allows the user to draw up a confusion matrix after the numerical matrix is calculated
# Params:
#   -confusion_matrix    -> a 2d numpy array: that has all the tally of predicted and real results
#   -CNN_name            -> string input: for title of the graph
#   -path                -> string input: the path to the directory where users want to save the graph(if they choose)
#   -norm                -> boolean input: the user can choose if they want to see the normalized values or not
#   -colour_scheme       -> string input: select colour schemes from the seaborn selection
def plot_confusion_matrix(confusion_matrix, CNN_name: str, path=None, norm = None, colour_scheme = "BuPu"):
    if norm is not None:
        confusion_matrix = np.transpose(confusion_matrix)
        for i in range(len(confusion_matrix)):
            row_sum = np.sum(confusion_matrix[i])
            for j in range(len(confusion_matrix[i])):
                if row_sum != 0:
                    confusion_matrix[i][j] = float(confusion_matrix[i][j]) / float(row_sum)
        confusion_matrix = np.transpose(confusion_matrix)
    names = ["Afraid", "Angry", "Disgust", "Happy", "Neutral", "Sad", "Surprised"]
    df_cm = pd.DataFrame(confusion_matrix, names, names)
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap= colour_scheme)  # font size
    ax = plt.axes()
    ax.set_title(CNN_name + ' Emotion Confusion Matrix')
    plt.xlabel('Actual Emotion')
    plt.ylabel('Predicted Emotion')

    if path is not None:
        plt.savefig(path + "/" + CNN_name + "-cm.png")
    plt.show()


# insertY : this allows the user to just append the y axis
# PARAMS:
#   -plotArray  -> which plot class will be used
#   -y          -> what the data that will be appended
def insertY(plotArray: genLinePlot, *ny):
    if len(ny) != plotArray.numOfLines:
        raise ValueError('Not enough arguments for the y value')
    if plotArray.numOfLines <= 1:
        plotArray.y.append(ny)
        plotArray.x.append(len(plotArray.y))
    else:
        for i, yVal in enumerate(ny):
            plotArray.y[i].append(yVal)
            plotArray.x[i].append(len(plotArray.y[i]))


# showPlot : Allows users to display the graphs (for loss and accuracy curve)
# PARAMS:
#   -*plots  -> *plots
#   -path    -> string input: the path to the directory where users want to save the graph(if they choose)
#   -name    -> string input: name of what the graph will be saved to
def showPlot(*plots: genLinePlot, path=None, name="default_graph"):
    # create figure
    fig = plt.figure()

    for num, plot in enumerate(plots):
        sPlt = fig.add_subplot(len(plots), 1, num + 1)
        sPlt.set_title(plot.title)
        sPlt.set_xlabel(plot.xlabel)
        sPlt.set_ylabel(plot.ylabel)

        if plot.numOfLines <= 1:
            sPlt.plot(plot.x, plot.y)
        else:
            for pltNum in range(plot.numOfLines):
                line, = sPlt.plot(plot.x[pltNum], plot.y[pltNum])
                line.set_label(plot.legendList[pltNum])
            sPlt.legend()

    if path is not None:
        plt.savefig(path + "/" + name + ".png")
    plt.show()


