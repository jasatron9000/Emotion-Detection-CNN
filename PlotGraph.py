# PLOT GRAPH
# ============================================================
# The task of this file is to make a class that makes it easier
# to use the python library for drawing plots
# ============================================================

import matplotlib.pyplot as plt


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


def plot_confusion_matrix(confusion_matrix, CNN_name: str):
    names = ["Afraid", "Angry", "Disgust", "Happy", "Neutral", "Sad", "Surprised"]
    df_cm = pd.DataFrame(confusion_matrix, names, names)
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="BuPu")  # font size
    ax = plt.axes()
    ax.set_title(CNN_name + 'Emotion Confusion Matrix')
    plt.xlabel('Actual Emotion')
    plt.ylabel('Predicted Emotion')
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


def showPlot(*plots: genLinePlot):
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

    plt.show()



