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
    def __init__(self, colour="blue", title="", xlabel="", ylabel="", average=1, x=None, y=None):
        if y is None:
            self.y = []
        else:
            self.y = y

        if x is None:
            self.x = []
        else:
            self.x = x
        self.colour = colour
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.average = average


# insertY : this allows the user to just append the y axis
# PARAMS:
#   -plotArray  -> which plot class will be used
#   -y          -> what the data that will be appended
def insertY(plotArray: genLinePlot, ny):
    plotArray.y.append(ny)
    plotArray.x.append(len(plotArray.y))


# insertY : this allows the user to just append the y axis
# PARAMS:
#   -plotArray  -> which plot class will be used
#   -y and x    -> what the data that will be appended
def insertDim(plotArray: genLinePlot, ny, nx):
    plotArray.y.append(ny)
    plotArray.x.append(nx)


def showPlot(*plots: genLinePlot):
    # create figure
    fig = plt.figure()

    for num, plot in enumerate(plots):
        newX = []
        newY = []

        # averaging method -> this determines
        if plot.average > 1:
            for i in range(len(plot.x)):
                if len(plot.x) % plot.average == 0:
                    newX.append(plot.x[i])
                    newY.append(plot.y[i])
        else:
            newX = plot.x
            newY = plot.y

        sPlt = fig.add_subplot(len(plots), 1, num + 1)
        sPlt.set_title(plot.title)
        sPlt.set_xlabel(plot.xlabel)
        sPlt.set_ylabel(plot.ylabel)
        sPlt.plot(newX, newY, color=plot.colour)

    plt.show()
