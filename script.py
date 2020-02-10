from datetime import datetime, timedelta

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import uncertainties.unumpy as unp
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from uncertainties import ufloat

# data points, cummulative 
# to add another date simply append the # of infected people
total_data_y = [45, 62, 121, 198, 291, 440, 571, 830, 1287, 
                1975, 2744, 4515, 5974, 7711, 9692, 11791, 
                14380, 17205, 20438, 24324, 28018, 31161, 
                34546, 37198, 40171]

# increase pyplot font size
font = {'family': 'normal', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
plt.rc('axes', labelsize=20)

# function definition for the exponential fit with parameters a, b
def exponential_fit_function(x, a, b):
    return a*np.exp(b*x)


# function definition for the sigmoidal fit with parameters a, b, c
def sigmoidal_fit_function(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))

regression_start = 5 # index to 
exponential_stop = len(total_data_y)+1 # index to stop plotting the exponential fit
sigmoidal_start = 13 # index to start plotting the sigmoidal fit

# x-axis range
xmin = 0
xmax = 28
# steps between major ticks on x-axi
xstep = 4

# create animation frames
for l in range(regression_start, len(total_data_y)+4):
    i = l # index of the last data point to be used
    n = 3 # number of previous fits to include
    
    if l > len(total_data_y): # used to fade out the last three plots
        i = len(total_data_y)
        n = 3 - (l - i)

    data_x = np.arange(0, i)
    data_y = total_data_y[0:i]

    # creation of pyplot plot
    fig, ax = plt.subplots()

    # setting the dimensions (basically resolution with 12by9 aspect ratio)
    fig.set_figheight(10)
    fig.set_figwidth(10*12/9)

    majxticks = ([], [])
    # generation of x-axis tick-names
    startdate = datetime(2020, 1, 16)
    for j in range(xmin, xmax+1, xstep):
        majxticks[0].append(j)
        majxticks[1].append((startdate + timedelta(j)).strftime("%d. %b"))

    # setting the x-axis ticks
    ax.set_xlim([xmin, xmax])
    plt.xticks(majxticks[0], majxticks[1])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", length=8, width=1.5)
    ax.tick_params(axis="both", which="minor", length=5, width=1)

    # setting the y-axis ticks
    plt.yticks([0, 10000, 20000, 30000, 40000, 50000], [
               "0", "10k", "20k", "30k", "40k", "50k"])
    ax.yaxis.set_minor_locator(MultipleLocator(5000))

    # setting the y-axis limit
    ax.set_ylim([0, 50000])
    
    # label axis
    plt.xlabel("date")
    plt.ylabel("total # of confirmed infections in Mainland China")

    #plot the original data
    plt.plot(data_x, data_y, "s", color="black",
             label="raw data collected from source")

    # create the exponential plots 
    for k in range(np.max([i-n, regression_start]), np.min([exponential_stop, i+1])):
        # fit the exponential function
        popt, pcov = scipy.optimize.curve_fit(
            exponential_fit_function,  data_x[0:k],  data_y[0:k])
        # get errors from trace of covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # create uncertainty floats for error bars, 2* means 2 sigma
        a = ufloat(popt[0], 2*perr[0])
        b = ufloat(popt[1], 2*perr[1])

        # get the values of the uncertain fit
        fit_x_unc = np.linspace(xmin, xmax, 300)
        fit_y_unc = a*unp.exp(b*fit_x_unc)
        nom_x = unp.nominal_values(fit_x_unc)
        nom_y = unp.nominal_values(fit_y_unc)
        std_y = unp.std_devs(fit_y_unc)

        # plot
        if k == i:
            print("[" + str(l) + "] Exponential fit-parameters:")
            print("a = " + str(a))
            print("b = " + str(b))

            plt.plot(nom_x, nom_y, color="blue", linewidth=2,
                     label="data fitted to an exponential function")
            ax.fill_between(nom_x, nom_y - std_y, nom_y + std_y, facecolor="blue",
                            alpha=0.3, label="area of uncertainty for the exponential fit")
        elif k == i-1:
            ax.fill_between(nom_x, nom_y - std_y, nom_y +
                            std_y, facecolor="blue", alpha=0.1)
        elif k == i-2:
            ax.fill_between(nom_x, nom_y - std_y, nom_y +
                            std_y, facecolor="blue", alpha=0.05)

    for k in range(np.max([i-n, sigmoidal_start]), i+1):
        # fit the sigmoidal function
        popt, pcov = scipy.optimize.curve_fit(
            sigmoidal_fit_function,  data_x[0:k],  data_y[0:k], p0=[60000, 0.4, 20])
        # get errors from trace of covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # create uncertainty floats for error bars, 2* means 2 sigma
        a = ufloat(popt[0], 2*perr[0])
        b = ufloat(popt[1], 2*perr[1])
        c = ufloat(popt[2], 2*perr[2])

        # get the values of the uncertain fit
        fit_x_unc = np.linspace(xmin, xmax, 300)
        fit_y_unc = a/(1+unp.exp(-b*(fit_x_unc-c)))
        nom_x = unp.nominal_values(fit_x_unc)
        nom_y = unp.nominal_values(fit_y_unc)
        std_y = unp.std_devs(fit_y_unc)

        # plot
        if k == i:
            print("[" + str(l) + "] Sigmoid fit-parameters:")
            print("a = " + str(a))
            print("b = " + str(b))
            print("c = " + str(c))
            plt.plot(nom_x, nom_y, color="orange", linewidth=2,
                     label="data fitted to an sigmoidal function")
            ax.fill_between(nom_x, nom_y - std_y, nom_y + std_y, facecolor="orange",
                            alpha=0.6, label="area of uncertainty for the sigmoidal fit")
        elif k == i-1:
            ax.fill_between(nom_x, nom_y - std_y, nom_y +
                            std_y, facecolor="orange", alpha=0.2)
        elif k == i-2:
            ax.fill_between(nom_x, nom_y - std_y, nom_y +
                            std_y, facecolor="orange", alpha=0.1)

    # format the border of the diagram
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('black')

    # these objects are used to create a consistent legend
    legendel_originaldata = Line2D([0], [0], marker='s', color='black',
                                   lw=0, label='Scatter', markerfacecolor='black', markersize=10)
    legendel_exponentialfit = Line2D(
        [0], [0], color='blue', lw=4, label='Line')
    legendel_sigmoidalfit = Line2D(
        [0], [0], color='orange', lw=4, label='Line')
    legendel_exponential_areaofuncertainty = Patch(
        facecolor='blue', alpha=0.5, label="e")
    legendel_sigmoidal_areaofuncertainty = Patch(
        facecolor='orange', alpha=0.5, label="d")

    # add the legend and object descriptions
    ax.legend([legendel_originaldata,
               legendel_exponentialfit,
               legendel_exponential_areaofuncertainty,
               legendel_sigmoidalfit,
               legendel_sigmoidal_areaofuncertainty],
              ["raw data collected from source",
               "data fitted to an exponential function",
               "area of uncertainty for the exponential fit",
               "data fitted to an sigmoidal function",
               "area of uncertainty for the sigmoidal fit"], loc='upper left').get_frame().set_edgecolor("black")
    plt.title("see comments for further explanations")

    # save the plot in the current folder
    plt.savefig(str(l) + ".png")

# batch the images to a video
initial_frame_repeatcount = 2 # number of times the initial frame is to be repeated
final_frame_repeatcount = 7 # number of times the final frame is to be repeated

video_name = 'video.mp4' # name of the exported video

# get video size data
frame = cv2.imread("./" + str(regression_start) + ".png")
height, width, layers = frame.shape

# create video writer
fps = 3
video = cv2.VideoWriter(video_name, 0, fps, (width, height))

# write initial frame
for i in range(0, initial_frame_repeatcount):
    video.write(cv2.imread("./" + str(regression_start) + ".png"))

# animation frames
for i in range(regression_start + 1, len(total_data_y)+3):
    video.write(cv2.imread("./" + str(i) + ".png"))

# write final frame repeatedly
for i in range(0, final_frame_repeatcount):
    video.write(cv2.imread("./" + str(len(total_data_y)+3) + ".png"))

# save video
cv2.destroyAllWindows()
video.release()
