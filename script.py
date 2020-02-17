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
                34546, 37198, 40171, 42638, 44653, 46551,
                48467, 49970, 51091, 70548]
total_data_y_including_cd = [58761, 63851, 66492, 68500, 70548]

relative_growth_y = []
for i in range(len(total_data_y)-1):
    relative_growth_y.append(total_data_y[i+1]/total_data_y[i]-1)

# increase pyplot font size
font = {'family': 'normal', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
plt.rc('axes', labelsize=20)

# plt.style.use('light_background')


# function definition for the exponential fit with parameters a, b
def exponential_fit_function(x, a, b):
    return a*np.exp(b*x)


# function definition for the sigmoidal fit with parameters a, b, c
def sigmoidal_fit_function(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))


regression_start = 5  # index to
# index to stop plotting the exponential fit
exponential_stop = len(total_data_y)+1
sigmoidal_start = 13  # index to start plotting the sigmoidal fit
cd_start = 27  # start of including clinical diagnosis

# x-axis range
xmin = 0
xmax = 32
# steps between major ticks on x-axi
xstep = 4

# colors
exponential_color = np.array([30, 136, 229]) / 255
sigmoidal_color = np.array([222, 167, 2]) / 255
data_color = np.array([0, 0, 0]) / 255
change_color = np.array([216, 27, 96]) / 255
cd_color = np.array([93, 93, 93]) / 255

# create animation frames
for l in range(regression_start, len(total_data_y)+4):
    i = l  # index of the last data point to be used
    n = 3  # number of previous fits to include

    if l > len(total_data_y):  # used to fade out the last three plots
        i = len(total_data_y)
        n = 3 - (l - i)

    data_x = np.arange(0, i)
    data_y = total_data_y[0:i]

    # creation of pyplot plot
    fig, ax2 = plt.subplots()
    ax1 = ax2.twinx()

    # setting the dimensions (basically resolution with 12by9 aspect ratio)
    fig.set_figheight(10)
    fig.set_figwidth(12)

    majxticks = ([], [])
    # generation of x-axis tick-names
    startdate = datetime(2020, 1, 16)
    for j in range(xmin, xmax+1, xstep):
        majxticks[0].append(j)
        majxticks[1].append((startdate + timedelta(j)).strftime("%d. %b"))

    # setting the x-axis ticks
    ax1.set_xlim([xmin, xmax])
    plt.xticks(majxticks[0], majxticks[1])
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.tick_params(axis="x", which="major", length=8, width=1.5)
    ax1.tick_params(axis="x", which="minor", length=5, width=1)

    # setting the y-axis ticks
    ax1.set_yticks([0, 10000, 20000, 30000, 40000, 50000, 60000, 70000])
    ax1.set_yticklabels(["0", "10k", "20k", "30k", "40k", "50k", "60k", "70k"])
    ax1.yaxis.set_minor_locator(MultipleLocator(5000))

    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    # setting the y-axis limit
    ax1.set_ylim([0, 70000])
    ax2.set_ylim([0, 1])

    # label axis
    plt.xlabel("date")
    ax1.set_ylabel("total # of confirmed infections in Mainland China")
    ax2.set_ylabel("relative growth")

    # plot the original data
    ax1.plot(data_x, data_y, "s", color=data_color,
             label="raw data collected from source")

    if i > cd_start:
        ax1.plot(
            data_x[cd_start:i], total_data_y_including_cd[0:i-cd_start], "s", color=cd_color)

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

            ax1.plot(nom_x, nom_y, color=exponential_color, linewidth=2,
                     label="data fitted to an exponential function")
            ax1.fill_between(nom_x, nom_y - std_y, nom_y + std_y, facecolor=exponential_color,
                             alpha=0.3, label="area of uncertainty for the exponential fit")
        elif k == i-1:
            ax1.fill_between(nom_x, nom_y - std_y, nom_y +
                             std_y, facecolor=exponential_color, alpha=0.2)
        elif k == i-2:
            ax1.fill_between(nom_x, nom_y - std_y, nom_y +
                             std_y, facecolor=exponential_color, alpha=0.05)

    for k in range(np.max([i-n, sigmoidal_start]), i+1):
        # fit the sigmoidal function
        popt, pcov = scipy.optimize.curve_fit(
            sigmoidal_fit_function,  data_x[8:k],  data_y[8:k], p0=[60000, 0.4, 20])
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
            ax1.plot(nom_x, nom_y, color=sigmoidal_color, linewidth=2,
                     label="data fitted to an sigmoidal function")
            ax1.fill_between(nom_x, nom_y - std_y, nom_y + std_y, facecolor=sigmoidal_color,
                             alpha=0.6, label="area of uncertainty for the sigmoidal fit")
        elif k == i-1:
            ax1.fill_between(nom_x, nom_y - std_y, nom_y +
                             std_y, facecolor=sigmoidal_color, alpha=0.2)
        elif k == i-2:
            ax1.fill_between(nom_x, nom_y - std_y, nom_y +
                             std_y, facecolor=sigmoidal_color, alpha=0.1)

    ax2.plot(data_x[1:i], relative_growth_y[0:i-1],
             color=change_color, alpha=0.4, lw=2)

    # format the border of the diagram
    ax1.spines['top'].set_color('white')
    ax2.spines['top'].set_color('white')

    # these objects are used to create a consistent legend
    legendel_originaldata = Line2D([0], [0], marker='s', color=data_color,
                                   lw=0, label='Scatter', markerfacecolor=data_color, markersize=10)
    legendel_cddata = Line2D([0], [0], marker='s', color=cd_color,
                             lw=0, label='Scatter', markerfacecolor=cd_color, markersize=10)
    legendel_exponentialfit = Line2D(
        [0], [0], color=exponential_color, lw=4, label='Line')
    legendel_sigmoidalfit = Line2D(
        [0], [0], color=sigmoidal_color, lw=4, label='Line')
    legendel_exponential_areaofuncertainty = Patch(
        facecolor=exponential_color, alpha=0.5, label="e")
    legendel_sigmoidal_areaofuncertainty = Patch(
        facecolor=sigmoidal_color, alpha=0.5, label="d")
    legendel_relchange = Line2D(
        [0], [0], color=change_color, lw=4, label='Line')

    # add the legend and object descriptions
    legend = ax2.legend([legendel_originaldata,
                         legendel_cddata,
                         legendel_exponentialfit,
                         legendel_exponential_areaofuncertainty,
                         legendel_sigmoidalfit,
                         legendel_sigmoidal_areaofuncertainty,
                         legendel_relchange],
                        ["data excluding clinical diagnosis",
                         "data including clinical diagnosis",
                         "data fitted to an exponential function",
                         "95% area of uncertainty for the exponential fit",
                         "data fitted to an sigmoidal function",
                         "95% area of uncertainty for the sigmoidal fit",
                         "relative growth"], loc='upper left')
    legend.get_frame().set_edgecolor("black")
    legend.set_zorder(20)
    plt.title("see comments for further explanations")

    plt.tight_layout()

    # save the plot in the current folder
    plt.savefig(str(l) + ".png")

# batch the images to a video
initial_frame_repeatcount = 2  # number of times the initial frame is to be repeated
final_frame_repeatcount = 7  # number of times the final frame is to be repeated

video_name = 'video.mp4'  # name of the exported video

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
