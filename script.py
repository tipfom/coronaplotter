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
pre_cd_data = np.array([45, 62, 121, 198, 291, 440, 571, 830, 1287,
                        1975, 2744, 4515, 5974, 7711, 9692, 11791,
                        14380, 17205, 20438, 24324, 28018, 31161,
                        34546, 37198, 40171, 42638, 44653, 46472])
post_cd_data = np.append(
    (pre_cd_data * 1.33),
    [63851, 66492, 68500, 70548, 72436, 74185,
     75003, 75891, 76288, 76936, 77150, 77658,
     78064])

# 25.Jan which means +9 from precd data
row_data = [23, 29, 37, 56, 68, 82, 106, 132, 146, 153, 159, 191, 216, 270, 288, 307,
            319, 395, 441, 447, 505, 526, 683, 794, 804, 924, 1073, 1200, 1402, 1769, 2069, 2459]

relative_growth_y = []
for i in range(len(post_cd_data)-1):
    relative_growth_y.append(post_cd_data[i+1]/post_cd_data[i]-1)

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


regression_start = 5  # index to
# index to stop plotting the exponential fit
exponential_stop = 16
sigmoidal_start = 16  # index to start plotting the sigmoidal fit
row_start = 9

# x-axis range
xmin = 0
xmax = 42
# steps between major ticks on x-axi
xstep = 7

# colors
exponential_color = np.array([30, 136, 229]) / 255
sigmoidal_color = np.array([222, 167, 2]) / 255
data_color = np.array([0, 0, 0]) / 255
change_color = np.array([216, 27, 96]) / 255
row_color = np.array([179, 0, 255]) / 255

# create animation frames
for l in range(regression_start, len(post_cd_data)+4):
    i = l  # index of the last data point to be used
    n = 3  # number of previous fits to include

    if l > len(post_cd_data):  # used to fade out the last three plots
        i = len(post_cd_data)
        n = 3 - (l - i)

    data_x = np.arange(0, i)
    data_y = post_cd_data[0:i]

    # creation of pyplot plot
    fig, ax_relgrow = plt.subplots()
    ax_abschina = ax_relgrow.twinx()
    ax_absrow = ax_relgrow.twinx()

    ax_absrow.spines["right"].set_position(("axes", 1.15))

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
    ax_abschina.set_xlim([xmin, xmax])
    plt.xticks(majxticks[0], majxticks[1])
    ax_abschina.xaxis.set_minor_locator(MultipleLocator(1))
    ax_abschina.tick_params(axis="x", which="major", length=8, width=1.5)
    ax_abschina.tick_params(axis="x", which="minor", length=5, width=1)

    # setting the y-axis ticks
    ax_abschina.set_yticks([0, 20000, 40000, 60000, 80000, 100000])
    ax_abschina.set_yticklabels(["0", "20k", "40k", "60k", "80k", "100k"])
    ax_abschina.yaxis.set_minor_locator(MultipleLocator(10000))

    ax_relgrow.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_relgrow.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax_relgrow.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax_absrow.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
    ax_absrow.set_yticklabels(["0", "1k", "2k", "3k", "4k", "5k"])
    ax_absrow.yaxis.set_minor_locator(MultipleLocator(500))

    # setting the y-axis limit
    ax_abschina.set_ylim([0, 100000])
    ax_relgrow.set_ylim([0, 1])
    ax_absrow.set_ylim([0, 5000])

    # label axis
    plt.xlabel("date")
    ax_abschina.set_ylabel("total # of confirmed infections in Mainland China")
    ax_relgrow.set_ylabel("relative growth of infections in China")
    ax_absrow.set_ylabel("total # of confirmed infections outside China")

    ax_relgrow.tick_params(axis="y",colors=change_color)
    ax_relgrow.yaxis.label.set_color(change_color)

    ax_absrow.tick_params(axis="y",colors=row_color)
    ax_absrow.yaxis.label.set_color(row_color)


    # plot the original data
    ax_abschina.plot(data_x, data_y, "s", color=data_color,
                     markersize=7, zorder=10)

    if i > row_start:
        ax_absrow.plot(data_x[row_start: i], row_data[0:i -
                                                      row_start], lw=4, color=row_color, zorder=10)

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

            ax_abschina.plot(nom_x, nom_y, color=exponential_color, linewidth=3,
                             label="data fitted to an exponential function")
            ax_abschina.fill_between(nom_x, nom_y - std_y, nom_y + std_y, facecolor=exponential_color,
                                     alpha=0.3, label="area of uncertainty for the exponential fit")
        elif k == i-1:
            ax_abschina.fill_between(nom_x, nom_y - std_y, nom_y +
                                     std_y, facecolor=exponential_color, alpha=0.2)
        elif k == i-2:
            ax_abschina.fill_between(nom_x, nom_y - std_y, nom_y +
                                     std_y, facecolor=exponential_color, alpha=0.05)

    for k in range(np.max([i-n, sigmoidal_start]), i+1):
        # fit the sigmoidal function
        popt, pcov = scipy.optimize.curve_fit(
            sigmoidal_fit_function,  data_x[8:k],  data_y[8:k], p0=[80000, 0.4, 20])
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
            ax_abschina.plot(nom_x, nom_y, color=sigmoidal_color, linewidth=3,
                             label="data fitted to an sigmoidal function")
            ax_abschina.fill_between(nom_x, nom_y - std_y, nom_y + std_y, facecolor=sigmoidal_color,
                                     alpha=0.6, label="area of uncertainty for the sigmoidal fit")
        elif k == i-1:
            ax_abschina.fill_between(nom_x, nom_y - std_y, nom_y +
                                     std_y, facecolor=sigmoidal_color, alpha=0.2)
        elif k == i-2:
            ax_abschina.fill_between(nom_x, nom_y - std_y, nom_y +
                                     std_y, facecolor=sigmoidal_color, alpha=0.1)

    ax_relgrow.plot(data_x[1:i], relative_growth_y[0:i-1],
                    color=change_color, alpha=0.4, lw=2)

    # format the border of the diagram
    ax_abschina.spines['top'].set_color('white')
    ax_relgrow.spines['top'].set_color('white')
    ax_absrow.spines['top'].set_color('white')

    ax_relgrow.spines['left'].set_color(change_color)
    ax_absrow.spines['left'].set_color([0, 0, 0, 0])
    ax_abschina.spines['left'].set_color([0, 0, 0, 0])

    ax_absrow.spines['right'].set_color(row_color)

    # these objects are used to create a consistent legend
    legendel_chinadata = Line2D([0], [0], marker='s', color=data_color,
                                lw=0, markerfacecolor=data_color, markersize=10)
    legendel_rowdata = Line2D([0], [0], color=row_color, lw=4)
    legendel_exponentialfit = Line2D(
        [0], [0], color=exponential_color, lw=4)
    legendel_sigmoidalfit = Line2D(
        [0], [0], color=sigmoidal_color, lw=4)
    legendel_exponential_areaofuncertainty = Patch(
        facecolor=exponential_color, alpha=0.5)
    legendel_sigmoidal_areaofuncertainty = Patch(
        facecolor=sigmoidal_color, alpha=0.5)
    legendel_relchange = Line2D(
        [0], [0], color=change_color, lw=4)

    # add the legend and object descriptions
    legend = ax_relgrow.legend([legendel_chinadata,
                                legendel_rowdata,
                                legendel_exponentialfit,
                                legendel_exponential_areaofuncertainty,
                                legendel_sigmoidalfit,
                                legendel_sigmoidal_areaofuncertainty,
                                legendel_relchange],
                               ["total infections in China (adjusted)",
                                "total infections outside China",
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
    plt.close()

# batch the images to a video
initial_frame_repeatcount = 2  # number of times the initial frame is to be repeated
final_frame_repeatcount = 7  # number of times the final frame is to be repeated

video_name = 'video.avi'  # name of the exported video

# get video size data
frame = cv2.imread("./" + str(regression_start) + ".png")
height, width, layers = frame.shape

# create video writer
fps = 4
video = cv2.VideoWriter(video_name, 0, fps, (width, height))

# write initial frame
for i in range(0, initial_frame_repeatcount):
    video.write(cv2.imread("./" + str(regression_start) + ".png"))

# animation frames
for i in range(regression_start + 1, len(post_cd_data)+3):
    video.write(cv2.imread("./" + str(i) + ".png"))

# write final frame repeatedly
for i in range(0, final_frame_repeatcount):
    video.write(cv2.imread("./" + str(len(post_cd_data)+3) + ".png"))

# save video
cv2.destroyAllWindows()
video.release()
