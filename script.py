import os
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

from colors import *
from load import get_data_from_file
from regions import *

if not os.path.exists("./images/"):
    os.mkdir("./images/")


jhu_submodule_path = "./JHU-data/csse_covid_19_data/csse_covid_19_time_series/"
datafile_confirmed = jhu_submodule_path + \
    "time_series_19-covid-Confirmed.csv"
datafile_deaths = jhu_submodule_path + "time_series_19-covid-Deaths.csv"
datafile_recovered = jhu_submodule_path + \
    "time_series_19-covid-Recovered.csv"

recovered_by_region, recovered_total = get_data_from_file(datafile_recovered)
confirmed_by_region, confirmed_total = get_data_from_file(datafile_confirmed)
dead_by_region, dead_total = get_data_from_file(datafile_deaths)
entries = len(recovered_total)

recovered_china = recovered_by_region[MAINLAND_CHINA]
confirmed_china = confirmed_by_region[MAINLAND_CHINA]

recovered_row = confirmed_total - recovered_china
confirmed_row = confirmed_total - confirmed_china


def row_fit_function(x, a, b):
    return a*np.exp(b*x)


def row_fit_jacobian(x, a, b):
    return np.transpose([np.exp(b*x), a*x*np.exp(b*x)])


def china_fit_function(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))


def china_fit_jacobian(x, a, b, c):
    return np.transpose([
        1/(1+np.exp(-b*(x-c))),
        -a/((1+np.exp(-b*(x-c)))**2)*(c-x)*np.exp(-b*(x-c)),
        -a/((1+np.exp(-b*(x-c)))**2)*b*np.exp(-b*(x-c))])


def generate_fits(x, y, start, p0, function, jacobian):
    result = []
    for i in range(start, len(x)+1):
        result.append(scipy.optimize.curve_fit(
            function, x[:i], y[:i], p0, jac=jacobian))
    return result


plot_start = 11

fit_data_x = np.arange(0, entries)
china_fits = generate_fits(fit_data_x, confirmed_china, plot_start, [
    80000, 0.4, 20], china_fit_function, china_fit_jacobian)
row_fits = generate_fits(fit_data_x, confirmed_row, plot_start, [
    5, 0.2], row_fit_function, row_fit_jacobian)

# increase pyplot font size
font = {'family': 'sans-serif', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
plt.rc('axes', labelsize=20)


# x-axis range
xmin = 0
xmax = 49
# steps between major ticks on x-axi
xstep = 7

# create animation frames
for l in range(plot_start, entries+4):
    current_date_index = l  # index of the last data point to be used
    desired_fit_count = 3  # number of previous fits to include

    if l > entries:  # used to fade out the last three plots
        current_date_index = entries
        desired_fit_count = 3 - (l - current_date_index)+1

    data_x = np.arange(0, current_date_index)
    china_data_y = confirmed_by_region[MAINLAND_CHINA][0:current_date_index]

    row_data_y = (confirmed_total -
                  confirmed_by_region[MAINLAND_CHINA])[0:current_date_index]

    # creation of pyplot plot
    fig, ax_shared = plt.subplots()
    ax_shared.yaxis.tick_right()
    ax_shared.yaxis.set_label_position("right")
    ax_shared.spines['left'].set_color("none")

    # setting the dimensions (basically resolution with 12by9 aspect ratio)
    fig.set_figheight(10)
    fig.set_figwidth(12)

    majxticks = ([], [])
    # generation of x-axis tick-names
    startdate = datetime(2020, 1, 22)
    for j in range(xmin, xmax+1, xstep):
        majxticks[0].append(j)
        majxticks[1].append((startdate + timedelta(j)).strftime("%d. %b"))

    # setting the x-axis ticks
    ax_shared.set_xlim([xmin, xmax])
    plt.xticks(majxticks[0], majxticks[1])
    ax_shared.xaxis.set_minor_locator(MultipleLocator(1))
    ax_shared.tick_params(axis="x", which="major", length=8, width=1.5)
    ax_shared.tick_params(axis="x", which="minor", length=5, width=1)

    # setting the y-axis ticks
    ax_shared.set_yticks([0, 2e4, 4e4, 6e4, 8e4, 10e4, 12e4])
    ax_shared.set_yticklabels(
        ["0", "20k", "40k", "60k", "80k", "100k", "120k"])
    ax_shared.yaxis.set_minor_locator(MultipleLocator(10000))

    # setting the y-axis limit
    ax_shared.set_ylim([0, 130000])

    # label axis
    plt.xlabel("date")
    ax_shared.set_ylabel("total # of confirmed infections")

    # format the border of the diagram
    ax_shared.spines['top'].set_color('white')

    # plot the original data
    ax_shared.plot(data_x, china_data_y, "s", color=china_color,
                   markersize=7, zorder=10)
    ax_shared.fill_between(data_x,
                           np.zeros(current_date_index),
                           recovered_by_region[MAINLAND_CHINA][:current_date_index],
                           color=china_recovered_color, ec=china_color, alpha=0.5, hatch="//")

    ax_shared.plot(data_x, row_data_y, "o",
                   color=row_color, markersize=7, zorder=10)
    ax_shared.fill_between(data_x, np.zeros(current_date_index),
                           recovered_total[:current_date_index] -
                           recovered_by_region[MAINLAND_CHINA][:current_date_index],
                           color=row_recovered_color, alpha=0.5, hatch="..")
    # create the exponential plots
    for k in range(0, np.min([desired_fit_count, current_date_index-plot_start])):
        # fit the exponential function
        popt, pcov = row_fits[current_date_index-plot_start-k]
        # get errors from trace of covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # create uncertainty floats for error bars, 2* means 2 sigma
        a = ufloat(popt[0], perr[0])
        b = ufloat(popt[1], perr[1])

        # get the values of the uncertain fit
        fit_x_unc = np.linspace(xmin, xmax, 300)
        fit_y_unc = a*unp.exp(b*fit_x_unc)
        nom_x = unp.nominal_values(fit_x_unc)
        nom_y = unp.nominal_values(fit_y_unc)
        std_y = unp.std_devs(fit_y_unc)

        # plot
        if k == 0:
            print("[" + (startdate + timedelta(current_date_index)
                         ).strftime("%d. %b") + "] ROW fit(y=a*exp(b*x))-parameters:")
            print("a = " + str(a))
            print("b = " + str(b))

            ax_shared.plot(
                nom_x, nom_y, color=row_regression_color, linewidth=3)
            ax_shared.fill_between(
                nom_x, nom_y - std_y, nom_y + std_y, facecolor=row_regression_color, alpha=0.5)
        elif k == 1:
            ax_shared.fill_between(nom_x, nom_y - std_y, nom_y +
                                   std_y, facecolor=row_regression_color, alpha=0.2)
        elif k == 2:
            ax_shared.fill_between(nom_x, nom_y - std_y, nom_y +
                                   std_y, facecolor=row_regression_color, alpha=0.05)

    for k in range(0, np.min([desired_fit_count, current_date_index-plot_start])):
        # fit the sigmoidal function
        popt, pcov = china_fits[current_date_index-plot_start-k]
        # get errors from trace of covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # create uncertainty floats for error bars, 2* means 2 sigma
        a = ufloat(popt[0], perr[0])
        b = ufloat(popt[1], perr[1])
        c = ufloat(popt[2], perr[2])

        # get the values of the uncertain fit
        fit_x_unc = np.linspace(xmin, xmax, 300)
        fit_y_unc = a/(1+unp.exp(-b*(fit_x_unc-c)))
        nom_x = unp.nominal_values(fit_x_unc)
        nom_y = unp.nominal_values(fit_y_unc)
        std_y = unp.std_devs(fit_y_unc)

        # plot
        if k == 0:
            print("[" + (startdate + timedelta(current_date_index)
                         ).strftime("%d. %b") + "] China fit(y=a/(1+exp(-b*(x-c))))-parameters:")
            print("a = " + str(a))
            print("b = " + str(b))
            print("c = " + str(c))
            ax_shared.plot(
                nom_x, nom_y, color=china_regression_color, linewidth=3)
            ax_shared.fill_between(
                nom_x, nom_y - std_y, nom_y + std_y, facecolor=china_regression_color, alpha=0.6)
        elif k == 1:
            ax_shared.fill_between(nom_x, nom_y - std_y, nom_y +
                                   std_y, facecolor=china_regression_color, alpha=0.2)
        elif k == 2:
            ax_shared.fill_between(nom_x, nom_y - std_y, nom_y +
                                   std_y, facecolor=china_regression_color, alpha=0.1)

    plt.tight_layout()

    ax_regional_development = plt.axes([.03, 0.6, .35, .35])
    ax_regional_development.set_title("regional distribution")
    regional_x = np.arange(0, 7)
    bottom = np.zeros(7)
    for i in range(1, REGION_COUNT):
        regional_data = np.zeros(7)
        for d in range(7):
            regional_data[d] = (confirmed_by_region[i][current_date_index-d-1]
                                ) / confirmed_row[current_date_index-d-1]
        ax_regional_development.fill_between(regional_x, bottom,
                                             bottom+regional_data,
                                             color=region_colors[i], alpha=0.7)
        bottom += regional_data
    ax_regional_development.set_ylim([0, 1])
    ax_regional_development.set_yticks([])
    ax_regional_development.set_xlim([0, 6])
    plt.xticks([0, 6], ["  " + (startdate + timedelta(current_date_index-1)
                         ).strftime("%d. %b"), "one week ago"])

    # these objects are used to create a consistent legend
    legendel_china = Patch(facecolor=region_colors[0])
    legendel_westerpacificregion = Patch(facecolor=region_colors[1])
    legendel_europeanregion = Patch(facecolor=region_colors[2])
    legendel_southeastasiaregion = Patch(
        facecolor=region_colors[3])
    legendel_easternmediterraneanregion = Patch(
        facecolor=region_colors[4])
    legendel_regionoftheamericans = Patch(
        facecolor=region_colors[5])
    legendel_africanregion = Patch(facecolor=region_colors[6])
    legendel_other = Patch(facecolor=region_colors[7])

    # add the legend and object descriptions
    piechart_legend = ax_regional_development.legend([legendel_westerpacificregion,
                                                      legendel_europeanregion,
                                                      legendel_southeastasiaregion,
                                                      legendel_easternmediterraneanregion,
                                                      legendel_regionoftheamericans,
                                                      legendel_africanregion,
                                                      legendel_other],
                                                     ["Western Pacific Region",
                                                      "European Region",
                                                      "South-East Asia Region",
                                                      "Eastern Mediterranean Region",
                                                      "Region of the Americans",
                                                      "African Region",
                                                      "Other"],
                                                     loc='upper center', bbox_to_anchor=(0.5, -0.1))
    piechart_legend.get_frame().set_edgecolor("black")
    piechart_legend.set_zorder(20)

    legendel_china_data = Line2D([0], [0], marker="s", color=china_color,
                                 lw=0, markerfacecolor=china_color, markersize=10)
    legendel_china_regression = Line2D(
        [0], [0], color=china_regression_color, lw=4)
    legendel_china_recovered = Patch(
        facecolor=china_recovered_color, alpha=0.5, hatch="//", edgecolor=china_recovered_color)
    legendel_spacer = Patch(facecolor="none")
    legendel_row_data = Line2D([0], [0], marker="o", color=row_color,
                               lw=0, markerfacecolor=row_color, markersize=10)
    legendel_row_regression = Line2D(
        [0], [0], color=row_regression_color, lw=4)
    legendel_row_recovered = Patch(
        facecolor=row_recovered_color, alpha=0.5, hatch="..", edgecolor=row_recovered_color)

    total_legend = ax_shared.legend([legendel_china_data,
                                     legendel_china_regression,
                                     legendel_china_recovered,
                                     legendel_row_data,
                                     legendel_row_regression,
                                     legendel_row_recovered],
                                    ["Infections in China",
                                     "Adoption to an sigmoidal fit",
                                     "Recovered cases in China",
                                     "Infections outside China",
                                     "Adoption to an exponential fit",
                                     "Recovered cases outside China"],
                                    loc='upper right')
    total_legend.get_frame().set_edgecolor("black")
    total_legend.set_zorder(20)

    # save the plot in the current folder
    plt.savefig("./images/" + str(l) + ".png")
    plt.close()

# batch the images to a video
initial_frame_repeatcount = 2  # number of times the initial frame is to be repeated
final_frame_repeatcount = 7  # number of times the final frame is to be repeated

video_name = 'video.mp4'  # name of the exported video

# get video size data
frame = cv2.imread("./images/" + str(plot_start) + ".png")
height, width, layers = frame.shape

# create video writer
fps = 4
video = cv2.VideoWriter(video_name, 0, fps, (width, height))

# write initial frame
for current_date_index in range(0, initial_frame_repeatcount):
    video.write(cv2.imread("./images/" + str(plot_start) + ".png"))

# animation frames
for current_date_index in range(plot_start + 1, entries+3):
    video.write(cv2.imread("./images/" + str(current_date_index) + ".png"))

# write final frame repeatedly
for current_date_index in range(0, final_frame_repeatcount):
    video.write(cv2.imread("./images/" + str(entries+3) + ".png"))

# save video
cv2.destroyAllWindows()
video.release()
