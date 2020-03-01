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

datafile_WHO_name = "./who_covid_19_sit_rep_time_series.csv"
datafile_deaths_name = "./time_series_19-covid-Deaths.csv"
datafile_recovered_name = "./time_series_19-covid-Recovered.csv"

china_total_infections = []
row_total_infections = []
row_distribution = []
infected_recovered_dead_distribution = []

with open(datafile_WHO_name) as datafile:
    data_WHO_raw = datafile.readlines()
data_WHO_raw = [l.split(",") for l in data_WHO_raw]

with open(datafile_WHO_name) as datafile:
    data_deaths_raw = datafile.readlines()
data_deaths_raw = [l.split(",") for l in data_deaths_raw]

with open(datafile_WHO_name) as datafile:
    data_recovered_raw = datafile.readlines()
data_recovered_raw = [l.split(",") for l in data_recovered_raw]

for i in range(3, len(data_WHO_raw[0])):
    row_distribution.append({"Western Pacific Region": 0, "South-East Asia Region": 0,
                             "Region of the Americas": 0, "European Region": 0, "Eastern Mediterranean Region": 0, "Other": 0})

for i in range(len(data_WHO_raw)):
    if(i == 2):
        for j in range(3, len(data_WHO_raw[i])):
            china_total_infections.append(int(data_WHO_raw[i][j]))
    elif (i == 3):
        for j in range(3, len(data_WHO_raw[i])):
            row_total_infections.append(int(data_WHO_raw[i][j]))
    elif (i > 43):
        for j in range(3, len(data_WHO_raw[i])):
            if(data_WHO_raw[i][j] != ""):
                row_distribution[j-3][data_WHO_raw[i][2]] += int(data_WHO_raw[i][j])


china_relgrowth = []
for current_date_index in range(len(china_total_infections)-1):
    china_relgrowth.append(
        china_total_infections[current_date_index+1]/china_total_infections[current_date_index]-1)

row_relgrowth = []
for current_date_index in range(len(row_total_infections)-1):
    row_relgrowth.append(
        row_total_infections[current_date_index+1]/row_total_infections[current_date_index]-1)


# increase pyplot font size
font = {'family': 'normal', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
plt.rc('axes', labelsize=20)

# function definition for the exponential fit with parameters a, b


def row_fit_function(x, a, b):
    return a*np.exp(b*x)  # exponential


# function definition for the sigmoidal fit with parameters a, b, c
def china_fit_function(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))  # sigmoidal


plot_start = 11
china_regression_start = 16  # index to start plotting the sigmoidal fit
row_regression_start = 11

# x-axis range
xmin = 0
xmax = 52
# steps between major ticks on x-axi
xstep = 7

# colors
china_color = np.array([18, 141, 179]) / 255
china_growth_color = np.array([51, 207, 255]) / 255
china_regression_color = np.array([56, 209, 255]) / 255

row_color = np.array([179, 98, 18]) / 255
row_growth_color = np.array([255, 166, 77]) / 255
row_regression_color = np.array([255, 152, 51]) / 255

piechart_colors = 0

# create animation frames
for l in range(plot_start, len(china_total_infections)+4):
    current_date_index = l  # index of the last data point to be used
    desired_fit_count = 3  # number of previous fits to include

    if l > len(china_total_infections):  # used to fade out the last three plots
        current_date_index = len(china_total_infections)
        desired_fit_count = 3 - (l - current_date_index)+1

    china_data_x = np.arange(0, current_date_index)
    china_data_y = china_total_infections[0:current_date_index]

    row_data_x = china_data_x[0:current_date_index]
    row_data_y = row_total_infections[0:current_date_index]

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
    startdate = datetime(2020, 1, 21)
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

    ax_absrow.set_yticks([0, 2000, 4000, 6000, 8000, 1e4])
    ax_absrow.set_yticklabels(["0", "2k", "4k", "6k", "8k", "10k"])
    ax_absrow.yaxis.set_minor_locator(MultipleLocator(1000))

    # setting the y-axis limit
    ax_abschina.set_ylim([0, 100000])
    ax_relgrow.set_ylim([0, 1])
    ax_absrow.set_ylim([0, 10000])

    # label axis
    plt.xlabel("date")
    ax_abschina.set_ylabel("total # of confirmed infections in Mainland China")
    ax_relgrow.set_ylabel("relative growth of infections")
    ax_absrow.set_ylabel("total # of confirmed infections outside China")

    # format the border of the diagram
    ax_abschina.spines['top'].set_color('white')
    ax_relgrow.spines['top'].set_color('white')
    ax_absrow.spines['top'].set_color('white')

    ax_abschina.spines['right'].set_color(china_color)
    ax_absrow.spines['right'].set_color(row_color)

    ax_abschina.tick_params(axis="y", colors=china_color)
    ax_abschina.yaxis.label.set_color(china_color)

    ax_absrow.tick_params(axis="y", colors=row_color)
    ax_absrow.yaxis.label.set_color(row_color)

    # plot the original data
    ax_abschina.plot(china_data_x, china_data_y, "s", color=china_color,
                     markersize=7, zorder=10)

    ax_absrow.plot(row_data_x, row_data_y, "o",
                   color=row_color, markersize=7, zorder=10)

    # create the exponential plots
    for k in range(0, np.min([desired_fit_count, current_date_index-row_regression_start])):
        # fit the exponential function
        popt, pcov = scipy.optimize.curve_fit(row_fit_function,  row_data_x[:current_date_index-k],
                                              row_data_y[:current_date_index-k], p0=[5, 0.2])
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

            ax_absrow.plot(
                nom_x, nom_y, color=row_regression_color, linewidth=3)
            ax_absrow.fill_between(
                nom_x, nom_y - std_y, nom_y + std_y, facecolor=row_regression_color, alpha=0.5)
        elif k == 1:
            ax_absrow.fill_between(nom_x, nom_y - std_y, nom_y +
                                   std_y, facecolor=row_regression_color, alpha=0.2)
        elif k == 2:
            ax_absrow.fill_between(nom_x, nom_y - std_y, nom_y +
                                   std_y, facecolor=row_regression_color, alpha=0.05)

    for k in range(0, np.min([desired_fit_count, current_date_index-china_regression_start])):
        # fit the sigmoidal function
        popt, pcov = scipy.optimize.curve_fit(
            china_fit_function,  china_data_x[0:current_date_index-k],  china_data_y[0:current_date_index-k], p0=[80000, 0.4, 20])
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
            ax_abschina.plot(
                nom_x, nom_y, color=china_regression_color, linewidth=3)
            ax_abschina.fill_between(
                nom_x, nom_y - std_y, nom_y + std_y, facecolor=china_regression_color, alpha=0.6)
        elif k == 1:
            ax_abschina.fill_between(nom_x, nom_y - std_y, nom_y +
                                     std_y, facecolor=china_regression_color, alpha=0.2)
        elif k == 2:
            ax_abschina.fill_between(nom_x, nom_y - std_y, nom_y +
                                     std_y, facecolor=china_regression_color, alpha=0.1)

    ax_relgrow.plot(china_data_x[1:current_date_index], china_relgrowth[0:current_date_index-1],
                    color=china_growth_color, alpha=0.8, lw=2)

    ax_relgrow.plot(row_data_x[1:current_date_index], row_relgrowth[0:current_date_index-1],
                    color=row_growth_color, alpha=0.8, lw=2)

    # these objects are used to create a consistent legend
    legendel_chinadata = Line2D([0], [0], marker="s", color=china_color,
                                lw=0, markerfacecolor=china_color, markersize=10)
    legendel_rowdata = Line2D([0], [0], marker="o", color=row_color,
                              lw=0, markerfacecolor=row_color, markersize=10)
    legendel_row_regression = Line2D(
        [0], [0], color=row_regression_color, lw=4)
    legendel_china_regression = Line2D(
        [0], [0], color=china_regression_color, lw=4)
    legendel_row_regression_areaofuncertainty = Patch(
        facecolor=row_regression_color, alpha=0.5)
    legendel_china_regression_areaofuncertainty = Patch(
        facecolor=china_regression_color, alpha=0.5)
    legendel_relchange_china = Line2D(
        [0], [0], color=china_growth_color, lw=4)
    legendel_relchange_row = Line2D(
        [0], [0], color=row_growth_color, lw=4)

    # add the legend and object descriptions
    # legend = ax_relgrow.legend([legendel_chinadata,
    #                            legendel_china_regression,
    #                            legendel_china_regression_areaofuncertainty,
    #                            legendel_relchange_china,
    #                            legendel_rowdata,
    #                            legendel_row_regression,
    #                            legendel_row_regression_areaofuncertainty,
    #                            legendel_relchange_row],
    #                           ["total infections in China",
    #                            "infections in China fitted to an logistic function",
    #                            "68% area of uncertainty for to the logistic function",
    #                            "relative growth in China",
    #                            "total infections outside China",
    #                            "infections outside China fitted to an exponential function",
    #                            "68% area of uncertainty for the exponential function",
    #                            "relative growth outside China"], loc='upper right')
    # legend.get_frame().set_edgecolor("black")
    # legend.set_zorder(20)
    plt.title("see comments for further explanations")

    plt.tight_layout()

    ax_pie = plt.axes([.1, .5, .4, .4])
    v = row_distribution[current_date_index-1].values()
    k = row_distribution[current_date_index-1].keys()
    ax_pie.pie(v, labels=k)
    ax_pie.axis("equal")

    # save the plot in the current folder
    plt.savefig(str(l) + ".png")
    plt.close()

# batch the images to a video
initial_frame_repeatcount = 2  # number of times the initial frame is to be repeated
final_frame_repeatcount = 7  # number of times the final frame is to be repeated

video_name = 'video.avi'  # name of the exported video

# get video size data
frame = cv2.imread("./" + str(plot_start) + ".png")
height, width, layers = frame.shape

# create video writer
fps = 4
video = cv2.VideoWriter(video_name, 0, fps, (width, height))

# write initial frame
for current_date_index in range(0, initial_frame_repeatcount):
    video.write(cv2.imread("./" + str(plot_start) + ".png"))

# animation frames
for current_date_index in range(plot_start + 1, len(china_total_infections)+3):
    video.write(cv2.imread("./" + str(current_date_index) + ".png"))

# write final frame repeatedly
for current_date_index in range(0, final_frame_repeatcount):
    video.write(cv2.imread("./" + str(len(china_total_infections)+3) + ".png"))

# save video
cv2.destroyAllWindows()
video.release()
