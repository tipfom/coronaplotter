from datetime import datetime, timedelta

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.optimize
import uncertainties.unumpy as unp
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from uncertainties import ufloat

# data points, cummulative
# to add another date simply append the # of infected people

jhu_submodule_path = "./JHU-data/csse_covid_19_data/csse_covid_19_time_series/"
datafile_confirmed_name = jhu_submodule_path + \
    "time_series_19-covid-Confirmed.csv"
datafile_deaths_name = jhu_submodule_path + "time_series_19-covid-Deaths.csv"
datafile_recovered_name = jhu_submodule_path + \
    "time_series_19-covid-Recovered.csv"

MAINLAND_CHINA = 0
WESTERN_PACIFIC_REGION = 1
EUROPEAN_REGION = 2
SOUTH_EAST_ASIA_REGION = 3
EASTERN_MEDITERRANEAN_REGION = 4
REGION_OF_THE_AMERICANS = 5
AFRICAN_REGION = 6
OTHER = 7

region_map = {
    "Mainland China": MAINLAND_CHINA,
    "Hong Kong": MAINLAND_CHINA,
    "Macau": MAINLAND_CHINA,
    "Taiwan": MAINLAND_CHINA,

    "South Korea": WESTERN_PACIFIC_REGION,
    "Japan": WESTERN_PACIFIC_REGION,
    "Singapore": WESTERN_PACIFIC_REGION,
    "Australia": WESTERN_PACIFIC_REGION,
    "Malaysia": WESTERN_PACIFIC_REGION,
    "Vietnam": WESTERN_PACIFIC_REGION,
    "Philippines": WESTERN_PACIFIC_REGION,
    "Cambodia": WESTERN_PACIFIC_REGION,
    "New Zealand": WESTERN_PACIFIC_REGION,

    "Italy": EUROPEAN_REGION,
    "France": EUROPEAN_REGION,
    "Germany": EUROPEAN_REGION,
    "Spain": EUROPEAN_REGION,
    "UK": EUROPEAN_REGION,
    "Switzerland": EUROPEAN_REGION,
    "Norway": EUROPEAN_REGION,
    "Sweden": EUROPEAN_REGION,
    "Austria": EUROPEAN_REGION,
    "Croatia": EUROPEAN_REGION,
    "Israel": EUROPEAN_REGION,
    "Netherlands": EUROPEAN_REGION,
    "Azerbaijan": EUROPEAN_REGION,
    "Denmark": EUROPEAN_REGION,
    "Georgia": EUROPEAN_REGION,
    "Greece": EUROPEAN_REGION,
    "Romania": EUROPEAN_REGION,
    "Finland": EUROPEAN_REGION,
    "Russia": EUROPEAN_REGION,
    "Belarus": EUROPEAN_REGION,
    "Belgium": EUROPEAN_REGION,
    "Estonia": EUROPEAN_REGION,
    "Ireland": EUROPEAN_REGION,
    "Lithuania": EUROPEAN_REGION,
    "Monaco": EUROPEAN_REGION,
    "North Macedonia": EUROPEAN_REGION,
    "San Marino": EUROPEAN_REGION,
    "Luxembourg": EUROPEAN_REGION,
    "Iceland": EUROPEAN_REGION,

    "Thailand": SOUTH_EAST_ASIA_REGION,
    "India": SOUTH_EAST_ASIA_REGION,
    "Nepal": SOUTH_EAST_ASIA_REGION,
    "Sri Lanka": SOUTH_EAST_ASIA_REGION,

    "Iran": EASTERN_MEDITERRANEAN_REGION,
    "Kuwait": EASTERN_MEDITERRANEAN_REGION,
    "Bahrain": EASTERN_MEDITERRANEAN_REGION,
    "United Arab Emirates": EASTERN_MEDITERRANEAN_REGION,
    "Iraq": EASTERN_MEDITERRANEAN_REGION,
    "Oman": EASTERN_MEDITERRANEAN_REGION,
    "Pakistan": EASTERN_MEDITERRANEAN_REGION,
    "Lebanon": EASTERN_MEDITERRANEAN_REGION,
    "Afghanistan": EASTERN_MEDITERRANEAN_REGION,
    "Egypt": EASTERN_MEDITERRANEAN_REGION,
    "Qatar": EASTERN_MEDITERRANEAN_REGION,

    "US": REGION_OF_THE_AMERICANS,
    "Canada": REGION_OF_THE_AMERICANS,
    "Brazil": REGION_OF_THE_AMERICANS,
    "Mexico": REGION_OF_THE_AMERICANS,
    "Ecuadoe": REGION_OF_THE_AMERICANS,

    "Algeria": AFRICAN_REGION,
    "Nigeria": AFRICAN_REGION,

    "Others": OTHER,
}

recovered_by_region = []
confirmed_by_region = []
dead_by_region = []

total_confirmed = np.array([])
total_recovered = np.array([])
total_dead = np.array([])

for i in range(OTHER):
    recovered_by_region.append(np.array([]))
    confirmed_by_region.append(np.array([]))
    dead_by_region.append(np.array([]))

data_confirmed_raw = []
with open(datafile_confirmed_name) as datafile:
    datafile_reader = csv.reader(datafile, delimiter=",", quotechar="\"")
    for row in datafile_reader:
        data_confirmed_raw.append(row)

data_deaths_raw = []
with open(datafile_deaths_name) as datafile:
    datafile_reader = csv.reader(datafile, delimiter=",", quotechar="\"")
    for row in datafile_reader:
        data_deaths_raw.append(row)

data_recovered_raw = []
with open(datafile_recovered_name) as datafile:
    datafile_reader = csv.reader(datafile, delimiter=",", quotechar="\"")
    for row in datafile_reader:
        data_recovered_raw.append(row)

entries = len(data_deaths_raw[0]) - 5
for i in range(4, len(data_deaths_raw[0])):
    column_recovered_by_region = np.zeros(8)
    column_dead_by_region = np.zeros(8)
    column_confirmed_by_region = np.zeros(8)

    for j in range(1, len(data_deaths_raw)):
        country = data_confirmed_raw[j][1]
        if region_map.__contains__(country):
            region = region_map[country]

            if(data_confirmed_raw[j][i] != ""):
                column_confirmed_by_region[region] += int(
                    data_confirmed_raw[j][i])
            if(data_deaths_raw[j][i] != ""):
                column_dead_by_region[region] += int(data_deaths_raw[j][i])
            if(data_recovered_raw[j][i] != ""):
                column_recovered_by_region[region] += int(
                    data_recovered_raw[j][i])
        else:
            print("could not find region for " + country)

    column_total_confirmed = 0
    column_total_recovered = 0
    column_total_dead = 0

    for i in range(OTHER):
        confirmed_by_region[i] = np.append(
            confirmed_by_region[i], column_confirmed_by_region[i])
        recovered_by_region[i] = np.append(
            recovered_by_region[i], column_recovered_by_region[i])
        dead_by_region[i] = np.append(
            dead_by_region[i], column_dead_by_region[i])

        column_total_confirmed += column_confirmed_by_region[i]
        column_total_recovered += column_recovered_by_region[i]
        column_total_dead += column_dead_by_region[i]

    total_confirmed = np.append(total_confirmed, column_total_confirmed)
    total_recovered = np.append(total_recovered, column_total_recovered)
    total_dead = np.append(total_dead, column_total_dead)

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
xmax = 42
# steps between major ticks on x-axi
xstep = 7

# colors
china_color = "#1866b4"  # np.array([18, 141, 179]) / 255
china_regression_color = "#5899DA"  # np.array([56, 209, 255]) / 255

row_color = "#596468"  # np.array([179, 98, 18]) / 255
row_regression_color = "#848f94"  # np.array([255, 152, 51]) / 255

piechart_colors = ["#66c2a3",  # recovered
                   "#19A979",
                   "#f5aa85",  # infected
                   "#E8743B",
                   "#bac1c4",  # dead
                   "#848f94"]

barchart_colors = [
    "#000000",  # MAINLAND_CHINA = 0
    "#5899DA",  # WESTERN_PACIFIC_REGION = 1
    "#E8743B",  # EUROPEAN_REGION = 2
    "#19A979",  # SOUTH_EAST_ASIA_REGION = 3
    "#ED4A7B",  # EASTERN_MEDITERRANEAN_REGION = 4
    "#945ECF",  # REGION_OF_THE_AMERICANS = 5
    "#13A4B4",  # AFRICAN_REGION = 6
    "#6C8893",  # OTHER = 7
]

# create animation frames
for l in range(plot_start, entries+4):
    current_date_index = l  # index of the last data point to be used
    desired_fit_count = 3  # number of previous fits to include

    if l > entries:  # used to fade out the last three plots
        current_date_index = entries
        desired_fit_count = 3 - (l - current_date_index)+1

    china_data_x = np.arange(0, current_date_index)
    china_data_y = confirmed_by_region[MAINLAND_CHINA][0:current_date_index]

    row_data_x = china_data_x
    row_data_y = (total_confirmed -
                  confirmed_by_region[MAINLAND_CHINA])[0:current_date_index]

    # creation of pyplot plot
    fig, ax_abschina = plt.subplots()
    ax_absrow = ax_abschina.twinx()

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
    ax_abschina.set_xlim([xmin, xmax])
    plt.xticks(majxticks[0], majxticks[1])
    ax_abschina.xaxis.set_minor_locator(MultipleLocator(1))
    ax_abschina.tick_params(axis="x", which="major", length=8, width=1.5)
    ax_abschina.tick_params(axis="x", which="minor", length=5, width=1)

    # setting the y-axis ticks
    ax_abschina.set_yticks([0, 2e4, 4e4, 6e4, 8e4, 10e4, 12e4])
    ax_abschina.set_yticklabels(
        ["0", "20k", "40k", "60k", "80k", "100k", "120k"])
    ax_abschina.yaxis.set_minor_locator(MultipleLocator(10000))

    ax_absrow.set_yticks([0, 2000, 4000, 6000, 8000, 1e4, 1.2e4])
    ax_absrow.set_yticklabels(["0", "2k", "4k", "6k", "8k", "10k", "12k"])
    ax_absrow.yaxis.set_minor_locator(MultipleLocator(1000))

    # setting the y-axis limit
    ax_abschina.set_ylim([0, 120000])
    ax_absrow.set_ylim([0, 12000])

    # label axis
    plt.xlabel("date")
    ax_abschina.set_ylabel("total # of confirmed infections in Mainland China")
    ax_absrow.set_ylabel("total # of confirmed infections outside China")

    # format the border of the diagram
    ax_abschina.spines['top'].set_color('white')
    ax_absrow.spines['top'].set_color('white')

    ax_abschina.spines['left'].set_color(china_color)

    ax_absrow.spines['left'].set_color("none")
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

    last_bottom = np.zeros(current_date_index)
    for i in range(1, OTHER):
        reg_data = confirmed_by_region[i][0:current_date_index]
        ax_absrow.bar(row_data_x, reg_data, width=0.3,
                      color=barchart_colors[i], bottom=last_bottom)
        last_bottom += reg_data

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

    plt.title("see comments for further explanations")
    plt.tight_layout()

    ax_pie = plt.axes([.1, .50, .35, .35])
    china_total_recovered = recovered_by_region[MAINLAND_CHINA][current_date_index]
    row_total_recovered = total_recovered[current_date_index] - \
        china_total_recovered

    china_total_infected = confirmed_by_region[MAINLAND_CHINA][current_date_index] - \
        dead_by_region[MAINLAND_CHINA][current_date_index] - \
        recovered_by_region[MAINLAND_CHINA][current_date_index]
    row_total_infected = total_confirmed[current_date_index] - \
        total_recovered[current_date_index] - \
        total_dead[current_date_index] - china_total_infected

    china_total_dead = dead_by_region[MAINLAND_CHINA][current_date_index]
    row_total_dead = total_dead[current_date_index] - \
        china_total_dead

    ax_pie.pie([china_total_recovered, row_total_recovered,
                china_total_infected, row_total_infected,
                china_total_dead, row_total_dead],
               labels=["Recovered", "", "Infected", "", "Dead", ""],
               colors=piechart_colors, startangle=90, radius=500)
    ax_pie.axis("equal")
    # these objects are used to create a consistent legend
    legendel_westerpacificregion = Patch(facecolor=barchart_colors[1])
    legendel_europeanregion = Patch(facecolor=barchart_colors[2])
    legendel_southeastasiaregion = Patch(
        facecolor=barchart_colors[3])
    legendel_easternmediterraneanregion = Patch(
        facecolor=barchart_colors[4])
    legendel_regionoftheamericans = Patch(
        facecolor=barchart_colors[5])
    legendel_africanregion = Patch(facecolor=barchart_colors[6])
    legendel_other = Patch(facecolor=barchart_colors[7])

    # add the legend and object descriptions
    legend = ax_abschina.legend([legendel_westerpacificregion,
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
                                loc='upper right')
    legend.get_frame().set_edgecolor("black")
    legend.set_zorder(20)

    # save the plot in the current folder
    plt.savefig(str(l) + ".png")
    plt.close()

# batch the images to a video
initial_frame_repeatcount = 2  # number of times the initial frame is to be repeated
final_frame_repeatcount = 7  # number of times the final frame is to be repeated

video_name = 'video.mp4'  # name of the exported video

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
for current_date_index in range(plot_start + 1, entries+3):
    video.write(cv2.imread("./" + str(current_date_index) + ".png"))

# write final frame repeatedly
for current_date_index in range(0, final_frame_repeatcount):
    video.write(cv2.imread("./" + str(entries+3) + ".png"))

# save video
cv2.destroyAllWindows()
video.release()
