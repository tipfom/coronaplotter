import os
from datetime import datetime, timedelta

import time
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
datafile_confirmed = jhu_submodule_path + "time_series_19-covid-Confirmed.csv"
datafile_deaths = jhu_submodule_path + "time_series_19-covid-Deaths.csv"
datafile_recovered = jhu_submodule_path + "time_series_19-covid-Recovered.csv"

recovered_by_region, recovered_total = get_data_from_file(datafile_recovered)
confirmed_by_region, confirmed_total = get_data_from_file(datafile_confirmed)
dead_by_region, dead_total = get_data_from_file(datafile_deaths)
entries = len(recovered_total)

recovered_china = recovered_by_region[MAINLAND_CHINA]
confirmed_china = confirmed_by_region[MAINLAND_CHINA]

recovered_row = confirmed_total - recovered_china
confirmed_row = confirmed_total - confirmed_china

new_cases_by_region = []
for confirmed_cases in confirmed_by_region:
    new_cases_by_region.append(
        np.append(
            [confirmed_cases[0]],
            np.array(confirmed_cases[1:]) - np.array(confirmed_cases[:-1]),
        )
    )


def row_fit_function(x, a, b):
    return a * np.exp(b * x)


def row_fit_jacobian(x, a, b):
    return np.transpose([np.exp(b * x), a * x * np.exp(b * x)])


def china_fit_function(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


def china_fit_jacobian(x, a, b, c):
    return np.transpose(
        [
            1 / (1 + np.exp(-b * (x - c))),
            -a / ((1 + np.exp(-b * (x - c))) ** 2) * (c - x) * np.exp(-b * (x - c)),
            -a / ((1 + np.exp(-b * (x - c))) ** 2) * b * np.exp(-b * (x - c)),
        ]
    )


def generate_fits(x, y, start, p0, function, jacobian):
    result = []
    for i in range(start, len(x) + 1):
        result.append(
            scipy.optimize.curve_fit(function, x[:i], y[:i], p0, jac=jacobian)
        )
    return result


plot_start = 15

fit_data_x = np.arange(0, entries)
china_fits = generate_fits(
    fit_data_x,
    confirmed_china,
    plot_start,
    [80000, 0.4, 20],
    china_fit_function,
    china_fit_jacobian,
)
row_fits = generate_fits(
    fit_data_x, confirmed_row, plot_start, [5, 0.2], row_fit_function, row_fit_jacobian
)

# increase pyplot font size
font = {"family": "sans-serif", "weight": "normal", "size": 16}
matplotlib.rc("font", **font)
plt.rc("axes", labelsize=20)


# x-axis range
xmin = 0
xmax = 53
# steps between major ticks on x-axi
xstep = 7


def get_yaxis_lim_ticks_labels(max_value):
    max_value = max_value * 1.05
    if max_value > 50000:
        return ([0, 80000], [0, 2e4, 4e4, 6e4, 8e4], ["0", "20k", "40k", "60k", "80k"])
    if max_value > 20000:
        return (
            [0, 50000],
            [0, 1e4, 2e4, 3e4, 4e4, 5e4],
            ["0", "10k", "20k", "30k", "40k", "50k"],
        )
    if max_value > 10000:
        return (
            [0, 20000],
            [0, 0.5e4, 1e4, 1.5e4, 2e4],
            ["0", "5k", "10k", "15k", "20k"],
        )
    if max_value > 5000:
        return (
            [0, 10000],
            [0, 2e3, 4e3, 6e3, 8e3, 10e3],
            ["0", "2k", "4k", "6k", "8k", "10k"],
        )
    if max_value > 2000:
        return (
            [0, 5000],
            [0, 1e3, 2e3, 3e3, 4e3, 5e3],
            ["0", "1k", "2k", "3k", "4k", "5k"],
        )
    if max_value > 1000:
        return (
            [0, 2000],
            [0, 500, 1000, 1500, 2000],
            ["0", "0.5k", "1k", "1.5k", "2k"],
        )
    if max_value > 500:
        return (
            [0, 1000],
            [0, 200, 400, 600, 800, 1000],
            ["0", "200", "400", "600", "800", "1000"],
        )
    if max_value > 200:
        return (
            [0, 500],
            [0, 100, 200, 300, 400, 500],
            ["0", "100", "200", "300", "400", "500"],
        )
    if max_value > 100:
        return (
            [0, 200],
            [0, 40, 80, 120, 160, 200],
            ["0", "40", "80", "120", "160", "200"],
        )
    if max_value > 50:
        return (
            [0, 100],
            [0, 20, 40, 60, 80, 100],
            ["0", "20", "40", "60", "80", "100"],
        )
    if max_value > 10:
        return ([0, 50], [0, 10, 20, 30, 40, 50], ["0", "10", "20", "30", "40", "50"])
    return ([0, 10], [0, 2, 4, 6, 8, 10], ["0", "2", "4", "6", "8", "10"])


def create_animation_frames(region):
    # create animation frames
    for l in range(plot_start, entries + 4):
        current_date_index = l  # index of the last data point to be used
        desired_fit_count = 3  # number of previous fits to include

        if l > entries:  # used to fade out the last three plots
            current_date_index = entries
            desired_fit_count = 3 - (l - current_date_index) + 1

        data_x = np.arange(0, current_date_index)
        china_data_y = confirmed_by_region[MAINLAND_CHINA][0:current_date_index]

        row_data_y = (confirmed_total - confirmed_by_region[MAINLAND_CHINA])[
            0:current_date_index
        ]

        # creation of pyplot plot
        fig, (ax_regional_development, ax_shared) = plt.subplots(2)
        ax_shared.yaxis.tick_right()
        ax_shared.yaxis.set_label_position("right")
        ax_shared.spines["left"].set_color("none")
        ax_regional_development.yaxis.tick_right()
        ax_regional_development.yaxis.set_label_position("right")
        ax_regional_development.spines["left"].set_color("none")

        # setting the dimensions (basically resolution with 12by9 aspect ratio)
        fig.set_figheight(12)
        fig.set_figwidth(12)

        majxticks = ([], [])
        # generation of x-axis tick-names
        startdate = datetime(2020, 1, 22)
        for j in range(xmin, xmax + 1, xstep):
            majxticks[0].append(j)
            majxticks[1].append((startdate + timedelta(j)).strftime("%d. %b"))

        # setting the x-axis ticks
        ax_shared.set_xlim([xmin, xmax])
        ax_shared.set_xticks(majxticks[0])
        ax_shared.set_xticklabels(majxticks[1])
        ax_shared.xaxis.set_minor_locator(MultipleLocator(1))

        # ax_regional_development.set_ylim([0, 1])
        ax_regional_development.set_xlim([xmin, xmax])
        ax_regional_development.set_xticks(majxticks[0])
        ax_regional_development.set_xticklabels(majxticks[1])
        ax_regional_development.xaxis.set_minor_locator(MultipleLocator(1))

        # setting the y-axis ticks
        ax_shared.set_yticks([0, 2e4, 4e4, 6e4, 8e4, 10e4, 12e4])
        ax_shared.set_yticklabels(["0", "20k", "40k", "60k", "80k", "100k", "120k"])
        ax_shared.yaxis.set_minor_locator(MultipleLocator(10000))

        regional_lim_ticks_labels = get_yaxis_lim_ticks_labels(
            confirmed_by_region[region][current_date_index-1]
        )
        ax_regional_development.set_ylim(regional_lim_ticks_labels[0])
        ax_regional_development.set_yticks(regional_lim_ticks_labels[1])
        ax_regional_development.set_yticklabels(regional_lim_ticks_labels[2])
        ax_regional_development.yaxis.set_minor_locator(
            MultipleLocator(regional_lim_ticks_labels[1][1] / 2)
        )

        # setting the y-axis limit
        ax_shared.set_ylim([0, 100000])

        # label axis
        plt.xlabel("date")
        ax_shared.set_ylabel("total # of confirmed infections")
        ax_regional_development.set_ylabel("total # of confirmed infections")

        # format the border of the diagram
        ax_shared.spines["top"].set_color("white")
        ax_regional_development.spines["top"].set_color("white")

        # plot the original data
        ax_shared.plot(
            data_x, china_data_y, "s", color=china_color, markersize=7, zorder=10
        )
        ax_shared.fill_between(
            data_x,
            np.zeros(current_date_index),
            recovered_by_region[MAINLAND_CHINA][:current_date_index],
            color=china_recovered_color,
            alpha=0.5,
            hatch="//",
        )
        ax_shared.fill_between(
            data_x,
            recovered_by_region[MAINLAND_CHINA][:current_date_index],
            recovered_by_region[MAINLAND_CHINA][:current_date_index]
            + dead_by_region[MAINLAND_CHINA][:current_date_index],
            color=china_dead_color,
            alpha=0.5,
            hatch="//",
        )
        ax_shared.fill_between(
            data_x,
            recovered_by_region[MAINLAND_CHINA][:current_date_index]
            + dead_by_region[MAINLAND_CHINA][:current_date_index],
            china_data_y,
            color=china_color,
            alpha=0.2,
            hatch="//",
        )

        recovered_row = (
            recovered_total[:current_date_index]
            - recovered_by_region[MAINLAND_CHINA][:current_date_index]
        )
        dead_row = (
            dead_total[:current_date_index]
            - dead_by_region[MAINLAND_CHINA][:current_date_index]
        )
        ax_shared.plot(data_x, row_data_y, "o", color=row_color, markersize=7, zorder=10)
        ax_shared.fill_between(
            data_x,
            np.zeros(current_date_index),
            recovered_row,
            color=row_recovered_color,
            alpha=0.5,
            hatch="..",
        )
        ax_shared.fill_between(
            data_x,
            recovered_row,
            recovered_row + dead_row,
            color=row_dead_color,
            alpha=0.5,
            hatch="..",
        )
        ax_shared.fill_between(
            data_x,
            recovered_row + dead_row,
            row_data_y,
            color=row_color,
            alpha=0.2,
            hatch="..",
        )
        # create the exponential plots
        for k in range(0, np.min([desired_fit_count, current_date_index - plot_start])):
            # fit the exponential function
            popt, pcov = row_fits[current_date_index - plot_start - k]
            # get errors from trace of covariance matrix
            perr = np.sqrt(np.diag(pcov))

            # create uncertainty floats for error bars, 2* means 2 sigma
            a = ufloat(popt[0], perr[0])
            b = ufloat(popt[1], perr[1])

            # get the values of the uncertain fit
            fit_x_unc = np.linspace(xmin, xmax, 300)
            fit_y_unc = a * unp.exp(b * fit_x_unc)
            nom_x = unp.nominal_values(fit_x_unc)
            nom_y = unp.nominal_values(fit_y_unc)
            std_y = unp.std_devs(fit_y_unc)

            # plot
            if k == 0:
                print(
                    "["
                    + (startdate + timedelta(current_date_index)).strftime("%d. %b")
                    + "] ROW fit(y=a*exp(b*x))-parameters:"
                )
                print("a = " + str(a))
                print("b = " + str(b))

                ax_shared.plot(nom_x, nom_y, color=row_regression_color, linewidth=3)
                ax_shared.fill_between(
                    nom_x,
                    nom_y - std_y,
                    nom_y + std_y,
                    facecolor=row_regression_color,
                    alpha=0.5,
                )
            elif k == 1:
                ax_shared.fill_between(
                    nom_x,
                    nom_y - std_y,
                    nom_y + std_y,
                    facecolor=row_regression_color,
                    alpha=0.2,
                )
            elif k == 2:
                ax_shared.fill_between(
                    nom_x,
                    nom_y - std_y,
                    nom_y + std_y,
                    facecolor=row_regression_color,
                    alpha=0.05,
                )

        for k in range(0, np.min([desired_fit_count, current_date_index - plot_start])):
            # fit the sigmoidal function
            popt, pcov = china_fits[current_date_index - plot_start - k]
            # get errors from trace of covariance matrix
            perr = np.sqrt(np.diag(pcov))

            # create uncertainty floats for error bars, 2* means 2 sigma
            a = ufloat(popt[0], perr[0])
            b = ufloat(popt[1], perr[1])
            c = ufloat(popt[2], perr[2])

            # get the values of the uncertain fit
            fit_x_unc = np.linspace(xmin, xmax, 300)
            fit_y_unc = a / (1 + unp.exp(-b * (fit_x_unc - c)))
            nom_x = unp.nominal_values(fit_x_unc)
            nom_y = unp.nominal_values(fit_y_unc)
            std_y = unp.std_devs(fit_y_unc)

            # plot
            if k == 0:
                print(
                    "["
                    + (startdate + timedelta(current_date_index)).strftime("%d. %b")
                    + "] China fit(y=a/(1+exp(-b*(x-c))))-parameters:"
                )
                print("a = " + str(a))
                print("b = " + str(b))
                print("c = " + str(c))
                ax_shared.plot(nom_x, nom_y, color=china_regression_color, linewidth=3)
                ax_shared.fill_between(
                    nom_x,
                    nom_y - std_y,
                    nom_y + std_y,
                    facecolor=china_regression_color,
                    alpha=0.6,
                )
            elif k == 1:
                ax_shared.fill_between(
                    nom_x,
                    nom_y - std_y,
                    nom_y + std_y,
                    facecolor=china_regression_color,
                    alpha=0.2,
                )
            elif k == 2:
                ax_shared.fill_between(
                    nom_x,
                    nom_y - std_y,
                    nom_y + std_y,
                    facecolor=china_regression_color,
                    alpha=0.1,
                )

        plt.tight_layout()

        ax_regional_development.set_title("regional development in " + region_names[region])
        ax_regional_development.plot(
            data_x,
            confirmed_by_region[region][:current_date_index],
            color=regional_total_color,
            lw=4,
        )
        ax_regional_development.bar(
            data_x,
            new_cases_by_region[region][:current_date_index],
            color=regional_change_color,
        )

        # these objects are used to create a consistent legend
        legendel_regional_total = Line2D([0], [0], color=regional_total_color, lw=4)
        legendel_regional_change = Patch(facecolor=regional_change_color)

        # add the legend and object descriptions
        piechart_legend = ax_regional_development.legend(
            [legendel_regional_total, legendel_regional_change,],
            ["Total cases", "New cases",],
            loc="upper left",
        )
        piechart_legend.get_frame().set_edgecolor("black")
        piechart_legend.set_zorder(20)

        legendel_china_data = Line2D(
            [0],
            [0],
            marker="s",
            color=china_color,
            lw=0,
            markerfacecolor=china_color,
            markersize=10,
        )
        legendel_china_regression = Line2D([0], [0], color=china_regression_color, lw=4)
        legendel_china_recovered = Patch(
            facecolor=china_recovered_color,
            alpha=0.5,
            hatch="//",
            edgecolor=china_recovered_color,
        )
        legendel_china_dead = Patch(
            facecolor=china_dead_color, alpha=0.5, hatch="//", edgecolor=china_dead_color,
        )
        legendel_spacer = Patch(facecolor="none")
        legendel_row_data = Line2D(
            [0],
            [0],
            marker="o",
            color=row_color,
            lw=0,
            markerfacecolor=row_color,
            markersize=10,
        )
        legendel_row_regression = Line2D([0], [0], color=row_regression_color, lw=4)
        legendel_row_recovered = Patch(
            facecolor=row_recovered_color,
            alpha=0.5,
            hatch="..",
            edgecolor=row_recovered_color,
        )
        legendel_row_dead = Patch(
            facecolor=row_dead_color, alpha=0.5, hatch="..", edgecolor=row_dead_color,
        )
        total_legend = ax_shared.legend(
            [
                legendel_china_data,
                legendel_china_regression,
                legendel_china_recovered,
                legendel_china_dead,
                legendel_row_data,
                legendel_row_regression,
                legendel_row_recovered,
                legendel_row_dead,
            ],
            [
                "Infections in China",
                "Adoption to an sigmoidal fit",
                "Recovered cases in China",
                "COVID-19 deaths in China",
                "Infections outside China",
                "Adoption to an exponential fit",
                "Recovered cases outside China",
                "COVID-19 deaths outside China",
            ],
            loc="upper left",
        )
        total_legend.get_frame().set_edgecolor("black")
        total_legend.set_zorder(20)

        # save the plot in the current folder
        plt.tight_layout()
        plt.savefig(f"./images/{region}_{l}.png")

        time.sleep(0.1)
        plt.close()

if __name__ == '__main__':    
    import concurrent.futures

    executor = concurrent.futures.ProcessPoolExecutor(6)
    futures = [executor.submit(create_animation_frames, item) for item in range(1, REGION_COUNT)]
    concurrent.futures.wait(futures)

    # batch the images to a video
    initial_frame_repeatcount = 2  # number of times the initial frame is to be repeated
    final_frame_repeatcount = 7  # number of times the final frame is to be repeated

    video_name = "video.mp4"  # name of the exported video

    # get video size data
    frame = cv2.imread(f"./images/{1}_{plot_start}.png")
    height, width, layers = frame.shape

    # create video writer
    fps = 4
    video = cv2.VideoWriter(video_name, 0, fps, (width, height))

    def write_frames(region):
        # write initial frame
        for current_date_index in range(0, initial_frame_repeatcount):
            video.write(cv2.imread(f"./images{region}_{plot_start}.png"))

        # animation frames
        for current_date_index in range(plot_start + 1, entries + 3):
            video.write(cv2.imread(f"./images/{region}_{current_date_index}.png"))

        # write final frame repeatedly
        for current_date_index in range(0, final_frame_repeatcount):
            video.write(cv2.imread(f"./images/{entries+3}_{plot_start}.png"))

    for i in range(1, REGION_COUNT):
        write_frames(i)

    # save video
    cv2.destroyAllWindows()
    video.release()
