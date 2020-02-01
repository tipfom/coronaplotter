import os
import cv2
import uncertainties.unumpy as unp
from uncertainties import ufloat

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy.optimize
import math
from datetime import datetime  
from datetime import timedelta  

font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)
plt.rc('axes', labelsize=20)


def exponential_fit_function(x, a, b):
    return a*np.exp(b*x)

def sigmoidal_fit_function(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))


total_data_y = [45, 62, 121, 198, 291, 440, 571,
                830, 1287, 1975, 2744, 4515, 5974, 7711, 9692, 11791]
regression_start = 5
sigmoidal_start = 13

xmin = 0
xmax = 18
xstep = 3

for l in range(regression_start, len(total_data_y)+4):
    i = l
    n = 3
    if l > len(total_data_y):
        i = len(total_data_y)
        n = 3 - (l - i)

    data_x = np.arange(0, i)
    data_y = total_data_y[0:i]

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10*12/9)
    majxticks = ([], [])

    startdate  = datetime(2020,1,16)
    for j in range(xmin, xmax+1, xstep):
        majxticks[0].append(j)
        majxticks[1].append((startdate + timedelta(j)).strftime("%d. %b"))

    ax.set_xlim([xmin, xmax])
    plt.xticks(majxticks[0], majxticks[1])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", length=8, width=1.5)
    ax.tick_params(axis="both", which="minor", length=5, width=1)

    plt.yticks([0, 5000, 10000, 15000, 20000], [
               "0", "5k", "10k", "15k", "20k"])
    ax.yaxis.set_minor_locator(MultipleLocator(1000))

    ax.set_ylim([0, 20000])
    plt.plot(data_x, data_y, "s", color="black",
             label="raw data collected from source")

    for k in range(np.max([i-n, regression_start]), i+1):
        popt, pcov = scipy.optimize.curve_fit(
            exponential_fit_function,  data_x[0:k],  data_y[0:k])
        perr = np.sqrt(np.diag(pcov))

        a = ufloat(popt[0], perr[0])
        b = ufloat(popt[1], perr[1])

        fit_x_unc = np.linspace(xmin, xmax, 300)
        fit_y_unc = a * unp.exp(b*fit_x_unc)
        nom_x = unp.nominal_values(fit_x_unc)
        nom_y = unp.nominal_values(fit_y_unc)
        std_y = unp.std_devs(fit_y_unc)

        if k == i:
            print("[" + str(l) +"] Exponential fit-parameters:")
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
        popt, pcov = scipy.optimize.curve_fit(
            sigmoidal_fit_function,  data_x[0:k],  data_y[0:k], p0=[13000, 0.54, 13])
        perr = np.sqrt(np.diag(pcov))

        a = ufloat(popt[0], perr[0])
        b = ufloat(popt[1], perr[1])
        c = ufloat(popt[2], perr[2])

        fit_x_unc = np.linspace(xmin, xmax, 300)
        fit_y_unc = a/(1+unp.exp(-b*(fit_x_unc-c)))
        nom_x = unp.nominal_values(fit_x_unc)
        nom_y = unp.nominal_values(fit_y_unc)
        std_y = unp.std_devs(fit_y_unc)

        if k == i:
            print("[" + str(l) +"] Sigmoid fit-parameters:")
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

    plt.xlabel("date")
    plt.ylabel("total # of confirmed infections in Mainland China")

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('black')

    legendel_originaldata = Line2D([0], [0], marker='s', color='black',
                                   lw=0, label='Scatter', markerfacecolor='black', markersize=10)
    legendel_exponentialfit = Line2D([0], [0], color='blue', lw=4, label='Line')
    legendel_sigmoidalfit = Line2D([0], [0], color='orange', lw=4, label='Line')
    legendel_exponential_areaofuncertainty = Patch(
        facecolor='blue', alpha=0.5, label="e")
    legendel_sigmoidal_areaofuncertainty = Patch(
        facecolor='orange', alpha=0.5, label="d")

    ax.legend([legendel_originaldata,
               legendel_exponentialfit,
               legendel_exponential_areaofuncertainty,
               legendel_sigmoidalfit,
               legendel_sigmoidal_areaofuncertainty],
              ["raw data collected from source",
               "data fitted to an exponential function",
               "area of uncertainty for the exponential fit",
               "data fitted to an sigmoidal function",
               "area of uncertainty for the sigmoidal fit"]).get_frame().set_edgecolor("black")
    plt.title("see comments for further explanations")

    plt.savefig(str(l) + ".png")


initial_frame_repeatcount = 2
final_frame_repeatcount = 7

video_name = 'video.mp4'

frame = cv2.imread("./" + str(regression_start) + ".png")
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 3, (width, height))

for i in range(0, initial_frame_repeatcount):
    video.write(cv2.imread("./" + str(regression_start) + ".png"))

for i in range(regression_start + 1, len(total_data_y)+3):
    video.write(cv2.imread("./" + str(i) + ".png"))

for i in range(0, final_frame_repeatcount):
    video.write(cv2.imread("./" + str(len(total_data_y)+3) + ".png"))

cv2.destroyAllWindows()
video.release()
