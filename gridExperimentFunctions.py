"""
Experiment-level processing functions
v0.0.2 for Grid2 calibration result analysis by ydx and ghz
"""

import gridBasicFunctions as grid
import datetime
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit, least_squares
import lmfit
import xlrd,csv
from scipy import interpolate
from fnmatch import fnmatch
from scipy.odr import ODR, Model, Data, RealData
import crc16,struct

#******************************************************************************************************************************************************
#*************************************************Experiment-level processing functions********************************************************
#******************************************************************************************************************************************************

#******************************************************************************************************************************************************
#*******************************************Temperature and bias functions for both experiments********************************************
#******************************************************************************************************************************************************

def tempBiasVariation(temp, bias, vmon, iMon, uscount, isTemp, singlech = False, groupScan = False, channel = -1, filenames = []):

    """
    Function for plotting the variation curves of temperature and bias with time,\
 as well as the variation of monitored voltage and current with temperature/bias
    :param temp: list of temperatures of all individual scans, orgaized in the form \
 of [channel][scan#][data#]
    :param bias: list of bisa of all individual scans, organized in the same way as \
 temperature
    :param vmon: monitored voltage, organized in the same way as temperature
    :param iMon: monitored current, organized in the same way as temperature
    :param uscount: uscount data, organized in the form of [scan#][data#]
    :param isTemp: True if the data given is from temperature scan, False if the \
 data is from bias scan
    :param singlech: boolean indicating whether the fit is for single channel
    :param groupScan: boolean indicating the way of grouping, True for grouping by\
 channel, False for grouping by scans
    :param channel: the channel number in range [0-3] for single channel plotting
    :param filenames: the file names for temperature scan
    :return: nothing
    """

    print('tempBiasVariation: plotting temperature and bias curves')
    if singlech:
        if not grid.isChannel(channel):
            raise Exception('tempBiasVariation: incorrect channel number form or channel number out of bound[0-3]')

    if isTemp:
        if not len(filenames) == len(uscount):
            raise Exception('tempBiasVariation: the amount of file names does not correspond with the data')
        else:
            timeBegin = []
            for file in filenames:
                timeBegin.append(float(file.split('\\')[-1][6:8]) + (float(file.split('\\')[-1][8:10]) + float(file.split('\\')[-1][10:12]) / 60.0) / 60.0)
                if timeBegin[-1] > 12:
                    timeBegin[-1] -= 24

    nScan = len(uscount)
    colorplot = ['y', 'r', 'g', 'k']
    
    #Temperature variation with time
    fig0 = plt.figure(figsize=(12, 8))
    gs0 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
    ax0 = fig0.add_subplot(gs0[0])
    if singlech:
        ich = channel
        lastTime = 0.0
        for isc in range(nScan):
            if isTemp:
                if groupScan:
                    if ich == 0:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, temp[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: ' + str('%.2f' % np.average(temp[ich][isc])) + '$^{\circ}$C'))
                    else:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, temp[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, temp[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, temp[ich][isc], c=colorplot[ich], s=1)
            else:
                if groupScan:
                    if ich == 0:
                        plt.scatter(lastTime + uscount[isc], temp[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(lastTime + uscount[isc], temp[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(lastTime + uscount[isc], temp[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(lastTime + uscount[isc], temp[ich][isc], c=colorplot[ich], s=1)
            lastTime += uscount[isc][-1]
    else:
        lastTime = 0.0
        for isc in range(nScan):
            for ich in range(4):
                if isTemp:
                    if groupScan:
                        if ich == 0:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, temp[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: '\
                               + str('%.2f' % np.average(temp[0][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[1][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[2][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[3][isc])) + '$^{\circ}$C'))
                        else:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, temp[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, temp[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, temp[ich][isc], c=colorplot[ich], s=1)
                else:
                    if groupScan:
                        if ich == 0:
                            plt.scatter(lastTime + uscount[isc], temp[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(lastTime + uscount[isc], temp[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(lastTime + uscount[isc], temp[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(lastTime + uscount[isc], temp[ich][isc], c=colorplot[ich], s=1)
            lastTime += uscount[isc][-1]
    if isTemp:
        ax0.set_xlabel('time/h')
    else:
        ax0.set_xlabel('time/s')
    ax0.set_ylabel('SiPM temperature/$^{\circ}$C')
    ax0.set_title('Variation of SiPM temperature with time')
    ax0.legend(loc=0)
    ax0.grid()
    plt.show()

    #Bias variation with time
    fig1 = plt.figure(figsize=(12, 8))
    gs1 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
    ax1 = fig1.add_subplot(gs1[0])
    if singlech:
        ich = channel
        lastTime = 0.0
        for isc in range(nScan):
            if isTemp:
                if groupScan:
                    if ich == 0:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, bias[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: ' + str('%.2f' % np.average(temp[ich][isc])) + '$^{\circ}$C'))
                    else:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, bias[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, bias[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, bias[ich][isc], c=colorplot[ich], s=1)
            else:
                if groupScan:
                    if ich == 0:
                        plt.scatter(lastTime + uscount[isc], bias[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(lastTime + uscount[isc], bias[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(lastTime + uscount[isc], bias[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(lastTime + uscount[isc], bias[ich][isc], c=colorplot[ich], s=1)
            lastTime += uscount[isc][-1]
    else:
        lastTime = 0.0
        for isc in range(nScan):
            for ich in range(4):
                if isTemp:
                    if groupScan:
                        if ich == 0:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, bias[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: '\
                               + str('%.2f' % np.average(temp[0][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[1][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[2][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[3][isc])) + '$^{\circ}$C'))
                        else:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, bias[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, bias[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, bias[ich][isc], c=colorplot[ich], s=1)
                else:
                    if groupScan:
                        if ich == 0:
                            plt.scatter(lastTime + uscount[isc], bias[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(lastTime + uscount[isc], bias[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(lastTime + uscount[isc], bias[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(lastTime + uscount[isc], bias[ich][isc], c=colorplot[ich], s=1)
            lastTime += uscount[isc][-1]
    if isTemp:
        ax1.set_xlabel('time/h')
    else:
        ax1.set_xlabel('time/s')
    ax1.set_ylabel('SiPM bias/V')
    ax1.set_title('Variation of SiPM bias with time')
    ax1.legend(loc=0)
    ax1.grid()
    plt.show()

    #Monotored voltage variation with time
    fig2 = plt.figure(figsize=(12, 8))
    gs2 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
    ax2 = fig2.add_subplot(gs2[0])
    if singlech:
        ich = channel
        lastTime = 0.0
        for isc in range(nScan):
            if isTemp:
                if groupScan:
                    if ich == 0:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: ' + str('%.2f' % np.average(temp[ich][isc])) + '$^{\circ}$C'))
                    else:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, vmon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, vmon[ich][isc], c=colorplot[ich], s=1)
            else:
                if groupScan:
                    if ich == 0:
                        plt.scatter(lastTime + uscount[isc], vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(lastTime + uscount[isc], vmon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(lastTime + uscount[isc], vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(lastTime + uscount[isc], vmon[ich][isc], c=colorplot[ich], s=1)
            lastTime += uscount[isc][-1]
    else:
        lastTime = 0.0
        for isc in range(nScan):
            for ich in range(4):
                if isTemp:
                    if groupScan:
                        if ich == 0:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: '\
                               + str('%.2f' % np.average(temp[0][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[1][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[2][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[3][isc])) + '$^{\circ}$C'))
                        else:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, vmon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, vmon[ich][isc], c=colorplot[ich], s=1)
                else:
                    if groupScan:
                        if ich == 0:
                            plt.scatter(lastTime + uscount[isc], vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(lastTime + uscount[isc], vmon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(lastTime + uscount[isc], vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(lastTime + uscount[isc], vmon[ich][isc], c=colorplot[ich], s=1)
            lastTime += uscount[isc][-1]
    if isTemp:
        ax2.set_xlabel('time/h')
    else:
        ax2.set_xlabel('time/s')
    ax2.set_ylabel('monitored voltage/V')
    ax2.set_title('Variation of monitored voltage with time')
    ax2.legend(loc=0)
    ax2.grid()
    plt.show()

    #Monotored current variation with time
    fig3 = plt.figure(figsize=(12, 8))
    gs3 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
    ax3 = fig3.add_subplot(gs3[0])
    if singlech:
        ich = channel
        lastTime = 0.0
        for isc in range(nScan):
            if isTemp:
                if groupScan:
                    if ich == 0:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: ' + str('%.2f' % np.average(temp[ich][isc])) + '$^{\circ}$C'))
                    else:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, iMon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, iMon[ich][isc], c=colorplot[ich], s=1)
            else:
                if groupScan:
                    if ich == 0:
                        plt.scatter(lastTime + uscount[isc], iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(lastTime + uscount[isc], iMon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(lastTime + uscount[isc], iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(lastTime + uscount[isc], iMon[ich][isc], c=colorplot[ich], s=1)
            lastTime += uscount[isc][-1]
    else:
        lastTime = 0.0
        for isc in range(nScan):
            for ich in range(4):
                if isTemp:
                    if groupScan:
                        if ich == 0:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: '\
                               + str('%.2f' % np.average(temp[0][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[1][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[2][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[3][isc])) + '$^{\circ}$C'))
                        else:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, iMon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(timeBegin[isc] + uscount[isc] / 3600.0, iMon[ich][isc], c=colorplot[ich], s=1)
                else:
                    if groupScan:
                        if ich == 0:
                            plt.scatter(lastTime + uscount[isc], iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(lastTime + uscount[isc], iMon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(lastTime + uscount[isc], iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(lastTime + uscount[isc], iMon[ich][isc], c=colorplot[ich], s=1)
            lastTime += uscount[isc][-1]
    if isTemp:
        ax3.set_xlabel('time/h')
    else:
        ax3.set_xlabel('time/s')
    ax3.set_ylabel('leak current/mA')
    ax3.set_title('Variation of monitored current with time')
    ax3.legend(loc=0)
    ax3.grid()
    plt.show()

    if isTemp:
        #Monitored voltage variation with temperature
        fig4 = plt.figure(figsize=(12, 8))
        gs4 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
        ax4 = fig4.add_subplot(gs4[0])
        if singlech:
            ich = channel
            for isc in range(nScan):
                if groupScan:
                    if ich == 0:
                        plt.scatter(temp[ich][isc], vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: ' + str('%.2f' % np.average(temp[ich][isc])) + '$^{\circ}$C'))
                    else:
                        plt.scatter(temp[ich][isc], vmon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(temp[ich][isc], vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(temp[ich][isc], vmon[ich][isc], c=colorplot[ich], s=1)
        else:
            for isc in range(nScan):
                for ich in range(4):
                    if groupScan:
                        if ich == 0:
                            plt.scatter(temp[ich][isc], vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: '\
                               + str('%.2f' % np.average(temp[0][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[1][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[2][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[3][isc])) + '$^{\circ}$C'))
                        else:
                            plt.scatter(temp[ich][isc], vmon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(temp[ich][isc], vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(temp[ich][isc], vmon[ich][isc], c=colorplot[ich], s=1)
        ax4.set_xlabel('SiPM temperature/$^{\circ}$C')
        ax4.set_title('Variation of monitored voltage with SiPM temperature')
        ax4.set_ylabel('monitored voltage/V')
        ax4.legend(loc=0)
        ax4.grid()
        plt.show()

        #Monitored current variation with temperature
        fig5 = plt.figure(figsize=(12, 8))
        gs5 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
        ax5 = fig5.add_subplot(gs5[0])
        if singlech:
            ich = channel
            for isc in range(nScan):
                if groupScan:
                    if ich == 0:
                        plt.scatter(temp[ich][isc], iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: ' + str('%.2f' % np.average(temp[ich][isc])) + '$^{\circ}$C'))
                    else:
                        plt.scatter(temp[ich][isc], iMon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(temp[ich][isc], iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(temp[ich][isc], iMon[ich][isc], c=colorplot[ich], s=1)
        else:
            for isc in range(nScan):
                for ich in range(4):
                    if groupScan:
                        if ich == 0:
                            plt.scatter(temp[ich][isc], iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Temperature: '\
                               + str('%.2f' % np.average(temp[0][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[1][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[2][isc])) + '$^{\circ}$C, ' + str('%.2f' % np.average(temp[3][isc])) + '$^{\circ}$C'))
                        else:
                            plt.scatter(temp[ich][isc], iMon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(temp[ich][isc], iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(temp[ich][isc], iMon[ich][isc], c=colorplot[ich], s=1)
        ax5.set_xlabel('SiPM temperature/$^{\circ}$C')
        ax5.set_title('Variation of monitored current with SiPM temperature')
        ax5.set_ylabel('leak current/mA')
        ax5.legend(loc=0)
        ax5.grid()
        plt.show()

    else:
        #Monitored voltage variation with bias
        fig6 = plt.figure(figsize=(12, 8))
        gs6 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
        ax6 = fig6.add_subplot(gs6[0])
        if singlech:
            ich = channel
            for isc in range(nScan):
                if groupScan:
                    if ich == 0:
                        plt.scatter(bias[ich][isc], vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(bias[ich][isc], vmon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(bias[ich][isc], vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(bias[ich][isc], vmon[ich][isc], c=colorplot[ich], s=1)
        else:
            for isc in range(nScan):
                for ich in range(4):
                    if groupScan:
                        if ich == 0:
                            plt.scatter(bias[ich][isc], vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(bias[ich][isc], vmon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(bias[ich][isc], vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(bias[ich][isc], vmon[ich][isc], c=colorplot[ich], s=1)
        ax6.set_xlabel('SiPM bias/V')
        ax6.set_title('Variation of monitored voltage with SiPM bias')
        ax6.set_ylabel('monitored voltage/V')
        ax6.legend(loc=0)
        ax6.grid()
        plt.show()

        #Monitored current variation with bias
        fig7 = plt.figure(figsize=(12, 8))
        gs7 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
        ax7 = fig7.add_subplot(gs7[0])
        if singlech:
            ich = channel
            for isc in range(nScan):
                if groupScan:
                    if ich == 0:
                        plt.scatter(bias[ich][isc], iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(bias[ich][isc], iMon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(bias[ich][isc], iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(bias[ich][isc], iMon[ich][isc], c=colorplot[ich], s=1)
        else:
            for isc in range(nScan):
                for ich in range(4):
                    if groupScan:
                        if ich == 0:
                            plt.scatter(bias[ich][isc], iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(bias[ich][isc], iMon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(bias[ich][isc], iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(bias[ich][isc], iMon[ich][isc], c=colorplot[ich], s=1)
        ax7.set_xlabel('SiPM bias/V')
        ax7.set_title('Variation of monitored current with SiPM bias')
        ax7.set_ylabel('leak current/mA')
        ax7.legend(loc=0)
        ax7.grid()
        plt.show()

    #TBD: add histogram for vmon

    print('tempBiasVariation: all figures plotted')
    return

def tempBiasFit(fitResults, temp, bias, isTemp, fileOutput = False, singlech = False, channel = -1, odr = False, corr = False, cont = False):

    """
    General function for fitting the temperature\bias responce curve
    :param fitResults: fit results, given in the form of list[list[dict]] for multiple channel\
 or list[dict] for single channel
    :param temp: list of temperatures of all individual scans, orgaized in the form \
 of [channel][scan#][data#]
    :param temp: list of bias of all individual scans, orgaized in the form \
 of [channel][scan#][data#]
    :param isTemp: True if the data given is from temperature scan, False if the \
 data is from bias scan
    :param fileOutput: the output style, False for only graph output, True for text file(.txt)\
 output as well as graph output
    :param singlech: boolean indicating whether the fit is for single channel
    :param channel: the channel number in range [0-3], used for single channel plots
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param corr: boolean indicating whether the fit will be correlative(surface) fit, True if the fit is done with 3-d data fit
    :param cont: boolean indicating whether the plot will be in 2-dimensional filled contour, if False the data will be presented with \
3-dimensional scatter plot and surface plot
    :return: fit results of the temperature\bias responce curve, in the form of a list[dict]\
 with the dictionary being
        {
            'a':        quadratic term,
            'b':        linear term,
            'c':        constant,
            'a_err':        error of quadratic term,
            'b_err':        error of linear term,
            'c_err':        error of constant,
        }
    and the fit form being 'channel = a * input ** 2 + b * input + c with input being temperature\
 or bias
    
    The current result of correlated fit does NOT give a good enough result(residual is too high at some opints), \
a better fit method(currently least-square) or perhaps a better model is required
    """
    
    if singlech:
        if not grid.isChannel(channel):
            raise Exception('tempBiasFit: incorrect channel number form or channel number out of bound[0-3]')

    #Correlative fits
    if corr:
        print('tempBiasFit: commencing temperature-bias responce surface fit')
        fig = plt.figure(figsize=(12, 8))
        if cont:
            gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
            ax = fig.add_subplot(gs[0])
        else:
            ax = plt.axes(projection = '3d')
        center = []
        centerErr = []
        tempAvg = []
        tempErr = []
        biasAvg = []
        biasErr = []
        residual = []
        colorplot = ['k', 'g', 'b', 'r']
        ecolorplot = ['y', 'k', 'm', 'c']

        #Single channel fit
        if singlech:
            init = [[87108.4771304815, 125.28875920325848, -6457.756181509147, -0.04591738291431899, 121.98774364166343, -4.733437624242255], \
                [88334.69975262291, 109.83009069210931, -6551.336547846865, -0.03477421425707115, 123.72984286722215, -4.148545801052957], \
                [87594.90364703887, 248.1181875550561, -6561.774721721633, -0.1787863140996599, 125.04309169080125, -9.456425004624986], \
                [108526.8948814171, 37.974851781620224, -7968.646453501684, -0.0033802649567535037, 148.84275015112078, -1.4186302300496079]]

            for ifit in fitResults:
                center.append(ifit['b'])
                centerErr.append(ifit['b_err'])
            for ibias in bias[channel]:
                biasAvg.append(np.average(ibias))
                biasErr.append(np.std(ibias))
            for itemp in temp[channel]:
                tempAvg.append(np.average(itemp))
                tempErr.append(np.std(itemp))
            center = np.array(center)
            tempAvg = np.array(tempAvg)
            biasAvg = np.array(biasAvg)
                        
            #Do 3d quadratic fit
            result = least_squares(grid.residualQuad3D, init[channel], args = (tempAvg, biasAvg, center))
            params = result.x
            tempFit = np.linspace(np.min(tempAvg), np.max(tempAvg), 100)
            biasFit = np.linspace(np.min(biasAvg), np.max(biasAvg), 100)
            tempFit, biasFit = np.meshgrid(tempFit, biasFit)
            centerFit = grid.quad3DFunction(params, tempFit, biasFit)
            residual.append(grid.residualQuad3D(params, tempAvg, biasAvg, center) / center * 100)

            if fileOutput:
                with open('temp_bias_fit_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', 'w') as fout:
                    fout.write('Channel ' + str(channel) + ':\nCh = ' + str(params[0]) + ' + ' + str(params[1]) + ' * temp + ' + str(params[2]) + ' * bias + ' + \
                        str(params[3]) + ' * temp ** 2 + ' + str(params[4]) + ' * bias ** 2 + ' + str(params[5]) + ' * temp * bias\n')

            #Plot part
            if cont:
                ax.scatter(tempAvg, biasAvg, c = colorplot[channel], s = 1, label=('ch' + str(channel)))
                cs = ax.tricontourf(tempAvg, biasAvg, center, alpha = 0.5)
                plt.colorbar(cs)
            else:
                ax.scatter3D(tempAvg, biasAvg, center, c = colorplot[channel], label = ('ch' + str(channel)))
                ax.plot_surface(tempFit, biasFit, centerFit, color = colorplot[ich], alpha = 0.1)

            ax.set_xlabel('SiPM temperature/$^{\circ}$C')
            ax.set_ylabel('SiPM bias/V')
            if cont:
                ax.set_title('Temperature-bias responce data')
            else:
                ax.set_title('Correlated fit of temperature-bias responce data')
                ax.set_zlabel('ADC/channel')
            ax.legend(loc = 0)
            plt.show()

            #Residual plots
            if not cont:
                fig = plt.figure(figsize=(12, 8))
                gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
                ax = fig.add_subplot(gs[0])
                cs = ax.tricontourf(tempAvg, biasAvg, residual)
                plt.colorbar(cs)
                ax.scatter(tempAvg, biasAvg, c = colorplot[channel], s = 1, label=('ch' + str(channel)))
                ax.set_title('Residual of correlated temperature-bias fit\nChannel' + str(channel))
                ax.set_xlabel('SiPM temperature/$^{\circ}$C')
                ax.set_ylabel('SiPM bias/V')
                ax.legend(loc = 0)
                plt.show()

        #Multiple channel fits
        else:
            if fileOutput:
                fout = open('temp_bias_fit_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', 'w')

            init = [[87108.4771304815, 125.28875920325848, -6457.756181509147, -0.04591738291431899, 121.98774364166343, -4.733437624242255], \
                [88334.69975262291, 109.83009069210931, -6551.336547846865, -0.03477421425707115, 123.72984286722215, -4.148545801052957], \
                [87594.90364703887, 248.1181875550561, -6561.774721721633, -0.1787863140996599, 125.04309169080125, -9.456425004624986], \
                [108526.8948814171, 37.974851781620224, -7968.646453501684, -0.0033802649567535037, 148.84275015112078, -1.4186302300496079]]
            for ich in range(4):
                center.append([])
                centerErr.append([])
                biasAvg.append([])
                biasErr.append([])
                tempAvg.append([])
                tempErr.append([])
                for ifit in fitResults[ich]:
                    center[ich].append(ifit['b'])
                    centerErr[ich].append(ifit['b_err'])
                for ibias in bias[ich]:
                    biasAvg[ich].append(np.average(ibias))
                    biasErr[ich].append(np.std(ibias))
                for itemp in temp[ich]:
                    tempAvg[ich].append(np.average(itemp))
                    tempErr[ich].append(np.std(itemp))
                center[ich] = np.array(center[ich])
                tempAvg[ich] = np.array(tempAvg[ich])
                biasAvg[ich] = np.array(biasAvg[ich])

                #Do 3d quadratic fit
                result = least_squares(grid.residualQuad3D, init[ich], args = (tempAvg[ich], biasAvg[ich], center[ich]))
                params = result.x
                tempFit = np.linspace(np.min(tempAvg[ich]), np.max(tempAvg[ich]), 100)
                biasFit = np.linspace(np.min(biasAvg[ich]), np.max(biasAvg[ich]), 100)
                tempFit, biasFit = np.meshgrid(tempFit, biasFit)
                centerFit = grid.quad3DFunction(params, tempFit, biasFit)
                residual.append(grid.residualQuad3D(params, tempAvg[ich], biasAvg[ich], center[ich]) / center[ich] * 100)

                if fileOutput:
                    fout.write('Channel ' + str(ich) + ':\nCh = ' + str(params[0]) + ' + ' + str(params[1]) + ' * temp + ' + str(params[2]) + ' * bias + ' + str(params[3]) + \
                        ' * temp ** 2 + ' + str(params[4]) + ' * bias ** 2 + ' + str(params[5]) + ' * temp * bias\n')

                #Plot part
                if cont:
                    ax.scatter(tempAvg[ich], biasAvg[ich], c = colorplot[ich], s = 1, label=('ch' + str(ich)))
                    cs = ax.tricontourf(tempAvg[ich], biasAvg[ich], center[ich], alpha = 0.5)
                    plt.colorbar(cs)
                else:
                    ax.scatter3D(tempAvg[ich], biasAvg[ich], center[ich], c = colorplot[ich], label = ('ch' + str(ich)))
                    ax.plot_surface(tempFit, biasFit, centerFit, color = colorplot[ich], alpha = 0.1)
                
            ax.set_xlabel('SiPM temperature/$^{\circ}$C')
            ax.set_ylabel('SiPM bias/V')
            if cont:
                ax.set_title('Temperature-bias responce data')
            else:
                ax.set_title('Correlated fit of temperature-bias responce data')
                ax.set_zlabel('ADC/channel')
            ax.legend(loc = 0)
            plt.show()

            #Residual plots
            if not cont:
                for ich in range(4):
                    fig = plt.figure(figsize=(12, 8))
                    gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
                    ax = fig.add_subplot(gs[0])
                    cs = ax.tricontourf(tempAvg[ich], biasAvg[ich], residual[ich])
                    plt.colorbar(cs)
                    ax.scatter(tempAvg[ich], biasAvg[ich], c = colorplot[ich], s = 1, label=('ch' + str(ich)))
                    ax.set_title('Residual of correlated temperature-bias fit\nChannel' + str(ich))
                    ax.set_xlabel('SiPM temperature/$^{\circ}$C')
                    ax.set_ylabel('SiPM bias/V')
                    ax.legend(loc = 0)
                    plt.show()

            if fileOutput:
                fout.close()

    #Independent fits
    else:
        if isTemp:
            input = temp
            print('tempBiasFit: commencing temperature responce curve fit')
        else:
            input = bias
            print('tempBiasFit: commencing bias responce curve fit')

        if isTemp:
            prefix = 'temp'
            lower = 10.
            upper = 20.
        else:
            prefix = 'bias'
            lower = 20.
            upper = 20.
    
        center = []
        centerErr = []
        inputAvg = []
        inputErr = []
        inputFit = []
        residual = []
        residualMax = 0.0
        colorplot = ['b', 'r', 'g', 'k']
        ecolorplot = ['y', 'k', 'm', 'c']
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        #Single channel fit
        if singlech:
            for ifit in fitResults:
                center.append(ifit['b'])
                centerErr.append(ifit['b_err'])
            for iin in input[channel]:
                inputAvg.append(np.average(iin))
                inputErr.append(np.std(iin))
            center = np.array(center)
            inputAvg = np.array(inputAvg)
            result = grid.doFitQuad(inputAvg, center, odr, inputErr, yerror = centerErr)
            a = result['fit_a']
            b = result['fit_b']
            c = result['fit_c']
            aErr = result['fit_a_err']
            bErr = result['fit_b_err']
            cErr = result['fit_c_err']
            inputFit.append({
                                        'a':        a,
                                        'b':        b,
                                        'c':        c,
                                        'a_err':        aErr,
                                        'b_err':        bErr,
                                        'c_err':        cErr,
                })

            if fileOutput:
                try:
                    with open(prefix + '_fit_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', 'w') as fout:
                        fout.write('Channel' + str(channel) + ': \n')
                        fout.write('ch = ' + str(a) + ' * ' + prefix + ' ** 2 + ' + str(b) + ' * ' + prefix + ' + ' + str(c) + '\n')
                        fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + ', c error: ' + str(cErr) + '\n')
                except:
                    raise Exception('tempBiasFit: Error writing output file')

            inputAll = np.arange(int(np.min(inputAvg) * 10.) - lower, int(np.max(inputAvg) * 10.) + upper, 1) / 10.
            centerAll = a * inputAll ** 2 + b * inputAll + c
            centerFit = a * inputAvg ** 2 + b * inputAvg + c
            ax0.errorbar(inputAvg, center, xerr=inputErr, yerr=centerErr, color=colorplot[1], fmt='s', mfc='white', ms=8, ecolor=colorplot[-1], \
               elinewidth=1, capsize=3, barsabove=True, zorder=1, label=('ch' + str(channel)))
            ax0.plot(inputAll, centerAll, color = colorplot[0], zorder=0, label=('ch' + str(channel) + ' fit, a = ' + str('%.3e' % a) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % b)\
                + ' $\pm$ ' + str('%.3e' % bErr) + ', c = ' + str('%.3e' % c) + ' $\pm$ ' + str('%.3e' % cErr)))
            residual = (center - centerFit) / center * 100
            ax1.plot(inputAvg, residual, marker='s', mfc='white', ms=8, color=colorplot[0], label=('ch' + str(channel)))
            if np.max(np.abs(residual)) > residualMax:
                residualMax = np.max(np.abs(residual))
            ax0.set_xlim([np.min(inputAll), np.max(inputAll)])
            ax0.set_ylabel('ADC/channel')
            if isTemp:
                ax0.set_title('Fit of temperature responce\nFit function: $ax^2 + bx + c$')
            else:
                ax0.set_title('Fit of bias responce\nFit function: $ax^2 + bx + c$')
            ax0.legend(loc=0)
            ax0.grid()
            ax1.set_xlim([np.min(inputAll), np.max(inputAll)])
            ax1.set_ylim([-1.2 * residualMax, 1.2 * residualMax])
            if isTemp:
                ax1.set_xlabel('SiPM temperature/$^{\circ}$C')
            else:
                ax1.set_xlabel('SiPM bias/V')
            ax1.set_ylabel('residual/%')
            ax1.legend(loc=0)
            ax1.grid()
            plt.show()

        #Multiple channel fits
        else:
            if fileOutput:
                try:
                    fout = open(prefix + '_fit_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', 'w')
                except:
                    raise Exception('tempBiasFit: Error opening output file')

            inputMax = 0.0
            inputMin = 0.0
            for ich in range(4):
                center.append([])
                centerErr.append([])
                inputAvg.append([])
                inputErr.append([])
                inputFit.append([])
                for ifit in fitResults[ich]:
                    center[ich].append(ifit['b'])
                    centerErr[ich].append(ifit['b_err'])
                for iin in input[ich]:
                    inputAvg[ich].append(np.average(iin))
                    inputErr[ich].append(np.std(iin))
                center[ich] = np.array(center[ich])
                inputAvg[ich] = np.array(inputAvg[ich])
                result = grid.doFitQuad(inputAvg[ich], center[ich], odr, inputErr[ich], yerror = centerErr[ich])
                a = result['fit_a']
                b = result['fit_b']
                c = result['fit_c']
                aErr = result['fit_a_err']
                bErr = result['fit_b_err']
                cErr = result['fit_c_err']
                inputFit[ich].append({
                                            'a':        a,
                                            'b':        b,
                                            'c':        c,
                                            'a_err':        aErr,
                                            'b_err':        bErr,
                                            'c_err':        cErr,
                   })

                if fileOutput:
                    try:
                        fout.write('Channel' + str(ich) + ': \n')
                        fout.write('ch = ' + str(a) + ' * ' + prefix + ' ** 2 + ' + str(b) + ' * ' + prefix + ' + ' + str(c) + '\n')
                        fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + ', c error: ' + str(cErr) + '\n')
                    except:
                        raise Exception('tempBiasFit: Error writing output file')

                inputAll = np.arange(int(np.min(inputAvg[ich]) * 10.) - lower, int(np.max(inputAvg[ich]) * 10.) + upper, 1) / 10.
                if ich == 0:
                    inputMax = np.max(inputAvg[ich])
                    inputMin = np.min(inputAvg[ich])
                else:
                    if inputMax < np.max(inputAvg[ich]):
                        inputMax = np.max(inputAvg[ich])
                    if inputMin > np.min(inputAvg[ich]):
                        inputMin = np.min(inputAvg[ich])
                centerAll = a * inputAll ** 2 + b * inputAll + c
                centerFit = a * inputAvg[ich] ** 2 + b * inputAvg[ich] + c
                ax0.errorbar(inputAvg[ich], center[ich], xerr=inputErr[ich], yerr=centerErr[ich], color=colorplot[ich], fmt='s', mfc='white', ms=8, ecolor=ecolorplot[ich], \
                   elinewidth=1, capsize=3, barsabove=True, zorder=1, label=('ch' + str(ich)))
                ax0.plot(inputAll, centerAll, color = colorplot[ich], zorder=0, label=('ch' + str(ich) + ' fit, a = ' + str('%.3e' % a) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % b)\
                    + ' $\pm$ ' + str('%.3e' % bErr) + ', c = ' + str('%.3e' % c) + ' $\pm$ ' + str('%.3e' % cErr)))
                residual = (center[ich] - centerFit) / center[ich] * 100
                ax1.plot(inputAvg[ich], residual, marker='s', mfc='white', ms=8, color=colorplot[ich], label=('ch' + str(ich)))
                if np.max(np.abs(residual)) > residualMax:
                    residualMax = np.max(np.abs(residual))

            ax0.set_xlim([0.995 * inputMin, 1.005 * inputMax])
            ax0.set_ylabel('ADC/channel')
            if isTemp:
                ax0.set_title('Fit of temperature responce\nFit function: $ax^2 + bx + c$')
            else:
                ax0.set_title('Fit of bias responce\nFit function: $ax^2 + bx + c$')
            ax0.legend(loc=0)
            ax0.grid()
            ax1.set_xlim([0.995 * inputMin, 1.005 * inputMax])
            ax1.set_ylim([-1.2 * residualMax, 1.2 * residualMax])
            if isTemp:
                ax1.set_xlabel('SiPM temperature/$^{\circ}$C')
            else:
                ax1.set_xlabel('SiPM bias/V')
            ax1.set_ylabel('residual/%')
            ax1.legend(loc=0)
            ax1.grid()
            plt.show()

            if fileOutput:
                fout.close()

        return inputFit

#*******************************************************************************************************************************************************
#******************************************************Leak current and overvoltage fitting******************************************************
#*******************************************************************************************************************************************************

def currentFit(input, iMon, isTemp, fileOutput = False, form = 'quad', odr = False):

    """
    General function for fitting the leak current curve
    :param input: list of temperatures\bias of all individual scans, orgaized in the form \
 of [channel][scan#][data#]
    :param iMon: monitored current, organized in the same way as temperature
    :param isTemp: True if the data given is from temperature scan, False if the \
 data is from bias scan
    :param fileOutput: the output style, False for only graph output, True for text file(.txt)\
 output as well as graph output
    :param form: formula used to fit the data, currently supported options are: 'quad':\
 quadratic fitting, 'exp': exponential fitting, 'linexp': linear-exponential fitting for temperature\
 fit only, 'fixexp': exponential fitting with fixed decay constant
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :return: fit results of the leak current curve, in the form of a list[dict]
    """

    forms = ['quad', 'exp', 'linexp', 'fixexp', 'mixexp']
    if not form in forms:
        raise Exception('currentFit: Unsupported fit form')

    if isTemp:
        print('currentFit: Commencing leak current-temperature curve fit')
    else:
        print('currentFit: Commencing leak current-bias curve fit')
    if fileOutput:
        try:
            fout = open('iMon_fit_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', 'w')
        except:
            raise Exception('currentFit: Error opening output file')

    inputAvg = []
    inputErr = []
    inputFit = []
    fitAvg = []
    iMonAvg = []
    iMonErr = []
    residual = []
    colorplot = ['b', 'r', 'g', 'k']
    ecolorplot = ['y', 'k', 'm', 'c']
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    if isTemp:
        if form == 'exp':
            prefix = '1 / (temp + 273.15)'
        elif form == 'linexp':
            prefix = '(temp + 273.15)'
        else:
            prefix = 'temp'
        lower = 10.
        upper = 20.
    else:
        prefix = 'bias'
        lower = 1.
        upper = 2.
    
    inputMax = 0.0
    inputMin = 0.0
    residualMax = 0.0
    
    for ich in range(4):
        inputAvg.append([])
        inputErr.append([])
        inputFit.append([])
        fitAvg.append([])
        iMonAvg.append([])
        iMonErr.append([])
        for isc in range(len(input[ich])):
            inputAvg[ich].append(np.average(input[ich][isc]))
            inputErr[ich].append(np.std(input[ich][isc]))
            if isTemp:
                fitAvg[ich].append(1. / (inputAvg[ich][isc] + 273.15))
            else:
                fitAvg[ich].append(inputAvg[ich][isc])
            iMonAvg[ich].append(np.average(iMon[ich][isc]))
            iMonErr[ich].append(np.std(iMon[ich][isc]))
        fitAvg[ich] = np.array(fitAvg[ich])
        iMonAvg[ich] = np.array(iMonAvg[ich])
        iMonErr[ich] = np.array(iMonErr[ich])
        #Quadratic fitting
        if form == 'quad':
            result = grid.doFitQuad(inputAvg[ich], iMonAvg[ich], odr, inputErr[ich], yerror = iMonErr[ich])
            a = result['fit_a']
            b = result['fit_b']
            c = result['fit_c']
            aErr = result['fit_a_err']
            bErr = result['fit_b_err']
            cErr = result['fit_c_err']
            inputFit[ich].append({
                                        'a':        a,
                                        'b':        b,
                                        'c':        c,
                                        'a_err':        aErr,
                                        'b_err':        bErr,
                                        'c_err':        cErr,
               })
        #Exponential-like form fitting
        elif 'exp' in form:
            result = grid.doFitExp(fitAvg[ich], iMonAvg[ich], (not isTemp) and odr, inputErr[ich], yerror = iMonErr[ich])
            a = result['fit_a']
            b = result['fit_b']
            c = result['fit_c']
            #Temperature fit
            if isTemp:
                #Reverse exponential fitting
                if form == 'exp':
                    pinit = (a, - 1. / b, c)
                    result = grid.doRevExpFit(inputAvg[ich], iMonAvg[ich], pinit, odr, inputErr[ich], yerror = iMonErr[ich])
                    a = result['fit_a']
                    b = result['fit_b']
                    aErr = result['fit_a_err']
                    bErr = result['fit_b_err']
                    inputFit[ich].append({
                                                        'a':            a,
                                                        'b':            b,
                                                        'a_err':        aErr,
                                                        'b_err':        bErr,
                        })
                #Mixed exponential fitting
                elif form == 'mixexp':
                    pinit = (a, - 1. / b, c)
                    result = grid.doMixedExpFit(inputAvg[ich], iMonAvg[ich], pinit, odr, inputErr[ich], yerror = iMonErr[ich])
                    a = result['fit_a']
                    b = result['fit_b']
                    c = result['fit_c']
                    aErr = result['fit_a_err']
                    bErr = result['fit_b_err']
                    cErr = result['fit_c_err']
                    inputFit[ich].append({
                                                        'a':            a,
                                                        'b':            b,
                                                        'c':            c,
                                                        'a_err':        aErr,
                                                        'b_err':        bErr,
                                                        'c_err':        cErr,
                        })
                #Linear-exponential fitting
                elif form == 'linexp':
                    pinit = (1e8, 1e2, 5802, 1e-3)
                    result = grid.doLinExpFit(inputAvg[ich], iMonAvg[ich], pinit, odr, inputErr[ich], yerror = iMonErr[ich])
                    a0 = result['fit_a']
                    b0 = result['fit_b']
                    c0 = result['fit_c']
                    d0 = result['fit_d']
                    aErr = result['fit_a_err']
                    bErr = result['fit_b_err']
                    cErr = result['fit_c_err']
                    dErr = result['fit_d_err']
                    inputFit[ich].append({
                                                        'a':            a0,
                                                        'b':            b0,
                                                        'c':            c0,
                                                        'd':            d0,
                                                        'a_err':        aErr,
                                                        'b_err':        bErr,
                                                        'c_err':        cErr,
                                                        'd_err':        dErr,
                        })
                #Fixed parameter exponential fitting
                else:
                    pinit = (a, c)
                    result = grid.doFixedExpFit(inputAvg[ich], iMonAvg[ich], pinit, odr, inputErr[ich], yerror = iMonErr[ich])
                    a0 = result['fit_a']
                    b0 = result['fit_b']
                    aErr = result['fit_a_err']
                    bErr = result['fit_b_err']
                    inputFit[ich].append({
                                                        'a':            a0,
                                                        'a':            b0,
                                                        'a_err':        aErr,
                                                        'a_err':        bErr,
                        })
            else:
                #Bias fit
                #Reverse linear-exponential fitting
                if form == 'linexp':
                    pinit = (1., 10., 24., 1e-6)
                    result = grid.doRevLinExpFit(inputAvg[ich], iMonAvg[ich], pinit, odr, inputErr[ich], yerror = iMonErr[ich])
                    a0 = result['fit_a']
                    b0 = result['fit_b']
                    c0 = result['fit_c']
                    d0 = result['fit_d']
                    aErr = result['fit_a_err']
                    bErr = result['fit_b_err']
                    cErr = result['fit_c_err']
                    dErr = result['fit_d_err']
                    inputFit[ich].append({
                                                        'a':            a0,
                                                        'b':            b0,
                                                        'c':            c0,
                                                        'd':            d0,
                                                        'a_err':        aErr,
                                                        'b_err':        bErr,
                                                        'c_err':        cErr,
                                                        'd_err':        dErr,
                        })
                #Exponential fitting
                else:
                    aErr = result['fit_a_err']
                    bErr = result['fit_b_err']
                    cErr = result['fit_c_err']
                    inputFit[ich].append({
                                                        'a':            a,
                                                        'b':            b,
                                                        'c':            c,
                                                        'a_err':        aErr,
                                                        'b_err':        bErr,
                                                        'c_err':        cErr,
                        })

        if fileOutput:
            try:
                fout.write('Channel' + str(ich) + ': \n')
                if form == 'exp':
                    fout.write('iMon = ' + str(a) + ' * exp(- ' + str(1. / np.abs(b)) + ' * ' + prefix + ')' + str(c) + '\n')
                    fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + ', c error: ' + str(cErr) + '\n')
                elif form == 'quad':
                    fout.write('ch = ' + str(a) + ' * ' + prefix + ' ** 2 + ' + str(b) + ' * ' + prefix + ' + ' + str(c) + '\n')
                    fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + ', c error: ' + str(cErr) + '\n')
                elif form == 'linexp':
                    if isTemp:
                        fout.write('ch = ' + str(a0) + ' * (' + str(b0) + ' + ' + prefix + ') * exp(- ' + str(c0) + ' * (1.1785 - 9.025e-5 * (t + 273.15) - 3.05e-7 * (t + 273.15) ** 2) / ' + prefix + ')' + str(d0) + '\n')
                        fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + ', c error: ' + str(cErr) + ', d error: ' + str(dErr) + '\n')
                    else:
                        fout.write('ch = ' + str(a0) + ' * (' + prefix + ' - ' + str(c0) + ') * exp(- ' + str(b0) + '  / (' + prefix + ' - ' + str(c0) + '))' + str(d0) + '\n')
                        fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + ', c error: ' + str(cErr) + ', d error: ' + str(dErr) + '\n')
                elif form == 'mixexp':
                    fout.write('ch = ' + str(a) + ' * exp(- ' + str(b) + ' * (1.1785 - 9.025e-5 * (t + 273.15) - 3.05e-7 * (t + 273.15) ** 2) / ' + prefix + ')' + str(c) + '\n')
                    fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + ', c error: ' + str(cErr) + '\n')
                else:
                    fout.write('ch = ' + str(a0) + ' * exp(- 1.1269 / (1.723e-4 * ' + prefix + '))' + str(b0) + '\n')
                    fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + '\n')
            except:
                raise Exception('currentFit: Error writing output file')
            
        inputAvg[ich] = np.array(inputAvg[ich])
        inputAll = np.arange(int(np.min(inputAvg[ich] * 10.) - lower), int(np.max(inputAvg[ich] * 10.) + upper), 1) / 10.
        if ich == 0:
            inputMax = np.max(inputAll)
            inputMin = np.min(inputAll)
        else:
            if inputMax < np.max(inputAll):
                inputMax = np.max(inputAll)
            if inputMin > np.min(inputAll):
                inputMin = np.min(inputAll)
        if 'exp' in form:
            if isTemp:
                if form == 'exp':
                    iMonAll = grid.revExpFunction([a, b], inputAll)
                elif form == 'linexp':
                    iMonAll = grid.linExpFunction([a0, b0, c0, d0], inputAll)
                elif form == 'mixexp':
                    iMonAll = grid.mixedExpFunction([a, b, c], inputAll)
                else:
                    iMonAll = grid.fixedExpFunction([a0, b0], inputAll)
            else:
                if form == 'linexp':
                    iMonAll = grid.revLinExpFunction([a0, b0, c0, d0], inputAll)
                else:
                    iMonAll = grid.expFunction([a, b, c], inputAll)
            if form == 'exp':
                if isTemp:
                    iMonFit = grid.revExpFunction([a, b], inputAvg[ich])
                else:
                    iMonFit = grid.expFunction([a, b, c], inputAvg[ich])
            elif form == 'linexp':
                if isTemp:
                    iMonFit = grid.linExpFunction([a0, b0, c0, d0], inputAvg[ich])
                else:
                    iMonFit = grid.revLinExpFunction([a0, b0, c0, d0], inputAvg[ich])
            elif form == 'mixexp':
                iMonFit = grid.mixedExpFunction([a, b, c], inputAvg[ich])
            else:
                iMonFit = grid.fixedExpFunction([a0, b0], inputAvg[ich])
        else:
            iMonAll = a * inputAll ** 2 + b * inputAll + c
            iMonFit = a * inputAvg[ich] ** 2 + b * inputAvg[ich] + c
        ax0.errorbar(inputAvg[ich], iMonAvg[ich], xerr=inputErr[ich], yerr=iMonErr[ich], color=colorplot[ich], fmt='s', mfc='white', ms=8, ecolor=ecolorplot[ich], \
           elinewidth=1, capsize=3, barsabove=True, zorder=1, label=('ch' + str(ich)))
        if form == 'exp':
            ax0.plot(inputAll, iMonAll, color = colorplot[ich], zorder=0, label=('ch' + str(ich) + ' fit, a = ' + str('%.3e' % a) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % (1. / np.abs(b)))\
                + ' $\pm$ ' + str('%.3e' % bErr) + ', c = ' + str('%.3e' % c) + ' $\pm$ ' + str('%.3e' % cErr)))
        elif form == 'quad':
            ax0.plot(inputAll, iMonAll, color = colorplot[ich], zorder=0, label=('ch' + str(ich) + ' fit, a = ' + str('%.3e' % a) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % b)\
                + ' $\pm$ ' + str('%.3e' % bErr) + ', c = ' + str('%.3e' % c) + ' $\pm$ ' + str('%.3e' % cErr)))
        elif form == 'linexp':
            if isTemp:
                ax0.plot(inputAll, iMonAll, color = colorplot[ich], zorder=0, label=('ch' + str(ich) + ' fit, a = ' + str('%.3e' % a0) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % b0)\
                    + ' $\pm$ ' + str('%.3e' % bErr) + ', c = ' + str('%.3e' % c0) + ' $\pm$ ' + str('%.3e' % cErr) + ', d = ' + str('%.3e' % d0) + ' $\pm$ ' + str('%.3e' % dErr)))
            else:
                ax0.plot(inputAll, iMonAll, color = colorplot[ich], zorder=0, label=('ch' + str(ich) + ' fit, a = ' + str('%.3e' % a0) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % b0)\
                    + ' $\pm$ ' + str('%.3e' % bErr) + ', c = ' + str('%.3e' % c0) + ' $\pm$ ' + str('%.3e' % cErr) + ', d = ' + str('%.3e' % d0) + ' $\pm$ ' + str('%.3e' % dErr)))
        elif form == 'mixexp':
            ax0.plot(inputAll, iMonAll, color = colorplot[ich], zorder=0, label=('ch' + str(ich) + ' fit, a = ' + str('%.3e' % a) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % b)\
                + ' $\pm$ ' + str('%.3e' % bErr) + ', c = ' + str('%.3e' % c) + ' $\pm$ ' + str('%.3e' % cErr)))
        else:
            ax0.plot(inputAll, iMonAll, color = colorplot[ich], zorder=0, label=('ch' + str(ich) + ' fit, a = ' + str('%.3e' % a0) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % b0)\
                + ' $\pm$ ' + str('%.3e' % bErr)))
        residual = (iMonAvg[ich] - iMonFit) / iMonAvg[ich] * 100
        ax1.plot(inputAvg[ich], residual, marker='s', mfc='white', ms=8, color=colorplot[ich], label=('ch' + str(ich)))
        if np.max(np.abs(residual)) > residualMax:
            residualMax = np.max(np.abs(residual))
        
    ax0.set_xlim([inputMin, inputMax])
    ax0.set_ylabel('leak current/mA')
    if form == 'exp':
        if isTemp:
            ax0.set_title('Fit of leak current\nFit function: $a e^{-b/T}$')
        else:
            ax0.set_title('Fit of leak current\nFit function: $a e^{Bx} + c$')
    elif form == 'quad':
        ax0.set_title('Fit of leak current\nFit function: $ax^2 + bx + c$')
    elif form == 'linexp':
        if isTemp:
            ax0.set_title('Fit of leak current\n' + r'Fit function: $a (b + T) e^{-c(1.1785 - 9.025\times10^{-5} T - 3.05\times10^{-7} T^{2})/{T}} + d$')
        else:
            ax0.set_title('Fit of leak current\n' + r'Fit function: $a (x - c) e^{-b/{(x - c)}} + d$')
    elif form == 'mixexp':
        ax0.set_title(r'Fit of leak current\nFit function: $a e^{-b (1.1785 - 9.025\times10^{-5} T - 3.05\times10^{-7} T^{2}) /T} + c$')
    else:
        ax0.set_title('Fit of leak current\nFit function: ' + r'$a e^{-1.1269/{1.723\times10^{-4} T}} + b$')
    ax0.legend(loc=0)
    ax0.grid()
    ax1.set_xlim([inputMin, inputMax])
    ax1.set_ylim([-1.2 * residualMax, 1.2 * residualMax])
    if isTemp:
        ax1.set_xlabel('SiPM temperature/$^{\circ}$C')
    else:
        ax1.set_xlabel('SiPM bias/V')
    ax1.set_ylabel('residual/%')
    ax1.legend(loc=0)
    ax1.grid()
    plt.show()

    if fileOutput:
        fout.close()
    return inputFit

def overvoltageFit(temp, bias, isTemp, fileOutput = False, odr = False):

    """
    Function for plotting and fitting the overvoltage curve
    :param temp: list of temperatures of all individual scans, orgaized in the form \
 of [channel][scan#][data#]
    :param bias: list of bisa of all individual scans, organized in the same way as \
 temperature
    :param isTemp: True if the data given is from temperature scan, False if the \
 data is from bias scan
    :param fileOutput: the output style, False for only graph output, True for text file(.txt)\
 output as well as graph output
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :return: fit results of the leak current curve, in the form of a list[dict]
    """
    
    if isTemp:
        prefix = 'temp'
        lower = 10.
        upper = 20.
        print('overvoltageFit: Commencing overvoltage-temperature curve fit')
    else:
        prefix = 'bias'
        lower = 1.
        upper = 2.
        print('overvoltageFit: Commencing overvoltage-bias curve fit')
    if fileOutput:
        try:
            fout = open('vov_fit_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', 'w')
        except:
            raise Exception('overvoltageFit: Error opening output file')

    inputAll = []
    inputAvg = []
    inputErr = []
    residual = []
    inputMax = 0.0
    inputMin = 0.0
    residualMax = 0.0
    tempAvg = []
    tempErr = []
    biasAvg = []
    biasErr = []
    vBd = 0.0
    vBdErr = 0.0
    vOv = []
    vOvErr = []
    fitResult = []
    colorplot = ['b', 'r', 'g', 'k']
    ecolorplot = ['y', 'k', 'm', 'c']
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    for ich in range(4):
        tempAvg.append([])
        tempErr.append([])
        biasAvg.append([])
        biasErr.append([])
        vOv.append([])
        vOvErr.append([])
        for isc in range(len(temp[ich])):
            tempAvg[ich].append(np.average(temp[ich][isc]))
            tempErr[ich].append(np.std(temp[ich][isc]))
            biasAvg[ich].append(np.average(bias[ich][isc]))
            biasErr[ich].append(np.std(bias[ich][isc]))
            vBd = 24.2 + 21.5e-3 * (tempAvg[ich][isc] - 21.0)
            vBdErr = 21.5e-3 * tempErr[ich][isc]
            vOv[ich].append(biasAvg[ich][isc] - vBd)
            vOvErr[ich].append(np.sqrt(biasErr[ich][isc] ** 2 + vBdErr ** 2))
        vOv[ich] = np.array(vOv[ich])
        if isTemp:
            tempAvg[ich] = np.array(tempAvg[ich])
            result = grid.doFitQuad(tempAvg[ich], vOv[ich], odr, tempErr[ich], yerror = vOvErr[ich])
        else:
            biasAvg[ich] = np.array(biasAvg[ich])
            result = grid.doFitQuad(biasAvg[ich], vOv[ich], odr, biasErr[ich], yerror = vOvErr[ich])
        a = result['fit_a']
        b = result['fit_b']
        c = result['fit_c']
        aErr = result['fit_a_err']
        bErr = result['fit_b_err']
        cErr = result['fit_c_err']
        fitResult.append({
                                    'a':            a,
                                    'b':            b,
                                    'c':            c,
                                    'a_err':        aErr,
                                    'b_err':        bErr,
                                    'c_err':        cErr,
            })

        if fileOutput:
            try:
                fout.write('Channel' + str(ich) + ': \n')
                fout.write('ch = ' + str(a) + ' * ' + prefix + ' ** 2 + ' + str(b) + ' * ' + prefix + ' + ' + str(c) + '\n')
                fout.write('a error: ' + str(aErr) + ', b error: ' + str(bErr) + ', c error: ' + str(cErr) + '\n')
            except:
                raise Exception('overvoltageFit: Error writing output file')

        if isTemp:
            inputAvg = np.array(tempAvg[ich])
            inputErr = tempErr[ich]
        else:
            inputAvg = np.array(biasAvg[ich])
            inputErr = biasErr[ich]
        inputAll = np.arange(int(np.min(inputAvg * 10.) - lower), int(np.max(inputAvg * 10.) + upper), 1) / 10.
        if ich == 0:
            inputMax = np.max(inputAll)
            inputMin = np.min(inputAll)
        else:
            if inputMax < np.max(inputAll):
                inputMax = np.max(inputAll)
            if inputMin > np.min(inputAll):
                inputMin = np.min(inputAll)
        vOvAll = a * inputAll ** 2 + b * inputAll + c
        vOvFit = a * inputAvg ** 2 + b * inputAvg + c
        ax0.errorbar(inputAvg, vOv[ich], xerr=inputErr[ich], yerr=vOvErr[ich], color=colorplot[ich], fmt='s', mfc='white', ms=8, ecolor=ecolorplot[ich], \
           elinewidth=1, capsize=3, barsabove=True, zorder=1, label=('ch' + str(ich)))
        ax0.plot(inputAll, vOvAll, color = colorplot[ich], zorder=0, label=('ch' + str(ich) + ' fit, a = ' + str('%.3e' % a) + ' $\pm$ ' + str('%.3e' % aErr) + ', b = ' + str('%.3e' % b)\
            + ' $\pm$ ' + str('%.3e' % bErr) + ', c = ' + str('%.3e' % c) + ' $\pm$ ' + str('%.3e' % cErr)))
        residual = (vOv[ich] - vOvFit) / vOv[ich] * 100
        ax1.plot(inputAvg, residual, marker='s', mfc='white', ms=8, color=colorplot[ich], label=('ch' + str(ich)))
        if np.max(np.abs(residual)) > residualMax:
            residualMax = np.max(np.abs(residual))

    ax0.set_xlim([inputMin, inputMax])
    ax0.set_ylabel('overvoltage/V')
    ax0.set_title('Fit of overvoltage\nFit function: $ax^2 + bx + c$')
    ax0.legend(loc=0)
    ax0.grid()
    ax1.set_xlim([inputMin, inputMax])
    ax1.set_ylim([-1.2 * residualMax, 1.2 * residualMax])
    if isTemp:
        ax1.set_xlabel('SiPM temperature/$^{\circ}$C')
    else:
        ax1.set_xlabel('SiPM bias/V')
    ax1.set_ylabel('residual/%')
    ax1.legend(loc=0)
    ax1.grid()
    plt.show()

    if fileOutput:
        fout.close()
    return fitResult

#********************************************************************************************************************************************************
#******************************************************************Angular responce******************************************************************
#********************************************************************************************************************************************************

def plotAngularResponce(fitResults, angle, source, fileOutput = False, singlech = False, channel = -1, rateCorr = False, simuFile = ''):
    
    """
    Function for plotting the angular responce
    :param fitResults: fit results, given in the form of list[list[dict]] for multiple channel\
 or list[dict] for single channel
    :param angle:  Corresponding angles of measurement in the form of list[list] for multiple \
 channel or list for single channel
    :param source: the name of the source
    :param fileOutput: the output style, False for only graph output, True for text file(.txt)\
 output as well as graph output
    :param singlech: boolean indicating whether the fit is for single channel
    :param channel: the channel number in range [0-3], used for single channel plots
    :param rateCorr: boolean indicating whether the dead time correction will be couducted
    :param simuFile: simulation result file name, used for comparison between experiment data and simulation data
    :return: nothing
    """

    from scipy.optimize import leastsq
    def getResidual(k, rS, rLnorm, isSimuL):
        """
        Inner auxiliary function to calculate the relative residual to get the optimal normalization coefficient k
        :param k: normalization coefficient
        :param rS: rateS in the function
        :param rLnorm: rateLnorm in the function
        :param isSimuL: simuL in the function
        """
        return (rLnorm - rS * k) if isSimuL else (rS - rLnorm * k)

    simulation = False
    if not simuFile == '':
        simulation = True
        try:
            with open(simuFile, 'r') as fin:
                lines = [line.rstrip() for line in fin]
            ichin = 0
            simuAngle = []
            simuRate = []
            efficiency = []
            for ich in range(4):
                simuAngle.append([])
                simuRate.append([])
                efficiency.append(0.0)
            for line in lines:
                if line.startswith('Channel'):
                    ichin = int(line.split('Channel')[1][0])
                elif line.startswith('Efficiency'):
                    efficiency[ichin] = float(line.split()[1])
                elif line[0].isdigit():
                    lineList = line.split()
                    simuAngle[ichin].append(float(lineList[0]))
                    simuRate[ichin].append(float(lineList[1]))
            efficiencyFull = True
            for ich in range(4):
                if efficiency[ich] == 0.0:
                    if singlech and ich == channel:
                        efficiencyFull = False
                        break
                    else:
                        efficiencyFull = False
                        break
            if not efficiencyFull:
                raise Exception('plotAngularResponce: Efficiency data not given in simulation file')
            simuAngle = np.array(simuAngle)
            simuRate = np.array(simuRate)
        except:
            raise Exception('plotAngularResponce: Error reading simulation file')

    if fileOutput:
        try:
            fout = open('angle_' + source + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', 'w')
        except:
            raise Exception('plotAngularResponce: Error opening output file')

    rate = []
    rateErr = []
    if singlech:
        if not grid.isChannel(channel):
            raise Exception('plotAngularResponce: incorrect channel number form or channel number out of bound[0-3]')
        elif not len(fitResults) == len(angle):
            raise Exception('plotAngularResponce: number of fit results does not match with angle data')
        else:
            for result in fitResults:
                rate.append(result['rate'])
                rateErr.append(result['rate_err'])
    else:
        for ich in range(4):
            if not len(fitResults[ich]) == len(angle):
                raise Exception('plotAngularResponce: number of fit results does not match with angle data')
            else:
                rate.append([])
                rateErr.append([])
                for result in fitResults[ich]:
                    rate[ich].append(result['rate'])
                    rateErr[ich].append(result['rate_err'])
    rate = np.array(rate)
    rateErr = np.array(rateErr)
                    
    colorplot = ['y', 'r', 'g', 'k']
    ecolorplot = ['y', 'k', 'm', 'c']
    fig = plt.figure(figsize=(12, 8))
    if simulation:
        gs = gridspec.GridSpec(2, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95, height_ratios=[4, 1])
    else:
        gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
    ax0 = plt.subplot(gs[0])
    if simulation:
        ax1 = plt.subplot(gs[1])

    #Single channel plot
    if singlech:
        if simulation:
            k = efficiency[channel] / rate[0]
            if len(simuAngle[channel]) >= len(angle):
                angleL = simuAngle[channel]
                angleS = angle
                rateL = simuRate[channel] * efficiency[channel] / simuRate[channel][0]
                rateS = rate
                simuL = True
            else:
                angleL = angle
                angleS = simuAngle[channel]
                rateL = rate
                rateS = simuRate[channel] * efficiency[channel] / simuRate[channel][0]
                simuL = False
            jl = 0
            js = 0
            rateLnorm = []
            while jl < len(angleL) and js < len(angleS):
                if angleS[js] >= angleL[jl]:
                    jl += 1
                    if jl == len(angleL):
                        if angleL[jl - 1] < angleS[js] or js < len(angleS) - 1:
                            raise Exception('plotAngularResponce: angle range of experiment data and simulation data does not match')
                        else:
                            rateLnorm.append(rateL[jl - 1])
                    continue
                else:
                    if jl == 0:
                        raise Exception('plotAngularResponce: angle range of experiment data and simulation data does not match')
                    elif angleS[js] == angleL[jl - 1]:
                        rateLnorm.append(rateL[jl - 1])
                    else:
                        currRate = rateL[jl - 1] + (rateL[jl] - rateL[jl - 1]) / (angleL[jl] - angleL[jl - 1]) * (angleS[js] - angleL[jl - 1])
                        rateLnorm.append(currRate)
                    js += 1
            rateS = np.array(rateS)
            rateLnorm = np.array(rateLnorm)
            angleS = np.array(angleS)
            qNorm = (angleS <= 90) + (angleS >= 270)
            k, kcov = leastsq(getResidual, k, args = (rateS[qNorm], rateLnorm[qNorm], simuL))
            residual = []
            for ir in range(len(rateS)):
                if simuL:
                    residual.append(rateLnorm[ir] - rateS[ir] * k)
                else:
                    residual.append(rateS[ir] - rateLnorm[ir] * k)
            ax0.plot(angle, rate * k, color=colorplot[1])
            ax0.errorbar(angle, rate * k, yerr=(rateErr * k), color=colorplot[1], fmt='s', mfc='white', \
                ms=8, ecolor=colorplot[-1], elinewidth=1, capsize=3, barsabove=True, zorder=1, label=('ch' + str(ich)))
            ax0.plot(simuAngle[channel], simuRate[channel] * efficiency[channel] / simuRate[channel][0], color=colorplot[1], label=('ch' + str(channel) + ' simulation'))
            ax1.plot(angleS, residual, marker='s', mfc='white', ms=8, color=colorplot[ich], label=('ch' + str(ich)))
        else:
            ax0.plot(angle, rate, color=colorplot[1])
            ax0.errorbar(angle, rate, yerr=rateErr, color=colorplot[1], fmt='s', mfc='white', ms=8, ecolor=colorplot[-1], elinewidth=1, capsize=3, \
                barsabove=True, zorder=1, label=('ch' + str(ich)))
        if fileOutput:
            fout.write('Channel' + str(channel) + ': \n')
            for isc in range(len(angle)):
                fout.write(str(angle[isc]) + '\t' + '%.3e' % rate[channel][isc] + '\n')

    #Multiple channel plot
    else:
        for ich in range(4):
            if simulation:
                k = efficiency[ich] / rate[ich][0]
                if len(simuAngle[ich]) >= len(angle):
                    angleL = simuAngle[ich]
                    angleS = angle
                    rateL = simuRate[ich] * efficiency[ich] / simuRate[ich][0]
                    rateS = rate[ich]
                    simuL = True
                else:
                    angleL = angle
                    angleS = simuAngle[ich]
                    rateL = rate[ich]
                    rateS = simuRate[ich] * efficiency[ich] / simuRate[ich][0]
                    simuL = False
                jl = 0
                js = 0
                rateLnorm = []
                while jl < len(angleL) and js < len(angleS):
                    if angleS[js] >= angleL[jl]:
                        jl += 1
                        if jl == len(angleL):
                            if angleL[jl - 1] < angleS[js] or js < len(angleS) - 1:
                                raise Exception('plotAngularResponce: angle range of experiment data and simulation data does not match')
                            else:
                                rateLnorm.append(rateL[jl - 1])
                        continue
                    else:
                        if jl == 0:
                            raise Exception('plotAngularResponce: angle range of experiment data and simulation data does not match')
                        elif angleS[js] == angleL[jl - 1]:
                            rateLnorm.append(rateL[jl - 1])
                        else:
                            currRate = rateL[jl - 1] + (rateL[jl] - rateL[jl - 1]) / (angleL[jl] - angleL[jl - 1]) * (angleS[js] - angleL[jl - 1])
                            rateLnorm.append(currRate)
                        js += 1
                rateS = np.array(rateS)
                rateLnorm = np.array(rateLnorm)
                angleS = np.array(angleS)
                qNorm = (angleS <= 90) + (angleS >= 270)
                k, kcov = leastsq(getResidual, k, args = (rateS[qNorm], rateLnorm[qNorm], simuL))
                residual = []
                for ir in range(len(rateS)):
                    if simuL:
                        residual.append(rateLnorm[ir] - rateS[ir] * k)
                    else:
                        residual.append(rateS[ir] - rateLnorm[ir] * k)
                ax0.plot(angle, rate[ich] * k, color=colorplot[ich])
                ax0.errorbar(angle, rate[ich] * k, yerr=(rateErr[ich] * k), color=colorplot[ich], fmt='s', \
                    mfc='white', ms=8, ecolor=ecolorplot[ich], elinewidth=1, capsize=3, barsabove=True, zorder=1, label=('ch' + str(ich)))
                ax0.plot(simuAngle[ich], simuRate[ich] * efficiency[ich] / simuRate[ich][0], color=colorplot[ich], label=('ch' + str(ich) + ' simulation'))
                ax1.plot(angleS, residual, marker='s', mfc='white', ms=8, color=colorplot[ich], label=('ch' + str(ich)))
            else:
                ax0.plot(angle, rate[ich], color=colorplot[ich])
                ax0.errorbar(angle, rate[ich], yerr=rateErr[ich], color=colorplot[ich], fmt='s', mfc='white', ms=8, ecolor=ecolorplot[ich], elinewidth=1, \
                    capsize=3, barsabove=True, zorder=1, label=('ch' + str(ich)))
            if fileOutput:
                fout.write('Channel' + str(ich) + ': \n')
                for isc in range(len(angle)):
                    fout.write(str(angle[isc]) + '\t' + '%.3e' % rate[ich][isc] + '\n')

    if fileOutput:
        fout.close()
    if simulation:
        ax1.set_xlabel('Angle/$^{\circ}$')
        ax0.set_ylabel('Effective area/$cm^{2}$')
    else:
        ax0.set_xlabel('Angle/$^{\circ}$')
        ax0.set_ylabel('Peak count rate/cps')
    ax0.set_title('Angular responce of ' + source)
    ax0.legend(loc=0)
    ax0.grid()
    if simulation:
        #ax1.set_ylabel('Absolute error/$cm^{2}$')
        ax1.set_ylabel('residual/$\sigma$')
        ax1.legend(loc=0)
        ax1.grid()
    plt.show()
    return

def plotEnergyChannel(ecFilepath, nbins = 8192, ch = 0, doCorr = True, rateCorr = True, isPlotSpec=False, isPlotEC= True, fitEC=False):
    """
    Function for plotting the E-C curve and the energy resolution curve of data from NIM.
    ONLY process data from single channel
    :param ecFilepath: filepath of GRID data
    :param nbins: number of bins, 0 < nbins <= 65536
    :param ch: the channel number in range [0-3] for single channel plotting
    :param doCorr: boolean indicating whether the temperature-bias correction will be done, to avoid warning info output
    :param rateCorr: boolean indicating whether the rate correction will be done. The default method is 's'.
    :param isPlotSpec: True to plot energy spectrums of each source file 
    :param isPlotEC: True to plot the final EC curve and the Energy Resolution curve
    :param fitEC: True to plot the fit curve of the EC curve and the Energy Resolution curve
    :return: result of the process on GRID data, in the form of a dictionary:
        {
            'energys' : energys of the source,
            'centers' : center ADC of the full-energy peak,
            'centersErr' : error of center ADC of the full-energy peak,
            'cpsPeak' : count rate of the full-energy peak (before rate correction),
            'cpsPeakErr' : error fo count rate of the full-energy peak (before rate correction),
            'energyResolutions' : energy resolution of the full-energy peak,
            'energyResolutionsErr' : error of energy resolution of the full-energy peak,
            'date' : date of the experiment,
            'extimeSrcs' : experiment time of the source file,
            'extimeBkgs' : experiment time of the background file,
            'cpsPeakCorr' : count rate of the full-energy peak (after rate correction),
            'cpsPeakCorrErr' : error of count rate of the full-energy peak (after rate correction)
        }
    """
    
    if not grid.isChannel(channel):
        raise Exception('plotEnergyChannel: channel number out of bound[0-3]')
    
    #Single channel plot
    
    ecFiles = os.listdir(ecFilepath)
    ecFiles.sort()
    energys = []
    centers = []
    centersErr = []
    amps = []
    ampsErr = []
    energyResolutions = []
    energyResolutionsErr = []
    date = []
    extimeSrcs = []
    extimeBkgs = []
    cpsTotalSrc = []
    cpsTotalBkg = []
    cpsRateCorrectSrc = []
    cpsRateCorrectBkg = []
    cpsRateCorrectErrSrc = []
    cpsRateCorrectErrBkg = []
    srcFiles = []
    bkgFiles = []

    for filepath in ecFiles:
        if not '.' in filepath:#to avoid '.DS_store' files in Mac OS
            filepathLv1 = os.listdir(ecFilepath+'\\'+filepath)
            filepathLv1.sort()
            
            if 'share' in filepath:
                for filename in filepathLv1:
                    if filename.endswith('.txt') and ('bkg' in filename):
                        bkgFile = ecFilepath+'\\'+filepath + '\\' + filename
                        ampBkg, tempSipmBkg, tempAdcBkg, vMonBkg, iMonBkg, biasBkg, uscountTeleBkg, uscountEvtBkg, rateCorrectBkg, effectiveCountBkg, missingCountBkg \
                            = grid.dataReadout(bkgFile, isHex = False, isCi = 0, isScan = False, scanRange = [], rateStyle = 'p')
                        extimeBkg = np.max(np.hstack(uscountEvtBkg))-np.min(np.hstack(uscountEvtBkg))
                        ampBkg[ch] = np.array(ampBkg[ch]) * grid.tempBiasCorrection(tempSipmBkg, biasBkg, corr = False, isTemp = False, doCorr = doCorr)[0][ch]

                        ctsBkg, xEdgeBkg = np.histogram(ampBkg[ch], bins = nbins, range = (0., 65536.))
                        xBkg = (xEdgeBkg[0:-1]+xEdgeBkg[1:])/2.
                        
                for filename in filepathLv1:
                    if filename.endswith('.txt') and ('bkg' not in filename):
                        srcFile = ecFilepath+ '\\' +filepath + '\\' + filename    
                        sourceEnergy = float(srcFile.split('_')[-2][:-3])
                        sourceDate = filepath.split('_')[1]
                        ampSrc, tempSipmSrc, tempAdcSrc, vMonSrc, iMonSrc, biasSrc, uscountTeleSrc, uscountEvtSrc, rateCorrectSrc, effectiveCountSrc, missingCountSrc \
                            = grid.dataReadout(srcFile, isHex = False, isCi = 0, isScan = False, scanRange = [], rateStyle = 'p')
                        extimeSrc = np.max(np.hstack(uscountEvtSrc))-np.min(np.hstack(uscountEvtSrc))
                        ampSrc[ch] = np.array(ampSrc[ch]) * grid.tempBiasCorrection(tempSipmSrc, biasSrc, corr = False, isTemp = False, doCorr = doCorr)[0][ch]

                        ctsSrc, xEdgeSrc = np.histogram(ampSrc[ch], bins = nbins, range = (0., 65536.))
                        xSrc = (xEdgeSrc[0:-1]+xEdgeSrc[1:])/2.
                        
                        ctsEff = ctsSrc/extimeSrc - ctsBkg/extimeBkg
                        ctsEff[ctsEff<0] = 0.
                        ctsEffErr = np.sqrt(grid.gehrelsErr(ctsSrc)**2/extimeSrc**2 + grid.gehrelsErr(ctsBkg)**2/extimeBkg**2)
                        
                        ''
                        q_1 = (xSrc > 0.) * (xSrc < 10000.)
                        res = grid.doFitPeak(xSrc[q_1], ctsEff[q_1], odr=False, yerror = ctsEffErr[q_1], quadBkg = False)
                        amp = res['peak_amplitude']
                        center = res['peak_center']
                        sigma = res['peak_sigma']
                        
                        q_2 = (xSrc >= center-3.*sigma)*(xSrc <= center+3.*sigma)
                        res_2 = grid.doFitPeak(xSrc[q_2], ctsEff[q_2], odr=False, yerror = ctsEffErr[q_2], quadBkg = False)
                        amp_2 = res_2['peak_amplitude']
                        center_2 = res_2['peak_center']
                        sigma_2 = res_2['peak_sigma']
                        amp_err_2 = res_2['peak_amplitude_err']
                        center_err_2 = res_2['peak_center_err']
                        sigma_err_2 = res_2['peak_sigma_err']
                        energy_resolution_2 = 2*np.sqrt(2*np.log(2))*sigma_2/center_2
                        energy_resolution_err_2 = 2*np.sqrt(2*np.log(2))*np.sqrt((sigma_err_2/center_2)**2 + (sigma_2 * center_err_2 / center_2**2)**2)
                                                
                        energys.append(sourceEnergy)
                        centers.append(center_2)
                        centersErr.append(center_err_2)
                        amps.append(amp_2*nbins/65536.)
                        ampsErr.append(amp_err_2*nbins/65536.)
                        energyResolutions.append(energy_resolution_2)
                        energyResolutionsErr.append(energy_resolution_err_2)
                        date.append(filepath.split('_')[1])

                        #rate correct
                        cpsTotalSrc.append(np.size(np.hstack(uscountEvtSrc))/extimeSrc)
                        cpsTotalBkg.append(np.size(np.hstack(uscountEvtBkg))/extimeBkg)
                        
                        rateAllSrc, rateAllErrSrcErr = grid.fitRateCorrect(srcFile.split('\\')[-1], rateCorrectSrc, plot = False, odr = False, rateStyle = 'p')
                        rateAllBkg, rateAllErrBkgErr = grid.fitRateCorrect(bkgFile.split('\\')[-1], rateCorrectBkg, plot = False, odr = False, rateStyle = 'p')
                        
                        cpsRateCorrectSrc.append(rateAllSrc)
                        cpsRateCorrectBkg.append(rateAllBkg)
                        cpsRateCorrectErrSrc.append(rateAllErrSrcErr)
                        cpsRateCorrectErrBkg.append(rateAllErrBkgErr)
                        
                        srcFiles.append(srcFile)
                        bkgFiles.append(bkgFile)

                        extimeSrcs.append(extimeSrc)
                        extimeBkgs.append(extimeBkg)
                        
                        if isPlotSpec:
                            fig = plt.figure(figsize=(12, 8))
                            gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
                            ax = fig.add_subplot(gs[0])
                        #    p1 = plt.step(xSrc, ctsSrc/extimeSrc, where = 'mid',label = 'src')
                        #    p2 = plt.step(xBkg, ctsBkg/extimeBkg, where = 'mid',label = 'bkg')
                            ax.step(xBkg, ctsEff, where = 'mid',label = 'eff')
                            param_2 = np.array([amp_2,center_2,sigma_2])
                            ax.plot(xSrc[q_2], grid.gaussianFunction(param_2,xSrc[q_2]), label = 'Gaussion fit')

                            ax.text(center*1.15, ctsEff.max()*0.25, \
                                    'Center: %.1f $\pm$  %.1f \n Energy Resolution: %.3f $\pm$  %.3f'%(center_2, center_err_2, energy_resolution_2, energy_resolution_err_2),  \
                                    fontsize=10, bbox=dict(facecolor='pink', alpha=0.1),\
                                    horizontalalignment='center', verticalalignment='center')
                        
                            ax.set_xlim([0,10000])
                            ax.set_xlabel('ADC')
                            ax.set_ylabel('cps')
                            
                            ax.legend(loc=0)
                            ax.grid()
                            fig.show()
                        
            elif 'same' in filepath:
                for filename in filepathLv1:
                    if ('src' in filename) and ('CH%d'%ch in filename):
                        srcFile = ecFilepath+ '\\' +filepath + '\\' + filename    
                        sourceEnergy = float(srcFile.split('_')[-2][:-3])
                        sourceDate = filepath.split('_')[1]
                        ampSrc, tempSipmSrc, tempAdcSrc, vMonSrc, iMonSrc, biasSrc, uscountTeleSrc, uscountEvtSrc, rateCorrectSrc, effectiveCountSrc, missingCountSrc \
                            = grid.dataReadout(srcFile, isHex = False, isCi = 0, isScan = False, scanRange = [], rateStyle = 'p')
                        extimeSrc = np.max(np.hstack(uscountEvtSrc))-np.min(np.hstack(uscountEvtSrc))
                        ampSrc[ch] = np.array(ampSrc[ch]) * grid.tempBiasCorrection(tempSipmSrc, biasSrc, corr = False, isTemp = False, doCorr = doCorr)[0][ch]

                        ctsSrc, xEdgeSrc = np.histogram(ampSrc[ch], bins = nbins, range = (0., 65536.))
                        xSrc = (xEdgeSrc[0:-1]+xEdgeSrc[1:])/2.
                        
                        for filename in filepathLv1:
                            if (srcFile.split('_')[-2] in filename) and (filename not in srcFile):
                                bkgFile = ecFilepath+'\\'+filepath + '\\' + filename
                                ampBkg, tempSipmBkg, tempAdcBkg, vMonBkg, iMonBkg, biasBkg, uscountTeleBkg, uscountEvtBkg, rateCorrectBkg, effectiveCountBkg, missingCountBkg \
                                    = grid.dataReadout(bkgFile, isHex = False, isCi = 0, isScan = False, scanRange = [], rateStyle = 'p')
                                extimeBkg = np.max(np.hstack(uscountEvtBkg))-np.min(np.hstack(uscountEvtBkg))
                                ampBkg[ch] = np.array(ampBkg[ch]) * grid.tempBiasCorrection(tempSipmBkg, biasBkg, corr = False, isTemp = False, doCorr = doCorr)[0][ch]

                                ctsBkg, xEdgeBkg = np.histogram(ampBkg[ch], bins = nbins, range = (0., 65536.))
                                xBkg = (xEdgeBkg[0:-1]+xEdgeBkg[1:])/2.
                                
                                ctsEff = ctsSrc/extimeSrc - ctsBkg/extimeBkg
                                ctsEff[ctsEff<0] = 0.
                                ctsEffErr = np.sqrt(grid.gehrelsErr(ctsSrc)**2/extimeSrc**2+grid.gehrelsErr(ctsBkg)**2/extimeBkg**2)
                                
                                ''
                                q_1 = (xSrc > 0.) * (xSrc < 10000.)
                                res = grid.doFitPeak(xSrc[q_1], ctsEff[q_1], odr=False, yerror = ctsEffErr[q_1], quadBkg = False)
                                amp = res['peak_amplitude']
                                center = res['peak_center']
                                sigma = res['peak_sigma']
                                
                                q_2 = (xSrc >= center-3.*sigma)*(xSrc <= center+3.*sigma)
                                res_2 = grid.doFitPeak(xSrc[q_2], ctsEff[q_2], odr=False, yerror = ctsEffErr[q_2], quadBkg = False)
                                amp_2 = res_2['peak_amplitude']
                                center_2 = res_2['peak_center']
                                sigma_2 = res_2['peak_sigma']
                                amp_err_2 = res_2['peak_amplitude_err']
                                center_err_2 = res_2['peak_center_err']
                                sigma_err_2 = res_2['peak_sigma_err']
                                energy_resolution_2 = 2*np.sqrt(2*np.log(2))*sigma_2/center_2
                                energy_resolution_err_2 = 2*np.sqrt(2*np.log(2))*np.sqrt((sigma_err_2/center_2)**2 + (sigma_2 * center_err_2 / center_2**2)**2)
                            
                                energys.append(sourceEnergy)
                                centers.append(center_2)
                                centersErr.append(center_err_2)
                                amps.append(amp_2*nbins/65536.)
                                ampsErr.append(amp_err_2*nbins/65536.)
                                energyResolutions.append(energy_resolution_2)
                                energyResolutionsErr.append(energy_resolution_err_2)
                                date.append(filepath.split('_')[1])

                                #rate correct
                                cpsTotalSrc.append(np.size(np.hstack(uscountEvtSrc))/extimeSrc)
                                cpsTotalBkg.append(np.size(np.hstack(uscountEvtBkg))/extimeBkg)
                                
                                rateAllSrc, rateAllErrSrcErr = grid.fitRateCorrect(srcFile.split('\\')[-1], rateCorrectSrc, plot = False, odr = False, rateStyle = 'p')
                                rateAllBkg, rateAllErrBkgErr = grid.fitRateCorrect(bkgFile.split('\\')[-1], rateCorrectBkg, plot = False, odr = False, rateStyle = 'p')
                                
                                cpsRateCorrectSrc.append(rateAllSrc)
                                cpsRateCorrectBkg.append(rateAllBkg)
                                cpsRateCorrectErrSrc.append(rateAllErrSrcErr)
                                cpsRateCorrectErrBkg.append(rateAllErrBkgErr)
                                
                                srcFiles.append(srcFile)
                                bkgFiles.append(bkgFile)

                                extimeSrcs.append(extimeSrc)
                                extimeBkgs.append(extimeBkg)
                                
                                if isPlotSpec:
                                    fig = plt.figure(figsize=(12, 8))
                                    gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
                                    ax = fig.add_subplot(gs[0])
                                #    p1 = plt.step(xSrc, ctsSrc/extimeSrc, where = 'mid',label = 'src')
                                #    p2 = plt.step(xBkg, ctsBkg/extimeBkg, where = 'mid',label = 'bkg')
                                    ax.step(xBkg, ctsEff, where = 'mid',label = 'eff')
                                    param_2 = np.array([amp_2,center_2,sigma_2])
                                    ax.plot(xSrc[q_2], grid.gaussianFunction(param_2,xSrc[q_2]), label = 'Gaussion fit')
                                    
                                    ax.text(center*1.15, ctsEff.max()*0.25, \
                                            'Center: %.1f $\pm$  %.1f \n Energy Resolution: %.3f $\pm$  %.3f'%(center_2, center_err_2, energy_resolution_2, energy_resolution_err_2),  \
                                            fontsize=10, bbox=dict(facecolor='pink', alpha=0.1),\
                                            horizontalalignment='center', verticalalignment='center')
                                
                                    ax.set_xlim([0,10000])
                                    ax.set_xlabel('ADC')
                                    ax.set_ylabel('cps')
                                    
                                    ax.legend(loc=0)
                                    ax.grid()
                                    fig.show()
            
            elif 'mutual' in filepath:
                for filename in filepathLv1:
                    if 'CH%d'%ch in filename:
                        srcFile = ecFilepath+ '\\' +filepath + '\\' + filename    
                        sourceEnergy = float(srcFile.split('_')[-2][:-3])
                        sourceDate = filepath.split('_')[1]
                        ampSrc, tempSipmSrc, tempAdcSrc, vMonSrc, iMonSrc, biasSrc, uscountTeleSrc, uscountEvtSrc, rateCorrectSrc, effectiveCountSrc, missingCountSrc \
                            = grid.dataReadout(srcFile, isHex = False, isCi = 0, isScan = False, scanRange = [], rateStyle = 'p')
                        extimeSrc = np.max(np.hstack(uscountEvtSrc))-np.min(np.hstack(uscountEvtSrc))
                        ampSrc[ch] = np.array(ampSrc[ch]) * grid.tempBiasCorrection(tempSipmSrc, biasSrc, corr = False, isTemp = False, doCorr = doCorr)[0][ch]

                        ctsSrc, xEdgeSrc = np.histogram(ampSrc[ch], bins = nbins, range = (0., 65536.))
                        xSrc = (xEdgeSrc[0:-1]+xEdgeSrc[1:])/2.
                        
                        for filename in filepathLv1:
                            if (srcFile.split('_')[-2] in filename) and (filename not in srcFile):
                                bkgFile = ecFilepath+'\\'+filepath + '\\' + filename
                                ampBkg, tempSipmBkg, tempAdcBkg, vMonBkg, iMonBkg, biasBkg, uscountTeleBkg, uscountEvtBkg, rateCorrectBkg, effectiveCountBkg, missingCountBkg \
                                    = grid.dataReadout(bkgFile, isHex = False, isCi = 0, isScan = False, scanRange = [], rateStyle = 'p')
                                extimeBkg = np.max(np.hstack(uscountEvtBkg))-np.min(np.hstack(uscountEvtBkg))
                                ampBkg[ch] = np.array(ampBkg[ch]) * grid.tempBiasCorrection(tempSipmBkg, biasBkg, corr = False, isTemp = False, doCorr = doCorr)[0][ch]

                                ctsBkg, xEdgeBkg = np.histogram(ampBkg[ch], bins = nbins, range = (0., 65536.))
                                xBkg = (xEdgeBkg[0:-1]+xEdgeBkg[1:])/2.
                                
                                ctsEff = ctsSrc/extimeSrc - ctsBkg/extimeBkg
                                ctsEff[ctsEff<0] = 0.
                                ctsEffErr = np.sqrt(grid.gehrelsErr(ctsSrc)**2/extimeSrc**2+grid.gehrelsErr(ctsBkg)**2/extimeBkg**2)
                                
                                ''
                                q_1 = (xSrc > 0.) * (xSrc < 10000.)
                                res = grid.doFitPeak(xSrc[q_1], ctsEff[q_1], odr=False, yerror = ctsEffErr[q_1], quadBkg = False)
                                amp = res['peak_amplitude']
                                center = res['peak_center']
                                sigma = res['peak_sigma']
                                
                                q_2 = (xSrc >= center-3.*sigma)*(xSrc <= center+3.*sigma)
                                res_2 = grid.doFitPeak(xSrc[q_2], ctsEff[q_2], odr=False, yerror = ctsEffErr[q_2], quadBkg = False)
                                amp_2 = res_2['peak_amplitude']
                                center_2 = res_2['peak_center']
                                sigma_2 = res_2['peak_sigma']
                                amp_err_2 = res_2['peak_amplitude_err']
                                center_err_2 = res_2['peak_center_err']
                                sigma_err_2 = res_2['peak_sigma_err']
                                energy_resolution_2 = 2*np.sqrt(2*np.log(2))*sigma_2/center_2
                                energy_resolution_err_2 = 2*np.sqrt(2*np.log(2))*np.sqrt((sigma_err_2/center_2)**2 + (sigma_2 * center_err_2 / center_2**2)**2)
                                
                                energys.append(sourceEnergy)
                                centers.append(center_2)
                                centersErr.append(center_err_2)
                                amps.append(amp_2*nbins/65536.)
                                ampsErr.append(amp_err_2*nbins/65536.)
                                energyResolutions.append(energy_resolution_2)
                                energyResolutionsErr.append(energy_resolution_err_2)
                                date.append(filepath.split('_')[1])

                                #rate correct
                                cpsTotalSrc.append(np.size(np.hstack(uscountEvtSrc))/extimeSrc)
                                cpsTotalBkg.append(np.size(np.hstack(uscountEvtBkg))/extimeBkg)
                                
                                rateAllSrc, rateAllErrSrcErr = grid.fitRateCorrect(srcFile.split('\\')[-1], rateCorrectSrc, plot = False, odr = False, rateStyle = 'p')
                                rateAllBkg, rateAllErrBkgErr = grid.fitRateCorrect(bkgFile.split('\\')[-1], rateCorrectBkg, plot = False, odr = False, rateStyle = 'p')
                                
                                cpsRateCorrectSrc.append(rateAllSrc)
                                cpsRateCorrectBkg.append(rateAllBkg)
                                cpsRateCorrectErrSrc.append(rateAllErrSrcErr)
                                cpsRateCorrectErrBkg.append(rateAllErrBkgErr)
                                
                                srcFiles.append(srcFile)
                                bkgFiles.append(bkgFile)

                                extimeSrcs.append(extimeSrc)
                                extimeBkgs.append(extimeBkg)
                                
                                if isPlotSpec:
                                    fig = plt.figure(figsize=(12, 8))
                                    gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
                                    ax = fig.add_subplot(gs[0])
                                #    p1 = plt.step(xSrc, ctsSrc/extimeSrc, where = 'mid',label = 'src')
                                #    p2 = plt.step(xBkg, ctsBkg/extimeBkg, where = 'mid',label = 'bkg')
                                    ax.step(xBkg, ctsEff, where = 'mid',label = 'eff')
                                    param_2 = np.array([amp_2,center_2,sigma_2])
                                    ax.plot(xSrc[q_2], grid.gaussianFunction(param_2,xSrc[q_2]), label = 'Gaussion fit')
                                    
                                    ax.text(center*1.15, ctsEff.max()*0.25, \
                                            'Center: %.1f $\pm$  %.1f \n Energy Resolution: %.3f $\pm$  %.3f'%(center_2, center_err_2, energy_resolution_2, energy_resolution_err_2),  \
                                            fontsize=10, bbox=dict(facecolor='pink', alpha=0.1),\
                                            horizontalalignment='center', verticalalignment='center')
                                
                                    ax.set_xlim([0,10000])
                                    ax.set_xlabel('ADC')
                                    ax.set_ylabel('cps')
                                    
                                    ax.legend(loc=0)
                                    ax.grid()
                                    fig.show()

    order = np.argsort(energys)
    energys = np.array(energys)[order]
    centers = np.array(centers)[order]
    centersErr = np.array(centersErr)[order]
    amps = np.array(amps)[order]
    ampsErr = np.array(ampsErr)[order]
    energyResolutions = np.array(energyResolutions)[order]
    energyResolutionsErr = np.array(energyResolutionsErr)[order]
    date = np.array(date)[order]
    cpsTotalSrc = np.array(cpsTotalSrc)[order]
    cpsTotalBkg = np.array(cpsTotalBkg)[order]
    cpsRateCorrectSrc = np.array(cpsRateCorrectSrc)[order]
    cpsRateCorrectBkg = np.array(cpsRateCorrectBkg)[order]
    cpsRateCorrectErrSrc = np.array(cpsRateCorrectErrSrc)[order]
    cpsRateCorrectErrBkg = np.array(cpsRateCorrectErrBkg)[order]

    cpsPeakCorr = amps / cpsTotalSrc * cpsRateCorrectSrc
    cpsPeakCorrErr = ampsErr / cpsTotalSrc * cpsRateCorrectSrc
    
    ecResult = {
        'energys' : energys,
        'centers' : centers,
        'centersErr' : centersErr,
        'cpsPeak' : amps,
        'cpsPeakErr' : ampsErr,
        'energyResolutions' : energyResolutions,
        'energyResolutionsErr' : energyResolutionsErr,
        'date' : date,
        'extimeSrcs' : extimeSrcs,
        'extimeBkgs' : extimeBkgs,
        'cpsPeakCorr' : cpsPeakCorr,
        'cpsPeakCorrErr' : cpsPeakCorrErr
    }

    if isPlotEC:
        # polyfit, quadratic
        def polyfit_quad(x, *p):
            y = p[0]*x**2 + p[1]*x + p[2]
            return y

        def polyfit_quad_odr(beta,x):
            return beta[0]*x**2+beta[1]*1+beta[2]

        def resolution_fit(x, *p):
            y = np.sqrt(p[0]*x**2 + p[1]*x + p[2])/x
            y[p[0]*x**2 + p[1]*x + p[2]<0] = 0.
            return y

        ##### For Grid 2, tempBiasCorrection #####
    
        Energy = np.array([59.5,239.,511.,661.7,1332.2])

        Center = np.array([[1582.566816707204, 1646.649163276882, 1641.7559080940837, 1774.6364105043729],
                            [7943.03970602043, 7795.72067633472, 8343.685387345404, 8820.084792525871],
                            [17618.609760806823, 17211.713722400196, 18655.9438271497, 19613.539329097064],
                            [22788.187875098924, 22152.732436646886, 24097.786085767468, 25377.559039844135],
                            [43135.91081719605, 49151.52616731882, 45325.40130988872, 51395.83399429588]]).T
    
        Center_err = np.array([[2.351047160631907, 1.4281001227808685, 4.222082636949857, 2.1303425320696503],    
                            [5.867163494576479, 4.615959765954061, 6.609635537247647, 5.88967272241771],
                            [3.488093959967798, 4.436788883750613, 6.006935498704928, 5.261026730761529],
                            [5.5829987456704355, 6.940954216123996, 5.745113119121808, 4.983243765439074],
                            [13.864630922728846, 16.52662249611073, 12.170675802570251, 12.693953462610999]]).T
    
        FWHM = np.array([[0.35670283971363326, 0.30395928325907895, 0.3374516830519281, 0.36229455564567636],
                        [0.13602310138886325, 0.1391400979779686, 0.15461732481217377, 0.14889970559760876],
                        [0.11187574905740798, 0.104056986148406, 0.11839431623679857, 0.11899612538405897],
                        [0.09446181564596827, 0.08885521300115116, 0.1066450188894006, 0.09956272687376323],
                        [0.11162847766514353, 0.08829055819316972, 0.09815209115948591, 0.07248397309063642]]).T
    
        FWHM_err = np.array([[0.002101660875722357, 0.001855765309002222, 0.002912469794361523, 0.0020301223985118595],
                            [0.009439440849099931, 0.009136894468900963, 0.012391718528702134, 0.011792088419629884],
                            [0.004423982351878793, 0.003984823602057866, 0.004936275953831328, 0.005364281268106144],
                            [0.003548631164151075, 0.003433900296957314, 0.004278673082443479, 0.005340570015328257],
                            [0.0009499049936753366, 0.0009559839926902354, 0.0012569744494991056, 0.0009013047234425496]]).T

        fig1 = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95,height_ratios=[4,1])
        ax0 = fig1.add_subplot(gs[0])
        ax0.errorbar(energys, centers, yerr = centersErr, fmt='s', mfc='white', ms=6, elinewidth=1, capsize=3, barsabove=True, zorder=1, label = 'X-ray tube CH%d'%ch)
        # ax0.errorbar(Energy, Center[ch], yerr = Center_err[ch], fmt='^', mfc='white', ms=6, elinewidth=1, capsize=3, barsabove=True, zorder=1, label = 'source CH%d'%ch)
        if fitEC:
            p = np.polyfit(Energy, Center[ch], 2)

            q_low = energys<49.
            p_low = np.polyfit(energys[q_low], centers[q_low], 2)
            par_low, pcov_low = curve_fit(polyfit_quad, energys[q_low], centers[q_low], sigma=centersErr[q_low], p0=p) #pre_fit
            perr_low = np.sqrt(np.diag(pcov_low))
            energy_set_low = np.arange(10,50,2)
            ax0.plot(energy_set_low, polyfit_quad(energy_set_low, *par_low), linestyle = '-', label='quadratic fit on X-ray tube ch%d, < 49keV'%ch)

            q_high = energys>55.
            p_high = np.polyfit(energys[q_high], centers[q_high], 2)
            par_high, pcov_high = curve_fit(polyfit_quad, energys[q_high], centers[q_high], sigma=centersErr[q_high], p0=p) #pre_fit
            perr_high = np.sqrt(np.diag(pcov_high))
            energy_set_high = np.arange(50,1400,10)
            ax0.plot(energy_set_high, polyfit_quad(energy_set_high, *par_high), linestyle = '-', label='quadratic fit on X-ray tube ch%d, > 55keV'%ch)

            qModel = lmfit.models.QuadraticModel(prefix = 'bk_')
            param1 = qModel.guess(energys[q_low], x = centers[q_low])
            data = RealData(centers[q_low], energys[q_low], sx = centersErr[q_low], fix = np.ones(3))
            model = Model(polyfit_quad_odr)
            odr = ODR(data, model, beta0=p_low)#[param1.valuesdict()['bk_a'], param1.valuesdict()['bk_b'], param1.valuesdict()['bk_c']])
            odr.set_job(fit_type = 0)
            result = odr.run()
            fitResult = {
                            'bk_a':         result.beta[0],
                            'bk_b':         result.beta[1],
                            'bk_c':         result.beta[2],
                }

        ax0.set_xlim([10,1400])
        ax0.set_xlabel('Energy(keV)')
        ax0.set_ylabel('ADC')
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax0.grid()
        ax0.legend(loc=0, fontsize = 12)

        if fitEC:
            ax1 = fig1.add_subplot(gs[1])
            ax1.plot(energys[q_low],(centers[q_low]- polyfit_quad(energys[q_low], *par_low))/centers[q_low]*100,'bs')
            ax1.plot(energys[q_high],(centers[q_high]-polyfit_quad(energys[q_high], *par_high))/centers[q_high]*100,'bs')
            ax1.plot(Energy,(Center[0]-polyfit_quad(Energy, *par_high))/Center[0]*100, '^',color='orange')
            ax1.set_xlabel('Energy (keV)')
            ax1.set_ylabel('Residual(%)')
            ax1.set_xscale('log')
            ax1.set_xlim([10,1400])
            ax1.set_ylim([-65,65])

        fig1.show()

        fig2 = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95,height_ratios=[4,1])
        ax0 = fig2.add_subplot(111)

        ax0.errorbar(energys, energyResolutions*100, yerr = energyResolutionsErr*100, fmt='s', mfc='white', ms=6, elinewidth=1, capsize=3, barsabove=True, zorder=1, label = 'CH%d'%ch)

        if fitEC:
            lbd = np.zeros(3)
            ubd = np.zeros(3)+np.inf
            p=np.ones(3)

            q_low = energys<49.
            par_low, pcov_low = curve_fit(resolution_fit, energys[q_low], energyResolutions[q_low], sigma=energyResolutionsErr[q_low], p0=p, bounds=(lbd, ubd)) #pre_fit
            perr_low = np.sqrt(np.diag(pcov_low))
            energy_set_low = np.arange(10,50,2)
            ax0.plot(energy_set_low, resolution_fit(energy_set_low, *par_low)*100, linestyle = '-', zorder=0, label='fit on ch%d, low energy'%ch)

            q_high = energys>55.
            par_high, pcov_high = curve_fit(resolution_fit, energys[q_high], energyResolutions[q_high], sigma=energyResolutionsErr[q_high], p0=p, bounds=(lbd, ubd)) #pre_fit
            perr_high = np.sqrt(np.diag(pcov_high))
            energy_set_high = np.arange(50,1400,10)
            ax0.plot(energy_set_high, resolution_fit(energy_set_high, *par_high)*100, linestyle = '-', zorder=0, label='fit on ch%d, high energy'%ch)

        ax0.set_xlabel('Energy (keV)')
        ax0.set_ylabel('R(%)')
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax0.legend(loc=0, fontsize = 12)
        ax0.grid()
        fig2.show()

    return ecResult

def processHPGe(HPGeFilepath, isPlotSpec=False):
    """
    Function for processing HPGe data from NIM.
    :param HPGeFilepath: filepath of HPGe data
    :param isPlotSpec: True to plot energy spectrums of each source file 
    :return: result of the process on GRID data, in the form of a dictionary:
        {
            'HPGeDates' : date of the experiment,
            'HPGeEnergys' : energys of the source,
            'HPGeCenters' : center ADC of the full-energy peak,
            'HPGeCentersErr' : error of center ADC of the full-energy peak,
            'HPGeEnergyResolutions' : energy resolution of the full-energy peak,
            'HPGeEnergyResolutionsErr' : error of center ADC of the full-energy peak,
            'HPGeCts' : counts of the full-energy peak,
            'HPGeCtsErr' : error of counts of the full-energy peak,
            'HPGeCps' : count rate of the full-energy peak,
            'HPGeCpsErr' : error of count rate of the full-energy peak,
            'HPGeTime' : experiment time,
        }
    """
    
    HPGeDates = []
    HPGeEnergys = []
    HPGeCenters = []
    HPGeCentersErr = []
    HPGeEnergyResolutions = []
    HPGeEnergyResolutionsErr = []
    HPGeCts = []
    HPGeCtsErr = []
    HPGeCps = []
    HPGeCpsErr = []
    HPGeTime = []
    HPGeFilenames = []

    HPGeFiles = os.listdir(HPGeFilepath)
    HPGeFiles.sort()
    for filepath in HPGeFiles:
        if not '.' in filepath:#to avoid '.DS_store' files in Mac OS
            filepathLv1 = os.listdir(HPGeFilepath+'\\'+filepath)
            filepathLv1.sort()
            HPGeDate = filepath[-4:]
            for filename in filepathLv1:
                if filename.endswith('TKA') and 'bkg' not in filename:
                    timeHPGe, ctsHPGe = grid.HPGeDataReadout(HPGeFilepath+'\\'+filepath+'\\'+filename)
                    xHPGe = np.arange(np.size(ctsHPGe))
                    qq = np.where(ctsHPGe == ctsHPGe.max())
                    q = (xHPGe >= xHPGe[qq]-150.) * (xHPGe <= xHPGe[qq]+150.)
                    res = grid.doFitPeak(xHPGe[q], ctsHPGe[q], odr=False, yerror = grid.gehrelsErr(ctsHPGe[q]), quadBkg = False)
                    amp = res['peak_amplitude']
                    center = res['peak_center']
                    sigma = res['peak_sigma']
                    amp_err = res['peak_amplitude_err']
                    center_err = res['peak_center_err']
                    sigma_err = res['peak_sigma_err']
                    energy_resolution = 2*np.sqrt(2*np.log(2))*sigma/center
                    energy_resolution_err = 2*np.sqrt(2*np.log(2))*np.sqrt((sigma_err/center)**2 + (sigma * center_err / center**2)**2)
                    if isPlotSpec:
                        fig = plt.figure(figsize=(12, 8))
                        plt.step(xHPGe, ctsHPGe, where = 'mid',label = 'cts')
                        param = np.array([amp,center,sigma])
                        plt.plot(xHPGe[q], grid.gaussianFunction(param,xHPGe[q]), label = 'Gaussion fit')
                        plt.title(filename)
                        plt.legend()
                        plt.grid()
                
                    HPGeDates.append(HPGeDate)
                    HPGeEnergys.append(float(filename.split('-')[0][:-3].replace('p','.')))
                    HPGeCenters.append(center)
                    HPGeCentersErr.append(center_err)
                    HPGeEnergyResolutions.append(energy_resolution)
                    HPGeEnergyResolutionsErr.append(energy_resolution_err)
                    HPGeCts.append(amp)
                    HPGeCtsErr.append(amp_err)
                    HPGeCps.append(amp/timeHPGe)
                    HPGeCpsErr.append(amp_err/timeHPGe)
                    HPGeTime.append(timeHPGe)
                    HPGeFilenames.append(HPGeFilepath+'\\'+filepath+'\\'+filename)
            
    HPGeDates = np.array(HPGeDates)
    HPGeEnergys = np.array(HPGeEnergys)
    HPGeCenters = np.array(HPGeCenters)
    HPGeCentersErr = np.array(HPGeCentersErr)
    HPGeEnergyResolutions = np.array(HPGeEnergyResolutions)
    HPGeEnergyResolutionsErr = np.array(HPGeEnergyResolutionsErr)
    HPGeCts = np.array(HPGeCts)
    HPGeCtsErr = np.array(HPGeCtsErr)
    HPGeCps = np.array(HPGeCps)
    HPGeCpsErr = np.array(HPGeCpsErr)
    HPGeTime = np.array(HPGeTime)
    HPGeFilenames = np.array(HPGeFilenames)

    HPGeResult={
        'HPGeDates' : HPGeDates,
        'HPGeEnergys' : HPGeEnergys,
        'HPGeCenters' : HPGeCenters,
        'HPGeCentersErr' : HPGeCentersErr,
        'HPGeEnergyResolutions' : HPGeEnergyResolutions,
        'HPGeEnergyResolutionsErr' : HPGeEnergyResolutionsErr,
        'HPGeCts' : HPGeCts,
        'HPGeCtsErr' : HPGeCtsErr,
        'HPGeCps' : HPGeCps,
        'HPGeCpsErr' : HPGeCpsErr,
        'HPGeTime' : HPGeTime,
    }

    return HPGeResult

def getEfficiency(ecResult, HPGeResult, isPlot=True):
    """
    Function for calculating detective efficiency data from NIM.
    :param ecResult: process result of GRID data
    :param HPGeResult: process result of HPGe data
    :param isPlot: True to plot the final effective curve
    :return: result of the process on GRID data and HPGe data, in the form of a dictionary:
        {
            'dates' : date of the experiment,
            'energys' : energys of the source,
            'relEff' : relative efficiency, before rate correct
            'relEffErr' : error of relative efficiency, before rate correct
            'relEffRateCorr' : relative efficiency, after rate correct
            'relEffRateCorrErr' : error of relative efficiency, after rate correct,
            'absoluteEfficiency' : absolute efficiency, after rate correct,
            'absoluteEfficiencyErr' : error of absolute efficiency, after rate correct,
        }
    """
    dateEff = []
    ergEff = []
    cpsPeakEff = []
    cpsPeakEffErr = []
    cpsPeakCorrEff = []
    cpsPeakCorrEffErr = []
    HPGeEff = []
    HPGeEffErr = []
    relEff = []
    relEffErr = []
    relEffRateCorr = []
    relEffRateCorrErr = []

    HPGeEnergys = HPGeResult['HPGeEnergys']
    HPGeCps = HPGeResult['HPGeCps']
    HPGeCpsErr = HPGeResult['HPGeCpsErr']
    HPGeDates = HPGeResult['HPGeDates']

    for date in set(HPGeResult['HPGeDates']):
        q = ecResult['date'] == date
        erg = ecResult['energys'][q]
        cpsPeak = ecResult['cpsPeak'][q]
        cpsPeakErr = ecResult['cpsPeakErr'][q]
        cpsPeakCorr = ecResult['cpsPeakCorr'][q]
        cpsPeakCorrErr = ecResult['cpsPeakCorrErr'][q]

        for i in range(np.size(erg)):
            if erg[i] in HPGeEnergys:
                if np.size(np.where((HPGeEnergys == erg[i]) * (HPGeDates == date)))>0:
                    h = np.where((HPGeEnergys == erg[i]) * (HPGeDates == date))[0][0]
                    dateEff.append(date)
                    ergEff.append(erg[i])
                    cpsPeakEff.append(cpsPeak[i])
                    cpsPeakEffErr.append(cpsPeakErr[i])
                    cpsPeakCorrEff.append(cpsPeakCorr[i])
                    cpsPeakCorrEffErr.append(cpsPeakCorrErr[i])
                    HPGeEff.append(HPGeCps[h])
                    HPGeEffErr.append(HPGeCpsErr[h])
                    relEff.append(cpsPeak[i]/HPGeCps[h])
                    relEffErr.append(np.sqrt((cpsPeakErr[i]/HPGeCps[h])**2 + (cpsPeak[i] * HPGeCpsErr[h] / HPGeCps[h]**2)**2))
                    relEffRateCorr.append(cpsPeakCorr[i]/HPGeCps[h])
                    relEffRateCorrErr.append(np.sqrt((cpsPeakCorrErr[i]/HPGeCps[h])**2 + (cpsPeakCorr[i] * HPGeCpsErr[h] / HPGeCps[h]**2)**2))
    
    orderEff = np.argsort(ergEff)
    dateEff = np.array(dateEff)[orderEff]
    ergEff = np.array(ergEff)[orderEff]
    cpsPeakEff = np.array(cpsPeakEff)[orderEff]
    cpsPeakEffErr = np.array(cpsPeakEffErr)[orderEff]
    cpsPeakCorrEff = np.array(cpsPeakCorrEff)[orderEff]
    cpsPeakCorrEffErr = np.array(cpsPeakCorrEffErr)[orderEff]
    HPGeEff = np.array(HPGeEff)[orderEff]
    HPGeEffErr = np.array(HPGeEffErr)[orderEff]
    relEff = np.array(relEff)[orderEff]
    relEffErr = np.array(relEffErr)[orderEff]
    relEffRateCorr = np.array(relEffRateCorr)[orderEff]
    relEffRateCorrErr = np.array(relEffRateCorrErr)[orderEff]
    
    ########## absolute efficiency correct ##########
    #workbook
    wb = xlrd.open_workbook('/Users/yangdx/Downloads/GRID/HPGe_detection_efficiency.xlsx')
    #worksheet
    ws = wb.sheet_by_index(0)
    HPGeEnergy = np.array(ws.col_values(0)[1:])
    HPGeEfficiency = np.array(ws.col_values(1)[1:])
    f = interpolate.interp1d(HPGeEnergy, HPGeEfficiency)

    absoluteEfficiency = f(ergEff)/100.
    absEffRateCorr = relEffRateCorr * absoluteEfficiency
    absEffRateCorrErr = relEffRateCorrErr * absoluteEfficiency

    efficiencyResult={
        'dates' : dateEff,
        'energys' : ergEff,
        'cpsPeakEff' : cpsPeakEff,
        'cpsPeakEffErr' : cpsPeakEffErr,
        'cpsPeakCorrEff' : cpsPeakCorrEff,
        'cpsPeakCorrEffErr' : cpsPeakCorrEffErr,
        'HPGeEff' : HPGeEff,
        'HPGeEffErr' : HPGeEffErr,
        'relEff' : relEff,
        'relEffErr' : relEffErr,
        'relEffRateCorr' : relEffRateCorr,
        'relEffRateCorrErr' : relEffRateCorrErr,
        'absoluteEfficiency' : absEffRateCorr,
        'absoluteEfficiencyErr' : absEffRateCorrErr,
    }

    if isPlot:
        ########## simulation data ##########
        simuData = pd.read_csv('/Users/yangdx/Downloads/GRID/cali_data_collect/result_crystal0_eff_full_20200310.csv')
        energySimu = np.array(simuData['energy'])
        absEffSimu = np.array(simuData['efficiency_full'])

        fig = plt.figure(figsize=(12, 8))
        ax0 = fig.add_subplot(111)
        ax0.errorbar(ergEff, absEffRateCorr, yerr = absEffRateCorrErr, fmt='.', mfc='white', ms=8, elinewidth=1, capsize=3, barsabove=True, zorder=1, label = 'X-ray tube data')
        ax0.plot(energySimu, absEffSimu, color='k', zorder=1, label = 'Simulation result, full-peak')
        ax0.set_xlabel('Energy (keV)')
        ax0.set_ylabel('Absolute Efficiency')
        #ax0.set_xscale('log')
        #ax0.set_yscale('log')
        ax0.set_xlim([0,130])
        # ax0.set_ylim([0.4,1.4])
        ax0.legend(loc=0, fontsize = 12)
        ax0.grid()
        fig.show()

    return efficiencyResult
