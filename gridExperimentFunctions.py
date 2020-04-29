"""
Experiment-level processing functions
v0.0.2 for Grid2 calibration result analysis by ydx and ghz
"""

import gridBasicFunctions as grid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lmfit
import datetime

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
                        plt.scatter(uscount[isc], temp[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(uscount[isc], temp[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(uscount[isc], temp[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(uscount[isc], temp[ich][isc], c=colorplot[ich], s=1)
    else:
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
                            plt.scatter(uscount[isc], temp[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(uscount[isc], temp[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(uscount[isc], temp[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(uscount[isc], temp[ich][isc], c=colorplot[ich], s=1)
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
                        plt.scatter(uscount[isc], bias[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(uscount[isc], bias[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(uscount[isc], bias[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(uscount[isc], bias[ich][isc], c=colorplot[ich], s=1)
    else:
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
                            plt.scatter(uscount[isc], bias[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(uscount[isc], bias[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(uscount[isc], bias[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(uscount[isc], bias[ich][isc], c=colorplot[ich], s=1)
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
                        plt.scatter(uscount[isc], vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(uscount[isc], vmon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(uscount[isc], vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(uscount[isc], vmon[ich][isc], c=colorplot[ich], s=1)
    else:
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
                            plt.scatter(uscount[isc], vmon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(uscount[isc], vmon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(uscount[isc], vmon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(uscount[isc], vmon[ich][isc], c=colorplot[ich], s=1)
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
                        plt.scatter(uscount[isc], iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[ich][isc])) + 'V'))
                    else:
                        plt.scatter(uscount[isc], iMon[ich][isc], c=colorplot[isc % 4], s=1)
                else:
                    if isc == 0:
                        plt.scatter(uscount[isc], iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                    else:
                        plt.scatter(uscount[isc], iMon[ich][isc], c=colorplot[ich], s=1)
    else:
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
                            plt.scatter(uscount[isc], iMon[ich][isc], c=colorplot[isc % 4], s=1, label=('Bias: ' + str('%.2f' % np.average(bias[0][isc])) + 'V, '\
                               + str('%.2f' % np.average(bias[1][isc])) + 'V, ' + str('%.2f' % np.average(bias[2][isc])) + 'V, ' + str('%.2f' % np.average(bias[3][isc])) + 'V'))
                        else:
                            plt.scatter(uscount[isc], iMon[ich][isc], c=colorplot[isc % 4], s=1)
                    else:
                        if isc == 0:
                            plt.scatter(uscount[isc], iMon[ich][isc], c=colorplot[ich], s=1, label=('ch' + str(ich)))
                        else:
                            plt.scatter(uscount[isc], iMon[ich][isc], c=colorplot[ich], s=1)
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

def tempBiasFit(fitResults, input, isTemp, fileOutput = False, singlech = False, channel = -1, odr = False):

    """
    General function for fitting the temperature\bias responce curve
    :param fitResults: fit results, given in the form of list[list[dict]] for multiple channel\
 or list[dict] for single channel
    :param input: list of temperatures\bias of all individual scans, orgaized in the form \
of [channel][scan#][data#]
    :param isTemp: True if the data given is from temperature scan, False if the \
data is from bias scan
    :param fileOutput: the output style, False for only graph output, True for text file(.txt)\
 output as well as graph output
    :param singlech: boolean indicating whether the fit is for single channel
    :param channel: the channel number in range [0-3], used for single channel plots
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
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
    """

    if isTemp:
        print('tempBiasFit: commencing temperature responce curve fit')
    else:
        print('tempBiasFit: commencing bias responce curve fit')
    if singlech:
        if not grid.isChannel(channel):
            raise Exception('tempBiasFit: incorrect channel number form or channel number out of bound[0-3]')

    if isTemp:
        prefix = 'temp'
        lower = 10.
        upper = 20.
    else:
        prefix = 'bias'
        lower = 1.
        upper = 2.
    
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
                inputMax = np.max(inputAll)
                inputMin = np.min(inputAll)
            else:
                if inputMax < np.max(inputAll):
                    inputMax = np.max(inputAll)
                if inputMin > np.min(inputAll):
                    inputMin = np.min(inputAll)
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

        ax0.set_xlim([inputMin, inputMax])
        ax0.set_ylabel('ADC/channel')
        if isTemp:
            ax0.set_title('Fit of temperature responce\nFit function: $ax^2 + bx + c$')
        else:
            ax0.set_title('Fit of bias responce\nFit function: $ax^2 + bx + c$')
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
            ax0.plot(angle, rate * efficiency[channel] / (rate[0] + rate[-1]) * 2, color=colorplot[1])
            ax0.errorbar(angle, rate * efficiency[channel] / (rate[0] + rate[-1]) * 2, yerr=(rateErr * efficiency[channel] / (rate[0] + rate[-1]) * 2), color=colorplot[1], fmt='s', mfc='white', \
                ms=8, ecolor=colorplot[-1], elinewidth=1, capsize=3, barsabove=True, zorder=1, label=('ch' + str(ich)))
            ax0.plot(simuAngle[channel], simuRate[channel] * efficiency[channel] / (simuRate[channel][0] + simuRate[channel][-1]) * 2, color=colorplot[1], label=('ch' + str(channel) + ' simulation'))
            if len(simuAngle[channel]) >= len(angle):
                angleL = simuAngle[channel]
                angleS = angle
                rateL = simuRate[channel] * efficiency[channel] / (simuRate[channel][0] + simuRate[channel][-1]) * 2
                rateS = rate / (rate[0] + rate[-1]) * 2 * efficiency[channel]
                simuL = True
            else:
                angleL = angle
                angleS = simuAngle[channel]
                rateL = rate / (rate[0] + rate[-1]) * 2 * efficiency[channel]
                rateS = simuRate[channel] * efficiency[channel] / (simuRate[channel][0] + simuRate[channel][-1]) * 2
                simuL = False
            jl = 0
            js = 0
            residual = []
            while jl < len(angleL) and js < len(angleS):
                if angleS[js] >= angleL[jl]:
                    jl += 1
                    if jl == len(angleL):
                        if angleL[jl - 1] < angleS[js] or js < len(angleS) - 1:
                            raise Exception('plotAngularResponce: angle range of experiment data and simulation data does not match')
                        else:
                            if simuL:
                                residual.append((rateL[jl - 1] - rateS[js]) / rateS[js] * 100)
                            else:
                                residual.append((rateS[js] - rateL[jl - 1]) / rateL[jl - 1] * 100)
                    continue
                else:
                    if jl == 0:
                        raise Exception('plotAngularResponce: angle range of experiment data and simulation data does not match')
                    elif angleS[js] == angleL[jl - 1]:
                        if simuL:
                            residual.append((rateL[jl - 1] - rateS[js]) / rateS[js] * 100)
                        else:
                            residual.append((rateS[js] - rateL[jl - 1]) / rateL[jl - 1] * 100)
                    else:
                        currRate = rateL[jl - 1] + (rateL[jl] - rateL[jl - 1]) / (angleL[jl] - angleL[jl - 1]) * (angleS[js] - angleL[jl - 1])
                        if simuL:
                            residual.append((currRate - rateS[js]) / rateS[js] * 100)
                        else:
                            residual.append((rateS[js] - currRate) / currRate * 100)
                    js += 1
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
                ax0.plot(angle, rate[ich] * efficiency[ich] / (rate[ich][0] + rate[ich][-1]) * 2, color=colorplot[ich])
                ax0.errorbar(angle, rate[ich] * efficiency[ich] / (rate[ich][0] + rate[ich][-1]) * 2, yerr=(rateErr[ich] * efficiency[ich] / (rate[ich][0] + rate[ich][-1]) * 2), color=colorplot[ich], fmt='s', \
                    mfc='white', ms=8, ecolor=ecolorplot[ich], elinewidth=1, capsize=3, barsabove=True, zorder=1, label=('ch' + str(ich)))
                ax0.plot(simuAngle[ich], simuRate[ich] * efficiency[ich] / (simuRate[ich][0] + simuRate[ich][-1]) * 2, color=colorplot[ich], label=('ch' + str(ich) + ' simulation'))
                if len(simuAngle[ich]) >= len(angle):
                    angleL = simuAngle[ich]
                    angleS = angle
                    rateL = simuRate[ich] * efficiency[ich] / (simuRate[ich][0] + simuRate[ich][-1]) * 2
                    rateS = rate[ich] / (rate[ich][0] + rate[ich][-1]) * 2 * efficiency[ich]
                    simuL = True
                else:
                    angleL = angle
                    angleS = simuAngle[ich]
                    rateL = rate[ich] / (rate[ich][0] + rate[ich][-1]) * 2 * efficiency[ich]
                    rateS = simuRate[ich] * efficiency[ich] / (simuRate[ich][0] + simuRate[ich][-1]) * 2
                    simuL = False
                jl = 0
                js = 0
                residual = []
                while jl < len(angleL) and js < len(angleS):
                    if angleS[js] >= angleL[jl]:
                        jl += 1
                        if jl == len(angleL):
                            if angleL[jl - 1] < angleS[js] or js < len(angleS) - 1:
                                raise Exception('plotAngularResponce: angle range of experiment data and simulation data does not match')
                            else:
                                if simuL:
                                    residual.append((rateL[jl - 1] - rateS[js]) / rateS[js] * 100)
                                else:
                                    residual.append((rateS[js] - rateL[jl - 1]) / rateL[jl - 1] * 100)
                        continue
                    else:
                        if jl == 0:
                            raise Exception('plotAngularResponce: angle range of experiment data and simulation data does not match')
                        elif angleS[js] == angleL[jl - 1]:
                            if simuL:
                                residual.append((rateL[jl - 1] - rateS[js]) / rateS[js] * 100)
                            else:
                                residual.append((rateS[js] - rateL[jl - 1]) / rateL[jl - 1] * 100)
                        else:
                            currRate = rateL[jl - 1] + (rateL[jl] - rateL[jl - 1]) / (angleL[jl] - angleL[jl - 1]) * (angleS[js] - angleL[jl - 1])
                            if simuL:
                                residual.append((currRate - rateS[js]) / rateS[js] * 100)
                            else:
                                residual.append((rateS[js] - currRate) / currRate * 100)
                        js += 1
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
    else:
        ax0.set_xlabel('Angle/$^{\circ}$')
    ax0.set_ylabel('Effective area/$cm^{2}$')
    ax0.set_title('Angular responce of ' + source)
    ax0.legend(loc=0)
    ax0.grid()
    if simulation:
        ax1.set_ylabel('residual/%')
        ax1.legend(loc=0)
        ax1.grid()
    plt.show()
    return