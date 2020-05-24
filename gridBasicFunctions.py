"""
Basic functions for Grid data processing and fit
v0.0.2 for Grid2 calibration result analysis by ydx and ghz
"""

import numpy as np
import lmfit
from scipy.odr import ODR, Model, Data, RealData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import struct
import crc16
import os
from copy import copy

#******************************************************************************************************************************************************
#*****************************************************Basic readout and fit functions*************************************************************
#******************************************************************************************************************************************************

#*****************************************************************************************************************************************************
#***************************************************************Auxiliariry functions**************************************************************
#*****************************************************************************************************************************************************

def isChannel(input):

    """
    Auxiliary function to determine whether the input is a channel number
    :param input: string
    :return: True if the input is a integer within [0-3], False if not
    """

    try:
        ch = int(input)
        if ch < 0 or ch > 3:
            raise Exception
    except:
        return False
    return True

def getSpectrum(amp, nbins = 65536, singlech = False):

    """
    Function for getting the spectrum of the amplitude data
    :param amp: amplitude of all 4 channels
    :param nbins: number of bins, 0 < nbins <= 65536
    :param singlech: boolean indicating whether the fit is for single channel
    :return: the corresponding spectrum of the input and the bin centers
    """

    if nbins <= 0 or nbins > 65536:
        raise Exception('getSpectrum: parameter \'nbins\' out of range [1-65536]')
    if not singlech:
        spectrum = []
        x = []
        for ich in range(4):
            specch, xch = np.histogram(amp[ich], bins=nbins, range=(0., 65536.))
            spectrum.append(specch)
            x.append((xch[:-1] + xch[1:]) / 2)
    else:
        spectrum, x = np.histogram(amp, bins=nbins, range=(0., 65536.))
        x = (x[:-1] + x[1:]) / 2
    return spectrum, x

#************************************************************************************************************************************************************
#**********************************************************************Basic I/O part*********************************************************************
#************************************************************************************************************************************************************

def getTime(filename):

    """
    Function for reading out the total time from the filename, CURRENTLY UNUSED
    :param filename: name of the input file
    :return: total time in seconds, or -1 if the filename cannot be parsed
    """

    time = 0
    lines = filename.split('_')
    for line in lines:
        if line.endswith('s'):
            try:
                time = int(line[:-1])
                return time
            except:
                pass
        elif line.endswith('m'):
            try:
                time = int(line[:-1]) * 60
                return time
            except:
                pass
        elif line.endswith('h'):
            try:
                time = int(line[:-1]) * 3600
                return time
            except:
                pass
    print('getTime: unable to parse filename ' + filename)
    return -1

def crcCheck(data, crc):

    """
    Function for checking crc
    :param data: data parts for calculating crc
    :param crc: crc parts in original data
    :return: True if crc check is correct, False if not
    """

    lineBytes = b''
    try:
        for x in data:
            lineBytes += struct.pack('B', x)
    except:
        return False
    crcCalculated = crc16.crc16xmodem(lineBytes)
    crcData = crc[0] * 256 + crc[1]
    if not crcCalculated == crcData:
        return False
    return True

def dataReadout(filename, isHex = False, isCi = 0, isScan = False, scanRange = [], rateStyle = '', newProgramme = False, timeCut = -1.0):
    
    """
    Function for reading out single Grid raw outout file
    :param filename: name of the output file, currently supporting only .txt files
    :param isHex: boolean indicating whether the input file is hexprint output
    :param isCi: int indicating whether the input file has CI part, with 0 for no CI, 1 for CI, 2 for multiple CI
    :param isScan: boolean indicating whether the input file has I-V scan part
    :param scanRange: list containing the scan range for multiple scans
    :param rateStyle: the style of calculating real count rate, '' for none, 's' for calculation with small data packs(512byte), \
'l' for calculation with large data packs(4096byte)
    :param newProgramme: boolean indicating whether the data comes from new hardware programme(6th ver.)
    :param timeCut: cut of time data in seconds, specially designed for temp-bias data with pid bias control(6th ver.)
    :return: all data extracted from the data file, including spectrums, SiPM&ADC temperatures, SiPM voltage&leak current,\
 uscount, correct live time, effective counts, missing counts, [CI data], [I-V scan data], all data in the form of ndarray
    """

    styleAvailable = ['s', 'p', '']
    if not rateStyle in styleAvailable:
        raise Exception('dataReadout: count rate calculation style \'' + rateStyle + '\' not available')

    dataBuffer = [] #to fix the problem of telemetry data crc check, checking the data before the current data
    lineBuffer = [] #also for the DAMN telemetry data crc check, checking the data after the current data
    bufferLen = 500
    lineLen = 500

    print('dataReadout: processing ' + filename)
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    
    amp = [] #after CI for data with CI
    ampCI = []
    tempSipm = []
    tempAdc = []
    vMon = []
    iMon = []
    bias = []
    uscount = [] #uscount in telemetry data
    uscountEvt = []
    uscountEvtCI = []
    vSet = [] #i-v scan data
    vScan = [] #i-v scan data
    iScan = [] #i-v scan data
    timeCorrect = []
    effectiveCount = []
    effectiveCountCI = []
    missingCount = []
    missingCountCI = []

    if isCi == 2:
        ranged = False
        if not len(scanRange) == 0:
            ranged = True
            lower = scanRange[0] - 1
            upper = scanRange[1] - 1

    for ich in range(4):
        amp.append([])
        ampCI.append([])
        tempSipm.append([])
        tempAdc.append([])
        vMon.append([])
        iMon.append([])
        bias.append([])
        uscountEvt.append([])
        uscountEvtCI.append([])
        vScan.append([])
        iScan.append([])

    bCi = True
    if isCi == 0:
        bCi = False
    if isHex: #in hexprint files there is no CI and I-V scan part
        isCi = 0
        isScan = False
    nScan = -1
    indexOut = 0 #count of events with channel index out of range[1-4]
    crcError = 0 #count of crc error data

    for line in lines:
        #I-V scan
        if isScan and 'Point' in line:
            lineList = line.split(',')
            scanData = []
            bScan = True #ensuring that corrupted data are disposed
            try:
                scanData.append(int(lineList[3]))
                scanList = lineList[4].split()
                for iscan in range(4):
                    scanData.append(float(scanList[2 * iscan]) / 4096.0 * 3.3 * 11.0)
                    scanData.append(float(scanList[1 + 2 * iscan]) / 4096.0 * 2.0 * 3.3)
            except:
                #print('Scan data error') #Debug
                bScan = False
            if bScan:
                vSet.append(scanData[0])
                for ich in range(4):
                    vScan[ich].append(scanData[1 + 2 * iscan])
                    iScan[ich].append([scanData[2 + 2 * iscan]])
            continue

        #begin and end of CI, also scan counts for multiple scans
        if isCi == 2 and 'Begin' in line:
            bCi = True
            nScan += 1
            if ranged and not (nScan >= lower and nScan <= upper):
                print('Skipping run #' + str(nScan + 1))
            else:
                print('Run #' + str(nScan + 1))
            uscount.append([])
            if not rateStyle == '':
                timeCorrect.append([])
            if newProgramme:
                effectiveCount.append([])
                effectiveCountCI.append([])
                missingCount.append([])
                missingCountCI.append([])
            for ich in range(4):
                amp[ich].append([])
                ampCI[ich].append([])
                tempSipm[ich].append([])
                tempAdc[ich].append([])
                vMon[ich].append([])
                iMon[ich].append([])
                bias[ich].append([])
                uscountEvt[ich].append([])
                uscountEvtCI[ich].append([])
            continue
        elif 'End' in line:
            bCi = False
            continue

        #Readout of single line
        lineList = line.split(' ')
        if len(lineList) > 502:
            if not isHex:
                #non-hexprint
                if isCi == 2 and ranged:
                    if not (nScan >= lower and nScan <= upper):
                        continue
                lineFloat = []
                try:
                    for linestr in lineList:
                        lineFloat.append(int(linestr))
                except:
                    pass

                for il in range(len(lineFloat)):
                    #Evevt data
                    if (lineFloat[il] == 170 and lineFloat[il + 1] == 187 and lineFloat[il + 2] == 204) and \
                        ((il + 502 <= len(lineFloat) and lineFloat[il + 499] == 221 and lineFloat[il + 500] == 238 and lineFloat[il + 501] == 255 and (not newProgramme)) \
                        or (il + 510 <= len(lineFloat) and lineFloat[il + 507] == 221 and lineFloat[il + 508] == 238 and lineFloat[il + 509] == 255 and newProgramme)):
                            timeEvtBegin = 0.0
                            timeEvtEnd = 0.0
                            timeIntv = []
                            if not newProgramme:
                                #crc check and crc buffer fill
                                if len(dataBuffer) >= bufferLen:
                                    del dataBuffer[0]
                                dataBuffer.append(lineFloat[496:504])
                                #check the previous telemetry data
                                for buf in lineBuffer:
                                    if buf[498:504] == lineFloat[498:504]:
                                        bufMatch = buf
                                        bufcrc = buf[496:498]
                                        bufMatch[496:498] = lineFloat[496:498]
                                        if not crcCheck(bufMatch[0:510], bufcrc):
                                            crcError += 1
                                            del lineBuffer[lineBuffer.index(buf)]
                                        else:
                                            if isCi == 2:
                                                for it in range(7):
                                                    uscount[nScan].append(float(sum([buf[15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                                    for ich in range(4):
                                                        tempSipm[ich][nScan].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                            if buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] > 2048 \
                                                            else tempSipm[ich][nScan].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it]) / 16.0)
                                                        tempAdc[ich][nScan].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                            if buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] > 2048 \
                                                            else tempAdc[ich][nScan].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it]) / 16.0)
                                                        vMon[ich][nScan].append(float(buf[39 + 2 * ich + 70 * it] * 256 + buf[40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                                        iMon[ich][nScan].append(float(buf[47 + 2 * ich + 70 * it] * 256 + buf[48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                                        bias[ich][nScan].append(vMon[ich][nScan][-1] - iMon[ich][nScan][-1])
                                            else:
                                                for it in range(7):
                                                    uscount.append(float(sum([buf[15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                                    for ich in range(4):
                                                        tempSipm[ich].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                            if buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] > 2048 \
                                                            else tempSipm[ich].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it]) / 16.0)
                                                        tempAdc[ich].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                            if buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] > 2048 \
                                                            else tempAdc[ich].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it]) / 16.0)
                                                        vMon[ich].append(float(buf[39 + 2 * ich + 70 * it] * 256 + buf[40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                                        iMon[ich].append(float(buf[47 + 2 * ich + 70 * it] * 256 + buf[48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                                        bias[ich].append(vMon[ich][-1] - iMon[ich][-1])
                                            del lineBuffer[lineBuffer.index(buf)]
                                        break
                                if not crcCheck(lineFloat[:502], lineFloat[502:504]):
                                    crcError += 1
                                    continue
                            else:
                                if not crcCheck(lineFloat[:510], lineFloat[510:512]):
                                    crcError += 1
                                    continue
                            ch = lineFloat[il + 3]
                            if newProgramme:
                                ch += 1
                            if isCi == 2:
                                if ch > 0 and ch < 5:
                                    if bCi:
                                        ampCI[ch - 1][nScan].append(lineFloat[il + 12] * 256 + lineFloat[il + 13])
                                        uscountEvtCI[ch - 1][nScan].append(float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                    else:
                                        amp[ch - 1][nScan].append(lineFloat[il + 12] * 256 + lineFloat[il + 13])
                                        uscountEvt[ch - 1][nScan].append(float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                else:
                                    indexOut += 1
                            else:
                                if ch > 0 and ch < 5:
                                    if bCi:
                                        ampCI[ch - 1].append(lineFloat[il + 12] * 256 + lineFloat[il + 13])
                                        uscountEvtCI[ch - 1].append(float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                    else:
                                        amp[ch - 1].append(lineFloat[il + 12] * 256 + lineFloat[il + 13])
                                        uscountEvt[ch - 1].append(float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                else:
                                    indexOut += 1
                            if rateStyle == 's' and not bCi:
                                timeIntv.append(float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                            elif rateStyle == 'p' and not bCi:
                                timeEvtBegin = float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6
                            for ie in range(43):
                                ch = lineFloat[il + 26 + 11 * ie]
                                if newProgramme:
                                    ch += 1
                                if isCi == 2:
                                    if ch > 0 and ch < 5:
                                        if bCi:
                                            ampCI[ch - 1][nScan].append(lineFloat[il + 35 + 11 * ie] * 256 + lineFloat[il + 36 + 11 * ie])
                                            uscountEvtCI[ch - 1][nScan].append(float(sum([lineFloat[il + 27 + 11 * ie + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                        else:
                                            amp[ch - 1][nScan].append(lineFloat[il + 35 + 11 * ie] * 256 + lineFloat[il + 36 + 11 * ie])
                                            uscountEvt[ch - 1][nScan].append(float(sum([lineFloat[il + 27 + 11 * ie + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                    else:
                                        indexOut += 1
                                else:
                                    if ch > 0 and ch < 5:
                                        if bCi:
                                            ampCI[ch - 1].append(lineFloat[il + 35 + 11 * ie] * 256 + lineFloat[il + 36 + 11 * ie])
                                            uscountEvtCI[ch - 1].append(float(sum([lineFloat[il + 27 + 11 * ie + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                        else:
                                            amp[ch - 1].append(lineFloat[il + 35 + 11 * ie] * 256 + lineFloat[il + 36 + 11 * ie])
                                            uscountEvt[ch - 1].append(float(sum([lineFloat[il + 27 + 11 * ie + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                    else:
                                        indexOut += 1
                                if rateStyle == 's' and not bCi:
                                    timeIntv.append(float(sum([lineFloat[il + 27 + 11 * ie + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                            if rateStyle == 's' and not bCi:
                                timeIntv = np.array(timeIntv)
                                if isCi == 2:
                                    timeCorrect[nScan] += list(timeIntv[1:] - timeIntv[:-1])
                                else:
                                    timeCorrect += list(timeIntv[1:] - timeIntv[:-1])
                            elif rateStyle == 'p' and not bCi:
                                timeEvtEnd = float(sum([lineFloat[il + 27 + 11 * 42 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6
                                if isCi == 2:
                                    timeCorrect[nScan].append(timeEvtEnd - timeEvtBegin)
                                else:
                                    timeCorrect.append(timeEvtEnd - timeEvtBegin)
                            if newProgramme:
                                if isCi == 2:
                                    if bCi:
                                        effectiveCountCI[nScan].append(int(sum([lineFloat[il + 499 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                                        missingCountCI[nScan].append(int(sum([lineFloat[il + 503 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                                    else:
                                        effectiveCount[nScan].append(int(sum([lineFloat[il + 499 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                                        missingCount[nScan].append(int(sum([lineFloat[il + 503 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                                else:
                                    if bCi:
                                        effectiveCountCI.append(int(sum([lineFloat[il + 499 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                                        missingCountCI.append(int(sum([lineFloat[il + 503 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                                    else:
                                        effectiveCount.append(int(sum([lineFloat[il + 499 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                                        missingCount.append(int(sum([lineFloat[il + 503 + ic] * 256 ** (3 - ic) for ic in range(4)])))

                    #Telemetry data
                    elif (lineFloat[il] == 1 and lineFloat[il + 1] == 35 and lineFloat[il + 2] == 69 and il + 502 <= len(lineFloat) and lineFloat[il + 493] == 103 and lineFloat[il + 494] == 137 \
                        and lineFloat[il + 495] == 16 and (not newProgramme)) or (lineFloat[il] == 18 and lineFloat[il + 1] == 52 and lineFloat[il + 2] == 86 and il + 502 <= len(lineFloat) and \
                        lineFloat[il + 493] == 120 and lineFloat[il + 494] == 154 and lineFloat[il + 495] == 188 and newProgramme):
                            if not newProgramme:
                                #check the data before the current data
                                crcCorrect = []
                                for databuf in dataBuffer:
                                    if databuf[2:] == lineFloat[498:504]:
                                        crcCorrect = databuf[:2]
                                        break
                                #check the previous telemetry data
                                for buf in lineBuffer:
                                    if buf[498:504] == lineFloat[498:504]:
                                        bufMatch = buf
                                        bufcrc = buf[496:498]
                                        bufMatch[496:498] = lineFloat[496:498]
                                        if not crcCheck(bufMatch[0:510], bufcrc):
                                            crcError += 1
                                            del lineBuffer[lineBuffer.index(buf)]
                                        else:
                                            if isCi == 2:
                                                for it in range(7):
                                                    uscount[nScan].append(float(sum([buf[15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                                    for ich in range(4):
                                                        tempSipm[ich][nScan].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                            if buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] > 2048 \
                                                            else tempSipm[ich][nScan].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it]) / 16.0)
                                                        tempAdc[ich][nScan].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                            if buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] > 2048 \
                                                            else tempAdc[ich][nScan].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it]) / 16.0)
                                                        vMon[ich][nScan].append(float(buf[39 + 2 * ich + 70 * it] * 256 + buf[40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                                        iMon[ich][nScan].append(float(buf[47 + 2 * ich + 70 * it] * 256 + buf[48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                                        bias[ich][nScan].append(vMon[ich][nScan][-1] - iMon[ich][nScan][-1])
                                            else:
                                                for it in range(7):
                                                    uscount.append(float(sum([buf[15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                                    for ich in range(4):
                                                        tempSipm[ich].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                            if buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] > 2048 \
                                                            else tempSipm[ich].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it]) / 16.0)
                                                        tempAdc[ich].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                            if buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] > 2048 \
                                                            else tempAdc[ich].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it]) / 16.0)
                                                        vMon[ich].append(float(buf[39 + 2 * ich + 70 * it] * 256 + buf[40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                                        iMon[ich].append(float(buf[47 + 2 * ich + 70 * it] * 256 + buf[48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                                        bias[ich].append(vMon[ich][-1] - iMon[ich][-1])
                                            del lineBuffer[lineBuffer.index(buf)]
                                        break
                                #crc check and crc buffer fill
                                if len(dataBuffer) >= bufferLen:
                                    del dataBuffer[0]
                                dataBuffer.append(lineFloat[496:504])
                                temp = lineFloat[496:498]
                                if not len(crcCorrect) == 0:
                                    lineFloat[496:498] = crcCorrect
                                else:
                                    if len(lineBuffer) >= lineLen:
                                        del lineBuffer[0]
                                        crcError += 1
                                    lineBuffer.append(lineFloat)
                                if not crcCheck(lineFloat[0:510], temp):
                                    crcError += 1
                                    continue
                            else:
                                if not crcCheck(lineFloat[0:496], lineFloat[496:498]):
                                    crcError += 1
                                    continue
                            for it in range(7):
                                if isCi == 2:
                                    uscount[nScan].append(float(sum([lineFloat[il + 15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                    for ich in range(4):
                                        tempSipm[ich][nScan].append(float(lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                            if lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it] > 2048 \
                                            else tempSipm[ich][nScan].append(float(lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it]) / 16.0)
                                        tempAdc[ich][nScan].append(float(lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                            if lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it] > 2048 \
                                            else tempAdc[ich][nScan].append(float(lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it]) / 16.0)
                                        vMon[ich][nScan].append(float(lineFloat[il + 39 + 2 * ich + 70 * it] * 256 + lineFloat[il + 40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                        iMon[ich][nScan].append(float(lineFloat[il + 47 + 2 * ich + 70 * it] * 256 + lineFloat[il + 48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                        bias[ich][nScan].append(vMon[ich][nScan][-1] - iMon[ich][nScan][-1] * 2.0)
                                else:
                                    uscount.append(float(sum([lineFloat[il + 15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                    for ich in range(4):
                                        tempSipm[ich].append(float(lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                            if lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it] > 2048 \
                                            else tempSipm[ich].append(float(lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it]) / 16.0)
                                        tempAdc[ich].append(float(lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                            if lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it] > 2048 \
                                            else tempAdc[ich].append(float(lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it]) / 16.0)
                                        vMon[ich].append(float(lineFloat[il + 39 + 2 * ich + 70 * it] * 256 + lineFloat[il + 40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                        iMon[ich].append(float(lineFloat[il + 47 + 2 * ich + 70 * it] * 256 + lineFloat[il + 48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                        bias[ich].append(vMon[ich][-1] - iMon[ich][-1] * 2.0)

            else: #hexprint
                lineFloat = []
                try:
                    for linestr in lineList:
                        lineFloat.append(int(linestr, 16))
                except:
                    pass
                il = 0
                while il + 502 <= len(lineFloat):
                    #Event data
                    if (lineFloat[il] == 170 and lineFloat[il + 1] == 187 and lineFloat[il + 2] == 204) and \
                        ((il + 502 <= len(lineFloat) and lineFloat[il + 499] == 221 and lineFloat[il + 500] == 238 and lineFloat[il + 501] == 255 and (not newProgramme)) \
                        or (il + 510 <= len(lineFloat) and lineFloat[il + 507] == 221 and lineFloat[il + 508] == 238 and lineFloat[il + 509] == 255 and newProgramme)):
                            timeEvtBegin = 0.0
                            timeEvtEnd = 0.0
                            timeIntv = []
                            if not newProgramme:
                                #crc check and crc buffer fill
                                if len(dataBuffer) >= bufferLen:
                                    del dataBuffer[0]
                                dataBuffer.append(lineFloat[il + 496:il + 504])
                                #check the previous telemetry data
                                for buf in lineBuffer:
                                    if buf[498:504] == lineFloat[il + 498:il + 504]:
                                        bufcrc = buf[496:498]
                                        buf[496:498] = lineFloat[il + 496:il + 498]
                                        if not crcCheck(buf[0:510], bufcrc):
                                            crcError += 1
                                            del lineBuffer[lineBuffer.index(buf)]
                                        else:
                                            for it in range(7):
                                                uscount.append(float(sum([buf[15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                                for ich in range(4):
                                                    tempSipm[ich].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                        if buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] > 2048 \
                                                        else tempSipm[ich].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it]) / 16.0)
                                                    tempAdc[ich].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                        if buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] > 2048 \
                                                        else tempAdc[ich].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it]) / 16.0)
                                                    vMon[ich].append(float(buf[39 + 2 * ich + 70 * it] * 256 + buf[40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                                    iMon[ich].append(float(buf[47 + 2 * ich + 70 * it] * 256 + buf[48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                                    bias[ich].append(vMon[ich][-1] - iMon[ich][-1])
                                            del lineBuffer[lineBuffer.index(buf)]
                                        break
                                if not crcCheck(lineFloat[il:il + 502], lineFloat[il + 502:il + 504]):
                                    crcError += 1
                                    il += 512
                                    continue
                            else:
                                if not crcCheck(lineFloat[il:il + 510], lineFloat[il + 510:il + 512]):
                                    crcError += 1
                                    il += 512
                                    continue
                            ch = lineFloat[il + 3]
                            if newProgramme:
                                ch += 1
                            if ch > 0 and ch < 5:
                                amp[ch - 1].append(lineFloat[il + 12] * 256 + lineFloat[il + 13])
                                uscountEvt[ch - 1].append(float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                            else:
                                indexOut += 1
                            if rateStyle == 's':
                                timeIntv.append(float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                            elif rateStyle == 'p':
                                timeEvtBegin = float(sum([lineFloat[il + 4 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6
                            for ie in range(43):
                                ch = lineFloat[il + 26 + 11 * ie]
                                if newProgramme:
                                    ch += 1
                                if ch > 0 and ch < 5:
                                    amp[ch - 1].append(lineFloat[il + 35 + 11 * ie] * 256 + lineFloat[il + 36 + 11 * ie])
                                    uscountEvt[ch - 1].append(float(sum([lineFloat[il + 27 + 11 * ie + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                else:
                                    indexOut += 1
                                if rateStyle == 's':
                                    timeIntv.append(float(sum([lineFloat[il + 27 + 11 * ie + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                            if rateStyle == 's':
                                timeIntv = np.array(timeIntv)
                                timeCorrect += list(timeIntv[1:] - timeIntv[:-1])
                            elif rateStyle == 'p':
                                timeEvtEnd = float(sum([lineFloat[il + 27 + 11 * 42 + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6
                                timeCorrect.append(timeEvtEnd - timeEvtBegin)
                            if newProgramme:
                                effectiveCount.append(int(sum([lineFloat[il + 499 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                                missingCount.append(int(sum([lineFloat[il + 503 + ic] * 256 ** (3 - ic) for ic in range(4)])))
                            il += 511
                    
                    #Telemetry data
                    elif (lineFloat[il] == 1 and lineFloat[il + 1] == 35 and lineFloat[il + 2] == 69 and il + 502 <= len(lineFloat) and lineFloat[il + 493] == 103 and lineFloat[il + 494] == 137 \
                        and lineFloat[il + 495] == 16 and (not newProgramme)) or (lineFloat[il] == 18 and lineFloat[il + 1] == 52 and lineFloat[il + 2] == 86 and il + 502 <= len(lineFloat) and \
                        lineFloat[il + 493] == 120 and lineFloat[il + 494] == 154 and lineFloat[il + 495] == 188 and newProgramme):
                            if not newProgramme:
                                #check the data before the current data
                                crcCorrect = []
                                for i in dataBuffer:
                                    if i[2:] == lineFloat[il + 498:il + 504]:
                                        crcCorrect = i[:2]
                                        break
                                #check the previous telemetry data
                                for buf in lineBuffer:
                                    if buf[498:504] == lineFloat[il + 498:il + 504]:
                                        bufcrc = buf[496:498]
                                        buf[496:498] = lineFloat[il + 496:il + 498]
                                        if not crcCheck(buf[0:510], bufcrc):
                                            crcError += 1
                                            del lineBuffer[lineBuffer.index(buf)]
                                        else:
                                            for it in range(7):
                                                uscount.append(float(sum([buf[15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                                for ich in range(4):
                                                    tempSipm[ich].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                        if buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it] > 2048 \
                                                        else tempSipm[ich].append(float(buf[23 + 2 * ich + 70 * it] * 256 + buf[24 + 2 * ich + 70 * it]) / 16.0)
                                                    tempAdc[ich].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                                        if buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it] > 2048 \
                                                        else tempAdc[ich].append(float(buf[31 + 2 * ich + 70 * it] * 256 + buf[32 + 2 * ich + 70 * it]) / 16.0)
                                                    vMon[ich].append(float(buf[39 + 2 * ich + 70 * it] * 256 + buf[40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                                    iMon[ich].append(float(buf[47 + 2 * ich + 70 * it] * 256 + buf[48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                                    bias[ich].append(vMon[ich][-1] - iMon[ich][-1])
                                            del lineBuffer[lineBuffer.index(buf)]
                                        break
                                #crc check and crc buffer fill
                                if len(dataBuffer) >= bufferLen:
                                    del dataBuffer[0]
                                dataBuffer.append(lineFloat[il + 496:il + 504])
                                temp = lineFloat[il + 496:il + 498]
                                if not len(crcCorrect) == 0:
                                    lineFloat[il + 496:il + 498] = crcCorrect
                                else:
                                    if len(lineBuffer) >= lineLen:
                                        del lineBuffer[0]
                                        crcError += 1
                                    lineBuffer.append(lineFloat[il:il + 512])
                                if not crcCheck(lineFloat[il:il + 510], temp):
                                    crcError += 1
                                    il += 512
                                    continue
                            else:
                                if not crcCheck(lineFloat[il:il + 496], lineFloat[il + 496:il + 498]):
                                    crcError += 1
                                    il += 512
                                    continue
                            for it in range(7):
                                uscount.append(float(sum([lineFloat[il + 15 + 70 * it + ius] * 256 ** (7 - ius) for ius in range(8)])) / 24.05e6)
                                for ich in range(4):
                                    tempSipm[ich].append(float(lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                        if lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it] > 2048 \
                                        else tempSipm[ich].append(float(lineFloat[il + 23 + 2 * ich + 70 * it] * 256 + lineFloat[il + 24 + 2 * ich + 70 * it]) / 16.0)
                                    tempAdc[ich].append(float(lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it] - 4096) / 16.0) \
                                        if lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it] > 2048 \
                                        else tempAdc[ich].append(float(lineFloat[il + 31 + 2 * ich + 70 * it] * 256 + lineFloat[il + 32 + 2 * ich + 70 * it]) / 16.0)
                                    vMon[ich].append(float(lineFloat[il + 39 + 2 * ich + 70 * it] * 256 + lineFloat[il + 40 + 2 * ich + 70 * it]) / 4096.0 * 3.3 * 11.0)
                                    iMon[ich].append(float(lineFloat[il + 47 + 2 * ich + 70 * it] * 256 + lineFloat[il + 48 + 2 * ich + 70 * it]) / 4096.0 * 3.3)
                                    bias[ich].append(vMon[ich][-1] - iMon[ich][-1])
                            il += 511

                    il += 1

    print(str(crcError + len(lineBuffer)) + ' data packs with crc error')
    print(str(indexOut) + ' events with channel out of bound[0-3]')

    #Transforming the data to ndarray(np.array)
    amp = np.array(amp)
    tempSipm = np.array(tempSipm)
    tempAdc = np.array(tempAdc)
    vMon = np.array(vMon)
    iMon = np.array(iMon)
    iMon = iMon / 2.0
    bias = np.array(bias)
    uscount = np.array(uscount)
    uscountEvt = np.array(uscountEvt)
    timeCorrect = np.array(timeCorrect)
    timeCorrect = timeCorrect[(timeCorrect > 0) * (timeCorrect < 20 * np.std(timeCorrect))]
    effectiveCount = np.array(effectiveCount)
    missingCount = np.array(missingCount)
    
    #Time cut
    if isCi == 2:
        for isc in range(len(uscount)):
            q1 = uscount[isc] > timeCut
            uscount[isc] = uscount[isc][q1]
            tempSipm[:, isc] = tempSipm[:, isc, q1]
            tempAdc[:, isc] = tempAdc[:, isc, q1]
            vMon[:, isc] = vMon[:, isc, q1]
            iMon[:, isc] = iMon[:, isc, q1]
            bias[:] = bias[:, isc, q1]
            for ich in range(4):
                q2 = uscountEvt[ich, isc] > timeCut
                uscountEvt[ich, isc] = uscountEvt[ich, isc][q2]
                amp[ich, isc] = amp[ich, isc][q2]
    else:
        q1 = uscount > timeCut
        uscount = uscount[q1]
        tempSipm = tempSipm[:, q1]
        tempAdc = tempAdc[:, q1]
        vMon = vMon[:, q1]
        iMon = iMon[:, q1]
        bias = bias[:, q1]
        for ich in range(4):
            q2 = np.array(uscountEvt[ich]) > timeCut
            uscountEvt[ich] = np.array(uscountEvt[ich])[q2]
            amp[ich] = np.array(amp[ich])[q2]
            
    #Output
    print('Data readout of ' + filename + ' complete')
    if isScan:
        vSet = np.array(vSet)
        vScan = np.array(vScan)
        iScan = np.array(iScan)
        if isCi == 0:
            return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount, vSet, vScan, iScan
        else:
            ampCI = np.array(ampCI)
            uscountEvtCI = np.array(uscountEvtCI)
            effectiveCountCI = np.array(effectiveCountCI)
            missingCountCI = np.array(missingCountCI)
            return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount, ampCI, uscountEvtCI, \
                effectiveCountCI, missingCountCI, vSet, vScan, iScan
    else:
        if isCi == 0:
            return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount
        else:
            ampCI = np.array(ampCI)
            uscountEvtCI = np.array(uscountEvtCI)
            effectiveCountCI = np.array(effectiveCountCI)
            missingCountCI = np.array(missingCountCI)
            return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount, ampCI, uscountEvtCI, \
                effectiveCountCI, missingCountCI

def HPGeDataReadout(filename):
    """
    Function for reading out single HPGe raw outout file
    :param filename: name of the input file, currently supporting only .txt files
    :return: data extracted from the data file, including spectrums and time, both in the form of ndarray
    """
    
    with open(filename) as f:
        # to remove \r\n using the following line
        lines = [line.rstrip() for line in f]
    
    cts = []
    for line in lines:
        cts.append(int(line))
    cts = np.array(cts)
    time = float(cts[0])
    cts[0], cts[1]=0, 0
        
    return time, cts

def deleteEmptyRun(amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount, ampCI, uscountEvtCI, \
    effectiveCountCI, missingCountCI, scanRange, rateStyle = '', newProgramme = False):

    """
    Auxiliary function to delete empty sublists for ranged multiple scan data
    Too lazy to write the specific information of all these input, so just use it as given in the main programme please
    :param amp: ADC amplitude data
    :param tempSipm: SiPM temperature data
    :param tempAdc: ADC temperature data
    :param vMon: monitored voltage
    :param iMon: monitored current
    :param bias: bias voltage
    :param uscount: uscount data
    :param uscountEvt: event uscount data
    :param timeCorrect: correct live time calculated when reading data
    :param effectiveCount: effective count data
    :param missingCount: missing count data
    :param ampCI: CI ADC amplitude data
    :param uscountEvtCI: CI event uscount data
    :param effectiveCountCI: CI effective count data
    :param missingCountCI: CI missing count data
    :param scanRange: list containing the scan range for multiple scans
    :param rateStyle: the style of calculating real count rate, '' for none, 's' for calculation with small data packs(512byte), \
'l' for calculation with large data packs(4096byte)
    :param newProgramme: boolean indicating whether the data comes from new hardware programme(6th ver.)
    :return: all the input except scanRange in excatly the same order, and the corresponding scan numbers
    """
    
    if len(scanRange) == 0:
        lower = 0
        upper = len(uscount) - 1
    else:
        lower = scanRange[0] - 1
        upper = scanRange[1] - 1
    nScan = len(uscount)

    amp = list(amp)
    tempSipm = list(tempSipm)
    tempAdc = list(tempAdc)
    vMon = list(vMon)
    iMon = list(iMon)
    bias = list(bias)
    uscount = list(uscount)
    uscountEvt = list(uscountEvt)
    if not rateStyle == '':
        timeCorrect = list(timeCorrect)
    ampCI = list(ampCI)
    uscountEvtCI = list(uscountEvtCI)
    if newProgramme:
        effectiveCount = list(effectiveCount)
        effectiveCountCI = list(effectiveCountCI)
        missingCount = list(missingCount)
        missingCountCI = list(missingCountCI)
    for isc in range(nScan):
        uscount[isc] = list(uscount[isc])
        if not rateStyle == '':
            timeCorrect[isc] = list(timeCorrect[isc])
        if newProgramme:
            effectiveCount[isc] = list(effectiveCount[isc])
            effectiveCountCI[isc] = list(effectiveCountCI[isc])
            missingCount[isc] = list(missingCount[isc])
            missingCountCI[isc] = list(missingCountCI[isc])
    for ich in range(4):
        amp[ich] = list(amp[ich])
        tempSipm[ich] = list(tempSipm[ich])
        tempAdc[ich] = list(tempAdc[ich])
        vMon[ich] = list(vMon[ich])
        iMon[ich] = list(iMon[ich])
        bias[ich] = list(bias[ich])
        uscountEvt[ich] = list(uscountEvt[ich])
        ampCI[ich] = list(ampCI[ich])
        uscountEvtCI[ich] = list(uscountEvtCI[ich])
        for isc in range(nScan):
            amp[ich][isc] = list(amp[ich][isc])
            tempSipm[ich][isc] = list(tempSipm[ich][isc])
            tempAdc[ich][isc] = list(tempAdc[ich][isc])
            vMon[ich][isc] = list(vMon[ich][isc])
            iMon[ich][isc] = list(iMon[ich][isc])
            bias[ich][isc] = list(bias[ich][isc])
            uscountEvt[ich][isc] = list(uscountEvt[ich][isc])
            ampCI[ich][isc] = list(ampCI[ich][isc])
            uscountEvtCI[ich][isc] = list(uscountEvtCI[ich][isc])

    for isc in range(lower):
        del uscount[0]
        if not rateStyle == '':
            del timeCorrect[0]
        if newProgramme:
            del effectiveCount[0], effectiveCountCI[0], missingCount[0], missingCountCI[0]
        for ich in range(4):
            del amp[ich][0], tempSipm[ich][0], tempAdc[ich][0], vMon[ich][0], iMon[ich][0], bias[ich][0], uscountEvt[ich][0], ampCI[ich][0], uscountEvtCI[ich][0]
    for isc in range(nScan - upper - 1):
        del uscount[-1]
        if not rateStyle == '':
            del timeCorrect[-1]
        if newProgramme:
            del effectiveCount[-1], effectiveCountCI[-1], missingCount[-1], missingCountCI[-1]
        for ich in range(4):
            del amp[ich][-1], tempSipm[ich][-1], tempAdc[ich][-1], vMon[ich][-1], iMon[ich][-1], bias[ich][-1], uscountEvt[ich][-1], ampCI[ich][-1], uscountEvtCI[ich][-1]
            
    amp = np.array(amp)
    tempSipm = np.array(tempSipm)
    tempAdc = np.array(tempAdc)
    vMon = np.array(vMon)
    iMon = np.array(iMon)
    bias = np.array(bias)
    uscount = np.array(uscount)
    uscountEvt = np.array(uscountEvt)
    if not rateStyle == '':
        timeCorrect = np.array(timeCorrect)
    ampCI = np.array(ampCI)
    uscountEvtCI = np.array(uscountEvtCI)
    if newProgramme:
        effectiveCount = np.array(effectiveCount)
        effectiveCountCI = np.array(effectiveCountCI)
        missingCount = np.array(missingCount)
        missingCountCI = np.array(missingCountCI)

    #Thanks to ndarray, ALL THESE matter is needed to delete a SINGLE sublist!
    return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount, ampCI, uscountEvtCI, \
        effectiveCountCI, missingCountCI

def fileOutput(filename, isCi = 0, isScan = False, scanRange = [], *data):

    """
    Function for writing designated readout data to files(text files in the form of '.txt')
    :param filename: name of the output file, currently supporting only .txt files
    :param isCi: int indicating whether the input file has CI part, with 0 for no CI, 1 for CI, 2 for multiple CI
    :param isScan: boolean indicating whether the input file has I-V scan part
    :param scanRange: list containing the scan range for multiple scans
    :param *data: tuple containing all the data to be written to the output files, in the order of\
    amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, [ampCI, uscountEvtCI], [vSet, vScan, iScan]
    :return: nope, nothing
    Output file naming rules(following the previous naming rules):
    Spectrum: out_[channel]_[[scancount]_][filename]
    ADC amplitude: out_ch[channel]_[[scancount]_][filename]
    SiPM temperature: out_t_[[scancount]_][filename]
    ADC temperature: out_ta_[[scancount]_][filename]
    Monitored voltage: v_[[scancount]_][filename]
    Monitored current: i_[[scancount]_][filename]
    Bias: bias_[[scancount]_][filename]
    Time data: out_time_[[scancount]_][filename]
    Event time data: out_timeevt_[[scancount]_][filename]
    Correct live time: livetime_[[scancount]_][filename]
    Effective count data: eff_[[scancount]_][filename]
    Missing count data: miss_[[scancount]_][filename]
    CI data: (output when isCi != 0)
    CI Spectrum: ci_[channel]_[[scancount]_][filename]
    CI ADC amplitude: ci_ch[channel]_[[scancount]_][filename]
    CI event time data: ci_timeevt_[[scancount]_][filename]
    CI effective count data: ci_eff_[[scancount]_][filename]
    CI missing count data: ci_miss_[[scancount]_][filename]
    Scan data: (output when isScan = True)
    Set voltage for voltage scan: vset_[filename]
    Scan voltage for voltage scan: vscan_[filename]
    Scan current for voltage scan: iscan_[filename]
    """

    if len(data) < 8:
        print('fileOutput: too few data given')
        return
    ranged = False
    if not len(scanRange) == 0:
        ranged = True
        lower = scanRange[0] - 1
        upper = scanRange[1] - 1
    print('fileOutput: writing output files of ' + filename)
    #Multiple scans
    if isCi == 2:
        nscan = len(data[6])
        for isc in range(nscan):
            if ranged and not (isc >= lower and isc <= upper):
                print('Skipping run #' + str(isc + 1))
                continue
            else:
                print('Run #' + str(isc + 1))
            #ADC amplitude
            for ich in range(4):
                try:
                    with open('out_ch' + str(ich + 1) + '_' + str(isc + 1) + '_' + filename, 'w') as foutamp:
                        for j in range(len(data[0][ich][isc])):
                            foutamp.write(str(data[0][ich][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing ADC amplitude file')
            
            #SiPM temperature
            try:
                with open('out_t_' + str(isc + 1) + '_' + filename, 'w') as fouttemp:
                    for j in range(len(data[1][0][isc])):
                        for ich in range(4):
                            fouttemp.write(str(data[1][ich][isc][j]) + '\t')
                        fouttemp.write('\n')
            except:
                print('fileOutput: Error writing SiPM temperature')
            
            #ADC temperature
            try:
                with open('out_ta_' + str(isc + 1) + '_' + filename, 'w') as fouttempadc:
                    for j in range(len(data[2][0][isc])):
                        for ich in range(4):
                            fouttempadc.write(str(data[2][ich][isc][j]) + '\t')
                        fouttempadc.write('\n')
            except:
                print('fileOutput: Error writing ADC temperature')
            
            #Monitored voltage
            try:
                with open('v_' + str(isc + 1) + '_' + filename, 'w') as foutvMon:
                    for j in range(len(data[3][0][isc])):
                        for ich in range(4):
                            foutvMon.write(str(data[3][ich][isc][j]) + '\t')
                        foutvMon.write('\n')
            except:
                print('fileOutput: Error writing monitored voltage')
            
            #Monitored current
            try:
                with open('i_' + str(isc + 1) + '_' + filename, 'w') as foutiMon:
                    for j in range(len(data[4][0][isc])):
                        for ich in range(4):
                            foutiMon.write(str(data[4][ich][isc][j]) + '\t')
                        foutiMon.write('\n')
            except:
                print('fileOutput: Error writing monitored current')
            
            #Bias
            try:
                with open('bias_' + str(isc + 1) + '_' + filename, 'w') as foutbias:
                    for j in range(len(data[5][0][isc])):
                        for ich in range(4):
                            foutbias.write(str(data[5][ich][isc][j]) + '\t')
                        foutbias.write('\n')
            except:
                print('fileOutput: Error writing bias')
            
            #Time data
            try:
                with open('out_time_' + str(isc + 1) + '_' + filename, 'w') as fouttime:
                    for j in range(len(data[6][isc])):
                        fouttime.write(str(data[6][isc][j]) + '\n')
            except:
                print('fileOutput: Error writing uscount')

            #Event time data
            for ich in range(4):
                try:
                    with open('out_timeevt_' + str(ich + 1) + '_' + str(isc + 1) + '_' + filename, 'w') as fouttimeevt:
                        for j in range(len(data[7][ich][isc])):
                            fouttimeevt.write(str(data[7][ich][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing event uscount')

            #Correct live time
            if not len(data[8]) == 0 and not len(data[8][0]) == 0:
                try:
                    with open('livetime_' + str(isc + 1) + '_' + filename, 'w') as fouttimecorr:
                        for j in range(len(data[8][isc])):
                            fouttimecorr.write(str(data[8][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing correct live time')

            #Effective count
            if not len(data[9]) == 0 and not len(data[9][0]) == 0:
                try:
                    with open('eff_' + str(isc + 1) + '_' + filename, 'w') as fouteff:
                        for j in range(len(data[9][isc])):
                            fouteff.write(str(data[9][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing effective count')
                    
            #Missing count
            if not len(data[10]) == 0 and not len(data[10][0]) == 0:
                try:
                    with open('miss_' + str(isc + 1) + '_' + filename, 'w') as foutmiss:
                        for j in range(len(data[10][isc])):
                            foutmiss.write(str(data[10][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing missing count')

            #CI ADC amplitude
            for ich in range(4):
                try:
                    with open('ci_ch' + str(ich + 1) + '_' + str(isc + 1) + '_' + filename, 'w') as foutampCI:
                        for j in range(len(data[11][ich][isc])):
                            foutampCI.write(str(data[11][ich][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing CI ADC amplitude file')
            
            #CI event time data
            for ich in range(4):
                try:
                    with open('ci_timeevt_' + str(ich + 1) + '_' + str(isc + 1) + '_' + filename, 'w') as fouttimeevtCI:
                        for j in range(len(data[12][ich][isc])):
                            fouttimeevtCI.write(str(data[12][ich][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing CI event uscount')
                    
            #CI effective count
            if not len(data[13]) == 0 and not len(data[13][0]) == 0:
                try:
                    with open('ci_eff_' + str(isc + 1) + '_' + filename, 'w') as fouteffCI:
                        for j in range(len(data[13][isc])):
                            fouteffCI.write(str(data[13][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing CI effective count')
                    
            #CI missing count
            if not len(data[14]) == 0 and not len(data[14][0]) == 0:
                try:
                    with open('ci_miss_' + str(isc + 1) + '_' + filename, 'w') as foutmissCI:
                        for j in range(len(data[14][isc])):
                            foutmissCI.write(str(data[14][isc][j]) + '\n')
                except:
                    print('fileOutput: Error writing CI missing count')

        ind = 15

    #Single scan
    else:
        #ADC amplitude
        for ich in range(4):
            try:
                with open('out_ch' + str(ich + 1) + '_' + filename, 'w') as foutamp:
                    for j in range(len(data[0][ich])):
                        foutamp.write(str(data[0][ich][j]) + '\n')
            except:
                print('fileOutput: Error writing ADC amplitude')
            
         #SiPM temperature
        try:
            with open('out_t_' + filename, 'w') as fouttemp:
                for j in range(len(data[1][0])):
                    for ich in range(4):
                        fouttemp.write(str(data[1][ich][j]) + '\t')
                    fouttemp.write('\n')
        except:
            print('fileOutput: Error writing SiPM temperature')
            
        #ADC temperature
        try:
            with open('out_ta_' + filename, 'w') as fouttempadc:
                for j in range(len(data[2][0])):
                    for ich in range(4):
                        fouttempadc.write(str(data[2][ich][j]) + '\t')
                    fouttempadc.write('\n')
        except:
            print('fileOutput: Error writing ADC temperature')
            
        #Monitored voltage
        try:
            with open('v_' + filename, 'w') as foutvMon:
                for j in range(len(data[3][0])):
                    for ich in range(4):
                        foutvMon.write(str(data[3][ich][j]) + '\t')
                    foutvMon.write('\n')
        except:
            print('fileOutput: Error writing monitored voltage')
            
        #Monitored current
        try:
            with open('i_' + filename, 'w') as foutiMon:
                for j in range(len(data[4][0])):
                    for ich in range(4):
                        foutiMon.write(str(data[4][ich][j]) + '\t')
                    foutiMon.write('\n')
        except:
            print('fileOutput: Error writing monitored current')
            
        #Bias
        try:
            with open('bias_' + filename, 'w') as foutbias:
                for j in range(len(data[5][0])):
                    for ich in range(4):
                        foutbias.write(str(data[5][ich][j]) + '\t')
                    foutbias.write('\n')
        except:
            print('fileOutput: Error writing bias')
            
        #Time data
        try:
            with open('out_time_' + filename, 'w') as fouttime:
                for j in range(len(data[6])):
                    fouttime.write(str(data[6][j]) + '\n')
        except:
            print('fileOutput: Error writing uscount')

        #Event time data
        for ich in range(4):
                try:
                    with open('out_timeevt_' + str(ich + 1) + '_' + filename, 'w') as fouttimeevt:
                        for j in range(len(data[7][ich])):
                            fouttimeevt.write(str(data[7][ich][j]) + '\n')
                except:
                    print('fileOutput: Error writing event uscount')
                    
        #Correct live time
        if not len(data[8]) == 0:
            try:
                with open('livetime_' + filename, 'w') as fouttimecorr:
                    for j in range(len(data[8])):
                        fouttimecorr.write(str(data[8][j]) + '\n')
            except:
                print('fileOutput: Error writing correct live time')
                
        #Effective count
        if not len(data[9]) == 0:
            try:
                with open('eff_' + filename, 'w') as fouteff:
                    for j in range(len(data[9])):
                        fouteff.write(str(data[9][j]) + '\n')
            except:
                print('fileOutput: Error writing effective count')

        #Missing count
        if not len(data[10]) == 0:
            try:
                with open('miss_' + filename, 'w') as foutmiss:
                    for j in range(len(data[10])):
                        foutmiss.write(str(data[10][j]) + '\n')
            except:
                print('fileOutput: Error writing missing count')

        ind = 11

        #CI data
        if isCi == 1:
            #CI ADC amplitude
            for ich in range(4):
                try:
                    with open('ci_ch' + str(ich + 1) + '_' + filename, 'w') as foutampCI:
                        for j in range(len(data[ind][ich])):
                            foutampCI.write(str(data[ind][ich][j]) + '\n')
                except:
                    print('fileOutput: Error writing CI ADC amplitude file')
            
            #CI event time data
            for ich in range(4):
                try:
                    with open('ci_timeevt_' + str(ich + 1) + '_' + filename, 'w') as fouttimeevtCI:
                        for j in range(len(data[ind][ich])):
                            fouttimeevtCI.write(str(data[ind][ich][j]) + '\n')
                except:
                    print('fileOutput: Error writing CI event uscount')
                ind += 1
                
            #CI effective count
            if not len(data[ind]) == 0:
                try:
                    with open('ci_eff_' + filename, 'w') as fouteffCI:
                        for j in range(len(data[ind])):
                            fouteffCI.write(str(data[ind][j]) + '\n')
                except:
                    print('fileOutput: Error writing CI effective count')
            ind += 1

            #CI missing count
            if not len(data[ind]) == 0:
                try:
                    with open('ci_miss_' + filename, 'w') as foutmissCI:
                        for j in range(len(data[ind])):
                            foutmissCI.write(str(data[ind][j]) + '\n')
                except:
                    print('fileOutput: Error writing CI missing count')

    #I-V scan
    if not isCi == 0 and isScan:
        #Set voltage
        try:
            with open('vset_' + filename, 'w') as foutvSet:
                for v in data[ind]:
                    foutvSet.write(str(v) + '\n')
        except:
            print('fileOutput: Error writing set voltage')
        finally:
            ind += 1

        #Scan voltage
        try:
            with open('vscan_' + filename, 'w') as foutvScan:
                for j in range(len(data[ind][0])):
                    for ich in range(4):
                        foutvScan.write(str(data[ind][ich][j]) + '\t')
                    foutvScan.write('\n')
        except:
            print('fileOutput: Error writing scan voltage')
        finally:
            ind += 1

        #Scan current
        try:
            with open('iscan_' + filename, 'w') as foutiScan:
                for j in range(len(data[ind][0])):
                    for ich in range(4):
                        foutiScan.write(str(data[ind][ich][j]) + '\t')
                    foutiScan.write('\n')
        except:
            print('fileOutput: Error writing scan current')

    print('File output complete')

    return

def importData(filename, importPath, isCi = 0, isScan = False, scanRange = []):

    """
    Function for importing data from output files
    :param filename: name of the output file, currently supporting only .txt files
    :param importPath: list of path of the import directories
    :param isCi: int indicating whether the input file has CI part, with 0 for no CI, 1 for CI, 2 for multiple CI
    :param isScan: boolean indicating whether the input file has I-V scan part
    :param scanRange: list containing the scan range for multiple scans
    :return: all data extracted from the data file, including spectrums, SiPM&ADC temperatures, SiPM voltage&leak current,\
 uscount, correct live time, effective count, missing count, [CI data], [I-V scan data], all data in the form of ndarray
    """

    rootname = filename.split('\\')[-1]

    amp = [] #after CI for data with CI
    ampCI = []
    tempSipm = []
    tempAdc = []
    vMon = []
    iMon = []
    bias = []
    uscount = [] #uscount in telemetry data
    uscountEvt = []
    uscountEvtCI = []
    vSet = [] #i-v scan data
    vScan = [] #i-v scan data
    iScan = [] #i-v scan data
    timeCorrect = []
    effectiveCount = []
    effectiveCountCI = []
    missingCount = []
    missingCountCI = []
    scanNum = []

    for ich in range(4):
        amp.append([])
        ampCI.append([])
        tempSipm.append([])
        tempAdc.append([])
        vMon.append([])
        iMon.append([])
        bias.append([])
        uscountEvt.append([])
        uscountEvtCI.append([])
        vScan.append([])
        iScan.append([])

    if isCi == 2:
        ranged = False
        if not len(scanRange) == 0:
            ranged = True
            lower = scanRange[0]
            upper = scanRange[1]
        for path in importPath:
            print('importData: reading files in ' + path)
            files = os.listdir(path)
            for file in files:
                #ADC amplitude files
                if file.startswith('out_ch') and file.endswith(rootname):
                    try:
                        ich = int(file.split('_')[1][-1]) - 1
                        iscan = int(file.split('_')[2])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        amp[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    amp[ich][-1].append(int(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                            
                #Event uscount files
                elif file.startswith('out_timeevt') and file.endswith(rootname):
                    try:
                        ich = int(file.split('_')[2]) - 1
                        iscan = int(file.split('_')[3])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        uscountEvt[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    uscountEvt[ich][-1].append(float(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                                
                #CI ADC amplitude files
                elif file.startswith('ci_ch') and file.endswith(rootname):
                    try:
                        ich = int(file.split('_')[1][-1]) - 1
                        iscan = int(file.split('_')[2])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        ampCI[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    ampCI[ich][-1].append(int(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                            
                #CI event uscount files
                elif file.startswith('ci_timeevt') and file.endswith(rootname):
                    try:
                        ich = int(file.split('_')[2]) - 1
                        iscan = int(file.split('_')[3])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        uscountEvtCI[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    uscountEvtCI[ich][-1].append(float(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)

                #Sipm temperature files
                elif file.startswith('out_t_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[2])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        for ich in range(4):
                            tempSipm[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    lineList = line.split()
                                    for ich in range(4):
                                        tempSipm[ich][-1].append(float(lineList[ich]))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                                    
                #ADC temperature files
                elif file.startswith('out_ta') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[2])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        for ich in range(4):
                            tempAdc[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    lineList = line.split()
                                    for ich in range(4):
                                        tempAdc[ich][-1].append(float(lineList[ich]))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                                    
                #Monitored voltage files
                elif file.startswith('v_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[1])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        for ich in range(4):
                            vMon[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    lineList = line.split()
                                    for ich in range(4):
                                        vMon[ich][-1].append(float(lineList[ich]))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                                    
                #Monitored current files
                elif file.startswith('i_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[1])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        for ich in range(4):
                            iMon[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    lineList = line.split()
                                    for ich in range(4):
                                        iMon[ich][-1].append(float(lineList[ich]))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                                        
                #SiPM bias files
                elif file.startswith('bias_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[1])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        for ich in range(4):
                            bias[ich].append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    lineList = line.split()
                                    for ich in range(4):
                                        bias[ich][-1].append(float(lineList[ich]))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                                    
                #Uscount files
                elif file.startswith('out_time_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[2])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        uscount.append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    uscount[-1].append(float(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                            
                #Correct live time files
                elif file.startswith('livetime_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[1])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        timeCorrect.append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    timeCorrect[-1].append(float(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                            
                #Effective count files
                elif file.startswith('eff_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[1])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        effectiveCount.append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    effectiveCount[-1].append(int(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)

                #Missing count files
                elif file.startswith('miss_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[1])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        missingCount.append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    missingCount[-1].append(int(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)
                            
                #CI effective count files
                elif file.startswith('ci_eff_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[2])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        effectiveCountCI.append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    effectiveCountCI[-1].append(int(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)

                #CI missing count files
                elif file.startswith('ci_miss_') and file.endswith(rootname):
                    try:
                        iscan = int(file.split('_')[2])
                        if not iscan in scanNum:
                            scanNum.append(iscan)
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    if not ranged or (ranged and iscan < upper and iscan >= lower):
                        missingCountCI.append([])
                        try:
                            with open(path + '\\' + file, 'r') as fin:
                                lines = [line.rstrip() for line in fin]
                                for line in lines:
                                    missingCountCI[-1].append(int(line))
                        except:
                            print('importData: error reading file ' + path + '\\' + file)

                #Set voltage files
                elif isScan and file.startswith('vset_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                vSet.append(float(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                            
                #Scan voltage files
                elif isScan and file.startswith('vscan_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    vScan[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                            
                #Scan current files
                elif isScan and file.startswith('iscan_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    iScan[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)

    else:
        for path in importPath:
            print('importData: reading files in ' + path)
            files = os.listdir(path)
            for file in files:
                #ADC amplitude files
                if file.startswith('out_ch') and file.endswith(rootname):
                    try:
                        ich = int(file.split('_')[1][-1]) - 1
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                amp[ich].append(int(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)

                #Event uscount files
                elif file.startswith('out_timeevt') and file.endswith(rootname):
                    try:
                        ich = int(file.split('_')[2][-1]) - 1
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                uscountEvt[ich].append(float(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)

                #Sipm temperature files
                elif file.startswith('out_t') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    tempSipm[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                                    
                #ADC temperature files
                elif file.startswith('out_ta') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    tempAdc[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                                    
                #Monitored voltage files
                elif file.startswith('v_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    vMon[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                                    
                #Monitored current files
                elif file.startswith('i_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    iMon[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                                    
                #SiPM bias files
                elif file.startswith('bias_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    bias[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                                    
                #Uscount files
                elif file.startswith('out_time_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                uscount.append(float(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)

                #Correct live time files
                elif file.startswith('livetime_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                timeCorrect.append(float(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                        
                #Effective count files
                elif file.startswith('eff_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                effectiveCount.append(int(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)

                #Missing count files
                elif file.startswith('miss_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                missingCount.append(int(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                        
                #CI effective count files
                elif file.startswith('ci_eff_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                effectiveCountCI.append(int(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)

                #CI missing count files
                elif file.startswith('ci_miss_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                missingCountCI.append(int(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)

                #CI ADC amplitude files
                elif isCi == 1 and file.startswith('ci_ch') and file.endswith(rootname):
                    try:
                        ich = int(file.split('_')[1][-1]) - 1
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                ampCI[ich].append(int(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                        
                #CI event uscount files
                elif isCi == 1 and file.startswith('ci_timeevt') and file.endswith(rootname):
                    try:
                        ich = int(file.split('_')[2][-1]) - 1
                    except:
                        raise Exception('importFile: unable to resolve filename ' + file)
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                uscountEvtCI[ich].append(float(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                        
                #Set voltage files
                elif isScan and file.startswith('vset_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                vSet.append(float(line))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                            
                #Scan voltage files
                elif isScan and file.startswith('vscan_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    vScan[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)
                            
                #Scan current files
                elif isScan and file.startswith('iscan_') and file.endswith(rootname):
                    try:
                        with open(path + '\\' + file, 'r') as fin:
                            lines = [line.rstrip() for line in fin]
                            for line in lines:
                                lineList = line.split()
                                for ich in range(4):
                                    iScan[ich].append(float(lineList[ich]))
                    except:
                        print('importData: error reading file ' + path + '\\' + file)

    print('Data readout complete')

    amp = np.array(amp)
    tempSipm = np.array(tempSipm)
    tempAdc = np.array(tempAdc)
    vMon = np.array(vMon)
    iMon = np.array(iMon)
    bias = np.array(bias)
    uscount = np.array(uscount)
    uscountEvt = np.array(uscountEvt)
    timeCorrect = np.array(timeCorrect)
    effectiveCount = np.array(effectiveCount)
    missingCount = np.array(missingCount)
    if isScan:
        vSet = np.array(vSet)
        vScan = np.array(vScan)
        iScan = np.array(iScan)

    if isScan:
        if isCi == 0:
            return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount, vSet, vScan, iScan
        else:
            ampCI = np.array(ampCI)
            uscountEvtCI = np.array(uscountEvtCI)
            effectiveCountCI = np.array(effectiveCountCI)
            missingCountCI = np.array(missingCountCI)
            return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount, ampCI, uscountEvtCI, \
                effectiveCountCI, missingCountCI, vSet, vScan, iScan, scanNum
    else:
        if isCi == 0:
            return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount
        else:
            ampCI = np.array(ampCI)
            uscountEvtCI = np.array(uscountEvtCI)
            effectiveCountCI = np.array(effectiveCountCI)
            missingCountCI = np.array(missingCountCI)
            return amp, tempSipm, tempAdc, vMon, iMon, bias, uscount, uscountEvt, timeCorrect, effectiveCount, missingCount, ampCI, uscountEvtCI, \
                effectiveCountCI, missingCountCI, scanNum

def plotRawData(filename, amp, nbins, corr, time, singlech = False, channel = -1, rateStyle = '', rateAll = 0.0, doCorr = True):
    
    """
    Function for plotting the processed, unfitted data
    :param filename: name of raw data file
    :param amp: ADC amplitude data of SINGLE CHANNEL
    :param nbins: number of bins to be used in the spectrum
    :param corr: temperature-bias correction factors for the data
    :param time: time taken to take the spectrum, usually calculated with event uscount
    :param singlech: boolean indicating whether the fit is for single channel
    :param channel: channel number for single channel fits, in range[0-3]
    :param rateStyle: the style of calculating real count rate, '' for none, 's' forsingle live time data, 'p' for calculation with small data packs(512byte)
    :param rateAll: correct count rate of all spectrum in all 4 channels calculated when reading data, only used when rateStyle is 's' or 'l'
    :param doCorr: boolean indicating whether the temperature-bias correction will be done, to avoid warning info output
    :return: nothing
    """

    if not doCorr:
        corr = [1.0, 1.0, 1.0, 1.0]

    styleAvailable = ['', 's', 'p']
    if not rateStyle in styleAvailable:
        raise Exception('plotRawData: unknown count rate correction style')
    elif rateAll == 0.0:
        raise Exception('plotRawData: corrected count rate not specified')

    if singlech:
        if not isChannel(channel):
            raise Exception('plotRawData: channel number out of bound[0-3]')
        spectrum, x = getSpectrum(amp[channel], nbins, singlech)
    else:
        spectrum, x = getSpectrum(amp, nbins, singlech)

    if not rateStyle == '':
        countAll = 0.0
        for ich in range(4):
            countAll += float(len(amp[ich]))
        rateFactor = rateAll / countAll

    fig = plt.figure(figsize=(12, 8))
    #Single channel plots
    if singlech:
        ich = channel
        gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
        ax = fig.add_subplot(gs[0])
        if rateStyle == '':
            plt.step(x * corr[ich], spectrum / time[ich], where='mid', label='raw data', zorder=1) #TBD: change the x data to energy with additional EC calculation funtcion
        else:
            plt.step(x * corr[ich], spectrum * rateFactor, where='mid', label='raw data', zorder=1) #TBD: change the x data to energy with additional EC calculation funtcion
        ax.set_xlim([0., 65535.])
        ax.set_title('Spectrum of raw data from ' + filename)
        ax.set_xlabel('ADC/channel')
        ax.set_ylabel('count rate/cps')
        ax.legend(loc=0)
        ax.grid()
    #Multiple channel plots
    else:
        gs = gridspec.GridSpec(4, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
        for ich in range(4):
            ax = fig.add_subplot(gs[ich])
            if rateStyle == '':
                plt.step(x[ich] * corr[ich], spectrum[ich] / time[ich], where='mid', label='raw data', zorder=1) #TBD: change the x data to energy with additional EC calculation funtcion
            else:
                plt.step(x[ich] * corr[ich], spectrum[ich] * rateFactor, where='mid', label='raw data', zorder=1) #TBD: change the x data to energy with additional EC calculation funtcion
            ax.set_xlim([0., 65535.])
            if ich == 0:
                ax.set_title('Spectrum of raw data from ' + filename)
            ax.set_xlabel('ADC/channel')
            ax.set_ylabel('count rate/cps')
            ax.legend(loc=0)
            ax.grid()
    plt.show()
    return

#***********************************************************************************************************************************************************
#************************************************Basic calibration (temperature, bias and EC) part************************************************
#***********************************************************************************************************************************************************
#TBD: add a input parameter to read the bin factor/fit range from input files
def getBiasnbinsFactor(scanNum, scanRange = []):
    #TBD: Add data for Am241
    """
    Function for calculating the binning factor for bias responce process
    :param scanNum: scan number
    :param scanRange: list containing the scan range for multiple scans
    :return: corresponding binning factor(nbins/512)
    """
    
    biasnbinsFactor = [8, 4, 2, 1]
    biasnbinsRange = [1, 5, 12, 18]

    num = scanNum
    if not len(scanRange) == 0:
        num += scanRange[0]
    ind = 0
    while ind < len(biasnbinsRange) - 1:
        if num >= biasnbinsRange[ind] and num < biasnbinsRange[ind + 1]:
            break
        ind += 1
    return biasnbinsFactor[ind]

def getBiasFitRange(scanNum, corr = False):
    #TBD: Add data for Am241
    """
    Function for calculating the fit range for bias responce process
    :param scanNum: scan number
    :param corr: boolean indicating whether the data is temperature-calibrated
    :return: corresponding fit range
    """

    if corr:
        biasRange = [1, 3, 7, 14]
        biasFitRange = [[[1000, 3250], [1250, 3800], [1000, 3250], [1200, 3500]], [[2500, 7000], [3000, 7500], [2500, 7000], [2500, 7000]], \
            [[5000, 15000], [6000, 16000], [5000, 15000], [5500, 15500]], [[12000, 26000], [13000, 29000], [12000, 28000], [13000, 29000]]]
        num = scanNum + 1
        ind = 0
        while ind < len(biasFitRange) - 1:
            if num >= biasRange[ind] and num < biasRange[ind + 1]:
                break
            ind += 1
        return biasFitRange[ind]
    else:
        biasFitRange = [[[1000, 1750], [1250, 2250], [1000, 1750], [1200, 2000]],
            [[1400, 2400], [1750, 3000], [1500, 2500], [1600, 2600]],
            [[2000, 3250], [2600, 3800], [2000, 3250], [2250, 3500]],
            [[2500, 4000], [3000, 5000], [2500, 4000], [2500, 4500]],
            [[3000, 5000], [3500, 5500], [3000, 5000], [3000, 5000]],
            [[3500, 6000], [4000, 6500], [3500, 6000], [4000, 6500]],
            [[4500, 7000], [5000, 7500], [4500, 7000], [4500, 7000]],
            [[5000, 8000], [6000, 8500], [5000, 8000], [5500, 8500]],
            [[6000, 9000], [6500, 10000], [6000, 9000], [6500, 9500]],
            [[7000, 10500], [8000, 12000], [7000, 12000], [7500, 12000]],
            [[8000, 12000], [8500, 13000], [8000, 13000], [8500, 13000]],
            [[9000, 14000], [11000, 15000], [9000, 14500], [9500, 15000]],
            [[10000, 15000], [11000, 16000], [10000, 16000], [11500, 16000]],
            [[11000, 17000], [12000, 18000], [11000, 18000], [11500, 18000]],
            [[12000, 18000], [13000, 20000], [13000, 20000], [13000, 20000]],
            [[13000, 20000], [14500, 21000], [14000, 21000], [14500, 22000]],
            [[14000, 22000], [15000, 23000], [14000, 23000], [14000, 24000]],
            [[16000, 24000], [17000, 25000], [18000, 25000], [17000, 26000]],
            [[17000, 26000], [20000, 27000], [17000, 27000], [18000, 28000]],
            [[19000, 28000], [20000, 29000], [18000, 29000], [19000, 30000]],
            [[20000, 30000], [24000, 32000], [23000, 32000], [24000, 33000]]]
        return biasFitRange[scanNum]

def ecCalibration(x, singlech = False, channel = -1):
    #TBD: EC calibiration for the energy value of the spectrum's x-axis

    """
    Function for calculating the EC calibrated values of the spectrum's x-axis(energy)
    :param x: the input bin center values
    :param singlech: boolean indicating whether the input data contains only data from one channel, True \
if the data is from one single channel
    :param channel: the channel number in range [0-3], if the data is single-channeled
    :return: the calibrated bin center values(converted from ADC channel to energy)
    """

    corrX = []
    return corrX

def tempBiasCorrection(temp, bias, corr = False, isTemp = False):
    #TBD: adding the correlation factors
    #ver 0.1 with only temperature correction

    """
    Function for calculating correction factor of temperature and bias
    :param temp: temperature(of SiPM), ndarray
    :param bias: SiPM bias, ndarray
    :param corr: method of correction, correlative correction if True, independent if False
    :param isTemp: True if the data given is from temperature scan, False if the \
data is from bias scan #TBD: reomve this part in the final version ,also note that in the main function the relation is given in REVERSE against the option
    :return: correction factors and corresponding error of all 4 channels
    """

    tempStandard = 25.0
    biasStandard = 28.5
    corrFactor = []
    corrErr = []
    tempA = [-0.2763584453760433, -0.229011562127076, -0.556186224390358, -0.48778267986537965]
    tempAErr = [0.006895897717006753, 0.0034202195348071126, 0.07324058325355964, 0.17655855018839478]
    tempB = [-5.143763434129346, -4.588208910593746, 5.374640351833634, 2.069306764051607]
    tempBErr = [0.20640178472497478, 0.10518963405039893, 2.107640099625546, 5.064259354642139]
    tempC = [1767.691653327116, 1731.9362852738936, 1772.234079395335, 1920.644877108622]
    tempCErr = [1.3852485895846538, 0.718417078682467, 13.512180927662211, 32.52547152274127]
    biasAt = [2398.4964219935587, 2543.474877801643, 2708.5722639602536, 3104.2570531444376]
    biasAtErr = [57.809633717969774, 82.06298490066145, 86.19374286192473, 112.13904107442131]
    biasBt = [-123941.41198532937, -132345.91260082374, -140199.98763738308, -161366.31381996674]
    biasBtErr = [3217.842782755337, 4612.241232022711, 4789.7245685028065, 6210.509841024584]
    biasCt = [1606126.7425641178, 1727035.5098439471, 1819758.2411229876, 2103180.263761908]
    biasCtErr = [44769.93662624587, 64791.24784579912, 66526.55210367328, 85972.931415768]
    biasA = [2069.0337145552735, 2228.0000425477992, 2188.319185650479, 2623.892898630916]
    biasAErr = [55.215024380923104, 68.2669047555512, 56.853844152465875, 84.05191033001722]
    biasB = [-106807.50521738746, -115748.29072658953, -112860.20934007895, -136192.37416283946]
    biasBErr = [3073.578040223645, 3836.9471891721187, 3159.9400080035007, 4655.477328200552]
    biasC = [1382898.3403082641, 1508250.928132551, 1459773.7409409452, 1772699.0674105799]
    biasCErr = [42764.95603300236, 53901.63697352948, 43898.32960048521, 64453.25711855276]
    tempAb = [0.061309172220172056, -0.08336701103031305, 0.004048775610269907, 0.08399523465218563]
    tempAbErr = [0.08211985960548235, 0.11892735488438218, 0.0723572881711063, 0.11887128393854622]
    tempBb = [-18.33761159544615, -12.245630171127809, -15.894270216984575, -18.63747886990456]
    tempBbErr = [2.6936591101708105, 3.8549964273339454, 2.4597749629970167, 4.005732897785914]
    tempCb = [2033.2592695580918, 1995.1454480486013, 2091.256524762956, 2237.254931940749]
    tempCbErr = [19.429792750152824, 26.160130056878476, 18.645570801178525, 28.384947145852912]
    #TBD: add the errors for the following coefficients
    tempAc = [-0.03594923117319518, -0.04593051646351449, -0.08449626753463194, 0.09429751988676158]
    tempAcErr = []
    tempBc = [124.622282167142, 104.0920375700276, 125.264006098604, 124.61353026819931]
    tempBcErr = []
    biasAc = [134.2821674425955, 136.2541684301311, 143.11481388547006, 167.3728511192398]
    biasAcErr = []
    biasBc = [-6629.88085155412, -6737.005948027494, -7099.806128025778, -8383.83796307196]
    biasBcErr = []
    tempBiasBc = [-5.015141502766973, -4.224695684754486, -4.954771463432692, -5.1692210016793965]
    tempBiasBcErr = []
    Cc = [82030.55115289443, 83455.88882892841, 88275.2920151818, 105333.17056814635]
    CcErr = []
    
    for ich in range(4):
        tempAvg = np.average(temp[ich])
        tempStd = np.std(temp[ich])
        biasAvg = np.average(bias[ich])
        biasStd = np.std(bias[ich])
        if not corr:
            if isTemp:
                tempFactor = (tempA[ich] * tempStandard ** 2 + tempB[ich] * tempStandard + tempC[ich]) / (tempA[ich] * tempAvg ** 2 + \
                     tempB[ich] * tempAvg + tempC[ich])
                biasFactor = (biasAt[ich] * biasStandard ** 2 + biasBt[ich] * biasStandard + biasCt[ich]) / (biasAt[ich] *  biasAvg ** 2 + biasBt[ich] * \
                    biasAvg + biasCt[ich])
                corrFactor.append(tempFactor * biasFactor)
                corrErr.append(np.sqrt(\
                    (corrFactor[-1] / (tempA[ich] * tempAvg ** 2 + tempB[ich] * tempAvg + tempC[ich]) * (2 * tempA[ich] * tempAvg + tempB[ich]) * tempStd) ** 2 + \
                    (corrFactor[-1] / (biasAt[ich] * biasAvg ** 2 + biasBt[ich] * biasAvg + biasCt[ich]) * (2 * biasAt[ich] * biasAvg + biasBt[ich]) * biasStd) ** 2 + \
                    (biasFactor * (tempB[ich] * tempAvg * tempStandard + tempC[ich] * (tempStandard + tempAvg)) * (tempStandard - tempAvg) / (tempA[ich] * tempAvg ** 2 + tempB[ich] * tempAvg + tempC[ich]) ** 2 * tempAErr[ich]) ** 2 + \
                    (tempFactor * (biasBt[ich] * biasAvg * biasStandard + biasCt[ich] * (biasStandard + biasAvg)) * (biasStandard - biasAvg) / (biasAt[ich] * biasAvg ** 2 + biasBt[ich] * biasAvg + biasCt[ich]) ** 2 * biasAtErr[ich]) ** 2 + \
                    (biasFactor * (- tempA[ich] * tempAvg * tempStandard - tempC[ich]) * (tempStandard - tempAvg) / (tempA[ich] * tempAvg ** 2 + tempB[ich] * tempAvg + tempC[ich]) ** 2 * tempBErr[ich]) ** 2 + \
                    (tempFactor * (- biasAt[ich] * biasAvg * biasStandard + biasCt[ich]) * (biasStandard - biasAvg) / (biasAt[ich] * biasAvg ** 2 + biasBt[ich] * biasAvg + biasCt[ich]) ** 2 * biasBtErr[ich]) ** 2 + \
                    (biasFactor * (tempA[ich] * (tempAvg + tempStandard) + tempB[ich]) * (tempStandard - tempAvg) / (tempA[ich] * tempAvg ** 2 + tempB[ich] * tempAvg + tempC[ich]) ** 2 * tempCErr[ich]) ** 2 + \
                    (tempFactor * (biasAt[ich] * (biasAvg + biasStandard) + biasBt[ich]) * (biasStandard - biasAvg) / (biasAt[ich] * biasAvg ** 2 + biasBt[ich] * biasAvg + biasCt[ich]) ** 2 * biasCtErr[ich]) ** 2))
            else:
                tempFactor = (tempAb[ich] * tempStandard ** 2 + tempBb[ich] * tempStandard + tempCb[ich]) / (tempAb[ich] * tempAvg ** 2 + \
                     tempBb[ich] * tempAvg + tempCb[ich])
                biasFactor = (biasA[ich] * biasStandard ** 2 + biasB[ich] * biasStandard + biasC[ich]) / (biasA[ich] *  biasAvg ** 2 + biasB[ich] * \
                    biasAvg + biasC[ich])
                corrFactor.append(tempFactor * biasFactor)
                corrErr.append(np.sqrt(\
                    (corrFactor[-1] / (tempAb[ich] * tempAvg ** 2 + tempBb[ich] * tempAvg + tempCb[ich]) * (2 * tempAb[ich] * tempAvg + tempBb[ich]) * tempStd) ** 2 + \
                    (biasFactor * (tempBb[ich] * tempAvg * tempStandard + tempCb[ich] * (tempStandard + tempAvg)) * (tempStandard - tempAvg) / (tempAb[ich] * tempAvg ** 2 + tempBb[ich] * tempAvg + tempCb[ich]) ** 2 * tempAbErr[ich]) ** 2 + \
                    (tempFactor * (biasB[ich] * biasAvg * biasStandard + biasC[ich] * (biasStandard + biasAvg)) * (biasStandard - biasAvg) / (biasA[ich] * biasAvg ** 2 + biasB[ich] * biasAvg + biasC[ich]) ** 2 * biasAErr[ich]) ** 2 + \
                    (biasFactor * (- tempAb[ich] * tempAvg * tempStandard - tempCb[ich]) * (tempStandard - tempAvg) / (tempAb[ich] * tempAvg ** 2 + tempBb[ich] * tempAvg + tempCb[ich]) ** 2 * tempBbErr[ich]) ** 2 + \
                    (tempFactor * (- biasA[ich] * biasAvg * biasStandard + biasC[ich]) * (biasStandard - biasAvg) / (biasA[ich] * biasAvg ** 2 + biasB[ich] * biasAvg + biasC[ich]) ** 2 * biasBErr[ich]) ** 2 + \
                    (biasFactor * (tempAb[ich] * (tempAvg + tempStandard) + tempBb[ich]) * (tempStandard - tempAvg) / (tempAb[ich] * tempAvg ** 2 + tempBb[ich] * tempAvg + tempCb[ich]) ** 2 * tempCbErr[ich]) ** 2 + \
                    (tempFactor * (biasA[ich] * (biasAvg + biasStandard) + biasB[ich]) * (biasStandard - biasAvg) / (biasA[ich] * biasAvg ** 2 + biasB[ich] * biasAvg + biasC[ich]) ** 2 * biasCErr[ich]) ** 2))
        else:
            corrFactor.append((Cc[ich] + tempAc[ich] * tempStandard ** 2 + tempBc[ich] * tempStandard + biasAc[ich] * biasStandard ** 2 + \
                biasBc[ich] * biasStandard + tempBiasBc[ich] * tempStandard * biasAvg) / (Cc[ich] + tempAc[ich] * tempAvg ** 2 + tempBc[ich] * tempAvg + \
                biasAc[ich] * biasAvg ** 2 + biasBc[ich] * biasAvg + tempBiasBc[ich] * tempAvg * biasAvg))
            corrErr.append(0.0) #TBD: add error
            
    return corrFactor, corrErr

#***********************************************************************************************************************************************************
#************************************************************Basic fit functions and fit part***********************************************************
#***********************************************************************************************************************************************************

def gehrelsErr(ydata):

    """
    Function for calcilating Poissons error of spectrum (y) data
    :param ydata: spectrum data (or other Poissions variables)
    :return: corresponding Poissons error
    """

    if np.size(ydata) == 1:
        yerr = np.sqrt(ydata) if ydata >= 5.0 else 1.0 + np.sqrt(ydata + 0.75)
    elif np.size(ydata) > 1:
        yerr = np.sqrt(ydata)
        q = np.where(ydata < 5.0)
        yerr[q] = 1.0 + np.sqrt(ydata[q] + 0.75)
    else:
        yerr = []
    return yerr

def gaussianFunction(param, x):

    """
    Auxiliary function to calculate the value of the gaussian function, used for odr fitting
    :param param: parameters of the gaussian and quadratic function, in the form of [amplitude, center, sigma]
    :param x: input x value
    :return: value of the gaussian function
    """
    
    return param[0] * np.exp(-(x - param[1]) ** 2 / (2 * (param[2] ** 2))) / (param[2] * np.sqrt(2 * np.pi))

def doFitGaussian(xdata, ydata, odr = False, xerror = [], yerror = []):
    
    """
    Function for fitting single gaussian peak, with no background
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :return: fit result
    For lsq fitting, in the form of a dictionary as
    result.best_values = {
        'fit_amplitude': area of the gaussian peak
        'fit_center': center of the gaussian peak
        'fit_sigma': standard deviation of the gaussian peak
        'fit_amplitude_err': error of area of the gaussian peak
        'fit_center_err': error of center of the gaussian peak
        'fit_sigma_err': error of standard deviation of the gaussian peak
    }
    with fit function as
    fit_amplitude * exp(-(x - fit_center) ** 2 / (2 * fit_sigma ** 2))
    For odr fitting in the form of a ODR Output as
    parameters: result.beta = [amplitude, center, sigma]
    standard deviations of parameters: result.sd_beta = [std_amplitude, std_center, std_sigma]
    """

    gModel = lmfit.models.GaussianModel(prefix = 'fit_')
    param = gModel.guess(ydata, x = xdata)
    if odr:
        if len(xerror) == 0:
            if len(yerror) == 0:
                data = RealData(xdata, ydata)
            else:
                data = RealData(xdata, ydata, sy = yerror)
        else:
            if len(yerror) == 0:
                data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
            else:
                data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
        model = Model(gaussianFunction)
        odrFit = ODR(data, model, [param.valuesdict()['fit_amplitude'], param.valuesdict()['fit_center'], param.valuesdict()['fit_sigma']])
        odrFit.set_job(fit_type = 0)
        result = odrFit.run()
        fitResult = {
                        'fit_amplitude':           result.beta[0],
                        'fit_center':                result.beta[1],
                        'fit_sigma':                 result.beta[2],
                        'fit_amplitude_err':           result.sd_beta[0],
                        'fit_center_err':                result.sd_beta[1],
                        'fit_sigma_err':                 result.sd_beta[2],
            }
    else:
        if len(yerror) == 0:
            result = gModel.fit(ydata, param, x = xdata)
        else:
            result = gModel.fit(ydata, param, x = xdata, weights = 1. / yerror)
        fitResult = {
                        'fit_amplitude':           result.best_values['fit_amplitude'],
                        'fit_center':                result.best_values['fit_center'],
                        'fit_sigma':                 result.best_values['fit_sigma'],
                        'fit_amplitude_err':           result.params['fit_amplitude'].stderr,
                        'fit_center_err':                result.params['fit_center'].stderr,
                        'fit_sigma_err':                 result.params['fit_sigma'].stderr,
            }
    return fitResult

def peakFunction(param, x):

    """
    Auxiliary function to calculate the value of the peak function, used for odr fitting
    :param param: parameters of the gaussian and quadratic function, in the form of [amplitude, center, sigma, a, b, c]
    :param x: input x value
    :return: value of the peak function, as an gaussian + quadratic
    """
    
    return param[0] * np.exp(-(x - param[1]) ** 2 / (2 * (param[2] ** 2))) / (param[2] * np.sqrt(2 * np.pi)) + param[3] * x ** 2 + param[4] * x + param[5]

def doFitPeak(xdata, ydata, odr = False, xerror = [], yerror = [], quadBkg = True):
    
    """
    Function for fitting single gaussian peak, with quadratic background
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :param quadBkg: boolean indicating whether the quadratic background is included in the fit function
    :return: fit result
    For lsq fitting, in the form of a dictionary as
    result.best_values = {
        'bk_a': quadratic term of the background
        'bk_b': linear term of the background
        'bk_c': constant term of the background
        'peak_amplitude': area of the gaussian peak
        'peak_center': center of the gaussian peak
        'peak_sigma': standard deviation of the gaussian peak
        'peak_amplitude_err': error of area of the gaussian peak
        'peak_center_err': error of center of the gaussian peak
        'peak_sigma_err': error of standard deviation of the gaussian peak
    }
    with fit function as
    peak_amplitude * exp(-(x - peak_center) ** 2 / (2 * peak_sigma ** 2)) + bk_a * x ** 2 + bk_b * x + bk_c
    For odr fitting in the form of a ODR Output as
    parameters: result.beta = [amplitude, center, sigma, a, b, c]
    standard deviations of parameters: result.sd_beta = [std_amplitude, std_center, std_sigma, std_a, std_b, std_c]
    """

    gModel = lmfit.models.GaussianModel(prefix = 'peak_')
    param1 = gModel.guess(ydata, x = xdata)
    if quadBkg:
        ydatabkg = ydata - gaussianFunction([param1.valuesdict()['peak_amplitude'], param1.valuesdict()['peak_center'], param1.valuesdict()['peak_sigma']], \
            xdata)
        qModel = lmfit.models.QuadraticModel(prefix = 'bk_')
        param2 = qModel.guess(ydatabkg, x = xdata)
        param = param1 + param2
    else:
        param = param1
    if odr:
        if len(xerror) == 0:
            if len(yerror) == 0:
                data = RealData(xdata, ydata)
            else:
                data = RealData(xdata, ydata, sy = yerror)
        else:
            if len(yerror) == 0:
                data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
            else:
                data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
        if quadBkg:
            model = Model(peakFunction)
            odrFit = ODR(data, model, [param.valuesdict()['peak_amplitude'], param.valuesdict()['peak_center'], param.valuesdict()['peak_sigma'], \
                param.valuesdict()['bk_a'], param.valuesdict()['bk_b'], param.valuesdict()['bk_c']])
        else:
            model = Model(gaussianFunction)
            odrFit = ODR(data, model, [param.valuesdict()['peak_amplitude'], param.valuesdict()['peak_center'], param.valuesdict()['peak_sigma']])
        odrFit.set_job(fit_type = 0)
        result = odrFit.run()
        if quadBkg:
            fitResult = {
                            'bk_a':         result.beta[3],
                            'bk_b':         result.beta[4],
                            'bk_c':         result.beta[5],
                            'peak_amplitude':           result.beta[0],
                            'peak_center':                result.beta[1],
                            'peak_sigma':                 result.beta[2],
                            'peak_amplitude_err':           result.sd_beta[0],
                            'peak_center_err':                result.sd_beta[1],
                            'peak_sigma_err':                 result.sd_beta[2],
                }
        else:
            fitResult = {
                            'peak_amplitude':           result.beta[0],
                            'peak_center':                result.beta[1],
                            'peak_sigma':                 result.beta[2],
                            'peak_amplitude_err':           result.sd_beta[0],
                            'peak_center_err':                result.sd_beta[1],
                            'peak_sigma_err':                 result.sd_beta[2],
                }
    else:
        if quadBkg:
            model = qModel + gModel
        else:
            model = gModel
        if len(yerror) == 0:
            result = model.fit(ydata, param, x = xdata)
        else:
            result = model.fit(ydata, param, x = xdata, weights = 1. / yerror)
        if quadBkg:
            fitResult = {
                            'bk_a':         result.best_values['bk_a'],
                            'bk_b':         result.best_values['bk_b'],
                            'bk_c':         result.best_values['bk_c'],
                            'peak_amplitude':           result.best_values['peak_amplitude'],
                            'peak_center':                result.best_values['peak_center'],
                            'peak_sigma':                 result.best_values['peak_sigma'],
                            'peak_amplitude_err':           result.params['peak_amplitude'].stderr,
                            'peak_center_err':                result.params['peak_center'].stderr,
                            'peak_sigma_err':                 result.params['peak_sigma'].stderr,
                }
        else:
            fitResult = {
                            'peak_amplitude':           result.best_values['peak_amplitude'],
                            'peak_center':                result.best_values['peak_center'],
                            'peak_sigma':                 result.best_values['peak_sigma'],
                            'peak_amplitude_err':           result.params['peak_amplitude'].stderr,
                            'peak_center_err':                result.params['peak_center'].stderr,
                            'peak_sigma_err':                 result.params['peak_sigma'].stderr,
                }
    return fitResult

def doFitDouble(xdata, ydata, range):
    
    """
    Fit function for fitting overlapping double-gaussian peaks
    CURRENTLY UNUSED AND NOT RECOMMENDED
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param range: list-like object containing the approximate ranges of the two peaks, \
in the form of [peak1_lower, peak1_upper, peak2_lower, peak2_upper] in ascending order
    :return: fit result
    """

    qModel = lmfit.models.QuadraticModel(prefix = 'bk_')
    param1 = qModel.guess(ydata, x = xdata)
    gModel1 = lmfit.models.GaussianModel(prefix = 'peak1_')
    param2 = gModel1.guess(ydata[:range[1] - range[0] + 1], x = xdata[:range[1] - range[0] + 1])
    gModel2 = lmfit.models.GaussianModel(prefix = 'peak2_')
    param3 = gModel2.guess(ydata[range[2] - range[0]:range[3] - range[0] + 1], x = xdata[range[2] - range[0]:range[3] - range[0] + 1])
    param = param1 + param2 + param3
    model = qModel + gModel1 + gModel2
    result = model.fit(ydata, param, x = xdata)
    return result

def quadFunction(param, x):

    """
    Auxiliary function to calculate the value of the quadratic function, used for odr fitting
    :param param: parameters of the quadratic function, in the form of [a, b, c]
    :param x: input x value
    :return: value of the quadratic function
    """
    
    return param[0] * x ** 2 + param[1] * x + param[2]

def doFitQuad(xdata, ydata, odr = False, xerror = [], yerror = []):

    """
    Function for fitting quadratic data for temperature and bias responce experiments
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :return: fit result, in the form of a dictionary as
    {
        'fit_a': quadratic term
        'fit_b': linear term
        'fit_c': constant
        'fit_a_err': error of quadratic term
        'fit_b_err': error of linear term
        'fit_c_err': error of constant
    }
    with fit function as
    fit_a * x ** 2 + fit_b * x + fit_c
    """

    qModel = lmfit.models.QuadraticModel(prefix = 'fit_')
    param = qModel.guess(ydata, x = xdata)
    if odr:
        if len(xerror) == 0:
            if len(yerror) == 0:
               data = RealData(xdata, ydata)
            else:
               data = RealData(xdata, ydata, sy = yerror)
        else:
            if len(yerror) == 0:
                data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
            else:
                data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
        model = Model(quadFunction)
        odrFit = ODR(data, model, [param.valuesdict()['fit_a'], param.valuesdict()['fit_b'], param.valuesdict()['fit_c']])
        odrFit.set_job(fit_type = 0)
        result = odrFit.run()
        fitResult = {
                        'fit_a':            result.beta[0],
                        'fit_b':            result.beta[1],
                        'fit_c':            result.beta[2],
                        'fit_a_err':            result.sd_beta[0],
                        'fit_b_err':            result.sd_beta[1],
                        'fit_c_err':            result.sd_beta[2],
            }
    else:
        if len(yerror) == 0:
            result = qModel.fit(ydata, param, x = xdata)
        else:
            result = qModel.fit(ydata, param, x = xdata, weights = 1. / np.array(yerror))
        fitResult = {
                        'fit_a':            result.best_values['fit_a'],
                        'fit_b':            result.best_values['fit_b'],
                        'fit_c':            result.best_values['fit_c'],
                        'fit_a_err':            result.params['fit_a'].stderr,
                        'fit_b_err':            result.params['fit_b'].stderr,
                        'fit_c_err':            result.params['fit_c'].stderr,
            }
    return fitResult

def expFunction(param, x):

    """
    Auxiliary function to calculate the value of the exponential function with a constant, used for odr fitting
    :param param: parameters of the quadratic function, in the form of [a, b]
    :param x: input x value
    :return: value of the exponential function
    """
    
    return param[0] * np.exp(x / param[1]) + param[2]

def expFunctionNoConst(param, x):

    """
    Auxiliary function to calculate the value of the exponential function with no constant, used for odr fitting
    :param param: parameters of the quadratic function, in the form of [a, b]
    :param x: input x value
    :return: value of the exponential function
    """
    
    C = 50e-6
    return param[0] * np.exp(- (x - C) / param[1])

def doFitExp(xdata, ydata, odr = False, xerror = [], yerror = [], fitRate = False):

    """
    Function for fitting exponential data for leak current-temperature and live time fitting
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :param fitRate: boolean indication whether the data to fit is corrected live time, if True the fit will be done with exponential function with no constant
    :return: fit result, in the form of a dictionary as
    {
        'fit_a': amplitude term
        'fit_b': decay term
        'fit_c': constant term
        'fit_a_error': error of amplitude term
        'fit_b_error': error of decay term
        'fit_c_error': error of constant term
    }
    with fit function as
    fit_a * exp(x / fit_b) + fit_c
    """

    eModel = lmfit.models.ExponentialModel(prefix = 'fit_')
    param = eModel.guess(ydata, x = xdata)
    if len(xerror) == 0:
        if len(yerror) == 0:
            data = RealData(xdata, ydata)
        else:
            data = RealData(xdata, ydata, sy = yerror)
    else:
        if len(yerror) == 0:
            data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
        else:
            data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
    if fitRate:
        model = Model(expFunctionNoConst)
        odrFit = ODR(data, model, [param.valuesdict()['fit_amplitude'], param.valuesdict()['fit_decay']])
    else:
        model = Model(expFunction)
        odrFit = ODR(data, model, [param.valuesdict()['fit_amplitude'], - param.valuesdict()['fit_decay'], 1e-6])
    if odr:
        odrFit.set_job(fit_type = 0)
    else:
        odrFit.set_job(fit_type = 2)
    result = odrFit.run()
    if fitRate:
        fitResult = {
                        'fit_a':            result.beta[0],
                        'fit_b':            result.beta[1],
                        'fit_a_err':            result.sd_beta[0],
                        'fit_b_err':            result.sd_beta[1],
            }
    else:
        fitResult = {
                        'fit_a':            result.beta[0],
                        'fit_b':            result.beta[1],
                        'fit_c':            result.beta[2],
                        'fit_a_err':            result.sd_beta[0],
                        'fit_b_err':            result.sd_beta[1],
                        'fit_c_err':            result.sd_beta[2],
            }
    return fitResult

def linExpFunction(param, x):

    """
    Auxiliary function for linear-exponential fitting of leak current-temperature curve
    :param param: parameters of the function, including a, b and c
    :param x: x value
    :return: the calculated funtion value as
        a * (b + (x + 273.15)) * exp(- c (1.1785 - 9.025e-5 * (x + 273.15) - 3.05e-7 * (x + 273.15) ** 2) / (x + 273.15)) + d
    """

    #return param[0] * (param[1] - (x + 273.15)) * np.exp(- param[2] / (x + 273.15 * np.ones(len(x))))
    return param[0] * (param[1] + (x + 273.15)) * np.exp(- param[2] * (1.1785 - 9.025e-5 * (x + 273.15 * np.ones(len(x))) - 3.05e-7 * (x + 273.15 * np.ones(len(x))) ** 2) / (x + 273.15 * np.ones(len(x)))) + param[3]

def doLinExpFit(xdata, ydata, init, odr = False, xerror = [], yerror = []):

    """
    Function for linear-exponential fitting of leak current-temperature curve
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param init: the initial parameter values
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :return: fit result, in the form of a dictionary as
    {
        'fit_a': amplitude
        'fit_b': linear term
        'fit_c': exponential term
        'fit_d': constant term
        'fit_a_err': error of amplitude
        'fit_b_err': error of linear term
        'fit_c_err': error of exponential term
        'fit_d_err': error of constant term
    }
    with fit function as
    fit_a * (fit_b + (x + 273.15)) * exp(- fit_c / (x + 273.15)) + fit_d
    """

    if len(xerror) == 0:
        if len(yerror) == 0:
            data = RealData(xdata, ydata)
        else:
            data = RealData(xdata, ydata, sy = yerror)
    else:
        if len(yerror) == 0:
            data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
        else:
            data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
    model = Model(linExpFunction)
    odrFit = ODR(data, model, list(init))
    if odr:
        odrFit.set_job(fit_type = 0)
    else:
        odrFit.set_job(fit_type = 2)
    result = odrFit.run()
    fitResult = {
                    'fit_a':            result.beta[0],
                    'fit_b':            result.beta[1],
                    'fit_c':            result.beta[2],
                    'fit_d':            result.beta[3],
                    'fit_a_err':            result.sd_beta[0],
                    'fit_b_err':            result.sd_beta[1],
                    'fit_c_err':            result.sd_beta[2],
                    'fit_d_err':            result.sd_beta[3],
        }
    return fitResult

def revExpFunction(param, x):

    """
    Auxiliary function for exponential fitting with reversed exponential term of leak current-temperature curve
    :param param: parameters of the function, including a and b
    :param x: x value
    :return: the calculated funtion value as
        a * exp(- b / (x + 273.15))
    """

    return param[0] * np.exp(- param[1] / (x + 273.15 * np.ones(len(x))))

def doRevExpFit(xdata, ydata, init, odr = False, xerror = [], yerror = []):

    """
    Function forexponential fitting with reversed exponential term of leak current-temperature curve
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param init: the initial parameter values
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :return: fit result, in the form of a dictionary as
    {
        'fit_a': amplitude
        'fit_b': exponential term
        'fit_a_err': error of amplitude
        'fit_b_err': error of exponential term
    }
    with fit function as
    fit_a * exp(- fit_b / (x + 273.15))
    """

    if len(xerror) == 0:
        if len(yerror) == 0:
            data = RealData(xdata, ydata)
        else:
            data = RealData(xdata, ydata, sy = yerror)
    else:
        if len(yerror) == 0:
            data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
        else:
            data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
    model = Model(revExpFunction)
    odrFit = ODR(data, model, list(init))
    if odr:
        odrFit.set_job(fit_type = 0)
    else:
        odrFit.set_job(fit_type = 2)
    result = odrFit.run()
    fitResult = {
                    'fit_a':            result.beta[0],
                    'fit_b':            result.beta[1],
                    'fit_a_err':            result.sd_beta[0],
                    'fit_b_err':            result.sd_beta[1],
        }
    return fitResult

def revLinExpFunction(param, x):

    """
    Auxiliary function for exponential fitting with reversed exponential term and a linear term of leak current-temperature curve
    :param param: parameters of the function, including a and b
    :param x: x value
    :return: the calculated funtion value as
        a * (x - c) * exp(- b / (x - c)) + d
    """

    return param[0] * (x - param[2]) * np.exp(- param[1] / (x - param[2])) + param[3]

def doRevLinExpFit(xdata, ydata, init, odr = False, xerror = [], yerror = []):

    """
    Function forexponential fitting with reversed exponential term and a linear term of leak current-temperature curve
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param init: the initial parameter values
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :return: fit result, in the form of a dictionary as
    {
        'fit_a': amplitude
        'fit_b': exponential term
        'fit_c': linear term
        'fit_d': constant term
        'fit_a_err': error of amplitude
        'fit_b_err': error of exponential term
        'fit_c_err': error of linear term
        'fit_d_err': error of constant term
    }
    with fit function as
    fit_a * (x - fit_c) * exp(- fit_b / (x - fit_c))
    """

    if len(xerror) == 0:
        if len(yerror) == 0:
            data = RealData(xdata, ydata)
        else:
            data = RealData(xdata, ydata, sy = yerror)
    else:
        if len(yerror) == 0:
            data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
        else:
            data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
    model = Model(revLinExpFunction)
    odrFit = ODR(data, model, list(init))
    if odr:
        odrFit.set_job(fit_type = 0)
    else:
        odrFit.set_job(fit_type = 2)
    result = odrFit.run()
    fitResult = {
                    'fit_a':            result.beta[0],
                    'fit_b':            result.beta[1],
                    'fit_c':            result.beta[2],
                    'fit_d':            result.beta[3],
                    'fit_a_err':            result.sd_beta[0],
                    'fit_b_err':            result.sd_beta[1],
                    'fit_c_err':            result.sd_beta[2],
                    'fit_d_err':            result.sd_beta[3],
        }
    return fitResult

def mixedExpFunction(param, x):

    """
    Auxiliary function for exponential fitting with mixed exponential term of leak current-temperature curve
    :param param: parameters of the function, including a and b
    :param x: x value
    :return: the calculated funtion value as
        a * exp(- b (1.1785 - 9.025e-5 * (x + 273.15) - 3.05e-7 * (x + 273.15) ** 2) / (x + 273.15)) + c
    """

    return param[0] * np.exp(- param[1] * (1.1785 - 9.025e-5 * (x + 273.15 * np.ones(len(x))) - 3.05e-7 * (x + 273.15 * np.ones(len(x))) ** 2) / (x + 273.15 * np.ones(len(x)))) + param[2]

def doMixedExpFit(xdata, ydata, init, odr = False, xerror = [], yerror = []):

    """
    Function forexponential fitting with mixed exponential term of leak current-temperature curve
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param init: the initial parameter values
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :return: fit result, in the form of a dictionary as
    {
        'fit_a': amplitude
        'fit_b': exponential term
        'fit_c': constant term
        'fit_a_err': error of amplitude
        'fit_b_err': error of exponential term
        'fit_c_err': error of constant term
    }
    with fit function as
    fit_a * exp(- fit_b * (1.1785 - 9.025e-5 * (x + 273.15) - 3.05e-7 * (x + 273.15) ** 2) * / (x + 273.15)) + fit_c
    """

    if len(xerror) == 0:
        if len(yerror) == 0:
            data = RealData(xdata, ydata)
        else:
            data = RealData(xdata, ydata, sy = yerror)
    else:
        if len(yerror) == 0:
            data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
        else:
            data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
    model = Model(mixedExpFunction)
    odrFit = ODR(data, model, list(init))
    if odr:
        odrFit.set_job(fit_type = 0)
    else:
        odrFit.set_job(fit_type = 2)
    result = odrFit.run()
    fitResult = {
                    'fit_a':            result.beta[0],
                    'fit_b':            result.beta[1],
                    'fit_c':            result.beta[2],
                    'fit_a_err':            result.sd_beta[0],
                    'fit_b_err':            result.sd_beta[1],
                    'fit_c_err':            result.sd_beta[2],
        }
    return fitResult

def fixedExpFunction(param, x):

    """
    Auxiliary function for fixed decay parameter exponential fitting of leak current-temperature curve
    :param param: parameters of the function, including only a
    :param x: x value
    :return: the calculated funtion value as
        a exp(- b / (x + 273.15)) + c
    where b = Eg(x)/2k = 1.1785 - 9.025e-5 * (x + 273.15) - 3.05e-7 * (x + 273.15) ** 2
    """

    return param[0] * np.exp(- 1.6e-19 * 1.1269 / (2 * 1.38e-23 * (x + 273.15 * np.ones(len(x))))) + param[1]

def doFixedExpFit(xdata, ydata, init, odr = False, xerror = [], yerror = []):

    """
    Function for linear-exponential fitting of leak current-temperature curve
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param init: the initial parameter values
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :return: fit result, in the form of a dictionary as
    {
        'fit_a': amplitude
        'fit_b': constant term
        'fit_a_err': error of amplitude
        'fit_b_err': error of constant term
    }
    with fit function as
    fit_a exp(- b / (x + 273.15)) + fit_b
    where b = Eg(x)/2k = 1.1785 - 9.025e-5 * (x + 273.15) - 3.05e-7 * (x + 273.15) ** 2 ~ 1.1269eV @ 0-25℃
    """

    if len(xerror) == 0:
        if len(yerror) == 0:
            data = RealData(xdata, ydata)
        else:
            data = RealData(xdata, ydata, sy = yerror)
    else:
        if len(yerror) == 0:
            data = RealData(xdata, ydata, sx = xerror, fix = np.ones(len(xerror)))
        else:
            data = RealData(xdata, ydata, sx = xerror, sy = yerror, fix = np.ones(len(xerror)))
    model = Model(fixedExpFunction)
    odrFit = ODR(data, model, list(init))
    if odr:
        odrFit.set_job(fit_type = 0)
    else:
        odrFit.set_job(fit_type = 2)
    result = odrFit.run()
    fitResult = {
                    'fit_a':            result.beta[0],
                    'fit_b':            result.beta[1],
                    'fit_a_err':            result.sd_beta[0],
                    'fit_b_err':            result.sd_beta[1],
        }
    return fitResult

def convExpFunction(param, x):

    """
    Auxiliary function for live time fitting, the log value of distribution being reverse distribution of convolution of 43 exponential distributions
    :param param: parameters of the function, including only a
    :param x: x value
    :return: the calculated funtion value as
        ln(A * (x - 43 * C) ** 42 * exp(- (x - 43 * C) / b) / b ** 43) = A + 42 * ln(x - 43 * C) - 43 * ln(b) - (x - 43 * C) / b
    where C = 43 * 50us is the internal dead time of the MCU
    """
    
    C = 50e-6
    return param[0] + 42.0 * np.log(x - 43 * C) - 43.0 * np.log(param[1]) - (x - 43 * C) / param[1]

def doConvExpFit(xdata, ydata, init, odr = False, xerror = [], yerror = []):

    """
    Function for convolution-of-exponential fitting of count rate correction
    :param xdata: the x data to be fit
    :param ydata: the y data to be fit
    :param init: the initial parameter values
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xerror: standard deviation of x data when using odr fit
    :param yerror: standard deviation of y data when using odr fit
    :return: fit result, in the form of a dictionary as
    {
        'fit_a': amplitude
        'fit_b': deacy
        'fit_a_err': error of amplitude
        'fit_b_err': error of decay
    }
    with fit function as
    A * (x - 43 * C) ** 42 * exp(- (x - 43 * C) / b)
    where C = 50us is the internal dead time of the MCU
    """

    if len(xerror) == 0:
        if len(yerror) == 0:
            data = RealData(xdata, np.log(ydata))
        else:
            data = RealData(xdata, np.log(ydata), sy = yerror / ydata)
    else:
        if len(yerror) == 0:
            data = RealData(xdata, np.log(ydata), sx = xerror, fix = np.ones(len(xerror)))
        else:
            data = RealData(xdata, np.log(ydata), sx = xerror, sy = yerror / ydata, fix = np.ones(len(xerror)))
    model = Model(convExpFunction)
    odrFit = ODR(data, model, list(init))
    if odr:
        odrFit.set_job(fit_type = 0)
    else:
        odrFit.set_job(fit_type = 2)
    result = odrFit.run()
    fitResult = {
                    'fit_a':            result.beta[0],
                    'fit_b':            result.beta[1],
                    'fit_a_err':            result.sd_beta[0],
                    'fit_b_err':            result.sd_beta[1],
        }
    return fitResult

def quad3DFunction(param, xdata, ydata):

    """
    Auxiliary function to calculate the value of 3D quadratic function for correlative temperature-bias fit
    :param param: parameters for the 3D quadratic model, described as in the function given below
    :param xdata: data of x dimension
    :param ydata: data of y dimension
    :return: vaule of quad3D(param, xdata, ydata)
    where quad3D(param, xdata, ydata) = 
        param[0] + param[1] * xdata + param[2] * ydata + param[3] * xdata ** 2 + param[4] * ydata ** 2 + param[5] * xdata * ydata
    """

    return param[0] + param[1] * xdata + param[2] * ydata + param[3] * xdata ** 2 + param[4] * ydata ** 2 + param[5] * xdata * ydata

def residualQuad3D(param, xdata, ydata, zdata):

    """
    Auxiliary function to calculate the residual of 3D quadratic surface model for correlative temperature-bias fit
    :param param: parameters for the 3D quadratic model, described as in the function given below
    :param xdata: data of x dimension
    :param ydata: data of y dimension
    :param zdata: data of z dimension
    :return: residual being zdata - quad3D(param, xdata, ydata)
    where quad3D(param, xdata, ydata) = 
        param[0] + param[1] * xdata + param[2] * ydata + param[3] * xdata ** 2 + param[4] * ydata ** 2 + param[5] * xdata * ydata
    """
    
    return zdata - quad3DFunction(param, xdata, ydata)

def fitRateCorrect(filename, timeCorrect, plot = True, odr = False, rateStyle = ''):

    """
    Function for calculating correct total count rate and its error with the timeCorrect calculated when reading the data
    :param filename: name of amplitude data file
    :param timeCorrect: correct live time calculated when reading data
    :param plot: boolean indicating the whether the plotting of the fit result will be done
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :return: correct total count rate and its error, as a tuple
    """

    styleAvailable = ['s', 'p']
    if not rateStyle in styleAvailable:
        raise Exception('fitRateCorrect: unknown count rate correction style ' + rateStyle)
    rateAll = 0.0
    rateAllErr = 0.0
    specTime, xdata = np.histogram(timeCorrect, bins=250, range=[np.min(timeCorrect), np.max(timeCorrect)])
    xdata = (xdata[:-1] + xdata[1:]) / 2
    C = 50e-6
    if rateStyle == 's':
        qFit = xdata > C
        result = doFitExp(xdata[qFit], specTime[qFit], odr = odr, yerror = gehrelsErr(specTime[qFit]), fitRate = True)
        a = result['fit_a']
        b = result['fit_b']
        bErr = result['fit_b_err']
    else:
        qFit = (xdata > 43 * C) * (specTime > 0)
        result = doConvExpFit(xdata[qFit], specTime[qFit], (1., C), odr = odr, yerror = gehrelsErr(specTime[qFit]))
        a = result['fit_a']
        b = result['fit_b']
        bErr = result['fit_b_err']
    rateAll = 1 / b
    rateAllErr = bErr / b ** 2
    if plot:
        if rateStyle == 's':
            fitSpec = expFunctionNoConst([a, b], xdata)
            qPlot = xdata > C
        else:
            fitSpec = np.exp(convExpFunction([a, b], xdata))
            qPlot = xdata > 43 * C
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
        ax = fig.add_subplot(gs[0])
        plt.step(xdata, specTime, where='mid', label='raw data', zorder=1)
        if rateStyle == 's':
            plt.plot(xdata[qPlot], fitSpec[qPlot], label='Exponential Fit')
        else:
            plt.plot(xdata[qPlot], fitSpec[qPlot], label='Exponential-convolution Fit')
        plt.text(np.average(timeCorrect), max(specTime) * 1.1, 'live time = ' + str('%.2e' % (b)) + ' $\pm$ ' + str('%.3e' % (bErr)) + 's\ncount rate = ' + \
            str('%.2e' % (rateAll)) + ' $\pm$ ' + str('%.3e' % (rateAllErr)) + 'cps', fontsize=10, bbox=dict(facecolor='pink', alpha=0.1), horizontalalignment='center', \
            verticalalignment='center')
        ax.set_ylim([0, 1.2 * np.max(specTime)])
        if rateStyle == 's':
            ax.set_title('Fit of live time of ' + filename + '\nFit function: ' + r'$A e^{- \frac{x - C}{b}}$')
        else:
            ax.set_title('Fit of live time of ' + filename + '\nFit function: ' + r'$\frac{A (x - 43 C)^{42}}{(42)!b^{43}} e^{- (x - 43C) / b}$')
        ax.set_xlabel('live time/s')
        ax.set_ylabel('count in bins')
        ax.legend(loc=0)
        ax.grid()
        plt.show()
    return rateAll, rateAllErr

def fitSpectrum(filename, amp, nbins, source, corr, time, fileOutput = False, singlech = False, bkg = False, odr = False, xRange = [],\
 channel = -1, corrErr = [], bkgAmp = [], bkgtime = [], maxiter = 1, bound = 3.0, plot = True, rateStyle = '', rateAll = 0.0, rateAllErr = 0.0,\
 bkgRate = 0.0, bkgRateErr = 0.0, quadBkg = True, doCorr = True):
    
    """
    Function for fitting the spectrum
    :param filename: name of amplitude data file
    :param amp: ADC amplitude data
    :param nbins: number of bins to be used in the spectrum
    :param source: the source of the spectrum, currently supporting sources: Am241, Ba133, Cs137, Na22, Co60, x
    :param corr: temperature-bias correction factors for the data
    :param time: time taken to take the spectrum, usually calculated with event uscount
    :param fileOutput: the output style, False for only graph output, True for text file(.txt) output as well as graph output
    :param singlech: boolean indicating whether the fit is for single channel
    :param bkg: boolean indicating if the corresponding background is given. The corresponding total measurement times(source \
and background) should be given in *args along with the background amplitude
    :param odr: boolean indicating the fit method, True if the fit is done with odr, False if the fit is done with least square
    :param xRange: specific fit range for x-ray fits or uncalibrated fits, in the form of [lower, upper]
    :param channel: channel number for single channel fits, in range[0-3]
    :param corrErr: error of temperature-bias correction factors used for odr fits
    :param bkgAmp: amplitude of background data, in the form of list[list]
    :param bkgtime: total measurement time of background data
    :param maxiter: maximum number of iterations, if auto-correction iteration is conducted
    :param bound: boundary for auto-correction in \sigma, with boundaries being \mu - bound * \sigma and \mu + bound * \sigma
    :param plot: boolean indicating the whether the plotting of the fit result will be done
    :param rateStyle: the style of calculating real count rate, '' for none, 's' for single live time data, 'p' for calculation with small data packs(512byte)
    :param rateAll: correct count rate of all spectrum in all 4 channels calculated when reading data, only used when rateStyle is 's' or 'l'
    :param rateAllErr: error of rateAll
    :param bkgRate: correct count rate of all background spectrum in all 4 channels calculated when reading data, only used when rateStyle is 's' or 'l'
    :param bkgRateErr: error of bkgRate
    :param quadBkg: boolean indicating whether the quadratic background is included in the fit function
    :param doCorr: boolean indicating whether the temperature-bias correction will be done, to avoid warning info output
    :return: fit parameters of the gaussian peak to be used for experiment-level processing, in the form of a dictionary:
        {
            'a':        amplitude,
            'b':        center,
            'c':        sigma,
            'a_err':    error of amplitude,
            'b_err':    error of center,
            'c_err':    error of sigma,
            'rate':        peak count rate,
            'rate_err':        error of peak count rate,
        }
    Suggested nbins for various sources at normal temperature and bias level:
    Am241, Ba133: 2048
    Cs137, Na22, Co60: 512
    Th228: 1024
    x: varies with different circumstances, try plotting and finding the range a bit more, MAY IMPLEMENT AN AUTOMATIC RANGE-\
FINDING WITH THE RESOLUTION FIT DATA
    """
    
    #Reference fit range data
    fitRange = {
            'Am241':    np.array([[1100, 2800], [1100, 2300], [1100, 2600], [1200, 2900]]),
            'Ba133':    np.array([[2500, 4500], [2800, 5500], [2800, 5500], [3200, 5500]]),
            'Na22':     np.array([[12800, 18000], [12000, 16700], [14000, 18000], [14000, 18600]]),
            'Cs137':    np.array([[16600, 23100], [16600, 21800], [17200, 25000], [18500, 25000]]),
            'Th228':    np.array([[5100, 7700], [5700, 8400], [5400, 8000], [5700, 8400]]),
            'Co60':    np.array([[35840, 41000], [35200, 39700], [38400, 44200], [40320, 46100]]),
        }

    if not source in fitRange or len(xRange) == 2:
        if source == 'x':
            if singlech:
                if not (len(xRange) == 2 and xRange[0] > 0 and xRange[1] > xRange[0]):
                    raise Exception('fitSpectrum: illegal format for fit range, please make sure the fit range is in the form of [lower, upper] and 0 < lower < upper')
                rangeLim = xRange
            else:
                for ich in range(4):
                    if not (len(xRange[ich]) == 2 and xRange[ich][0] > 0 and xRange[ich][1] > xRange[ich][0]):
                        raise Exception('fitSpectrum: illegal format for fit range, please make sure the fit range is in the form of [lower, upper] and 0 < lower < upper')
                    rangeLim = xRange
        else:
            raise Exception('fitSpectrum: unknown source, supported sources: Am241, Ba133, Cs137, Na22, Co60, x')
    else:
        if singlech:
            rangeLim = fitRange[source][channel]
            if not doCorr:
                try:
                    for ir in range(2):
                        rangeLim[ir]  = rangeLim[ir] / corr[channel]
                except:
                    pass
                corr = [1.0, 1.0, 1.0, 1.0]
                corrErr = []
        else:
            rangeLim = fitRange[source]
            if not doCorr:
                try:
                    for ich in range(4):
                        for ir in range(2):
                            rangeLim[ich][ir] = rangeLim[ich][ir] / corr[ich]
                except:
                    pass
                corr = [1.0, 1.0, 1.0, 1.0]
                corrErr = []

    if fileOutput:
        try:
            fout = open('fit_' + filename, 'w')
        except:
            raise Exception('fitSpectrum: Error opening output file')
        
    if singlech:
        if not isChannel(channel):
            raise Exception('fitSpectrum: channel number out of bound[0-3]')
        spectrum, xraw = getSpectrum(amp[channel], nbins, singlech)
        spectrumStatErr = gehrelsErr(spectrum)
    else:
        spectrum, xraw = getSpectrum(amp, nbins, singlech)
        spectrumStatErr = []
        for ich in range(4):
            spectrumStatErr.append(gehrelsErr(spectrum[ich]))
            
    if not rateStyle == '':
        countAll = 0.0
        for ich in range(4):
            countAll += float(len(amp[ich]))
        rateFactor = rateAll / countAll
        rateFactorErr = rateAllErr / countAll
        if singlech:
            spectrumErr = np.sqrt((spectrum * rateFactorErr) ** 2 + (spectrumStatErr * rateFactor) ** 2)
            spectrum = spectrum * rateFactor
        else:
            spectrumErr = []
            for ich in range(4):
                spectrumErr.append([])
                spectrumErr[ich] = np.sqrt((spectrum[ich] * rateFactorErr) ** 2 + (spectrumStatErr[ich] * rateFactor) ** 2)
            spectrumErr = np.array(spectrumErr)
            for ich in range(4):
                spectrum[ich] = spectrum[ich] * rateFactor
    else:
        spectrumErr = np.array(spectrumStatErr)
        
    if bkg:
        timeScale = []
        if len(bkgAmp) == 0 or (rateStyle == '' and len(bkgtime) == 0):
            raise Exception('fitSpectrum: background data not given')
        if rateStyle == '':
            for ich in range(4):
                timeScale.append(float(time[ich]) / float(bkgtime[ich]))
        else:
            bkgCountAll = 0.0
            for ich in range(4):
                bkgCountAll += float(len(bkgAmp[ich]))
            bkgRateFactor = bkgRate / bkgCountAll
            bkgRateFactorErr = bkgRateErr / bkgCountAll
        if singlech:
            bkgSpectrum = getSpectrum(bkgAmp[channel], nbins, singlech)[0]
            bkgSpectrumStatErr = gehrelsErr(bkgSpectrum)
            if rateStyle == '':
                spectrum = spectrum - bkgSpectrum * timeScale[channel]
                spectrumErr = np.sqrt(spectrumErr ** 2 + (bkgSpectrumStatErr * timeScale[channel]) ** 2)
            else:
                spectrum = spectrum - bkgSpectrum * bkgRateFactor
                spectrumErr = np.sqrt(spectrumErr ** 2 + (bkgSpectrum * bkgRateFactorErr) ** 2 + (bkgSpectrumStatErr * bkgRateFactor) ** 2)
        else:
            bkgSpectrum = getSpectrum(bkgAmp, nbins, singlech)[0]
            bkgSpectrumStatErr = []
            for ich in range(4):
                bkgSpectrumStatErr.append(gehrelsErr(bkgSpectrum[ich]))
            if rateStyle == '':
                for ich in range(4):
                    spectrum[ich] = spectrum[ich] - bkgSpectrum[ich] * timeScale[ich]
                    spectrumErr[ich] = np.sqrt(spectrumErr[ich] ** 2 + (bkgSpectrumStatErr[ich] * timeScale[channel]) ** 2)
            else:
                for ich in range(4):
                    spectrum[ich] = spectrum[ich] - bkgSpectrum[ich] * bkgRateFactor
                    spectrumErr[ich] = np.sqrt(spectrumErr[ich] ** 2 + (bkgSpectrum[ich] * bkgRateFactorErr) ** 2 + (bkgSpectrumStatErr[ich] * bkgRateFactor) ** 2)
        if not rateStyle == '':
            rateFactor -= bkgRateFactor
            rateFactorErr = np.sqrt(rateFactorErr ** 2 + bkgRateFactorErr ** 2)
                    
    minDiff = 1e-4
    #multi-channel fits
    if not singlech:
        if plot:
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(4, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
        fitResult = []

        x = copy(xraw)
        for ich in range(4):
            x[ich] *= corr[ich] #temperature-bias correction
            spectrum[ich] = spectrum[ich] * nbins / 65535
            spectrumErr[ich] = spectrumErr[ich] * nbins / 65535
            q1 = (x[ich] >= rangeLim[ich][0] * corr[ich]) * (x[ich] < rangeLim[ich][1] * corr[ich])
            if not len(corrErr) == 0:
                result = doFitPeak(x[ich][q1], spectrum[ich][q1], odr, x[ich][q1] * corrErr[ich], yerror = spectrumErr[ich][q1], quadBkg = quadBkg)
            else:
                result = doFitPeak(x[ich][q1], spectrum[ich][q1], odr, yerror = spectrumErr[ich][q1], quadBkg = quadBkg)
            amplitude = result['peak_amplitude']
            center = result['peak_center']
            sigma = result['peak_sigma']
            if quadBkg:
                a = result['bk_a']
                b = result['bk_b']
                c = result['bk_c']
            lastCenter, lastSigma = center, sigma

            #Iteration part
            for iit in range(maxiter):
                q2 = (x[ich] >= center - bound * sigma) * (x[ich] < center + bound * sigma)
                if not len(corrErr) == 0:
                    result = doFitPeak(x[ich][q2], spectrum[ich][q2], odr, x[ich][q2] * corrErr[ich], yerror = spectrumErr[ich][q2], quadBkg = quadBkg)
                else:
                    result = doFitPeak(x[ich][q2], spectrum[ich][q2], odr, yerror = spectrumErr[ich][q2], quadBkg = quadBkg)
                amplitude = result['peak_amplitude']
                center = result['peak_center']
                sigma = result['peak_sigma']
                if quadBkg:
                    a = result['bk_a']
                    b = result['bk_b']
                    c = result['bk_c']
                if np.abs((lastCenter - center) / lastCenter) < minDiff and np.abs((lastSigma - sigma) / lastSigma) < minDiff:
                    break
                lastCenter, lastSigma = center, sigma

            ampErr = result['peak_amplitude_err']
            centerErr = result['peak_center_err']
            sigmaErr = result['peak_sigma_err']
            resolution = 2 * np.sqrt(2 * np.log(2)) * sigma / center
            resolutionErr = 2 * np.sqrt(2 * np.log(2)) * np.sqrt((sigmaErr / center) ** 2 + (sigma * centerErr / center ** 2) ** 2)
            #Count rate calculation part
            qRate = (x[ich] >= center - bound * sigma) * (x[ich] < center + bound * sigma)
            resultRate = doFitPeak(xraw[ich][qRate], spectrum[ich][qRate], odr, yerror = spectrumErr[ich][qRate], quadBkg = quadBkg)
            ampRate = resultRate['peak_amplitude']
            if rateStyle == '':
                rate = ampRate / time[ich]
                rateErr = np.sqrt(ampRate) / time[ich]
            else:
                rate = ampRate
                rateErr = np.sqrt(ampRate * rateFactor + (ampRate * rateFactorErr / rateFactor) ** 2)

            if fileOutput:
                try:
                    fout.write('Channel ' + str(ich) +': \n')
                    if quadBkg:
                        fout.write(str(amplitude / (sigma * np.sqrt(2 * np.pi))) + ' * exp(-((x - ' + str(center) + ')^2 / ' + str(2 * (sigma ** 2)) + ')) + '\
                           + str(a) + ' * x^2 + ' + str(b) + ' * x + ' + str(c) + '\n')
                    else:
                        fout.write(str(amplitude / (sigma * np.sqrt(2 * np.pi))) + ' * exp(-((x - ' + str(center) + ')^2 / ' + str(2 * (sigma ** 2)) + '))\n')
                    fout.write('Peak count rate ' + str(rate) + '\n')
                    fout.write('Center error: ' + str(centerErr) + ', variance error: ' + str(sigmaErr) + ', count rate error: '  + str(rateErr) + '\n')
                    fout.write('Energy resolution: ' + str(resolution) + '\n')
                    fout.write('Resolution error: ' + str(resolutionErr) + '\n')
                except:
                    raise Exception('fitSpectrum: Error writing output file ' + fout.name)

            fitResult.append({
                                        'a':            amplitude,
                                        'b':            center,
                                        'c':            sigma,
                                        'a_err':        ampErr,
                                        'b_err':        centerErr,
                                        'c_err':        sigmaErr,
                                        'rate':        rate,
                                        'rate_err':        rateErr,
                })

            #Plot part
            if plot:
                fitPeak = amplitude * np.exp(-(x[ich] - center) ** 2 / (2 * (sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
                if quadBkg:
                    fitBk = a * x[ich] ** 2 + b * x[ich] + c
                    fitTotal = fitBk + fitPeak
                else:
                    fitTotal = fitPeak
                qplot = (x[ich] >= center - bound * sigma) * (x[ich] < center + bound * sigma)
                ax = fig.add_subplot(gs[ich])
                if rateStyle == '':
                    plt.step(x[ich], spectrum[ich] / time[ich], where='mid', label='raw data', zorder=1) #TBD: change the x data to energy with additional EC calculation funtcion
                    plt.plot(x[ich][qplot], fitPeak[qplot] / time[ich], label='Gaussian Peak Fit')
                    if quadBkg:
                        plt.plot(x[ich][qplot], fitBk[qplot] / time[ich], label='Background Fit')
                        plt.plot(x[ich][qplot], fitTotal[qplot] / time[ich], label='Gaussian Peak and Background Fit')
                    plt.text(center - 4 * sigma, max(fitPeak[qplot] / time[ich]) * 0.8, 'count rate = ' + str('%.2e' % (rate)) + ' $\pm$ ' + str('%.3e' % (rateErr)) + \
                        ', center = ' + str('%.3e' % center) + ' $\pm$ ' + str('%.3e' % centerErr) + '\nsigma = ' + str('%.3e' % sigma) + ' $\pm$ ' + str('%.3e' % sigmaErr) + ', resolution = ' + \
                        str('%.3e' % resolution) + ' $\pm$ ' + str('%.3e' % resolutionErr), fontsize=10, bbox=dict(facecolor='pink', alpha=0.1), horizontalalignment='center', verticalalignment='center')
                    ax.set_ylim([0, 1.2 * np.max(spectrum[ich][qplot] / time[ich])])
                else:
                    plt.step(x[ich], spectrum[ich], where='mid', label='raw data', zorder=1) #TBD: change the x data to energy with additional EC calculation funtcion
                    plt.plot(x[ich][qplot], fitPeak[qplot], label='Gaussian Peak Fit')
                    if quadBkg:
                        plt.plot(x[ich][qplot], fitBk[qplot], label='Background Fit')
                        plt.plot(x[ich][qplot], fitTotal[qplot], label='Gaussian Peak and Background Fit')
                    plt.text(center - 4 * sigma, max(fitPeak[qplot]) * 0.8, 'count rate = ' + str('%.2e' % (rate)) + ' $\pm$ ' + str('%.3e' % (rateErr)) + ', center = ' + str('%.3e' % center) + \
                        ' $\pm$ ' + str('%.3e' % centerErr) + '\nsigma = ' + str('%.3e' % sigma) + ' $\pm$ ' + str('%.3e' % sigmaErr) + ', resolution = ' + str('%.3e' % resolution) + ' $\pm$ ' + \
                        str('%.3e' % resolutionErr), fontsize=10, bbox=dict(facecolor='pink', alpha=0.1), horizontalalignment='center', verticalalignment='center')
                    ax.set_ylim([0, 1.2 * np.max(spectrum[ich][qplot])])
                lower = center - 6 * sigma
                if lower < 0:
                    lower = 0
                upper = center + 6 * sigma
                if upper > 65535:
                    upper = 65535
                ax.set_xlim([lower, upper])
                if ich == 0:
                    ax.set_title('Spectrum fit of data from ' + filename + '\nFit function: ' + r'$\frac{a}{\sqrt{2\pi}\sigma} e^{[{-{(x - \mu)^2}/{{2\sigma}^2}}]}$')
                ax.set_xlabel('ADC/channel')
                ax.set_ylabel('count rante/cps')
                ax.legend(loc=0)
                ax.grid()
        if plot:
            plt.show()

        if fileOutput:
            fout.close()

    #single channel fits
    else:
        ich = channel
        if ich < 0 or ich > 3:
            raise Exception('fitSpectrum: channel number out of bound[0-3]')
        x = xraw * corr[ich] #temperature-bias correction
        spectrum = spectrum * nbins / 65535
        spectrumErr = spectrumErr * nbins / 65535
        q1 = (x >= rangeLim[0] * corr[ich]) * (x < rangeLim[1] * corr[ich])
        if not len(corrErr) == 0:
            result = doFitPeak(x[q1], spectrum[q1], odr, x[q1] * corrErr[ich], yerror = spectrumErr[q1], quadBkg = quadBkg)
        else:
            result = doFitPeak(x[q1], spectrum[q1], odr, yerror = spectrumErr[q1], quadBkg = quadBkg)
        amplitude = result['peak_amplitude']
        center = result['peak_center']
        sigma = result['peak_sigma']
        if quadBkg:
            a = result['bk_a']
            b = result['bk_b']
            c = result['bk_c']
        lastCenter, lastSigma = center, sigma

        #Iteration part
        for iit in range(maxiter):
            q2 = (x >= center - bound * sigma) * (x < center + bound * sigma)
            if not len(corrErr) == 0:
                result = doFitPeak(x[q2], spectrum[q2], odr, x[q2] * corrErr[ich], yerror = spectrumErr[q2], quadBkg = quadBkg)
            else:
                result = doFitPeak(x[q2], spectrum[q2], odr, yerror = spectrumErr[q2], quadBkg = quadBkg)
            amplitude = result['peak_amplitude']
            center = result['peak_center']
            sigma = result['peak_sigma']
            if quadBkg:
                a = result['bk_a']
                b = result['bk_b']
                c = result['bk_c']
            if np.abs((lastCenter - center) / lastCenter) < minDiff and np.abs((lastSigma - sigma) / lastSigma) < minDiff:
                break
            lastCenter, lastSigma = center, sigma
            
        ampErr = result['peak_amplitude_err']
        centerErr = result['peak_center_err']
        sigmaErr = result['peak_sigma_err']
        resolution = 2 * np.sqrt(2 * np.log(2)) * sigma / center
        resolutionErr = 2 * np.sqrt(2 * np.log(2)) * np.sqrt((sigmaErr / center) ** 2 + (sigma * centerErr / center ** 2) ** 2)
        #Count rate calculation part
        qRate = (x >= center - bound * sigma) * (x < center + bound * sigma)
        resultRate = doFitPeak(xraw[qRate], spectrum[qRate], odr, yerror = spectrumErr[qRate], quadBkg = quadBkg)
        ampRate = resultRate['peak_amplitude']
        if rateStyle == '':
            rate = ampRate / time[ich]
            rateErr = np.sqrt(ampRate) / time[ich]
        else:
            rate = ampRate
            rateErr = np.sqrt(ampRate * rateFactor + (ampRate * rateFactorErr / rateFactor) ** 2)

        if fileOutput:
            fout.write('Channel ' + str(ich) +': \n')
            if quadBkg:
                fout.write(str(amplitude / (sigma * np.sqrt(2 * np.pi))) + ' * exp(-((x - ' + str(center) + ')^2 / ' + str(2 * (sigma ** 2)) + ')) + '\
                   + str(a) + ' * x^2 + ' + str(b) + ' * x + ' + str(c) + '\n')
            else:
                fout.write(str(amplitude / (sigma * np.sqrt(2 * np.pi))) + ' * exp(-((x - ' + str(center) + ')^2 / ' + str(2 * (sigma ** 2)) + '))\n')
            fout.write('Peak count rate ' + str(rate) + '\n')
            fout.write('Center error: ' + str(centerErr) + ', variance error: ' + str(sigmaErr) + ', count rate error: '  + str(rateErr) + '\n')
            fout.write('Energy resolution: ' + str(resolution) + '\n')
            fout.write('Resolution error: ' + str(resolutionErr) + '\n')
            fout.close()

        fitResult = {
                                        'a':            amplitude,
                                        'b':            center,
                                        'c':            sigma,
                                        'a_err':        ampErr,
                                        'b_err':        centerErr,
                                        'c_err':        sigmaErr,
                                        'rate':        rate,
                                        'rate_err':        rateErr,
                }

        #Plot part
        if plot:
            fitPeak = amplitude * np.exp(-(x - center) ** 2 / (2 * (sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
            if quadBkg:
                fitBk = a * x ** 2 + b * x + c
                fitTotal = fitBk + fitPeak
            else:
                fitTotal = fitPeak
            qplot = (x >= center - bound * sigma) * (x < center + bound * sigma)
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.2, left=0.13, right=0.95)
            ax = fig.add_subplot(gs[0])
            if rateStyle == '':
                plt.step(x, spectrum / time[ich], where='mid', label='raw data', zorder=1) #TBD: change the x data to energy with additional EC calculation funtcion
                plt.plot(x[qplot], fitPeak[qplot] / time[ich], label='Gaussian Peak Fit')
                if quadBkg:
                    plt.plot(x[qplot], fitBk[qplot] / time[ich], label='Background Fit')
                    plt.plot(x[qplot], fitTotal[qplot] / time[ich], label='Gaussian Peak and Background Fit')
                plt.text(center - 4 * sigma, max(fitPeak[qplot] / time[ich]) * 0.8, 'count rate = ' + str('%.2e' % (rate)) + ' $\pm$ ' + str('%.3e' % (rateErr)) + \
                    ', center = ' + str('%.3e' % center) + ' $\pm$ ' + str('%.3e' % centerErr) + '\nsigma = ' + str('%.3e' % sigma) + ' $\pm$ ' + str('%.3e' % sigmaErr) + ', resolution = ' + \
                    str('%.3e' % resolution) + ' $\pm$ ' + str('%.3e' % resolutionErr), fontsize=10, bbox=dict(facecolor='pink', alpha=0.1), horizontalalignment='center', verticalalignment='center')
                ax.set_ylim([0, 1.2 * np.max(spectrum[qplot] / time[ich])])
            else:
                plt.step(x, spectrum, where='mid', label='raw data', zorder=1) #TBD: change the x data to energy with additional EC calculation funtcion
                plt.plot(x[qplot], fitPeak[qplot], label='Gaussian Peak Fit')
                if quadBkg:
                    plt.plot(x[qplot], fitBk[qplot], label='Background Fit')
                    plt.plot(x[qplot], fitTotal[qplot], label='Gaussian Peak and Background Fit')
                plt.text(center - 4 * sigma, max(fitPeak[qplot]) * 0.8, 'count rate = ' + str('%.2e' % (rate)) + ' $\pm$ ' + str('%.3e' % (rateErr)) + ', center = ' + str('%.3e' % center) + \
                    ' $\pm$ ' + str('%.3e' % centerErr) + '\nsigma = ' + str('%.3e' % sigma) + ' $\pm$ ' + str('%.3e' % sigmaErr) + ', resolution = ' + str('%.3e' % resolution) + ' $\pm$ ' + \
                    str('%.3e' % resolutionErr), fontsize=10, bbox=dict(facecolor='pink', alpha=0.1), horizontalalignment='center', verticalalignment='center')
                ax.set_ylim([0, 1.2 * np.max(spectrum[qplot])])
            lower = center - 6 * sigma
            if lower < 0:
                lower = 0
            upper = center + 6 * sigma
            if upper > 65535:
                upper = 65535
            ax.set_xlim([lower, upper])
            if ich == 0:
                ax.set_title('Spectrum fit of data from channel ' + str(ich) + ' in ' + filename + '\nFit function: ' + r'$\frac{a}{\sqrt{2\pi}\sigma} e^{[{-{(x - \mu)^2}/{{2\sigma}^2}}]}$')
            ax.set_xlabel('ADC/channel')
            ax.set_ylabel('count rate/cps')
            ax.legend(loc=0)
            ax.grid()
            plt.show()

    return fitResult
