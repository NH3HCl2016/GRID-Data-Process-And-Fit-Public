
#Grid Data Processor v0.0.2 by ydx and ghz
#Used for Grid2 calibration
#This is the main module, with the basic functions stored in 'gridBasicFunctions.py'

import gridBasicFunctions as grid
import gridExperimentFunctions as experiment
import sys
import os
import numpy as np

#******************************************************************************************************************************************************
#***************************************************************Auxiliary functions*****************************************************************
#******************************************************************************************************************************************************

def printUsage():
    #TBD: update the options when adding new options
    """
    Function to print usage, including options and sources available
    :return: nothing
    """

    print('Usage: python [.\\]GridDataProcessor [option] [--nbins nbins] [--src source [--range fitRange]] [--ch channel] [--dir/--mul] filename1 [--scan scanRange] [--dir/--mul] filename2 [--scan scanRange] [--bkg backgroundfile] ...')
    print('Available options:')
    print('\'h\': Hexprint files')
    print('\'c\': Files with (single) CI(charge injection)')
    print('\'s\': Files with I-V scan')
    print('\'o\': Save the process results in text files(.txt)')
    print('\'f\': Fit the data spectrum')
    print('\'g\': Do only gaussian peak fitting with no quadratic background in fit session')
    print('\'p\': Plot the processed data from the input file(s)')
    print('\'t\': Do temperature analysis and give the calibration results')
    print('\'b\': Do bias analysis and give the calibration results')
    #print('\'i\': Import data from output files rather than original file(usually used for multiple scans)')
    print('\'v\': Plot the temperature, bias and monitored voltage and current curves, and do fitting for monotored current and overvoltage against temperature/bias')
    print('\'r\': Do temperature/bias responce analysis and give the fit results')
    print('\'O\': Use ODR(Orthogonal distance regression) fit method')
    print('\'n\': New hardware programme(6th ver.) with added data of effective and missing counts')
    print('\'a\': Plot angular responce')
    print('\'m\': Plot EC, Energy resolution and absolute efficiency of data from NIM. Necessary options include \'--ch\', \'--gridfile\' and \'--hpgefile\'')
    print('Other options:')
    print('\'--dir\': Process all text(.txt) files of SINGLE RUN in the given directory. Please DO NOT give additional filenames other than the directory and DO NOT give directories containing multiple scan files')
    print('\'--nbins\': Number of bins for the spectrum, in the range (0-65536]')
    print('\'--src\': Name of source')
    print('  Available sources:')
    print('  \'Am241\', \'Cs137\', \'Na22\', \'Th228\', \'Co60\', \'x\'')
    print('\'--mul\': Files with multiple scans. The scan numbers of interest can be specified with a following option \'--scan\'')
    print('\'--scan\': Scan numbers to be processed for specified multiple scan file')
    print('  For multiple scan files the scans of interest can be specified in the form of \'[lower upper]\' or just \'lower upper\', only scans from #lower to #upper will be processed')
    print('\'--bkg\': set the background file')
    print('  For single channel fits or x-ray spectrum fits, please specify the channel number with a following option \'--ch\' or the fit range with a following option \'--range\'')
    print('\'--ch\': Channel number in range [0-3]')
    print('\'--range\': Fit range in ADC channels')
    print('  For x-ray fits, please give the fit range in the form of \'[lower upper]\' or just \'lower upper\'')
    print('\'--nocorr\': Do not do temperature-bias correction')
    print('\'--iter\': Maximum number of iteration for spectrum fitting, 0 for no iteration')
    print('\'--sigma\': Boundary for auto-correction fit in \sigma, with boundaries being \mu - bound * \sigma and \mu + bound * \sigma')
    print('\'--noplot\': Do not plot the spectrum during fit session')
    print('\'--simu\': Give simulation result for angular responce')
    print('\'--rate\': Style of calculating correct count rate from raw data, \'s\' for calculating with single time interval')
    print('\'--gridfile\': Give the filepath of GRID data, When processing nim data')
    print('\'--hpgefile\': Give the filepath of HPGe data, When processing nim data')
    print('\'--cut\': Time cut in seconds to cut off the data in initial fwe seconds, used to cut off data before the bias is stablized(6th ver.)')
    print('Supported file type: text file(.txt)')
    return

#******************************************************************************************************************************************************
#******************************************************************Main function********************************************************************
#******************************************************************************************************************************************************

isHex = False
isScan = False
fileOutput = False
singlech = False
odr = False
fitPlot = True
quadBkg = True
newProgramme = False
isCi = 0
option = ''
maxiter = 1
iarg = 1

#*****************************************************************************************************************************************************
#****************************************************************Options readout******************************************************************
#*****************************************************************************************************************************************************

#h: hexprint, c: with (single) CI, s: with I-V scan, o:file output, f:fit, p: plot raw data, t: temperature calibration, B: bias calibration
#i: import data from file, v: temp&bias curves & imon fit, r: temp/bias responce, O: odr fitting, a: angular responce
#TBD: Think of and implement more options
optionsAvailable = ['h', 'c', 's', 'o', 'f', 'p', 't', 'b', 'v', 'r', 'O', 'a', 'g', 'n', 'm']
sourceAvailable = ['Am241', 'Ba133', 'Cs137', 'Na22', 'Th228', 'Co60', 'x']
nbinsRef = {
                'Am241':        2048,
                'Ba133':        2048,
                'Th228':        1024,
                'Cs137':        512,
                'Na22':         512,
                'Co60':         512,
    }

if len(sys.argv) == 1:
    printUsage()
    sys.exit()

if sys.argv[iarg][0] == '-' and not len(sys.argv[iarg]) == 1:
    option = sys.argv[iarg][1:]
    iarg += 1
    for s in option:
        if not s in optionsAvailable:
            print('GridDataProcessor: unsupported option ' + s)
            printUsage()
            sys.exit()
    if 's' in option:
        isScan = True
    if 'c' in option:
        isCi = 1
    if 'h' in option:
        isHex = True
        isCi = 0
        isScan = False
    if 'o' in option:
        fileOutput = True
    if 'O' in option:
        odr = True
    if 'r' in option or 'a' in option:
        if not 'f' in option:
            option += 'f'
    if 'g' in option:
        quadBkg = False
    if 'n' in option:
        newProgramme = True

#Source, nbins, range, scans and files input part
#v0.0.2 by ghz
bkg = False
simulation = False
plotNIM = False
binSpecified = False
iterSpecified = False
boundSpecified = False
rateStyleSpecified = False
timeCutSpecified = False
corr = True
source = ''
fitRange = []
scanRange = []
channel = -1
nbins = 65536
bound = 3.0
timeCut = 0.0
filename = []
mulfilename = []
simuFilename = ''
gridFilepath = ''
hpgeFilepath = ''
rateStyle = ''
rateStyles = ['s', 'p']
if 'i' in option:
    importFilename = []
    importPath = []
    importMulFilename = []

while iarg < len(sys.argv):
    #nbins
    if sys.argv[iarg] == '--nbins':
        iarg += 1
        if binSpecified:
            print('GridDataProcessor: please do not specify nbins more than once. The first nbins given will be taken as the parameter for fit and plot')
            iarg += 1
            continue
        try:
            nbins = int(sys.argv[iarg])
            if nbins <= 0 or nbins > 65536:
                raise Exception
        except:
            print('GridDataProcessor: nbins should be in integer form and within range [1-65536]')
            sys.exit()
        binSpecified = True
        iarg += 1

    #Source
    elif sys.argv[iarg] == '--src':
        iarg += 1
        if source in sourceAvailable:
            print('GridDataProcessor: please do not specify the source more than once. The first source given will be taken as the source for fit')
            iarg += 1
            continue
        source = sys.argv[iarg]
        if not source in sourceAvailable:
            print('GridDataProcessor: unknown source or source not given')
            if 'f' in option:
                print('GridDataProcessor: please specify the source in order to make the fit')
            printUsage()
            sys.exit()
        elif source == 'x':
            singlech = True
        iarg += 1

    #Channel
    elif sys.argv[iarg] == '--ch':
        iarg += 1
        if singlech and grid.isChannel(channel):
            print('GridDataProcessor: please do not specify the channel more than once. The first channel given will be taken as the channel of interest')
            iarg += 1
            continue
        try:
            channel = int(sys.argv[iarg])
            if channel < 0 or channel > 3:
                raise Exception
        except:
            print('GridDataProcessor: channel number not given in integer form or out of bound [0-3]')
            if 'f' in option:
                print('GridDataProcessor: please give the channel number in order to fit the spectrum of the specific channel')
            printUsage()
            sys.exit()
        singlech = True
        iarg += 1

    #Fit range
    elif sys.argv[iarg] == '--range':
        iarg += 1
        if not len(fitRange) == 0:
            print('GridDataProcessor: please do not specify the fit range more than once. The first range given will be taken as the fit range')
            iarg += 1
            continue
        try:
            fitRange.append(int(sys.argv[iarg].split('[')[-1]))
            iarg += 1
            fitRange.append(int(sys.argv[iarg].split(']')[0]))
            if fitRange[0] >= fitRange[1] or fitRange[0] < 0:
                raise Exception
        except:
            print('GridDataProcessor: fit range is not given in correct form \'[lower upper]\' or \'lower upper\' with lower and upper being both non-negative integers')
            printUsage()
            sys.exit()
        iarg += 1

    #Maximum number of iteration
    elif sys.argv[iarg] == '--iter':
        iarg += 1
        if iterSpecified:
            print('GridDataProcessor: please do not specify maximum number of iteration more than once. The first maximum iteration number given will be taken as the parameter for spectrum fit')
            iarg += 1
            continue
        try:
            maxiter = int(sys.argv[iarg])
            if maxiter < 0:
                raise Exception
        except:
            print('GridDataProcessor: maximum number of iteration should be a non-negative integer')
            sys.exit()
        iterSpecified = True
        if maxiter == 0:
            print('GridDataProcessor: no iteration will be conducted with \'maxiter = 0\' given')
        iarg += 1

    #Boundary of iteration
    elif sys.argv[iarg] == '--sigma':
        iarg += 1
        if boundSpecified:
            print('GridDataProcessor: please do not specify iteration boundary more than once. The first boundary given will be taken as the parameter for fit')
            iarg += 1
            continue
        try:
            bound = float(sys.argv[iarg])
            if bound <= 0:
                raise Exception
        except:
            print('GridDataProcessor: boundary should be in positive float form')
            sys.exit()
        boundSpecified = True
        iarg += 1

    #Correct count rate calculation style
    elif sys.argv[iarg] == '--rate':
        iarg += 1
        if rateStyleSpecified:
            print('GridDataProcessor: please do not specify style of correct count rate calculation more than once. The first style given will be taken as the final calculation style')
            iarg += 1
            continue
        if not sys.argv[iarg] in rateStyles:
            print('GridDataProcessor: rate calculation style \'' + sys.argv[iarg] + '\' not supported')
            printUsage()
            sys.exit()
        rateStyle = sys.argv[iarg]
        rateStyleSpecified = True
        iarg += 1

    #Time cut:
    elif sys.argv[iarg] == '--cut':
        iarg += 1
        if timeCutSpecified:
            print('GridDataProcessor: please do not specify cut time more than once. The first time given will be taken as the final cut time')
            iarg += 1
            continue
        try:
            timeCut = float(sys.argv[iarg])
            if timeCut <= 0.0:
                raise Exception
        except:
            print('GridDataProcessor: cut time should be in positive float form')
            printUsage()
            sys.exit()
        timeCutSpecified = True
        iarg += 1

    #No temperature-bias correction
    elif sys.argv[iarg] == '--nocorr':
        iarg += 1
        corr = False
        
    #No plotting during fitting
    elif sys.argv[iarg] == '--noplot':
        iarg += 1
        fitPlot = False

    #Multiple scan files
    elif sys.argv[iarg] == '--mul':
        iarg += 1
        if not os.path.exists(sys.argv[iarg]):
            print('GridDataProcessor: multiple scan file \'' + sys.argv[iarg] + '\' not found, this multiple scan file (and the corresponding scan range) will be automatically omitted')
            iarg += 1
            if sys.argv[iarg] == '--scan':
                iarg += 3
        mulfilename.append(sys.argv[iarg])
        iarg += 1
        if iarg >= len(sys.argv):
            scanRange.append([])
            break
        if sys.argv[iarg] == '--scan':
            iarg += 1
            curscanRange = []
            try:
                curscanRange.append(int(sys.argv[iarg].split('[')[-1]))
                iarg += 1
                curscanRange.append(int(sys.argv[iarg].split(']')[0]))
                if curscanRange[0] > curscanRange[1] or curscanRange[0] <= 0:
                    raise Exception()
            except:
                print('GridDataProcessor: scan range not given in correct form \'[lower upper]\' or \'lower upper\' with lower and upper being both integers')
                printUsage()
                sys.exit()
            scanRange.append(curscanRange)
            iarg += 1
        else:
            scanRange.append([])

    #Full directories
    elif sys.argv[iarg] == '--dir':
        iarg += 1
        if not os.path.isdir(sys.argv[iarg]):
            print('GridDataProcessor: directory \'' + sys.argv[iarg] + '\' does not exist, this directory will be automatically omitted')
            iarg += 1
            continue
        else:
            files = os.listdir(sys.argv[iarg])
            path = os.path.realpath(sys.argv[iarg])
            if 'i' in option:
                importPath.append(path)
            else:
                for file in files:
                    if file.endswith('.txt'):
                        filename.append(path + '\\' + file)
            iarg += 1

    #Background files
    elif sys.argv[iarg] == '--bkg':
        iarg += 1
        if not bkg:
            bkgFilename = []
        if not os.path.exists(sys.argv[iarg]):
            print('GridDataProcessor: background file \'' + sys.argv[iarg] + '\' not found, this background file will be automatically omitted')
            iarg += 1
            continue
        bkg = True
        bkgFilename.append(sys.argv[iarg])
        iarg += 1

    #Simulation files
    elif sys.argv[iarg] == '--simu':
        iarg += 1
        if simulation:
            print('GridDataProcessor: please do not specify simulation file more than once. The first file given will be taken as the data for processing')
            iarg += 1
            continue
        if not os.path.exists(sys.argv[iarg]):
            print('GridDataProcessor: simulation file \'' + sys.argv[iarg] + '\' not found, this simulation file will be automatically omitted')
            iarg += 1
            continue
        simulation = True
        simuFilename = sys.argv[iarg]
        iarg += 1

    #GRID filepath for NIM data
    elif sys.argv[iarg] == '--gridfile':
        iarg += 1
        if simulation:
            print('GridDataProcessor: please do not specify GRID filepath more than once. The first file given will be taken as the data for processing')
            iarg += 1
            continue
        if not os.path.exists(sys.argv[iarg]):
            print('GridDataProcessor: GRID filepath \'' + sys.argv[iarg] + '\' not found, this GRID filepath will be automatically omitted')
            iarg += 1
            continue
        plotNIM = True
        gridFilepath = sys.argv[iarg]
        iarg += 1

    #GRID filepath for NIM data
    elif sys.argv[iarg] == '--hpgefile':
        iarg += 1
        if simulation:
            print('GridDataProcessor: please do not specify HPGe filepath more than once. The first file given will be taken as the data for processing')
            iarg += 1
            continue
        if not os.path.exists(sys.argv[iarg]):
            print('GridDataProcessor: HPGe filepath \'' + sys.argv[iarg] + '\' not found, this HPGe filepath will be automatically omitted')
            iarg += 1
            continue
        plotNIM = True
        hpgeFilepath = sys.argv[iarg]
        iarg += 1

    #Single files
    elif os.path.exists(sys.argv[iarg]):
        filename.append(sys.argv[iarg])
        iarg += 1

    #Wrong option
    elif sys.argv[iarg].startswith('--'):
        print('GridDataProcessor: \'' + sys.argv[iarg] + '\' is not in the options list, this option and the following value(s) will be automatically omitted')
        if iarg < len(sys.argv) - 2:
            iarg += 2
        else:
            iarg = len(sys.argv)
        if iarg < len(sys.argv) and sys.argv[iarg].endswith(']'):
            iarg += 1

    #Wrong input
    else:
        print('GridDataProcessor: unable to parse input \'' + sys.argv[iarg] + '\' or file not found, this input value will be automatically omitted')
        print('GridDataProcessor: if this input value is an option, please add \'--\' before this value to indicate an possible option')
        iarg += 1

#nbins not specified if plot and fit is to be conducted
if not binSpecified:
    if source in nbinsRef:
        binSpecified = True
        nbins = nbinsRef[source]
    elif 'f' in option or 'p' in option:
        print('GridDataProcessor: to make the fit for data or plot the spectrum, nbins must be specified')
        print('GridDataProcessor: nbins should be in integer form and within range [1-65536]')
        printUsage()
        sys.exit()

#Unknown source
if not source in sourceAvailable:
    if 'f' in option and not 'b' in option:
        print('GridDataProcessor: to make the fit for data, the source must be specified')
        printUsage()
        sys.exit()

#Channel or fit range not specified for x-ray data fit
elif source == 'x':
    if 'f' in option:
        if not (grid.isChannel(channel) and not len(fitRange) == 0):
            print('GridDataProcessor: to make the fit for x-ray data, the channel and fit range must be specified')
            print('GridDataProcessor: the channel should be within range [0-3], and fit range in the form of \'[lower upper]\' \
                or \'lower upper\' with lower and upper being both non-negative integers')
            printUsage()
            sys.exit()

#Import path not specified for processed data import
if 'i' in option and len(importPath) == 0:
    print('GridDataProcessor: import path specified, setting the import path as default (current path)')
    importPath.append(os.getcwd())

#****************************************************************************************************************************************************
#**********************************************************Data readout and fit part************************************************************
#****************************************************************************************************************************************************
#v0.0.2 by ghz

#Background readout
bamp = []
for ich in range(4):
    bamp.append([])
bkgTime = np.ones(4)
brateAll = 0.0
brateAllErr = 0.0
curbamp = []
curbtimeCorrect = []
brateData = []
brateDataErr = []
if bkg:
    bkgTime = np.zeros(4)
    for bkfile in bkgFilename:
        for file in filename:
            if file.endswith(bkfile):
                del filename[filename.index(file)]
        bkgdata = grid.dataReadout(bkfile, isHex, isCi = 1, rateStyle = rateStyle)
        curbamp, curbuscountEvt, curbtimeCorrect = bkgdata[0], bkgdata[7], bkgdata[8]
        for ich in range(4):
            bamp[ich] += list(curbamp[ich])
            bkgTime[ich] += curbuscountEvt[ich][-1] - curbuscountEvt[ich][0]
        if not rateStyle == '':
            curbrateData = grid.fitRateCorrect(bkfile, curbtimeCorrect, fitPlot, odr, rateStyle = rateStyle)
            brateData.append(curbrateData[0])
            brateDataErr.append(curbrateData[1])
    bamp = np.array(bamp)
    if not rateStyle == '':
        brateAll = np.average(np.array(brateData))
        brateAllErr = np.sqrt(np.std(np.array(brateData)) ** 2 + np.sum(np.array(brateDataErr) ** 2) / len(brateData) ** 2)

#Data readout and fit
amp = []
tempSipm = []
tempAdc = []
vMon = []
iMon = []
bias = []
uscount = []
uscountEvt = []
timeCorrect = []
effectiveCount = []
missingCount = []
if not isCi == 0:
    ampCI = []
    uscountEvtCI = []
    effectiveCountCI = []
    missingCountCI = []
if isScan:
    vSet = []
    vScan = []
    iScan = []
for ich in range(4):
    amp.append([])
    tempSipm.append([])
    tempAdc.append([])
    vMon.append([])
    iMon.append([])
    bias.append([])
    uscountEvt.append([])
    if isCi == 1:
        ampCI.append([])
        uscountEvtCI.append([])
    if isScan:
        vScan.append([])
        iScan.append([])

if 'f' in option:
    fitResults = []
    if not singlech:
        for ich in range(4):
            fitResults.append([])

curamp = []
curtempSipm = []
curtempAdc = []
curvMon = []
curiMon = []
curbias = []
curuscount = []
curuscountEvt = []
curtimeCorrect = []
cureffectiveCount = []
curmissingCount = []
if not isCi == 0:
    curampCI = []
    curuscountEvtCI = []
    cureffectiveCountCI = []
    curmissingCountCI = []
if isScan:
    curvSet = []
    curvScan = []
    curiScan = []

for file in mulfilename + filename:
    #Data readout
    curCi = isCi
    curscanRange = []
    scanNum = []
    if file in mulfilename:
        curCi = 2
        curscanRange = scanRange[mulfilename.index(file)]
    #Readout from processed files
    if 'i' in option:
        #UNUSED
        if isScan:
            if curCi == 0:
                curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount, curvSet, \
                    curvScan, curiScan = grid.importData(file, importPath, curCi, isScan, curscanRange)
            else:
                curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount, curampCI, \
                    curuscountEvtCI, cureffectiveCountCI, curmissingCountCI, curvSet, curvScan, curiScan, scanNum = grid.importData(file, importPath, curCi, isScan, curscanRange)
        else:
            if curCi == 0:
                curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount = grid.importData(\
                    file, importPath, curCi, isScan, curscanRange)
            else:
                curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount, curampCI, \
                    curuscountEvtCI, cureffectiveCountCI, curmissingCountCI, scanNum = grid.importData(file, importPath, curCi, isScan, curscanRange)
    #Readout from raw data
    else:
        if isScan:
            if curCi == 0:
                curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount, curvSet, curvScan, \
                    curiScan = grid.dataReadout(file, isHex, curCi, isScan, curscanRange, rateStyle, newProgramme, timeCut = timeCut)
                if fileOutput:
                    grid.fileOutput(file.split('\\')[-1], curCi, isScan, curscanRange, *[curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, \
                        curtimeCorrect, cureffectiveCount, curmissingCount, curvSet, curvScan, curiScan])
            else:
                curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount, curampCI, curuscountEvtCI, \
                    cureffectiveCountCI, curmissingCountCI, curvSet, curvScan, curiScan = grid.dataReadout(file, isHex, curCi, isScan, curscanRange, rateStyle, newProgramme, timeCut = timeCut)
                if fileOutput:
                    grid.fileOutput(file.split('\\')[-1], curCi, isScan, curscanRange, *[curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, \
                        cureffectiveCount, curmissingCount, curampCI, curuscountEvtCI, cureffectiveCountCI, curmissingCountCI, curvSet, curvScan, curiScan])
        else:
            if curCi == 0:
                curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount = grid.dataReadout(\
                    file, isHex, curCi, isScan, curscanRange, rateStyle, newProgramme, timeCut = timeCut)
                if fileOutput:
                    grid.fileOutput(file.split('\\')[-1], curCi, isScan, curscanRange, *[curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, \
                        cureffectiveCount, curmissingCount])
            else:
                curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount, curampCI, curuscountEvtCI, \
                    cureffectiveCountCI, curmissingCountCI = grid.dataReadout(file, isHex, curCi, isScan, curscanRange, rateStyle, newProgramme, timeCut = timeCut)
                if fileOutput:
                    grid.fileOutput(file.split('\\')[-1], curCi, isScan, curscanRange, *[curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, \
                        cureffectiveCount, curmissingCount, curampCI, curuscountEvtCI, cureffectiveCountCI, curmissingCountCI])

    #Plot raw spectrum
    rateAll = 1.0
    if 'p' in option:
        #Multiple scans
        if curCi == 2:
            for isc in range(len(curuscount)):
                if not len(curscanRange) == 0 and not (isc >= curscanRange[0] - 1 and isc <= curscanRange[1] - 1):
                    continue
                if rateStyleSpecified:
                    rateAll, rateAllErr = grid.fitRateCorrect('', curtimeCorrect[isc], False, odr, rateStyle = rateStyle)
                timeSpec = []
                for ich in range(4):
                    timeSpec.append(curuscountEvt[ich][isc][-1] - curuscountEvt[ich][isc][0])
                grid.plotRawData('Run #' + str(isc + 1) + ' of ' + file.split('\\')[-1], curamp[:, isc], nbins * grid.getBiasnbinsFactor(isc + 1), grid.tempBiasCorrection(\
                    curtempSipm[:, isc], curbias[:, isc], False, not 't' in option)[0], timeSpec, singlech, channel = channel, rateStyle = rateStyle, rateAll = rateAll, \
                    doCorr = corr)
        #Single scan
        else:
            if rateStyleSpecified:
                rateAll, rateAllErr = grid.fitRateCorrect('', curtimeCorrect, False, odr, rateStyle = rateStyle)
            timeSpec = []
            for ich in range(4):
                timeSpec.append(curuscountEvt[ich][-1] - curuscountEvt[ich][0])
            grid.plotRawData(file.split('\\')[-1], curamp, nbins, grid.tempBiasCorrection(curtempSipm, curbias, False, not 't' in option)[0], timeSpec, singlech, \
                channel = channel, rateStyle = rateStyle, rateAll = rateAll, doCorr = corr)

    #Fit session
    if 'f' in option:
        #Multiple scans
        if curCi == 2:
            if source == 'x':
                for isc in range(len(curuscount)):
                    if not len(curscanRange) == 0 and not (isc >= curscanRange[0] - 1 and isc <= curscanRange[1] - 1):
                        continue
                    rateAll = 0.0
                    rateAllErr = 0.0
                    if rateStyleSpecified:
                        rateAll, rateAllErr = grid.fitRateCorrect(str(isc + 1) + '_' + file.split('\\')[-1], curtimeCorrect[isc], fitPlot, odr, rateStyle = rateStyle)
                    timeSpec = []
                    for ich in range(4):
                        timeSpec.append(curuscountEvt[ich][isc][-1] - curuscountEvt[ich][isc][0])
                    currfitResults = grid.fitSpectrum(str(isc + 1) + '_' + file.split('\\')[-1], curamp[:, isc], nbins * grid.getBiasnbinsFactor(isc + 1), source, \
                        grid.tempBiasCorrection(curtempSipm[:, isc], curbias[:, isc], False, False)[0], timeSpec, fileOutput, singlech, bkg, \
                        xRange = fitRange, channel = channel, bkgAmp = bamp, bkgtime = bkgTime, corrErr = grid.tempBiasCorrection(curtempSipm[:, isc], \
                        curbias[:, isc], False, False)[1], odr = odr, maxiter = maxiter, bound = bound, plot = fitPlot, rateStyle = rateStyle, rateAll = rateAll, \
                        rateAllErr = rateAllErr, bkgRate = brateAll, bkgRateErr = brateAllErr, quadBkg = quadBkg, doCorr = corr)
                    for ich in range(4):
                        fitResults[ich].append(currfitResult[ich])
            else:
                for isc in range(len(curuscount)):
                    if not len(curscanRange) == 0 and not (isc >= curscanRange[0] - 1 and isc <= curscanRange[1] - 1):
                        continue
                    rateAll = 0.0
                    rateAllErr = 0.0
                    if rateStyleSpecified:
                        rateAll, rateAllErr = grid.fitRateCorrect(str(isc + 1) + '_' + file.split('\\')[-1], curtimeCorrect[isc], fitPlot, odr, rateStyle = rateStyle)
                    timeSpec = []
                    for ich in range(4):
                        timeSpec.append(curuscountEvt[ich][isc][-1] - curuscountEvt[ich][isc][0])
                    if 'b' in option:
                        curfitResults = grid.fitSpectrum(str(isc + 1) + '_' + file.split('\\')[-1], curamp[:, isc], nbins * grid.getBiasnbinsFactor(isc + 1), \
                            'x', grid.tempBiasCorrection(curtempSipm[:, isc], curbias[:, isc], False, False)[0], timeSpec, fileOutput, singlech, bkg, \
                            xRange = grid.getBiasFitRange(isc, False), channel = channel, bkgAmp = bamp, bkgtime = bkgTime, corrErr = grid.tempBiasCorrection(\
                            curtempSipm[:, isc], curbias[:, isc], False, False)[1], odr = odr, maxiter = maxiter, bound = bound, plot = fitPlot, rateStyle = rateStyle, \
                            rateAll = rateAll, rateAllErr = rateAllErr, bkgRate = brateAll, bkgRateErr = brateAllErr, quadBkg = quadBkg, doCorr = corr)
                    else:
                        curfitResults = grid.fitSpectrum(str(isc + 1) + '_' + file.split('\\')[-1], curamp[:, isc], nbins * grid.getBiasnbinsFactor(isc + 1), \
                            source, grid.tempBiasCorrection(curtempSipm[:, isc], curbias[:, isc], False, False)[0], timeSpec, fileOutput, singlech, bkg, \
                            channel = channel, bkgAmp = bamp, bkgtime = bkgTime, corrErr = grid.tempBiasCorrection(curtempSipm[:, isc], curbias[:, isc], False, \
                            False)[1], odr = odr, maxiter = maxiter, bound = bound, plot = fitPlot, rateStyle = rateStyle, rateAll = rateAll, rateAllErr = \
                            rateAllErr, bkgRate = brateAll, bkgRateErr = brateAllErr, quadBkg = quadBkg, doCorr = corr)
                    if singlech:
                        fitResults.append(curfitResults)
                    else:
                        for ich in range(4):
                            fitResults[ich].append(curfitResults[ich])

        #Single scan
        else:
            if source == 'x':
                rateAll = 0.0
                rateAllErr = 0.0
                if rateStyleSpecified:
                    rateAll, rateAllErr = grid.fitRateCorrect(file.split('\\')[-1], curtimeCorrect, fitPlot, odr, rateStyle = rateStyle)
                timeSpec = []
                for ich in range(4):
                    timeSpec.append(curuscountEvt[ich][-1] - curuscountEvt[ich][0])
                fitResults.append(grid.fitSpectrum(file.split('\\')[-1], curamp, nbins, source, grid.tempBiasCorrection(curtempSipm, curbias, False, False)[0], \
                    timeSpec, fileOutput, singlech, bkg, xRange = fitRange, channel = channel, bkgAmp = bamp, bkgtime = bkgTime, corrErr = grid.tempBiasCorrection(\
                    curtempSipm, curbias, False, False)[1], odr = odr, maxiter = maxiter, bound = bound, plot = fitPlot, rateStyle = rateStyle, rateAll = rateAll, \
                    rateAllErr = rateAllErr, bkgRate = brateAll, bkgRateErr = brateAllErr, quadBkg = quadBkg, doCorr = corr))
            else:
                if not singlech:
                    for ich in range(4):
                        fitResults.append([])
                rateAll = 0.0
                rateAllErr = 0.0
                if rateStyleSpecified:
                    rateAll, rateAllErr = grid.fitRateCorrect(file.split('\\')[-1], curtimeCorrect, fitPlot, odr, rateStyle = rateStyle)
                timeSpec = []
                for ich in range(4):
                    timeSpec.append(curuscountEvt[ich][-1] - curuscountEvt[ich][0])
                curfitResults = grid.fitSpectrum(file.split('\\')[-1], curamp, nbins, source, grid.tempBiasCorrection(curtempSipm, curbias, False, False)[0], \
                    timeSpec, fileOutput, singlech, bkg, channel = channel, bkgAmp = bamp, bkgtime = bkgTime, corrErr = grid.tempBiasCorrection(curtempSipm, \
                    curbias, False, False)[1], odr = odr, maxiter = maxiter, bound = bound, plot = fitPlot, rateStyle = rateStyle, rateAll = rateAll, \
                    rateAllErr = rateAllErr, bkgRate = brateAll, bkgRateErr = brateAllErr, quadBkg = quadBkg, doCorr = corr)
                if singlech:
                    fitResults.append(curfitResults)
                else:
                    for ich in range(4):
                        fitResults[ich].append(curfitResults[ich])

    #Remove empty runs for multiple scan files
    if curCi == 2:
        if not len(scanRange) == 0:
            curamp, curtempSipm, curtempAdc, curvMon, curiMon, curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount, \
                curampCI, curuscountEvtCI, cureffectiveCountCI, curmissingCountCI = grid.deleteEmptyRun(curamp, curtempSipm, curtempAdc, curvMon, curiMon, \
                curbias, curuscount, curuscountEvt, curtimeCorrect, cureffectiveCount, curmissingCount, curampCI, curuscountEvtCI, cureffectiveCountCI, curmissingCountCI, \
                curscanRange, rateStyle, newProgramme)

    #Add processed data to data list
    for ich in range(4):
        if curCi == 2:
            for isc in range(len(curuscount)):
                amp[ich].append(curamp[ich][isc])
                tempSipm[ich].append(curtempSipm[ich][isc])
                tempAdc[ich].append(curtempAdc[ich][isc])
                vMon[ich].append(curvMon[ich][isc])
                iMon[ich].append(curiMon[ich][isc])
                bias[ich].append(curbias[ich][isc])
                uscountEvt[ich].append(curuscountEvt[ich][isc])
        else:
            amp[ich].append(curamp[ich])
            tempSipm[ich].append(curtempSipm[ich])
            tempAdc[ich].append(curtempAdc[ich])
            vMon[ich].append(curvMon[ich])
            iMon[ich].append(curiMon[ich])
            bias[ich].append(curbias[ich])
            uscountEvt[ich].append(curuscountEvt[ich])
            if not isCi == 0:
                ampCI[ich].append(curampCI[ich])
                uscountEvtCI[ich].append(curuscountEvtCI[ich])
        if isScan:
            vScan[ich].append(curvScan[ich])
            iScan[ich].append(curiScan[ich])
    if curCi == 2:
        for isc in range(len(curuscount)):
            uscount.append(curuscount[isc])
            if not rateStyle == '':
                timeCorrect.append(curtimeCorrect[isc])
            if newProgramme:
                effectiveCount.append(cureffectiveCount[isc])
                effectiveCountCI.append(cureffectiveCountCI[isc])
                missingCount.append(curmissingCount[isc])
                missingCountCI.append(curmissingCountCI[isc])
    else:
        uscount.append(curuscount)
        if not rateStyle == '':
            timeCorrect.append(curtimeCorrect)
        if newProgramme:
            effectiveCount.append(cureffectiveCount)
            missingCount.append(curmissingCount)
            if not isCi == 0:
                effectiveCountCI.append(cureffectiveCountCI)
                missingCountCI.append(curmissingCountCI)
    if isScan:
        vSet.append(curvSet)

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
if newProgramme:
    effectiveCount = np.array(effectiveCount)
    missingCount = np.array(missingCount)
if not isCi == 0:
    ampCI = np.array(ampCI)
    uscountEvtCI = np.array(uscountEvtCI)
    if newProgramme:
        effectiveCountCI = np.array(effectiveCountCI)
        missingCountCI = np.array(missingCountCI)
if isScan:
    vSet = np.array(vSet)
    vScan = np.array(vScan)
    iScan = np.array(iScan)

if 'f' in option:
    fitResults = np.array(fitResults)

#*****************************************************************************************************************************************************
#**********************************************************Temperature and bias part************************************************************
#*****************************************************************************************************************************************************

#Temperature and bias curves, leak current and overvoltage fit
if ('b' in option or 't' in option) and 'v' in option:
    experiment.tempBiasVariation(tempSipm, bias, vMon, iMon, uscount, 't' in option, singlech, False, filenames = filename)
    #if 't' in option:
    #    currentInput = tempSipm
    #else:
    #    currentInput = bias
    #experiment.currentFit(currentInput, iMon, 't' in option, fileOutput, form = 'mixexp', odr = odr)
    #experiment.overvoltageFit(tempSipm, bias, 't' in option, fileOutput, odr = odr)

#Temperature\bias responce fit
if 'r' in option:
    if 't' in option and 'b' in option:
        experiment.tempBiasFit(fitResults, tempSipm, bias, True, fileOutput, singlech, channel = channel, odr = odr, corr = True, cont = False)
    elif 't' in option:
        tempFit = experiment.tempBiasFit(fitResults, tempSipm, bias, True, fileOutput, singlech, channel = channel, odr = odr)
    elif 'b' in option:
        biasFit = experiment.tempBiasFit(fitResults, tempSipm, bias, False, fileOutput, singlech, channel = channel, odr = odr)

#****************************************************************************************************************************************************
#************************************************************Angular responce part**************************************************************
#****************************************************************************************************************************************************

angle = np.arange(0, 375, 15)
#Plot angular responce
if 'a' in option:
    experiment.plotAngularResponce(fitResults, angle, source, fileOutput, singlech, channel = channel, rateCorr = True, simuFile = \
        simuFilename)

#****************************************************************************************************************************************************
#************************************************************ NIM data part **************************************************************
#****************************************************************************************************************************************************

#Plot nim data, including EC, Resolution and efficiency
if 'm' in option:
    gridResult = experiment.plotEnergyChannel(gridFilepath, ch = channel, doCorr = corr, isPlotSpec = False, isPlotEC = plotNIM, rateCorr = True, fitEC = False)
    hpgeResult = experiment.processHPGe(hpgeFilepath, isPlotSpec = False)
    efficiencyResult = experiment.getEfficiency(gridResult, hpgeResult, isPlot = plotNIM)

#Ending line
print('GridDataProcessor: all files processed')
