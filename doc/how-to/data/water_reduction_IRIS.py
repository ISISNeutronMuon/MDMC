# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np


ISISIndirectEnergyTransferWrapper(InputFiles=r'IRS26173.raw', Instrument='IRIS', Analyser='graphite', Reflection='002', Efixed='1.845', SpectraRange='3,53', FoldMultipleFrames='0', GroupingMethod='Individual', OutputWorkspace='iris26173_graphite002_Reduced')
Rebin(InputWorkspace='iris26173_graphite002_red', OutputWorkspace='iris26173_graphite002_r', Params='-0.5,0.005,0.5')

# Child algorithms of SofQW
SofQWNormalisedPolygon(InputWorkspace='iris26173_graphite002_r', OutputWorkspace='iris26173_graphite002_sqw',  QAxisBinning='0.5,0.1,1.8', EMode='Indirect', EFixed='1.845', ReplaceNaNs='1')
# End of child algorithms of SofQW

AddSampleLog(Workspace='iris26173_graphite002_sqw', LogName='rebin_type', LogText='NormalisedPolygon')
#SaveNexus(InputWorkspace='iris26173_graphite002_sqw', Filename='C:/Users/My PC/Documents/Science/Data/IRIS/Water/iris26173_graphite002_sqw.nxs')
#Load(Filename='/home/jc15575/Desktop/water_files/iris26173_graphite002_sqw.nxs', OutputWorkspace='iris26173_graphite002_sqw')



# Child algorithms of ISISIndirectEnergyTransfer
Load(Filename=r'IRS26176.raw', OutputWorkspace='IRS26176')
LoadParameterFile(Workspace='IRS26176', Filename='C:/MantidInstall/instrument/IRIS_graphite_002_Parameters.xml')
ExtractSingleSpectrum(InputWorkspace='IRS26176', OutputWorkspace='IRS26176_mon', WorkspaceIndex='0')
CropWorkspace(InputWorkspace='IRS26176', OutputWorkspace='IRS26176', StartWorkspaceIndex='2', EndWorkspaceIndex='52')
SetInstrumentParameter(Workspace='IRS26176', ComponentName='graphite', ParameterName='Efixed', ParameterType='Number', Value='1.845')
UnwrapMonitor(InputWorkspace='IRS26176_mon', OutputWorkspace='IRS26176_mon', LRef='37.859999999999999', JoinWavelength='6.144443290949126')
RemoveBins(InputWorkspace='IRS26176_mon', OutputWorkspace='IRS26176_mon', XMin='6.1434432909491257', XMax='6.1454432909491263', Interpolation='Linear')
FFTSmooth(InputWorkspace='IRS26176_mon', OutputWorkspace='IRS26176_mon', IgnoreXBins='1')
OneMinusExponentialCor(InputWorkspace='IRS26176_mon', OutputWorkspace='IRS26176_mon', C='0.20750000000000002', C1='0.001276')
Scale(InputWorkspace='IRS26176_mon', OutputWorkspace='IRS26176_mon', Factor='9.9999999999999995e-07')
ConvertUnits(InputWorkspace='IRS26176', OutputWorkspace='IRS26176', Target='Wavelength', EMode='Indirect')
RebinToWorkspace(WorkspaceToRebin='IRS26176', WorkspaceToMatch='IRS26176_mon', OutputWorkspace='IRS26176')
Divide(LHSWorkspace='IRS26176', RHSWorkspace='IRS26176_mon', OutputWorkspace='IRS26176')
DeleteWorkspace(Workspace='IRS26176_mon')
ConvertUnits(InputWorkspace='IRS26176', OutputWorkspace='IRS26176', Target='DeltaE', EMode='Indirect')
CorrectKiKf(InputWorkspace='IRS26176', OutputWorkspace='IRS26176', EMode='Indirect')
RebinToWorkspace(WorkspaceToRebin='IRS26176', WorkspaceToMatch='IRS26176', OutputWorkspace='IRS26176')
ExponentialCorrection(InputWorkspace='IRS26176', OutputWorkspace='IRS26176', C1='6.4560518004586457e-308', Operation='Multiply')
RenameWorkspace(InputWorkspace='IRS26176', OutputWorkspace='iris26176_graphite002_red')
GroupWorkspaces(InputWorkspaces='iris26176_graphite002_red', OutputWorkspace='IndirectEnergyTransfer_Workspaces')
# End of child algorithms of ISISIndirectEnergyTransfer

Rebin(InputWorkspace='iris26176_graphite002_red', OutputWorkspace='iris26176_graphite002_r', Params='-0.5,0.005,0.5')

# Child algorithms of SofQW
SofQWPolygon(InputWorkspace='iris26176_graphite002_r', OutputWorkspace='iris26176_graphite002_sqw', QAxisBinning='0.5,0.1,1.8', EMode='Indirect', EFixed='1.845')
# End of child algorithms of SofQW

AddSampleLog(Workspace='iris26176_graphite002_sqw', LogName='rebin_type', LogText='Polygon')
#SaveNexus(InputWorkspace='iris26176_graphite002_sqw', Filename=r'C:\Users\Public\Documents\Science\IRIS\Water\iris26176_graphite002_sqw.nxs')
#Load(Filename='/home/jc15575/Desktop/water_files/iris26176_graphite002_sqw.nxs', OutputWorkspace='iris26176_graphite002_sqw')
