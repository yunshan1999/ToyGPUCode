import numpy as np
import json
import matplotlib.pyplot as plt
file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)
binning0 = [[100, 0.,100.],[100, 0.5, np.log10(10000)]]

binningE = [100,0,50]

oriTreeFile  = config['python3.6']['files']['oriSimuNpz']
oriTree = np.load(oriTreeFile)
simu_tree = oriTree['oriTree']
histogram2d      = np.zeros((binning0[0][0],binning0[1][0]),dtype = np.float32)
histogramE      = np.zeros((binningE[0]),dtype = np.float32)
s1 = np.zeros((simu_tree.shape[0],),dtype = np.float32)
s2 = np.zeros((simu_tree.shape[0],),dtype = np.float32)
energyA = np.zeros((simu_tree.shape[0],),dtype = np.float32)

def get_bin_number(x,y,binning):
    xBinning, xmin, xmax = binning[0]
    yBinning, ymin, ymax = binning[1]
    xStep = (xmax - xmin) / xBinning
    yStep = (ymax - ymin) / yBinning
    xbin = np.floor((x - xmin)/xStep)
    ybin = np.floor((y - ymin)/yStep) 
    if -1 < xbin < xBinning and -1 < ybin <yBinning:
        return [np.int(xbin),np.int(ybin)]
    else:
        return [-1, -1]

def get_bin_number_1d(x,binning):
    xBinning, xmin, xmax = binning
    xStep = (xmax - xmin) / xBinning
    xbin = np.floor((x - xmin)/xStep)
    if -1 < xbin < xBinning:
        return np.int(xbin)
    else:
        return -1



for i in range(1000000):
    x = simu_tree[i][0]
    s1[i] = x
    y = np.log10(simu_tree[i][14])
    s2[i] = y
    energyA[i]=(simu_tree[i][20])
    #weight = simu_tree[i][10]
    weight = 1.
    [xbin0,ybin0] = get_bin_number(x,y,binning0)
    if xbin0 > 0 and ybin0 > 0:
        histogram2d[ybin0][xbin0] = histogram2d[ybin0][xbin0] + weight
    energyBin = get_bin_number_1d(energyA[i],binningE)
    if energyBin > 0:
        histogramE[energyBin] = histogramE[energyBin] + weight

x0 = np.linspace(binning0[0][1], binning0[0][2],num=int(binning0[0][0]))
y0 = np.linspace(binning0[1][1], binning0[1][2],num=int(binning0[1][0]))

xv, yv = np.meshgrid(x0,y0)

fig,axs = plt.subplots(2)
xE = np.linspace(binningE[1],binningE[2],num = int(binningE[0]))
print(xE)
print(histogramE)
axs[1].plot(xE,histogramE.T)
c = axs[0].pcolor(xv,yv,histogram2d,cmap='magma',vmin=0, vmax = histogram2d.max())
axs[0].axis([binning0[0][1], binning0[0][2],binning0[1][1],binning0[1][2]])
fig.colorbar(c,ax = axs[0])

fig.savefig('testh2Ori.png')

plt.hist(simu_tree[:,0],weights = simu_tree[:,10])

