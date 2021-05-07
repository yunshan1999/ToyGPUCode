import numpy as np
import json
import uproot
import matplotlib.pyplot as plt
binning0 = [[100, 0.,120.],[100, 0.5, np.log10(30000)]]

histogram2d      = np.zeros((binning0[0][0],binning0[1][0]),dtype = np.float32)

file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)
DATA_PATH = config['calDataRootFile'];
data_tree = uproot.open(DATA_PATH)["out_tree"]
s1= np.asarray(data_tree["qS1C_max"].array(),dtype=np.float32)
s2 = np.asarray(data_tree["qS2BC_max"].array(),dtype=np.float32)
h_s1 = np.zeros(binning0[0][0],dtype=np.float32)

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


for i in range(s1.shape[0]):
    weight = 1. 
    [xbin0,ybin0] = get_bin_number(s1[i],np.log10(s2[i]),binning0)
    if xbin0 > 0 and ybin0 > 0:
        histogram2d[ybin0][xbin0] = histogram2d[ybin0][xbin0] + weight
        h_s1[xbin0] =  h_s1[xbin0] + weight

x0 = np.linspace(binning0[0][1], binning0[0][2],num=int(binning0[0][0]))
y0 = np.linspace(binning0[1][1], binning0[1][2],num=int(binning0[1][0]))

fig,axs = plt.subplots(2)
axs[0].plot(x0,h_s1)

xv, yv = np.meshgrid(x0,y0)
axs[1].pcolor(xv,yv,histogram2d)
#extent0 = [binning0[0][1], binning0[0][2],binning0[1][1],binning0[1][2]]
#plt.imshow(histogram2d,origin = 'lower', extent = extent0, interpolation='none')
axs[1].axis([binning0[0][1], binning0[0][2],binning0[1][1],binning0[1][2]])
fig.savefig('testh2_data.png')
    
