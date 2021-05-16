import numpy as np
PATH = '../FittingGPU/outputs/'
mc_outputs = np.loadtxt(PATH+'mcoutputs.dat')

N = 2**20

cS1mm = mc_outputs[3*N:4*N]
cS2mm = mc_outputs[4*N:5*N]
wmm = mc_outputs[5*N:6*N]
print(np.average(cS1mm, weights = wmm))
print(np.average(cS2mm, weights = wmm))
