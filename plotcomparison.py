import numpy as np
import matplotlib.pyplot as plt
import uproot
from scipy.stats.distributions import chi2

def chi2test(data, exp):
    chisquare = np.sum(np.square(data-exp)/exp)
    dof = data.shape[0] - 1
    return chisquare, chi2.sf(chisquare, dof), dof

ROOT_PATH = '../FittingGPU/data/reduce_ana3_p4_run1_tritium_5kV.root'

ana_tree = uproot.open(ROOT_PATH)["out_tree"]

cS1dd = ana_tree["qS1C_max"].array()
cS2dd = ana_tree["qS2BC_max"].array()

mc_outputs = np.loadtxt("../FittingGPU/outputs/mcoutputs.dat")

N = 2**20

##define cuts and hist min&max ##
s1min, s1max, s1step = 2.,120.,2.
s2min, s2max, s2step = 0., 7500., 75.

cS1mm = mc_outputs[3*N:4*N]
cS2mm = mc_outputs[4*N:5*N]
wmm = mc_outputs[5*N:6*N]

#to record the data&mc with applied cut
cS1d = []
cS2d = []
cS1m = []
cS2m = []
wm = []

for i in range(len(cS1dd)):
    if cS1dd[i]>s1min and cS1dd[i]<s1max:
        cS1d.append(cS1dd[i])
        cS2d.append(cS2dd[i])
for i in range(len(cS1mm)):
    if cS1mm[i]>s1min and cS1mm[i]<s1max:
        cS1m.append(cS1mm[i])
        cS2m.append(cS2mm[i])        
        wm.append(wmm[i])
        
cS1d = np.array(cS1d)
cS2d = np.array(cS2d)
cS1m = np.array(cS1m)
cS2m = np.array(cS2m)
wm = np.array(wm)


##cS1 comparison plotter##

binning_s1 = np.arange(s1min,s1max+s1step,s1step)

bincontentd = plt.hist(cS1d, bins = binning_s1)[0]
bincontentm = plt.hist(cS1m, bins = binning_s1, weights=wm)[0]

errm = np.sqrt(np.histogram(cS1m, bins = binning_s1, weights=wm**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
plt.close()

binning_s1 = np.arange(s1min,s1max+s1step,s1step)
bincontentm = plt.hist(cS1m, bins = binning_s1, weights=wm/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.7)[0]
errm = np.sqrt(np.histogram(cS1m, bins = binning_s1, weights=(wm/integralm * integrald)**2)[0])
binning_s1 = np.delete(binning_s1,0)
binning_s1 -= s1step*0.5
plt.errorbar(binning_s1,bincontentm,yerr=errm,color='red',linewidth = .7, fmt="none")

binning_s1 = np.arange(s1min,s1max+s1step,s1step)
errd = np.sqrt(plt.hist(cS1d, bins = binning_s1,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 1.2)[0])

binning_s1 = np.delete(binning_s1,0)
binning_s1 -= s1step*0.5

plt.errorbar(binning_s1, bincontentd, yerr=errd,color='black', linewidth = 1.2, fmt="none")
chisquare, p, dof = chi2test(bincontentd, bincontentm)
plt.title('cS1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
plt.legend()
plt.savefig("../FittingGPU/outputs/cS1.pdf")
plt.close()
##cS2 comparison plotter##

binning_s2 = np.arange(s2min,s2max+s2step,s2step)

bincontentd = plt.hist(cS2d, bins = binning_s2)[0]
bincontentm = plt.hist(cS2m, bins = binning_s2, weights=wm)[0]

errm = np.sqrt(np.histogram(cS2m, bins = binning_s2, weights=wm**2)[0])

#binning_s1 = np.delete(binning_s1,0)
#binning_s1 -= 0.5

#plt.errorbar(binning_s1, bincontentd, yerr=errd,color='black',linewidth = 1.2, fmt="none")
#plt.errorbar(binning_s1, bincontentm, yerr=errm,color='red',linewidth = .7, fmt="none")

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
plt.close()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = plt.hist(cS2m, bins = binning_s2, weights=wm/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.7)[0]
errm = np.sqrt(np.histogram(cS2m, bins = binning_s2, weights=(wm/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
plt.errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .7, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(plt.hist(cS2d, bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 1.2)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

plt.errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 1.2, fmt="none")
plt.legend()
chisquare, p, dof = chi2test(bincontentd, bincontentm)
plt.title('cS2'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
plt.savefig("../FittingGPU/outputs/cS2.pdf")
plt.close()
## divide s1s to 3 parts to see their distributions seperately##

cS1d1 = []
cS1d2 = []
cS1d3 = []

cS2d1 = []
cS2d2 = []
cS2d3 = []

cS1m1 = []
cS1m2 = []
cS1m3 = []

cS2m1 = []
cS2m2 = []
cS2m3 = []

weightm1 = []
weightm2 = []
weightm3 = []

for i in range(len(cS1m)):
    if s1min<cS1m[i]<40:
        cS1m1.append(cS1m[i])
        cS2m1.append(cS2m[i])
        weightm1.append(wm[i])
    elif cS1m[i]<100:
        cS1m2.append(cS1m[i])
        cS2m2.append(cS2m[i])
        weightm2.append(wm[i])
    elif cS1m[i]<120:
        cS1m3.append(cS1m[i])
        cS2m3.append(cS2m[i])
        weightm3.append(wm[i])
        
for i in range(len(cS1d)):
    if s1min<cS1d[i]<40:
        cS1d1.append(cS1d[i])
        cS2d1.append(cS2d[i])
    elif cS1d[i]<100:
        cS1d2.append(cS1d[i])
        cS2d2.append(cS2d[i])
    elif cS1d[i]<120:
        cS1d3.append(cS1d[i])
        cS2d3.append(cS2d[i])

weightm1 = np.array(weightm1)
weightm2 = np.array(weightm2)
weightm3 = np.array(weightm3)

##cS2s with different cS1 regions##

##cS2s in cS1 region 1 plotter##

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = plt.hist(cS2d1, bins = binning_s2)[0]
bincontentm = plt.hist(cS2m1, bins = binning_s2, weights=weightm1)[0]

errm = np.sqrt(np.histogram(cS2m1, bins = binning_s2, weights=weightm1**2)[0])

#binning_s1 = np.delete(binning_s1,0)
#binning_s1 -= 0.5

#plt.errorbar(binning_s1, bincontentd, yerr=errd,color='black',linewidth = 1.2, fmt="none")
#plt.errorbar(binning_s1, bincontentm, yerr=errm,color='red',linewidth = .7, fmt="none")

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
plt.close()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = plt.hist(cS2m1, bins = binning_s2, weights=weightm1/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.7)[0]
errm = np.sqrt(np.histogram(cS2m1, bins = binning_s2, weights=(weightm1/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
plt.errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .7, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(plt.hist(cS2d1, bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 1.2)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

plt.errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 1.2, fmt="none")
plt.legend()
chisquare, p, dof = chi2test(bincontentd, bincontentm)
plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
plt.savefig("../FittingGPU/outputs/cS2_region1.pdf")
plt.close()
##cS2s in cS1 region 2 plotter##

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = plt.hist(cS2d2, bins = binning_s2)[0]
bincontentm = plt.hist(cS2m2, bins = binning_s2, weights=weightm2)[0]

errm = np.sqrt(np.histogram(cS2m2, bins = binning_s2, weights=weightm2**2)[0])

#binning_s1 = np.delete(binning_s1,0)
#binning_s1 -= 0.5

#plt.errorbar(binning_s1, bincontentd, yerr=errd,color='black',linewidth = 1.2, fmt="none")
#plt.errorbar(binning_s1, bincontentm, yerr=errm,color='red',linewidth = .7, fmt="none")

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
plt.close()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = plt.hist(cS2m2, bins = binning_s2, weights=weightm2/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.7)[0]
errm = np.sqrt(np.histogram(cS2m2, bins = binning_s2, weights=(weightm2/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
plt.errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .7, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(plt.hist(cS2d2, bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 1.2)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
plt.errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 1.2, fmt="none")
plt.legend()
chisquare, p, dof = chi2test(bincontentd, bincontentm)
plt.title('cS2_region2'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
plt.savefig("../FittingGPU/outputs/cS2_region2.pdf")
plt.close()
##cS2s in cS1 region 3 plotter##

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = plt.hist(cS2d3, bins = binning_s2)[0]
bincontentm = plt.hist(cS2m3, bins = binning_s2, weights=weightm3)[0]

errm = np.sqrt(np.histogram(cS2m3, bins = binning_s2, weights=weightm3**2)[0])

#binning_s1 = np.delete(binning_s1,0)
#binning_s1 -= 0.5

#plt.errorbar(binning_s1, bincontentd, yerr=errd,color='black',linewidth = 1.2, fmt="none")
#plt.errorbar(binning_s1, bincontentm, yerr=errm,color='red',linewidth = .7, fmt="none")

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
plt.close()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = plt.hist(cS2m3, bins = binning_s2, weights=weightm3/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.7)[0]
errm = np.sqrt(np.histogram(cS2m3, bins = binning_s2, weights=(weightm3/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
plt.errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .7, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(plt.hist(cS2d3, bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 1.2)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
plt.errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 1.2, fmt="none")
plt.legend()
chisquare, p, dof = chi2test(bincontentd, bincontentm)
plt.title('cS2_region3'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
plt.savefig("../FittingGPU/outputs/cS2_region3.pdf")
plt.close()
