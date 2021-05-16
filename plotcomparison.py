import numpy as np
import matplotlib.pyplot as plt
import uproot
from scipy.stats.distributions import chi2
from matplotlib.ticker import FixedLocator, FixedFormatter

def chi2test(data, exp):
    inds = np.where((exp > 0.)&(data > 0.))[0]
    chisquare = np.sum(np.square(data[inds]-exp[inds])/exp[inds])
    dof = data[inds].shape[0] - 1
    return chisquare, chi2.sf(chisquare, dof), dof

PATH = '../FittingGPU/outputs/'
ROOT_PATH = '/home/yunshan/data/reduce_ana3_p4_run1_tritium_5kV.root'

ana_tree = uproot.open(ROOT_PATH)["out_tree"]

cS1dd = ana_tree["qS1C_max"].array()
cS2dd = ana_tree["qS2BC_max"].array()

mc_outputs = np.loadtxt(PATH+'mcoutputs.dat')

N = 2**20

##define cuts and hist min&max ##
s1min, s1max, s1step = 2.,120.,2.
s2min, s2max, s2step = 0., np.power(10.,3.5), 60.
new_ticks = np.linspace(s2min, s2max, 7)
#new_ticks = new_ticks.tolist()

cS1mm = mc_outputs[0*N:1*N]
cS2mm = mc_outputs[1*N:2*N]
wmm = mc_outputs[2*N:3*N]

#to record the data&mc with applied cut
inds_d = np.where((cS1dd>=2)&(cS1dd<120.)&(cS2dd<np.power(10.,3.5)))[0]
inds_m = np.where((cS1mm>=2)&(cS1mm<120.)&(cS2mm<np.power(10.,3.5)))[0]
cS1d = cS1dd[inds_d]
cS2d = cS2dd[inds_d]
cS1m = cS1mm[inds_m]
cS2m = cS2mm[inds_m]
wm = wmm[inds_m]

#outliers data with small likelihoods#
#inds_out = [10   ,19   ,22  , 24  , 26  , 46  , 50   ,56   ,60   ,67   ,69   ,76  ,105  ,132,138 , 168 , 170,  172,  183,  208,  209 , 232 , 242 , 247 , 251 , 253,  263,  279,  293 , 309 , 312,  328,  339,  348,  406 , 411 , 415 , 446 , 470 , 471,  513,  528,  537 , 540 , 564,  571,  609,  613,  653 , 681 , 697 , 705 , 717 , 742,  786,  817,  877 , 888 , 912,  913,  914,  953,  964 , 969 , 992 ,1004 ,1016 ,1029, 1040, 1065,1072 ,1080 ,1091, 1094, 1114, 1141, 1153 ,1155 ,1175 ,1188 ,1192 ,1193, 1196, 1197, 1201 ,1212 ,1214, 1258, 1269, 1284]
cS1d = np.array(cS1d)
cS2d = np.array(cS2d)

#cS1d = np.delete(cS1d, inds_out)
#cS2d = np.delete(cS2d, inds_out)
##cS1 comparison plotter##

binning_s1 = np.arange(s1min,s1max+s1step,s1step)

bincontentd = plt.hist(cS1d, bins = binning_s1)[0]
bincontentm = plt.hist(cS1m, bins = binning_s1, weights=wm)[0]

errm = np.sqrt(np.histogram(cS1m, bins = binning_s1, weights=wm**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
plt.cla()

binning_s1 = np.arange(s1min,s1max+s1step,s1step)
bincontentm = plt.hist(cS1m, bins = binning_s1, weights=wm/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS1m, bins = binning_s1, weights=(wm/integralm * integrald)**2)[0])
binning_s1 = np.delete(binning_s1,0)
binning_s1 -= s1step*0.5
plt.errorbar(binning_s1,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s1 = np.arange(s1min,s1max+s1step,s1step)
errd = np.sqrt(plt.hist(cS1d, bins = binning_s1,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s1 = np.delete(binning_s1,0)
binning_s1 -= s1step*0.5

plt.errorbar(binning_s1, bincontentd, yerr=errd,color='black', linewidth = 0.3, fmt="none")
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#plt.title('cS1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
plt.title('cS1')
plt.legend(loc='upper right', fontsize=5)
plt.savefig(PATH+'cS1.pdf')
plt.cla()
##cS2 comparison plotter##

binning_s2 = np.arange(s2min,s2max+s2step,s2step)

bincontentd = plt.hist(cS2d, bins = binning_s2)[0]
bincontentm = plt.hist(cS2m, bins = binning_s2, weights=wm)[0]

errm = np.sqrt(np.histogram(cS2m, bins = binning_s2, weights=wm**2)[0])

#binning_s1 = np.delete(binning_s1,0)
#binning_s1 -= 0.5

#plt.errorbar(binning_s1, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
#plt.errorbar(binning_s1, bincontentm, yerr=errm,color='red',linewidth = .2, fmt="none")

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
plt.cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = plt.hist(cS2m, bins = binning_s2, weights=wm/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m, bins = binning_s2, weights=(wm/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
plt.errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(plt.hist(cS2d, bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

plt.errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
plt.legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#plt.title('cS2b'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
plt.title('cS2b')
plt.savefig(PATH+'cS2b.pdf')
plt.cla()
## divide s1s to 9 parts to see their distributions seperately##
indsm1 = np.where((cS1m>=2)&(cS1m<10))[0]
indsm2 = np.where((cS1m>=10)&(cS1m<20))[0]
indsm3 = np.where((cS1m>=20)&(cS1m<30))[0]
indsm4 = np.where((cS1m>=30)&(cS1m<40))[0]
indsm5 = np.where((cS1m>=40)&(cS1m<50))[0]
indsm6 = np.where((cS1m>=50)&(cS1m<60))[0]
indsm7 = np.where((cS1m>=60)&(cS1m<80))[0]
indsm8 = np.where((cS1m>=80)&(cS1m<100))[0]
indsm9 = np.where((cS1m>=100)&(cS1m<120))[0]

indsd1 = np.where((cS1d>=2)&(cS1d<10))[0]
indsd2 = np.where((cS1d>=10)&(cS1d<20))[0]
indsd3 = np.where((cS1d>=20)&(cS1d<30))[0]
indsd4 = np.where((cS1d>=30)&(cS1d<40))[0]
indsd5 = np.where((cS1d>=40)&(cS1d<50))[0]
indsd6 = np.where((cS1d>=50)&(cS1d<60))[0]
indsd7 = np.where((cS1d>=60)&(cS1d<80))[0]
indsd8 = np.where((cS1d>=80)&(cS1d<100))[0]
indsd9 = np.where((cS1d>=100)&(cS1d<120))[0]

fig, axs = plt.subplots(3, 3)
##cS2s with different cS1 regions##
#plt.setp(axs, xticks=new_ticks, fontsize=5)
##cS2s in cS1 region 1 plotter##

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[0, 0].hist(cS2d[indsd1], bins = binning_s2)[0]
bincontentm = axs[0, 0].hist(cS2m[indsm1], bins = binning_s2, weights=wm[indsm1])[0]

errm = np.sqrt(np.histogram(cS2m[indsm1], bins = binning_s2, weights=wm[indsm1]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[0, 0].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[0, 0].hist(cS2m[indsm1], bins = binning_s2, weights=wm[indsm1]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm1], bins = binning_s2, weights=(wm[indsm1]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[0, 0].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[0, 0].hist(cS2d[indsd1], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[0, 0].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[0, 0].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#axs[0, 0].set_title('cS2b_cS1[2, 10]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
axs[0, 0].set_title('cS2b_cS1[2, 10]',fontsize=7)
axs[0, 0].set_xticks(np.linspace(s2min, s2max, 7))
axs[0, 0].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[0, 0].xaxis.set_major_locator(x_locator)
#axs[0, 0].xaxis.set_major_formatter(x_formatter, labelsize=4)

##cS2s in cS1 region 2 plotter##
binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[0, 1].hist(cS2d[indsd2], bins = binning_s2)[0]
bincontentm = axs[0, 1].hist(cS2m[indsm2], bins = binning_s2, weights=wm[indsm2])[0]

errm = np.sqrt(np.histogram(cS2m[indsm2], bins = binning_s2, weights=wm[indsm2]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[0, 1].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[0, 1].hist(cS2m[indsm2], bins = binning_s2, weights=wm[indsm2]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm2], bins = binning_s2, weights=(wm[indsm2]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[0, 1].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[0, 1].hist(cS2d[indsd2], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[0, 1].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[0, 1].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#axs[0, 1].set_title('cS2b_cS1[10, 20]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
#plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
axs[0, 1].set_title('cS2b_cS1[10, 20]',fontsize=7)
axs[0, 1].set_xticks(np.linspace(s2min, s2max, 7))
axs[0, 1].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[0, 1].xaxis.set_major_locator(x_locator)
#axs[0, 1].xaxis.set_major_formatter(x_formatter, labelsize=4)

##cS2s in cS1 region 3 plotter##
binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[0, 2].hist(cS2d[indsd3], bins = binning_s2)[0]
bincontentm = axs[0, 2].hist(cS2m[indsm3], bins = binning_s2, weights=wm[indsm3])[0]

errm = np.sqrt(np.histogram(cS2m[indsm3], bins = binning_s2, weights=wm[indsm3]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[0, 2].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[0, 2].hist(cS2m[indsm3], bins = binning_s2, weights=wm[indsm3]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm3], bins = binning_s2, weights=(wm[indsm3]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[0, 2].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[0, 2].hist(cS2d[indsd3], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[0, 2].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[0, 2].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
axs[0, 2].set_title('cS2b_cS1[20, 30]',fontsize=7)
#axs[0, 2].set_title('cS2b_cS1[20, 30]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
axs[0, 2].set_xticks(np.linspace(s2min, s2max, 7))
axs[0, 2].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[0, 2].xaxis.set_major_locator(x_locator)
#axs[0, 2].xaxis.set_major_formatter(x_formatter, labelsize=4)

##cS2s in cS1 region 4 plotter##
binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[1, 0].hist(cS2d[indsd4], bins = binning_s2)[0]
bincontentm = axs[1, 0].hist(cS2m[indsm4], bins = binning_s2, weights=wm[indsm4])[0]

errm = np.sqrt(np.histogram(cS2m[indsm4], bins = binning_s2, weights=wm[indsm4]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[1, 0].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[1, 0].hist(cS2m[indsm4], bins = binning_s2, weights=wm[indsm4]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm4], bins = binning_s2, weights=(wm[indsm4]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[1, 0].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[1, 0].hist(cS2d[indsd4], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[1, 0].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[1, 0].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#axs[1, 0].set_title('cS2b_cS1[30, 40]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
#plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
axs[1, 0].set_title('cS2b_cS1[30, 40]',fontsize=7)
axs[1, 0].set_xticks(np.linspace(s2min, s2max, 7))
axs[1, 0].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[1, 0].xaxis.set_major_locator(x_locator)
#axs[1, 0].xaxis.set_major_formatter(x_formatter, labelsize=4)

##cS2s in cS1 region 5 plotter##
binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[1, 1].hist(cS2d[indsd5], bins = binning_s2)[0]
bincontentm = axs[1, 1].hist(cS2m[indsm5], bins = binning_s2, weights=wm[indsm5])[0]

errm = np.sqrt(np.histogram(cS2m[indsm5], bins = binning_s2, weights=wm[indsm5]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[1, 1].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[1, 1].hist(cS2m[indsm5], bins = binning_s2, weights=wm[indsm5]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm5], bins = binning_s2, weights=(wm[indsm5]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[1, 1].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[1, 1].hist(cS2d[indsd5], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[1, 1].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[1, 1].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#axs[1, 1].set_title('cS2b_cS1[40, 50]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
#plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
axs[1, 1].set_title('cS2b_cS1[40, 50]',fontsize=7)
axs[1, 1].set_xticks(np.linspace(s2min, s2max, 7))
axs[1, 1].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[1, 1].xaxis.set_major_locator(x_locator)
#axs[1, 1].xaxis.set_major_formatter(x_formatter, labelsize=4)

##cS2s in cS1 region 6 plotter##
binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[1, 2].hist(cS2d[indsd6], bins = binning_s2)[0]
bincontentm = axs[1, 2].hist(cS2m[indsm6], bins = binning_s2, weights=wm[indsm6])[0]

errm = np.sqrt(np.histogram(cS2m[indsm6], bins = binning_s2, weights=wm[indsm6]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[1, 2].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[1, 2].hist(cS2m[indsm6], bins = binning_s2, weights=wm[indsm6]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm6], bins = binning_s2, weights=(wm[indsm6]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[1, 2].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[1, 2].hist(cS2d[indsd6], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[1, 2].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[1, 2].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#axs[1, 2].set_title('cS2b_cS1[50, 60]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
#plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
axs[1, 2].set_title('cS2b_cS1[50, 60]',fontsize=7)
axs[1, 2].set_xticks(np.linspace(s2min, s2max, 7))
axs[1, 2].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[1, 2].xaxis.set_major_locator(x_locator)
#axs[1, 2].xaxis.set_major_formatter(x_formatter, labelsize=4)

##cS2s in cS1 region 7 plotter##
binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[2, 0].hist(cS2d[indsd7], bins = binning_s2)[0]
bincontentm = axs[2, 0].hist(cS2m[indsm7], bins = binning_s2, weights=wm[indsm7])[0]

errm = np.sqrt(np.histogram(cS2m[indsm7], bins = binning_s2, weights=wm[indsm7]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[2, 0].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[2, 0].hist(cS2m[indsm7], bins = binning_s2, weights=wm[indsm7]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm7], bins = binning_s2, weights=(wm[indsm7]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[2, 0].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[2, 0].hist(cS2d[indsd7], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[2, 0].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[2, 0].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#axs[2, 0].set_title('cS2b_cS1[60, 80]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
#plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
axs[2, 0].set_title('cS2b_cS1[60, 80]',fontsize=7)
axs[2, 0].set_xticks(np.linspace(s2min, s2max, 7))
axs[2, 0].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[2, 0].xaxis.set_major_locator(x_locator)
#axs[2, 0].xaxis.set_major_formatter(x_formatter, labelsize=4)

##cS2s in cS1 region 8 plotter##
binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[2, 1].hist(cS2d[indsd8], bins = binning_s2)[0]
bincontentm = axs[2, 1].hist(cS2m[indsm8], bins = binning_s2, weights=wm[indsm8])[0]

errm = np.sqrt(np.histogram(cS2m[indsm8], bins = binning_s2, weights=wm[indsm8]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[2, 1].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[2, 1].hist(cS2m[indsm8], bins = binning_s2, weights=wm[indsm8]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm8], bins = binning_s2, weights=(wm[indsm8]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[2, 1].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[2, 1].hist(cS2d[indsd8], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[2, 1].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[2, 1].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#axs[2, 1].set_title('cS2b_cS1[80, 100]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
#plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
axs[2, 1].set_title('cS2b_cS1[80, 100]',fontsize=7)
axs[2, 1].set_xticks(np.linspace(s2min, s2max, 7))
axs[2, 1].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[2, 1].xaxis.set_major_locator(x_locator)
#axs[2, 1].xaxis.set_major_formatter(x_formatter, labelsize=4)

##cS2s in cS1 region 9 plotter##
binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentd = axs[2, 2].hist(cS2d[indsd9], bins = binning_s2)[0]
bincontentm = axs[2, 2].hist(cS2m[indsm9], bins = binning_s2, weights=wm[indsm9])[0]

errm = np.sqrt(np.histogram(cS2m[indsm9], bins = binning_s2, weights=wm[indsm9]**2)[0])

integrald = 1. * sum(bincontentd)
integralm = 1. * sum(bincontentm)
axs[2, 2].cla()

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
bincontentm = axs[2, 2].hist(cS2m[indsm9], bins = binning_s2, weights=wm[indsm9]/integralm * integrald, histtype = 'step', label = 'mc', color = 'red', linewidth = 0.2)[0]
errm = np.sqrt(np.histogram(cS2m[indsm9], bins = binning_s2, weights=(wm[indsm9]/integralm * integrald)**2)[0])
binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5
axs[2, 2].errorbar(binning_s2,bincontentm,yerr=errm,color='red',linewidth = .2, fmt="none")

binning_s2 = np.arange(s2min,s2max+s2step,s2step)
errd = np.sqrt(axs[2, 2].hist(cS2d[indsd9], bins = binning_s2,histtype = 'step', label = 'tritium data', color = 'black', linewidth = 0.3)[0])

binning_s2 = np.delete(binning_s2,0)
binning_s2 -= s2step*0.5

axs[2, 2].errorbar(binning_s2, bincontentd, yerr=errd,color='black',linewidth = 0.3, fmt="none")
axs[2, 2].legend(loc='upper right', fontsize=5)
chisquare, p, dof = chi2test(bincontentd, bincontentm)
#axs[2, 2].set_title('cS2b_cS1[100, 120]'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof), fontsize=6)
#plt.title('cS2_region1'+' chi2='+str(chisquare)+', p='+str(p)+', dof='+str(dof))
axs[2, 2].set_title('cS2b_cS1[100, 120]',fontsize=7)
axs[2, 2].set_xticks(np.linspace(s2min, s2max, 7))
axs[2, 2].tick_params(labelsize=4)
#x_formatter = FixedFormatter(new_ticks)
#x_locator = FixedLocator(new_ticks)
#axs[2, 2].xaxis.set_major_locator(x_locator)
#axs[2, 2].xaxis.set_major_formatter(x_formatter, labelsize=4)

fig.tight_layout()
plt.savefig(PATH+'cS2b_regions.pdf')
