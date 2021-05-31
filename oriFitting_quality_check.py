import numpy as np
import uproot
import json
import matplotlib.pyplot as plt
import corner
import datetime
from matplotlib.backends.backend_pdf import PdfPages


#inFileName = "./outputs/fitting_result_1.npy.npz"
#fitting_oriData = np.load(inFileName)
#samples = fitting_oriData['samples']
#flat_samples = samples[-50:,:,:]
#flat_samples = flat_samples.reshape(flat_samples.shape[0] * flat_samples.shape[1],flat_samples.shape[2])
#ndim = flat_samples.shape[1]
#print('check ndim ',ndim)
#
##quality check
#with PdfPages('fitting_quality.pdf') as pdf:
#    plt.figure(figsize = (6,5))
#    lnls = fitting_oriData['lnls']
#    plt.plot(lnls.T, linewidth = 0.1, color = "black")
#    plt.title('likelihood')
#    pdf.savefig()
#    plt.close()
#
#    plt.figure(figsize = (6,5))
#    acceptance = fitting_oriData['acceptance']
#    plt.hist(acceptance, bins = 40,color = "black")
#    plt.title('acceptance')
#    pdf.savefig()
#    plt.close()
# 
#    plt.figure(figsize = (12,10))
#    labelsG=["g1", "eee", "seg","seb_frac","p0Recomb","p1Recomb","p0FlucRecomb","flatE"]
#    figCorner = corner.corner(flat_samples,labels=labelsG);
#    pdf.savefig()
#    plt.close()
#
#    plt.figure(figsize = (12,20))
#    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
#    for i in range(ndim):
#      ax = axes[i]
#      ax.plot(samples[:, :, i], "k", alpha=0.3)
#      ax.set_xlim(0, len(samples))
#      ax.set_ylabel(labelsG[i])
#      ax.yaxis.set_label_coords(-0.1, 0.5)
#      axes[-1].set_xlabel("step number");
#    pdf.savefig()
#    plt.close()
#
#    d = pdf.infodict()
#    d['Title'] = 'Multipage PDF Example'
#    d['Author'] = 'dan'
#    d['Subject'] = 'fitting check'
#    d['Keywords'] = 'PdfPages multipage keywords author title subject'
#    d['CreationDate'] = datetime.datetime(2009, 11, 13)
#    d['ModDate'] = datetime.datetime.today()
#

#oriData comparison
file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)

#boundary conditions
s1min = np.float32(config['s1min'])
s1u_c = np.float32(config['s1u_c'])
s2min = np.float32(config['s2min'])
s2minRaw = np.float32(config['s2minRaw'])
s2max = np.float32(config['s2max'])
s2u_c = np.float32(config['s2u_c'])
nHitsS1min = np.float32(config['nHitsS1min'])

def get_2dData():
  DATA_PATH = config['calDataRootFile'];
  oriData_tree = uproot.open(DATA_PATH)["out_tree"]
  cS1_0= np.asarray(oriData_tree["qS1C_max"].array(),dtype=np.float32)
  cS2_0 = np.asarray(oriData_tree["qS2BC_max"].array(),dtype=np.float32)
  nPMTS1_max = np.asarray(oriData_tree["nPMTS1_max"].array(),dtype=np.float32)
  cS1_0_raw = np.asarray(oriData_tree["qS1_max"].array(),dtype=np.float32)
  cS2_0_raw = np.asarray(oriData_tree["qS2_max"].array(),dtype=np.float32)
  NoriData_0 = cS1_0.shape[0]
  cS1 = np.array([],dtype=np.float32)
  cS2 = np.array([],dtype=np.float32)

  for i in range(NoriData_0):
    if (nPMTS1_max[i] > nHitsS1min) and (cS1_0[i] > s1min ) and (cS1_0[i] < s1u_c) and (cS2_0_raw[i] > s2minRaw) and (cS2_0_raw[i] < s2max) and (cS2_0[i] > s2min) and (cS2_0[i] < s2u_c ):
      cS1 = np.append(cS1,[cS1_0[i]])
      cS2 = np.append(cS2,[cS2_0[i]])
  return cS1,cS2

cS1,cS2 = get_2dData()
NoriData = np.asarray([cS1.shape[0]],dtype=np.int)
print("survived oriData: ",NoriData)

binS1 = np.uint(60)
binS2b = np.uint(60)
cS1Hist, bins1_edges = np.histogram(cS1,bins=binS1,range = [s1min,s1u_c],normed = True)
cS2bHist, bins2b_edges = np.histogram(cS2,bins=binS2b,range = [s2min,s2u_c],normed = True)

#input best fitting
simuInputFile = config['python3.6']['files']['oriBestSimuNpz']
oriData = np.load(simuInputFile)
oriTree = oriData['oriTree']
ori_v_weight = oriTree[:,10] 
fig0,ax0=plt.subplots(2,2,figsize=(12,10))
ax0[0,0].hist(oriTree[:,20],bins=40,range=[0,40],weights = ori_v_weight)
ax0[0,1].hist2d(oriTree[:,0],np.log10(oriTree[:,14]),bins=[binS1,binS2b],range=[[s1min,s1u_c],[np.log10(s2min),np.log10(s2u_c)]],weights = ori_v_weight)
ax0[1,0].hist(oriTree[:,0],bins=binS1,range=[s1min,s1u_c],weights = ori_v_weight, density = True)
ax0[1,0].plot(bins1_edges[:-1],cS1Hist,linewidth = 2.,color = 'red')
ax0[1,1].hist((oriTree[:,14]),bins=binS2b,range=[s2min,s2u_c],weights = ori_v_weight, density = True)
ax0[1,1].plot(bins2b_edges[:-1],cS2bHist,linewidth = 2.,color = 'red')

#plt.yscale('log')
fig0.savefig('ori_reweighted_e.pdf')


stepS1 = 10.
v_slice_s1 = np.arange(s1min,s1u_c,stepS1)

simuS1 = oriTree[:,0]
simuS2b = oriTree[:,14]
with PdfPages('ori_s1_slice.pdf') as pdf:
    for i in range(v_slice_s1.shape[0] - 1):
      plt.figure(figsize = (6,5))
      iLocal_l = np.where( cS1 > v_slice_s1[i])
      iLocal_u = np.where( cS1 < v_slice_s1[i+1])
      cS2bHist_i, bins2b_edges_i = np.histogram(cS2[iLocal_l and iLocal_u],bins=binS2b,range = [s2min,s2u_c],normed = True)
      iLocalSimu_l = np.where( simuS1 > v_slice_s1[i])
      iLocalSimu_u = np.where( simuS1 < v_slice_s1[i+1])
      i_ori_v_weight = ori_v_weight[iLocalSimu_l and iLocalSimu_u]
      plt.hist(simuS2b[iLocalSimu_l and iLocalSimu_u],bins=binS2b,range=[s2min,s2u_c],weights = i_ori_v_weight, density = True)
      plt.plot(bins2b_edges_i[:-1],cS2bHist_i,linewidth = 2.,color = 'red') 
      plt.title('s1 (%.2f,%.2f)'%(v_slice_s1[i],v_slice_s1[i+1]))
      pdf.savefig()
      plt.close()

