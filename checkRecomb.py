import numpy as np
import uproot
import json
import matplotlib.pyplot as plt
import corner
import datetime
from matplotlib.backends.backend_pdf import PdfPages


#data recombination comparison
colors = ['black','red','green']
file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)

#boundary conditions
energymin = np.float32(config['E_eeTh'])
energyMax = np.float32(config['E_eeMaxSimu'])
#energyMax = 12.
re_min = 0. 
#re_minRaw = np.float32(config['re_minRaw'])
#s2max = np.float32(config['s2max'])
re_max = 1. 
#nHitsEmin = np.float32(config['nHitsEmin'])

def get_2dData():
  DATA_PATH = config['calDataRootFile'];
  data_tree = uproot.open(DATA_PATH)["out_tree"]
  cE_0= np.asarray(data_tree["qS1C_max"].array(),dtype=np.float32)
  cS2_0 = np.asarray(data_tree["qS2BC_max"].array(),dtype=np.float32)
  #nPMTE_max = np.asarray(data_tree["nPMTE_max"].array(),dtype=np.float32)
  cE_0_raw = np.asarray(data_tree["qS1_max"].array(),dtype=np.float32)
  cS2_0_raw = np.asarray(data_tree["qS2_max"].array(),dtype=np.float32)
  Ndata_0 = cE_0.shape[0]
  #cE = np.array([],dtype=np.float32)
  #cS2 = np.array([],dtype=np.float32)

 # for i in range(Ndata_0):
 #   if (nPMTE_max[i] > nHitsEmin) and (cE_0[i] > energymin ) and (cE_0[i] < energyMax) and (cS2_0_raw[i] > re_minRaw) and (cS2_0_raw[i] < s2max) and (cS2_0[i] > re_min) and (cS2_0[i] < re_max ):
  #    cE = np.append(cE,[cE_0[i]])
   #   cS2 = np.append(cS2,[cS2_0[i]])
  return cE_0,cS2_0

cE,cS2 = get_2dData()
Ndata = np.asarray([cE.shape[0]],dtype=np.int)
print("survived data: ",Ndata)

binE = np.uint(60)
binRecomb = np.uint(60)
cEHist, binenergy_edges = np.histogram(cE,bins=binE,range = [energymin,energyMax],normed = True)
cRecombHist, bins2b_edges = np.histogram(cS2,bins=binRecomb,range = [re_min,re_max],normed = True)

#pool simulation events check
poolInputFile  = config['python3.6']['files']['oriSimuNpz']
poolData = np.load(poolInputFile)
poolTree = poolData['oriTree']


#input original simulation with best fit parameters
simuInputFile = config['python3.6']['files']['oriBestSimuNpz']
oriData = np.load(simuInputFile)
oriTree = oriData['oriTree']
ori_v_weight = oriTree[:,10]

#input reweighting best fitting
simuInputFile = config['python3.6']['files']['bestNpz']
data = np.load(simuInputFile)
#oriTree = data['qmc_tree']
v_weight = data['new_weight']

normalization = data['normalization']
sum_ori = sum(ori_v_weight)
sum_new = sum(v_weight)
sum_norm = sum(normalization)
print('sum check:',sum_ori,sum_new,sum_norm)
scale = sum_ori/sum_new 
v_weight = v_weight * scale
print('scale',scale)

scale1 = sum_ori/sum_norm 
normalization = normalization *scale1
print('scale1',scale1)

fig0,ax0=plt.subplots(2,3,figsize=(12,10))
ax0[0,0].hist(poolTree[:,20],bins=40,range=[0,40],weights = v_weight, density = False, linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)
ax0[0,0].hist(poolTree[:,20],bins=40,range=[0,40],weights = normalization, density = False, linewidth = 2,edgecolor='blue', histtype = 'step',fill=False)
ax0[0,0].hist(oriTree[:,20],bins=40,range=[0,40],weights = ori_v_weight, density = False, linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)

ax0[0,1].hist2d(poolTree[:,20],(poolTree[:,17]),bins=[binE,binRecomb],range=[[energymin,energyMax],[(re_min),(re_max)]],weights = v_weight)
ax0[0,2].hist2d(oriTree[:,20],(oriTree[:,17]),bins=[binE,binRecomb],range=[[energymin,energyMax],[(re_min),(re_max)]],weights = ori_v_weight)


ax0[1,0].hist(poolTree[:,20],bins=binE,range=[energymin,energyMax],weights = v_weight, density = False, linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)
ax0[1,0].hist(oriTree[:,20],bins=binE,range=[energymin,energyMax],weights = ori_v_weight, density = False, linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)
#ax0[1,0].plot(binenergy_edges[:-1],cEHist,linewidth = 2.,color = colors[2])

ax0[1,1].hist((poolTree[:,17]),bins=binRecomb,range=[re_min,re_max],weights = v_weight, density = False,linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)
ax0[1,1].hist((oriTree[:,17]),bins=binRecomb,range=[re_min,re_max],weights = ori_v_weight, density = False,linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)
#ax0[1,1].plot(bins2b_edges[:-1],cRecombHist,linewidth = 2.,color = colors[2])
ax0[1,2].hist2d(poolTree[:,20],(poolTree[:,17]),bins=[binE,binRecomb],range=[[energymin,energyMax],[(re_min),(re_max)]])

#plt.yscale('log')
fig0.savefig('whole_reweighted_e_recomb.pdf')


stepE = 1.
v_slice_energy = np.arange(energymin,energyMax,stepE)

simuE = poolTree[:,20]
simuRecomb = 1. - poolTree[:,17]

oriSimuE = oriTree[:,20]
oriSimuRecomb =1. - oriTree[:,17]

with PdfPages('whole_energy_slice_RecombRatio.pdf') as pdf:
    for i in range(v_slice_energy.shape[0] - 1):
      plt.figure(figsize = (6,5))
      iLocal_l = np.where( cE > v_slice_energy[i])
      iLocal_u = np.where( cE < v_slice_energy[i+1])
      #cRecombHist_i, bins2b_edges_i = np.histogram(cS2[iLocal_l and iLocal_u],bins=binRecomb,range = [re_min,re_max],normed = True)
      iLocalSimu_l = np.where( simuE > v_slice_energy[i])
      iLocalSimu_u = np.where( simuE < v_slice_energy[i+1])
      i_v_weight = v_weight[iLocalSimu_l and iLocalSimu_u]
      plt.hist(simuRecomb[iLocalSimu_l and iLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_v_weight, density = False,linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)

      iOriLocalSimu_l = np.where( oriSimuE > v_slice_energy[i])
      iOriLocalSimu_u = np.where( oriSimuE < v_slice_energy[i+1])
      i_ori_v_weight = ori_v_weight[iOriLocalSimu_l and iOriLocalSimu_u]
      plt.hist(oriSimuRecomb[iOriLocalSimu_l and iOriLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_ori_v_weight, density = False,linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)

      #plt.plot(bins2b_edges_i[:-1],cRecombHist_i,linewidth = 2.,color = colors[2]) 
      plt.title('energy (%.2f,%.2f)'%(v_slice_energy[i],v_slice_energy[i+1]))
      pdf.savefig()
      plt.close()

exit()
re_min=2
re_max=120
oriSimuRecomb = oriTree[:,0]
simuRecomb = poolTree[:,0]
#with PdfPages('whole_energy_slice_s1.pdf') as pdf:
#    for i in range(v_slice_energy.shape[0] - 1):
#      plt.figure(figsize = (6,5))
#      iLocal_l = np.where( cE > v_slice_energy[i])
#      iLocal_u = np.where( cE < v_slice_energy[i+1])
#      #cRecombHist_i, bins2b_edges_i = np.histogram(cS2[iLocal_l and iLocal_u],bins=binRecomb,range = [re_min,re_max],normed = True)
#      iLocalSimu_l = np.where( simuE > v_slice_energy[i])
#      iLocalSimu_u = np.where( simuE < v_slice_energy[i+1])
#      i_v_weight = v_weight[iLocalSimu_l and iLocalSimu_u]
#      plt.hist(simuRecomb[iLocalSimu_l and iLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_v_weight, density = False,linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)
#
#      iOriLocalSimu_l = np.where( oriSimuE > v_slice_energy[i])
#      iOriLocalSimu_u = np.where( oriSimuE < v_slice_energy[i+1])
#      i_ori_v_weight = ori_v_weight[iOriLocalSimu_l and iOriLocalSimu_u]
#      plt.hist(oriSimuRecomb[iOriLocalSimu_l and iOriLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_ori_v_weight, density = False,linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)
#
#      #plt.plot(bins2b_edges_i[:-1],cRecombHist_i,linewidth = 2.,color = colors[2]) 
#      plt.title('energy (%.2f,%.2f)'%(v_slice_energy[i],v_slice_energy[i+1]))
#      pdf.savefig()
#      plt.close()

oriSimuRecomb = oriTree[:,14]
simuRecomb = poolTree[:,14]
re_min = 100
re_max = 3000
with PdfPages('whole_energy_slice_s2.pdf') as pdf:
    for i in range(v_slice_energy.shape[0] - 1):
      plt.figure(figsize = (6,5))
      iLocal_l = np.where( cE > v_slice_energy[i])
      iLocal_u = np.where( cE < v_slice_energy[i+1])
      #cRecombHist_i, bins2b_edges_i = np.histogram(cS2[iLocal_l and iLocal_u],bins=binRecomb,range = [re_min,re_max],normed = True)
      iLocalSimu_l = np.where( simuE > v_slice_energy[i])
      iLocalSimu_u = np.where( simuE < v_slice_energy[i+1])
      i_v_weight = v_weight[iLocalSimu_l and iLocalSimu_u]
      plt.hist(simuRecomb[iLocalSimu_l and iLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_v_weight, density = False,linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)

      iOriLocalSimu_l = np.where( oriSimuE > v_slice_energy[i])
      iOriLocalSimu_u = np.where( oriSimuE < v_slice_energy[i+1])
      i_ori_v_weight = ori_v_weight[iOriLocalSimu_l and iOriLocalSimu_u]
      plt.hist(oriSimuRecomb[iOriLocalSimu_l and iOriLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_ori_v_weight, density = False,linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)

      #plt.plot(bins2b_edges_i[:-1],cRecombHist_i,linewidth = 2.,color = colors[2])
      plt.title('energy (%.2f,%.2f)'%(v_slice_energy[i],v_slice_energy[i+1]))
      pdf.savefig()
      plt.close()

oriSimuRecomb = oriTree[:,9]
simuRecomb = poolTree[:,9]
re_min = 0
re_max = 500
with PdfPages('whole_energy_slice_Nee.pdf') as pdf:
    for i in range(v_slice_energy.shape[0] - 1):
      plt.figure(figsize = (6,5))
      iLocal_l = np.where( cE > v_slice_energy[i])
      iLocal_u = np.where( cE < v_slice_energy[i+1])
      #cRecombHist_i, bins2b_edges_i = np.histogram(cS2[iLocal_l and iLocal_u],bins=binRecomb,range = [re_min,re_max],normed = True)
      iLocalSimu_l = np.where( simuE > v_slice_energy[i])
      iLocalSimu_u = np.where( simuE < v_slice_energy[i+1])
      i_v_weight = v_weight[iLocalSimu_l and iLocalSimu_u]
      plt.hist(simuRecomb[iLocalSimu_l and iLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_v_weight, density = False,linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)

      iOriLocalSimu_l = np.where( oriSimuE > v_slice_energy[i])
      iOriLocalSimu_u = np.where( oriSimuE < v_slice_energy[i+1])
      i_ori_v_weight = ori_v_weight[iOriLocalSimu_l and iOriLocalSimu_u]
      plt.hist(oriSimuRecomb[iOriLocalSimu_l and iOriLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_ori_v_weight, density = False,linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)
      plt.title('energy (%.2f,%.2f)'%(v_slice_energy[i],v_slice_energy[i+1]))
      pdf.savefig()
      plt.close()


oriSimuRecomb = oriTree[:,8]
simuRecomb = poolTree[:,8]
re_min = 0
re_max = 500
with PdfPages('whole_energy_slice_Ne.pdf') as pdf:
    for i in range(v_slice_energy.shape[0] - 1):
      plt.figure(figsize = (6,5))
      iLocal_l = np.where( cE > v_slice_energy[i])
      iLocal_u = np.where( cE < v_slice_energy[i+1])
      #cRecombHist_i, bins2b_edges_i = np.histogram(cS2[iLocal_l and iLocal_u],bins=binRecomb,range = [re_min,re_max],normed = True)
      iLocalSimu_l = np.where( simuE > v_slice_energy[i])
      iLocalSimu_u = np.where( simuE < v_slice_energy[i+1])
      i_v_weight = v_weight[iLocalSimu_l and iLocalSimu_u]
      plt.hist(simuRecomb[iLocalSimu_l and iLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_v_weight, density = False,linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)

      iOriLocalSimu_l = np.where( oriSimuE > v_slice_energy[i])
      iOriLocalSimu_u = np.where( oriSimuE < v_slice_energy[i+1])
      i_ori_v_weight = ori_v_weight[iOriLocalSimu_l and iOriLocalSimu_u]
      plt.hist(oriSimuRecomb[iOriLocalSimu_l and iOriLocalSimu_u],bins=binRecomb,range=[re_min,re_max],weights = i_ori_v_weight, density = False,linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)
      plt.title('energy (%.2f,%.2f)'%(v_slice_energy[i],v_slice_energy[i+1]))
      pdf.savefig()
      plt.close()

