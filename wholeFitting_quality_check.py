import numpy as np
import uproot
import json
import matplotlib.pyplot as plt
import corner
import datetime
from matplotlib.backends.backend_pdf import PdfPages



#data comparison
colors = ['black','red','green']
file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)

#boundary conditions
s1min = np.float32(config['s1min'])
#s1u_c = np.float32(config['s1u_c'])
s1u_c = 100.
s2min = np.float32(config['s2min'])
s2minRaw = np.float32(config['s2minRaw'])
s2max = np.float32(config['s2max'])
s2u_c = np.float32(config['s2u_c'])
nHitsS1min = np.float32(config['nHitsS1min'])

def get_2dData():
  DATA_PATH = config['calDataRootFile'];
  data_tree = uproot.open(DATA_PATH)["out_tree"]
  cS1_0= np.asarray(data_tree["qS1C_max"].array(),dtype=np.float32)
  cS2_0 = np.asarray(data_tree["qS2BC_max"].array(),dtype=np.float32)
  nPMTS1_max = np.asarray(data_tree["nPMTS1_max"].array(),dtype=np.float32)
  cS1_0_raw = np.asarray(data_tree["qS1_max"].array(),dtype=np.float32)
  cS2_0_raw = np.asarray(data_tree["qS2_max"].array(),dtype=np.float32)
  Ndata_0 = cS1_0.shape[0]
  cS1 = np.array([],dtype=np.float32)
  cS2 = np.array([],dtype=np.float32)

  for i in range(Ndata_0):
    if (nPMTS1_max[i] > nHitsS1min) and (cS1_0[i] > s1min ) and (cS1_0[i] < s1u_c) and (cS2_0_raw[i] > s2minRaw) and (cS2_0_raw[i] < s2max) and (cS2_0[i] > s2min) and (cS2_0[i] < s2u_c ):
      cS1 = np.append(cS1,[cS1_0[i]])
      cS2 = np.append(cS2,[cS2_0[i]])
  return cS1,cS2

cS1,cS2 = get_2dData()
Ndata = np.asarray([cS1.shape[0]],dtype=np.int)
print("survived data: ",Ndata)

binS1 = np.uint(60)
binS2b = np.uint(60)
cS1Hist, bins1_edges = np.histogram(cS1,bins=binS1,range = [s1min,s1u_c],normed = False)
cS2bHist, bins2b_edges = np.histogram(cS2,bins=binS2b,range = [s2min,s2u_c],normed = False)

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
#poolTree = data['qmc_tree']
v_weight = data['new_weight']


simuS1 = poolTree[:,0]
simuS2b = poolTree[:,14]
simuE = poolTree[:,20]
simuRange = np.where((simuS1 > s1min) & (simuS1 < s1u_c) & (simuS2b > s2min) & (simuS2b < s2u_c))

simuS1_cut = simuS1[simuRange]
simuS2b_cut = simuS2b[simuRange]
simuE_cut = simuE[simuRange]
v_weight_cut = v_weight[simuRange]
sum_v_weight_cut = sum(v_weight_cut)
v_weight_cut = Ndata/sum_v_weight_cut * v_weight_cut
simuLog10S2bS1_cut = np.log10(simuS2b_cut / simuS1_cut)
belowNRMedianRange = np.where (simuLog10S2bS1_cut < 1.27372+0.383333*np.exp(-simuS1_cut/11.2784)+(-0.00380928*simuS1_cut))
v_weight_cut_belowNRMedianRange = v_weight_cut[belowNRMedianRange]
print('reweight below NR median : ',sum(v_weight_cut_belowNRMedianRange))
print('reweight Total Tritium + flatER: ', sum(v_weight_cut))


print('test simu Range ',v_weight_cut.shape[0])

oriSimuS1 = oriTree[:,0]
oriSimuS2b = oriTree[:,14]
oriSimuE = oriTree[:,20]
oriRange = np.where( (oriSimuS1 > s1min) & (oriSimuS1 < s1u_c) & (oriSimuS2b > s2min) & (oriSimuS2b < s2u_c) )
oriSimuS1_cut = oriSimuS1[oriRange]
oriSimuS2b_cut = oriSimuS2b[oriRange]
oriSimuE_cut = oriSimuE[oriRange]
ori_v_weight_cut = ori_v_weight[oriRange]
sum_ori_v_weight_cut = sum(ori_v_weight_cut)
ori_v_weight_cut = Ndata/sum_ori_v_weight_cut *ori_v_weight_cut

print('test oriSimu Range ',ori_v_weight_cut.shape[0])
oriSimuLog10S2bS1_cut = np.log10(oriSimuS2b_cut / oriSimuS1_cut)
belowNRMedianRange = np.where(oriSimuLog10S2bS1_cut < 1.27372+0.383333*np.exp(-oriSimuS1_cut/11.2784)+(-0.00380928*oriSimuS1_cut))

ori_v_weight_cut_belowNRMedianRange = ori_v_weight_cut[belowNRMedianRange]
print('below NR median : ',sum(ori_v_weight_cut_belowNRMedianRange))
print('Total Tritium + flatER: ', sum(ori_v_weight_cut))

s1_ana = np.arange(s1min,s1u_c,1)
log10s2b_s1_ana = 1.27372+0.383333*np.exp(-s1_ana/11.2784)+(-0.00380928*s1_ana)

fig0,ax0=plt.subplots(2,3,figsize=(12,10))
ax0[0,0].hist(simuE_cut,bins=40,range=[0,40],weights = v_weight_cut, density = False, linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)
ax0[0,0].hist(oriSimuE_cut,bins=40,range=[0,40],weights = ori_v_weight_cut, density = False, linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)

#ax0[0,1].hist2d(poolTree[:,0],np.log10(poolTree[:,14]),bins=[binS1,binS2b],range=[[s1min,s1u_c],[np.log10(s2min),np.log10(s2u_c)]],weights = v_weight)
#ax0[0,2].hist2d(oriTree[:,0],np.log10(oriTree[:,14]),bins=[binS1,binS2b],range=[[s1min,s1u_c],[np.log10(s2min),np.log10(s2u_c)]],weights = ori_v_weight)

ax0[0,1].hist2d(simuS1_cut,simuLog10S2bS1_cut,bins=[binS1,binS2b],range=[[s1min,s1u_c],[0.5,2.4]],weights = v_weight_cut)
ax0[0,1].plot(s1_ana,log10s2b_s1_ana,color='blue',linewidth=2)
ax0[0,2].hist2d(oriSimuS1_cut,oriSimuLog10S2bS1_cut,bins=[binS1,binS2b],range=[[s1min,s1u_c],[0.5,2.4]],weights = ori_v_weight_cut)
ax0[0,2].plot(s1_ana,log10s2b_s1_ana,color='blue',linewidth=2)


ax0[1,0].hist(simuS1_cut,bins=binS1,range=[s1min,s1u_c],weights = v_weight_cut, density = False, linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)
ax0[1,0].hist(oriSimuS1_cut,bins=binS1,range=[s1min,s1u_c],weights = ori_v_weight_cut, density = False, linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)
ax0[1,0].plot(bins1_edges[:-1],cS1Hist,linewidth = 2.,color = colors[2])

ax0[1,1].hist(simuS2b_cut,bins=binS2b,range=[s2min,s2u_c],weights = v_weight_cut, density = False,linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)
ax0[1,1].hist(oriSimuS2b_cut,bins=binS2b,range=[s2min,s2u_c],weights = ori_v_weight_cut, density = False,linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)
ax0[1,1].plot(bins2b_edges[:-1],cS2bHist,linewidth = 2.,color = colors[2])
ax0[1,2].hist2d(poolTree[:,0],np.log10(poolTree[:,14]),bins=[binS1,binS2b],range=[[s1min,s1u_c],[np.log10(s2min),np.log10(s2u_c)]])


#plt.yscale('log')
fig0.savefig('whole_reweighted_e.pdf')

#exit()

stepS1 = 10.
v_slice_s1 = np.arange(s1min,s1u_c,stepS1)


with PdfPages('whole_s1_slice.pdf') as pdf:
    for i in range(v_slice_s1.shape[0] - 1):
      plt.figure(figsize = (6,5))
      iLocal_l = np.where( cS1 > v_slice_s1[i])
      iLocal_u = np.where( cS1 < v_slice_s1[i+1])
      cS2bHist_i, bins2b_edges_i = np.histogram(cS2[iLocal_l and iLocal_u],bins=binS2b,range = [s2min,s2u_c],normed = False)
      iLocalSimu_l = np.where( simuS1_cut > v_slice_s1[i])
      iLocalSimu_u = np.where( simuS1_cut < v_slice_s1[i+1])
      i_v_weight = v_weight_cut[iLocalSimu_l and iLocalSimu_u]
      plt.hist(simuS2b_cut[iLocalSimu_l and iLocalSimu_u],bins=binS2b,range=[s2min,s2u_c],weights = i_v_weight, density = False,linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)

      iOriLocalSimu_l = np.where( oriSimuS1_cut > v_slice_s1[i])
      iOriLocalSimu_u = np.where( oriSimuS1_cut < v_slice_s1[i+1])
      i_ori_v_weight = ori_v_weight_cut[iOriLocalSimu_l and iOriLocalSimu_u]
      plt.hist(oriSimuS2b_cut[iOriLocalSimu_l and iOriLocalSimu_u],bins=binS2b,range=[s2min,s2u_c],weights = i_ori_v_weight, density = False,linewidth = 2,edgecolor=colors[1], histtype = 'step',fill=False)

      plt.plot(bins2b_edges_i[:-1],cS2bHist_i,linewidth = 2.,color = colors[2]) 
      plt.title('s1 (%.2f,%.2f)'%(v_slice_s1[i],v_slice_s1[i+1]))
      pdf.savefig()
      plt.close()

