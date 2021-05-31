import numpy as np
import json
import matplotlib.pyplot as plt
file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)
inFileName = np.str(config['python3.6']['files']['bestNpz'])
data = np.load(inFileName)

poolInputFile  = config['python3.6']['files']['oriSimuNpz']
poolData = np.load(poolInputFile)
simu_tree = poolData['oriTree']
#plt.hist2d(simu_tree[:,20],simu_tree[:,17],bins=[50,50],range=[[1,50],[0.015,4]],weights = v_weight)

#boundary conditions
s1min = np.float32(config['s1min'])
s1u_c = np.float32(config['s1u_c'])
s2min = np.float32(config['s2min'])
s2minRaw = np.float32(config['s2minRaw'])
s2max = np.float32(config['s2max'])
s2u_c = np.float32(config['s2u_c'])
nHitsS1min = np.float32(config['nHitsS1min'])

v_weight = data['new_weight']
#energy spectrum 
colors = ['black','red','green']
#exit()
fig0,ax0=plt.subplots(2,2,figsize=(12,5))
print(simu_tree[:,20].shape,v_weight.shape)
v_energy = simu_tree[:,20]
ax0[0,0].hist(v_energy,bins=40,range=[0,40],weights = v_weight, density = True, linewidth = 2,edgecolor=colors[0], histtype = 'step',fill=False)
binS1 = np.uint(60)
binS2b = np.uint(60)
ax0[0,1].hist2d(simu_tree[:,0],np.log10(simu_tree[:,14]),bins=[binS1,binS2b],range=[[s1min,s1u_c],[np.log10(s2min),np.log10(s2u_c)]],weights = v_weight)
fig0.savefig('reweighted_e.pdf')
#import numpy as np
#import matplotlib.pyplot as plt
#data1 = np.load('outputs/bestOriSimu_tritium_5kV.npy.npz')
#simu_tree = data1['oriTree']
#plt.hist(simu_tree[:,0],bins=100,range=[2,120],weights = v_weight)
#plt.show()
Ninterval = np.int(16)
energyrange = np.linspace(0,30,Ninterval)
v_energy = simu_tree[:,20]

interest_variables = np.asarray([4,7,11,13],dtype=np.uint)
label_variables = ['g1','eee','seg','seb_fraction']
column = np.uint(3)
row = np.int(np.floor((Ninterval - 2)/column)+1)
fig,ax=plt.subplots(3,6,figsize=(17,15))
print('test',column,row,ax.shape)


for j in interest_variables:
  
  v_interest = simu_tree[:,j]
  for i in range(Ninterval-1):
    row_i = np.int(np.floor(i/column))
    column_i =  np.int(i - row_i * column)
    el_id = np.where(v_energy>energyrange[i])
    eu_id = np.where(v_energy<energyrange[i+1])
    weight = v_weight[el_id and eu_id]
    interestTemp = v_interest[el_id and eu_id]
    ax[column_i,row_i].clear() 
    ax[column_i,row_i].hist(interestTemp,bins = 50,weights = weight)
    ax[column_i,row_i].set_title('e (%.f, %.f) keV'%(energyrange[i],energyrange[i+1]))
    print(interestTemp.shape)
  
  fig.savefig('v_interest_%d.pdf'%j)
