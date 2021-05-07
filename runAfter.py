import numpy as np
import matplotlib.pyplot as plt
data = np.load('outputs/bestFittingResult_tritium_5kV.npy.npz')
simu_tree = data['qmc_tree']
#plt.hist2d(simu_tree[:,20],simu_tree[:,17],bins=[50,50],range=[[1,50],[0.015,4]],weights = simu_tree[:,10])
#plt.hist(simu_tree[:,4],bins=100,weights = simu_tree[:,10])
plt.show()


import numpy as np
import matplotlib.pyplot as plt
data1 = np.load('outputs/bestOriSimu_tritium_5kV.npy.npz')
simu_tree = data1['oriTree']
plt.hist(simu_tree[:,0],bins=100,range=[2,120],weights = simu_tree[:,10])
plt.show()

