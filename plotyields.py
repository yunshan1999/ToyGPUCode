import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import random

PATH = '../FittingGPU/outputs/'

nest_avo = 6.0221409e+23
molar_mass = 131.293
atom_num = 54.
E_drift = 115.6
density = 2.8611
eDensity = (density/molar_mass) * nest_avo * atom_num
Wq_eV = 18.7263 - 1.01e-23 * eDensity
Wq_eV *= 1.1716263232

## yield calculation function of NEST2.0 ##
def get_yield_pars(energy, free_pars):

    simuTypeNR = False
    
    if simuTypeNR == True:
        NuisParam = [11., 1.1, 0.0480, -0.0533, 12.6, 0.3, 2., 0.3, 2., 0.5, 1.]
        Nq = NuisParam[0] * np.power(energy, NuisParam[1])
        ThomasImel = NuisParam[2] * np.power(E_drift, NuisParam[3]) * np.power(density / 2.90, 0.3)
        Qy = 1. / (ThomasImel*np.power(energy + NuisParam[4], NuisParam[9]))
        Qy *= 1. - 1. / np.power(1. + np.powerer((energy / NuisParam[5]), NuisParam[6]),NuisParam[10])
        Ly = Nq / energy - Qy
        if Qy < 0.0:
            Qy = 0.0  
        elif Ly < 0.0:
            Ly = 0.0  
        Ne = Qy * energy  
        Nph = Ly * energy * (1. - 1. / (1. + np.power((energy / NuisParam[7]), NuisParam[8])))  
        Nq = Nph + Ne  
        Ni = (4. / ThomasImel) * (np.exp(Ne * ThomasImel / 4.) - 1.)  
        Nex = (-1. / ThomasImel) * (4. * np.exp(Ne * ThomasImel / 4.) -(Ne + Nph) * ThomasImel - 4.)  
        r0 = 1. - Ne / Ni  
        deltar = free_pars[6] + free_pars[7] * energy + free_pars[8] * energy * energy
        r = r0 + deltar
        if r > 1.:
            r = 1.
        elif r < 0.:
            r = 0.
        Ycharge = (1- r)/(1-r0) * Qy
        Ylight = (Ly + Qy) - Ycharge
        return Ycharge, Ylight

    else:
        QyLvllowE = 1e3 / Wq_eV + 6.5 * (1. - 1. / (1. + np.power(E_drift / 47.408, 1.9851)))  
        HiFieldQy = 1. + 0.4607 / np.power(1. + np.power(E_drift / 621.74, -2.2717), 53.502)  
        QyLvlmedE = 32.988 - 32.988 /(1. + np.power(E_drift / (0.026715 * np.exp(density / 0.33926)), 0.6705))  
        QyLvlmedE *= HiFieldQy  
        DokeBirks = 1652.264 + (1.415935e10 - 1652.264) / (1. + np.power(E_drift / 0.02673144, 1.564691))  
        Nq = energy * 1e3 / Wq_eV     
        LET_power = -2.  
        QyLvlhighE = 28.  
        Qy = QyLvlmedE + (QyLvllowE - QyLvlmedE) / np.power(1. + 1.304 * np.power(energy, 2.1393), 0.35535) + QyLvlhighE / (1. + DokeBirks * np.power(energy, LET_power))  
        #inds = np.where((Qy > QyLvllowE)&(energy > 1.)&E_drift > 1e4)[0]
        #Qy[inds] = QyLvllowE  
        Ly = Nq / energy - Qy 
        inds_l = np.where(Ly<0.)[0]
        Ly[inds_l] = 0.
        inds_c = np.where(Qy<0.)[0]
        Qy[inds_c] = 0.
        Ne = Qy * energy  
        Nph = Ly * energy  
        alpha = 0.067366 + density * 0.039693  
        NexONi = alpha * scipy.special.erf(0.05 * energy)  
        Nex = Nq * (NexONi) / (1. + NexONi)  
        Ni = Nq * 1. / (1. + NexONi)  
        r0 = 1 - Ne / Ni  
        deltar = free_pars[6] + free_pars[7] * energy + free_pars[8] * energy * energy
        r = r0 + deltar
        inds = np.where((r>1.))[0]
        r[inds] = 1.
        inds = np.where((r<0.))[0]
        r[inds] = 0.

        Ycharge = (1-r)/(1-r0) * Qy
        Ylight = (Ly + Qy) - Ycharge
        return Ycharge, Ylight  

npoints = 50+1
ndim = 9
ncov = 2500
energy = np.linspace(1., 27, npoints)

# get walker positions from saved array 
samples = np.loadtxt(PATH+'samples.dat')
samples_orig = samples.reshape(samples.shape[0], samples.shape[1] // ndim, ndim)
acceptances = np.loadtxt(PATH+'acceptances.dat')
lnls = np.loadtxt(PATH+'lnls.dat')
inds = np.where((-7476.757684<=lnls[:,-1])&(-2000>=lnls[:,-1]))[0]
samples_useful = samples_orig[inds,ncov:,:].reshape((-1, ndim))
samples_useful = samples_useful.tolist()
samples_sub = random.sample(samples_useful, 1000)
samples_sub = np.array(samples_sub)

charge_yield_outputs = []
light_yield_outputs = []

for i in range(1000):

    # define a input array

    position = samples_sub[i,:]

    nuisance_par_array = np.asarray(position, dtype=np.float32)
    cy, ly = get_yield_pars(energy, nuisance_par_array)

    charge_yield_outputs.append(cy)
    light_yield_outputs.append(ly)

charge_yield_outputs = np.array(charge_yield_outputs)
light_yield_outputs = np.array(light_yield_outputs)

charge_yield_errup = np.quantile(charge_yield_outputs, 0.84, axis = 0) - np.quantile(charge_yield_outputs, 0.5, axis = 0)
charge_yield_median = np.quantile(charge_yield_outputs, 0.5, axis = 0)
charge_yield_errdown = np.quantile(charge_yield_outputs, 0.5, axis = 0) - np.quantile(charge_yield_outputs, 0.16, axis = 0)
light_yield_errup = np.quantile(light_yield_outputs, 0.84, axis = 0) - np.quantile(light_yield_outputs, 0.5, axis = 0)
light_yield_median = np.quantile(light_yield_outputs, 0.5, axis = 0)
light_yield_errdown = np.quantile(light_yield_outputs, 0.5, axis = 0) - np.quantile(light_yield_outputs, 0.16, axis = 0)

charge_yield_err_asym = [charge_yield_errdown, charge_yield_errup]
light_yield_err_asym = [light_yield_errdown, light_yield_errup]
##calculate non_tuned yields##
position0 = np.zeros(ndim, np.float32)
cy0, ly0 = get_yield_pars(energy, position0)
#print(get_yield_pars(np.array([10],np.float32), position0))
## do plotting ##
plt.xscale('log')
fig, axs = plt.subplots(2, 1)
axs[0].errorbar(energy, light_yield_median, yerr = light_yield_err_asym, color='black', linewidth = 0.1, fmt = '.', label = 'tuned')
axs[0].plot(energy, ly0, color = 'red', label = 'NESTv2')
axs[0].set_title('light yield',fontsize=7)
#axs[0].set_xlabel('energy(keV)',fontsize=7)
#axs[0].set_xscale('log')
axs[0].legend()

axs[1].errorbar(energy, charge_yield_median, yerr = light_yield_err_asym, color='black', linewidth = 0.1, fmt = '.', label = 'tuned')
axs[1].plot(energy, cy0, color = 'red', label = 'NESTv2')
axs[1].set_title('charge yield',fontsize=7)
axs[1].set_xlabel('energy(keV)',fontsize=7)
#axs[1].set_xscale('log')
axs[1].legend()

fig.savefig(PATH+'yields.pdf')
