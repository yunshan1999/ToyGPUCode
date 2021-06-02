###################################
## DEMO version for GPU fitting
###################################
## Created @ Apr. 12th, 2021
## By Yunshan Cheng
## contact: yunshancheng1@gmail.com
###################################
import numpy as np
import time
import matplotlib.pyplot as plt
import uproot

try:
    from pycuda.compiler import SourceModule
    import pycuda.driver as drv
    import pycuda.tools
    import pycuda.gpuarray
    import pycuda.autoinit
except ImportError:
    print(
        'Warning: Cuda libraries not found - RunManager.init() not available!'
     )

#g1, g2b, segb, for input initial pos&prior pdfs#
nuisance_mean = np.asarray([0.09196, 3.905, 5.41429], np.float32)
nuisance_sigma = np.asarray([0.01285, 0.227, 1.44129e-2], np.float32)

# data are in root ttree, we use uproot lib to get them as arrays
DATA_PATH = '../data/reduced_ana_DD_5KV_merged_newtag_newEL2_fom.root'
data_tree = uproot.open(DATA_PATH)["out_tree"]
cS1 = data_tree["qS1C_max"].array()
cS2b = data_tree["qS2BC_max"].array()
cS2 = data_tree["qS2C_max"].array()
chargeloss_array = np.loadtxt("../data/chargeloss_cpdf.dat", dtype=np.float32)
bls_array = np.loadtxt("../data/bls_cpdf.dat", dtype=np.float32)
from gpu_modules.TestSignalSimCuda import pandax4t_signal_sim

start = drv.Event()
end = drv.Event()
GPUFunc     = SourceModule(pandax4t_signal_sim, no_extern_c=True).get_function("signal_simulation")
GPUFunc_er  = SourceModule(pandax4t_signal_sim, no_extern_c=True).get_function("signal_simulation_er")

def get_2dband(theta, binning):

    #########################################
    ## Initialization
    ########################################

    #dev         = drv.Device(0) # hard coded, because we only have 1 GPU
    #gpuContext  = dev.make_context()
    
    xbin, xmin, xmax = binning[0]
    ybin, ymin, ymax = binning[1]
    
    ########################################
    ## Run the GPU sim
    ########################################

    #gpuContext.push() # start

    # define a input array under cpu level
    gpu_seed            = int(time.time()*100)
    num_trials          = 2**23
    GPUSeed             = np.asarray(gpu_seed, dtype=np.int32)
    input_array         = np.asarray([num_trials, xbin, xmin, xmax, ybin, ymin, ymax], dtype=np.float32)
    output_array        = np.zeros(11+xbin*ybin*2, dtype=np.double)
    output_array_er     = np.zeros(11+xbin*ybin*2, dtype=np.double)
    par_array = np.asarray(theta, dtype=np.float32)
    
    # define the block & grid
    # this is some GPU specific parameter
    # don't need to change
    # HARDCODE WARNING
    d_gpu_scale = {}
    nThreads = 128
    d_gpu_scale['block'] = (nThreads, 1, 1)
    numBlocks = np.floor(float(num_trials) / np.power(float(nThreads), 1.))
    d_gpu_scale['grid'] = (int(numBlocks), 1)
    # important step, need to push (or copy) it to GPU memory so that GPU function can use it
    # this step take time so shall minimize the number of times calling it
    tArgs               = [
        drv.In(GPUSeed),
        drv.In(input_array),
        drv.In(par_array),
        drv.In(bls_array),
        drv.In(chargeloss_array),
        drv.InOut(output_array)
    ]
    GPUFunc(*tArgs, **d_gpu_scale)
    tArgs_er               = [
        drv.In(GPUSeed),
        drv.In(input_array),
        drv.In(par_array),
        drv.In(bls_array),
        drv.In(chargeloss_array),
        drv.InOut(output_array_er)
    ]
    GPUFunc_er(*tArgs_er, **d_gpu_scale)
    #gpuContext.pop() #finish
    #print(output_array[1], output_array_er[1])
    return output_array, output_array_er

def get_bin_num(x, y, binning):
    xBinning, xmin, xmax = binning[0]
    yBinning, ymin, ymax = binning[1]
    xStep = (xmax - xmin) / xBinning
    yStep = (ymax - ymin) / yBinning
    xbin = np.asarray((x - xmin)/xStep + 1, dtype = np.int32)
    ybin = np.asarray((y - ymin)/yStep + 1, dtype = np.int32)
    index = (10 + (ybin-1) * xBinning + xbin)
    return index

def get_lnlikelihood(theta):

    # encoded mc simu hist
    binning = [
        [50, 2.,120.],
        [50, 0., 3.5]
    ]
    #MC_starter =  time.time()
    hmc, hmc_er = get_2dband(theta, binning)
    #print(hmc[1], hmc_er[1])
    #MC_time = time.time() - MC_starter
    #print(f"MC in this step is{MC_time:1.3e}")
    
    #lnL_starter = time.time()
    lnLikelihood_total = 0.
    
    # calculate likelihood 
    hmc_total = np.asarray(hmc) + np.asarray(hmc_er) + 1e-35 # the small value of 1e-35 is for avoiding calculation infinity
    Nmc = hmc_total[1]
    # this value is determined after test the normal Nmc, just for this tritium weight#
    # to aviod almost empty hist #
    # for other energy spectrum input, need to test again#
    # just print out normal Nmc is okay #
    if Nmc<1e6:    
        return -np.inf
    else:
        
        global cS1, cS2b, cS2
        
        #be sure the data inside hist#
        
        inds = np.where((cS1>=binning[0][1])&(cS1<binning[0][2])&(cS2<8000)&(np.log10(cS2b)>=binning[1][1])&(np.log10(cS2b)<binning[1][2]))[0]
        cS1_sel = cS1[inds]
        cS2b_sel = cS2b[inds]

        bin_inds = get_bin_num(cS1_sel,np.log10(cS2b_sel),binning)

        lnLikelihood_total=np.sum(np.log(hmc_total[bin_inds]/Nmc))
        #lnL_time = time.time()-lnL_starter
        #print(f"lnL in this step is {lnL_time:1.3e}")
        return -2. * lnLikelihood_total

def get_lnprior(theta):
    g1, g2b, segb = theta[0], theta[1], theta[2]
    if 0.< g1 <0.2 and g2b > 0. and segb > 0. and theta[10] > 0.:
        return -0.5*np.sum(np.power((theta[0:3]-nuisance_mean),2)/np.power(nuisance_sigma,2))
    else:
        return -np.inf

def get_lnprob(theta, scale):
    lp = get_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnprob =  lp + get_lnlikelihood(theta)
    return lnprob/scale


import emcee

nwalkers =  100
ndim_nonzero, ndim_zero = 3+1, 3+3+1
ndim = ndim_nonzero + ndim_zero
updateFlag = False
scaleFactor = 1.
p0 = [[nuisance_mean[0], nuisance_mean[1], nuisance_mean[2], 1., -0.15, 0.38, 0.11, -0.01, 0.13, 0.2, 2e-3]*(1 + 0.1*np.random.randn(ndim)) for i in range(nwalkers)]

#p0a = [np.concatenate(nuisance_mean,np.asarray([20.,1.],np.float32))*(1 + 0.1*np.random.randn(ndim_nonzero)) for i in range(nwalkers)]
#p0a = [[nuisance_mean[0], nuisance_mean[1], nuisance_mean[2], 1.]*(1 + 0.1*np.random.randn(ndim_nonzero)) for i in range(nwalkers)]
#p0b = [[0.,0.,0.,0.,0.,0.,0.] + 0.05*np.random.randn(ndim_zero) for i in range(nwalkers)]
#p0 = np.concatenate((p0a, p0b[:,:ndim_zero-1]), axis=1)
#p0 = np.concatenate((p0, np.absolute(p0b[:,-1])), axis=1)

#samplesFromLastRun = np.loadtxt("./outputs/samples.dat")
#samplesFromLastRunOriginal = samplesFromLastRun.reshape(samplesFromLastRun.shape[0], samplesFromLastRun.shape[1] // ndim, ndim)
#p00 = samplesFromLastRunOriginal[:,-1,:]
#p0 = np.concatenate((p00[:,:9], np.absolute(p00[:,-1].reshape(samplesFromLastRun.shape[0], 1))), axis=1)


# get walkers initial positions from the last run
#samplesFromLastRun = np.loadtxt("./outputs/samples.dat")
#samplesFromLastRunOriginal = samplesFromLastRun.reshape(samplesFromLastRun.shape[0], samplesFromLastRun.shape[1] // ndim, ndim)
#p0 = samplesFromLastRunOriginal[:,-1,:]

if updateFlag:
    updateTimes = 5
    iteration = 500
    iterationForDeltaLnL = 100
    acceptances = []
    sample_list = []
    for i in range(updateTimes):
        # Update scale factor in every 50 steps #
        sampler = emcee.EnsembleSampler(nwalkers, ndim, get_lnprob, args=(scaleFactor,))
        sampler.run_mcmc(p0,iteration)
        p0 = sampler.chain[:,-1,:]
        parForDeltaLnL = sampler.chain.reshape((-1,ndim))
        sample_list.append(sampler.chain.reshape(1,-1))
        
        parForDeltaLnL = np.mean(parForDeltaLnL, axis = 0)
        acceptances.append(sampler.acceptance_fraction)
        # calculate DeltaLnL #
        LnLs = []
        IfSuccess = True
        for j in range(iterationForDeltaLnL):
            OneLnL = get_lnprob(parForDeltaLnL,scaleFactor)
            if not np.isfinite(OneLnL):
                IfSuccess = False
                break
            LnLs.append(OneLnL)
        if IfSuccess:
            DeltaLnL = np.std(LnLs)
            print("####### Finishing checking DeltaLnL of "+str(i)+"th update ########")
            print("####### Updated DeltaLnL to "+str(DeltaLnL)+" ########")
            scaleFactor = np.sqrt(1+DeltaLnL**2)
    
    sample_list = np.array(sample_list)
    samples = np.reshape(sample_list,(nwalkers,updateTimes*iteration,ndim))            

else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, get_lnprob, args=(scaleFactor,))
    sampler.run_mcmc(p0,500,progress=True)
    samples = sampler.chain
    lnls = sampler.lnprobability

samples_reshaped = samples.reshape(nwalkers, -1)
np.savetxt("./outputs/samples.dat",samples_reshaped)
np.savetxt("./outputs/acceptances.dat",sampler.acceptance_fraction)
    
plt.hist(sampler.acceptance_fraction,histtype = "step")
plt.title("acceptance ratio")
plt.savefig("./outputs/acceptances.pdf")

np.savetxt("./outputs/lnls.dat",lnls)
plt.plot(lnls.T, linewidth = 0.1, color = "black")
plt.title("lnlikelihoods")
plt.savefig("./outputs/lnls.pdf")

fig, axs = plt.subplots(4, 3)

plt.yticks(size = 5)
plt.xticks(size = 5)

axs[0, 0].plot(samples[:,:,0].T, linewidth = 0.1, color = "black")
axs[0, 0].set_title('g1',fontsize=7)

axs[0, 1].plot(samples[:,:,1].T, linewidth = 0.1, color = "black")
axs[0, 1].set_title('g2b',fontsize=7)

axs[0, 2].plot(samples[:,:,2].T, linewidth = 0.1, color = "black")
axs[0, 2].set_title('SEG_b',fontsize=7)

axs[1, 0].plot(samples[:,:,3].T, linewidth = 0.1, color = "black")
axs[1, 0].set_title('scale deltar',fontsize=7)

axs[1, 1].plot(samples[:,:,4].T, linewidth = 0.1, color = "black")
axs[1, 1].set_title('p0_r',fontsize=7)

axs[1, 2].plot(samples[:,:,5].T, linewidth = 0.1, color = "black")
axs[1, 2].set_title('p1_r',fontsize=7)

axs[2, 0].plot(samples[:,:,6].T, linewidth = 0.1, color = "black")
axs[2, 0].set_title('p2_r',fontsize=7)

axs[2, 1].plot(samples[:,:,7].T, linewidth = 0.1, color = "black")
axs[2, 1].set_title('p0_l',fontsize=7)

axs[2, 2].plot(samples[:,:,8].T, linewidth = 0.1, color = "black")
axs[2, 2].set_title('p1_l',fontsize=7)

#axs[2, 1].plot(samples[:,:,7].T, linewidth = 0.1, color = "black")
#axs[2, 1].set_title('p2_',fontsize=7)

axs[3, 0].plot(samples[:,:,9].T, linewidth = 0.1, color = "black")
axs[3, 0].set_title('p2_l',fontsize=7)

axs[3, 1].plot(samples[:,:,10].T, linewidth = 0.1, color = "black")
axs[3, 1].set_title('flat ER',fontsize=7)

fig.tight_layout()

plt.savefig("./outputs/parameters.pdf")

