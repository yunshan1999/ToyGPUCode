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
nuisance_sigma = np.asarray([0.01285, 0.227, 2.36688], np.float32)

from gpu_modules.SignalSimCuda import pandax4t_signal_sim

start = drv.Event()
end = drv.Event()
GPUFunc     = SourceModule(pandax4t_signal_sim, no_extern_c=True).get_function("signal_simulation")

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
    num_trials          = 2**28
    GPUSeed             = np.asarray(gpu_seed, dtype=np.int32)
    input_array         = np.asarray([num_trials, 0, xbin, xmin, xmax, ybin, ymin, ymax], dtype=np.float32)
    output_array        = np.zeros(11+xbin*ybin*2, dtype=np.double)
    par_array = np.asarray(theta, dtype=np.float32)

    # important step, need to push (or copy) it to GPU memory so that GPU function can use it
    # this step take time so shall minimize the number of times calling it
    tArgs               = [
        drv.In(GPUSeed),
        drv.In(input_array),
        drv.In(par_array),
        drv.InOut(output_array)
    ]

    # define the block & grid
    # this is some GPU specific parameter
    # don't need to change
    # HARDCODE WARNING
    d_gpu_scale = {}
    nThreads = 128
    d_gpu_scale['block'] = (nThreads, 1, 1)
    numBlocks = np.floor(float(num_trials) / np.power(float(nThreads), 1.))
    d_gpu_scale['grid'] = (int(numBlocks), 1)
  # Run the GPU code
    GPUFunc(*tArgs, **d_gpu_scale)
    #gpuContext.pop() #finish

    return output_array

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

    # data are in root ttree, we use uproot lib to get them as arrays
    DATA_PATH = '../data/reduce_ana3_p4_run1_tritium_5kV.root'
    data_tree = uproot.open(DATA_PATH)["out_tree"]
    cS1 = data_tree["qS1C_max"].array()
    cS2b = data_tree["qS2BC_max"].array()

    # encoded mc simu hist
    binning = [
        [50, 2.,120.],
        [50, 0., 3.5]
    ]
    #MC_starter =  time.time()
    hmc = get_2dband(theta, binning)
    #MC_time = time.time() - MC_starter
    #print(f"MC in this step is{MC_time:1.3e}")

    #lnL_starter = time.time()
    lnLikelihood_total = 0.

    # calculate likelihood 
    hmc = np.asarray(hmc) + 1e-35 # the small value of 1e-35 is for avoiding calculation infinity
    Nmc = hmc[1]
    print(Nmc) 
    # this value is determined after test the normal Nmc, just for this tritium weight#
    # to aviod almost empty hist #
    # for other energy spectrum input, need to test again#
    # just print out normal Nmc is okay #
    #if Nmc<1e6:
        #return -np.inf
    if True:
        inds = np.where((cS1>=2)&(cS1<120.)&(np.log10(cS2b)<3.5))[0]
        ccS1 = cS1[inds]
        ccS2b = cS2b[inds]

        bin_inds = get_bin_num(ccS1,np.log10(ccS2b),binning)
        lnLikelihood_total=np.sum(np.log(hmc[bin_inds]/Nmc))
        #print(np.where(np.log(hmc[bin_inds]/Nmc)<-7)[0])
        #events_for_delete = np.array([513, 537, 681, 705, 817, 964, 1072, 1197, 1269])
        events_for_delete = np.array([10   ,19   ,22  , 24  , 26  , 46  , 50   ,56   ,60   ,67   ,69   ,76  ,105  ,132,138 , 168 , 170,  172,  183,  208,  209 , 232 , 242 , 247 , 251 , 253,  263,  279,  293 , 309 , 312,  328,  339,  348,  406 , 411 , 415 , 446 , 470 , 471,  513,  528,  537 , 540 , 564,  571,  609,  613,  653 , 681 , 697 , 705 , 717 , 742,  786,  817,  877 , 888 , 912,  913,  914,  953,  964 , 969 , 992 ,1004 ,1016 ,1029, 1040, 1065,1072 ,1080 ,1091, 1094, 1114, 1141, 1153 ,1155 ,1175 ,1188 ,1192 ,1193, 1196, 1197, 1201 ,1212 ,1214, 1258, 1269, 1284])
        #print(events_for_delete)
        #cS1_del_in_original = np.where()
        
        #print(ccS1[events_for_delete])
        #print(ccS2b[events_for_delete])
        #np.savetxt('./s1_for_delete.dat',ccS1[events_for_delete])
        #np.savetxt('./s2_for_delete.dat',ccS2b[events_for_delete])
        #lnL_time = time.time()-lnL_starter
        #print(f"lnL in this step is {lnL_time:1.3e}")
        return ccS1[events_for_delete],ccS2b[events_for_delete]

def get_lnprior(theta):
    g1, g2b, segb = theta[0], theta[1], theta[2]
    if 0.< g1 <0.2 and g2b > 0. and segb > 0. and theta[3] > 0. and theta[4] > 0.:
        return -0.5*np.sum(np.power((theta[0:3]-nuisance_mean),2)/np.power(nuisance_sigma,2))
    else:
        return -np.inf

def get_lnprob(theta, scale):
    lp = get_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnprob =  lp + get_lnlikelihood(theta)
    return lnprob/scale

PATH = '../FittingGPU/outputs/'
par_bestfit         = np.loadtxt(PATH+'bestfit.dat')[1].T
aa, bb = get_lnlikelihood(par_bestfit)
aa = np.array(aa)
bb = np.array(bb)
np.savetxt('./s1_for_delete.dat',aa)
np.savetxt('./s2_for_delete.dat',bb)
