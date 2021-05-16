import numpy as np
import time
import matplotlib.pyplot as plt
import sys

PATH = '../FittingGPU/outputs/'

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


from gpu_modules.TestSignalSimCuda import pandax4t_signal_sim

#########################################
## Initialization
########################################


#dev         = drv.Device(0) # hard coded, because we only have 1 GPU
#gpuContext  = dev.make_context()
start = drv.Event()
end = drv.Event()

GPUFunc     = SourceModule(pandax4t_signal_sim, no_extern_c=True).get_function("signal_simulation")


########################################
## Run the GPU sim
########################################

#gpuContext.push() # start

# define a input array under cpu level
gpu_seed            = int(time.time()*100)
num_trials          = 2**20
GPUSeed             = np.asarray(gpu_seed, dtype=np.int32)
# iteration number and binning settings
xbinning, xmin, xmax = [100, 2., 120.]
ybinning, ymin, ymax = [100, 0., 3.5]
# histFlag gives intstruction about what kink of hists it will plot
# 0 - 2d band, which requires 6 pars of hist setting
# 1 - energy&Nph&Ne&cS1&cS2 comparison, which would be easily to decode to store samples, but requires larger storage space, better set num_trials~1e6 if someone need to do this.
histFlag = int(sys.argv[1])
input_array         = np.asarray([num_trials, histFlag, xbinning, xmin, xmax, ybinning, ymin, ymax], dtype=np.float32)

output_trials = 0
if histFlag == 0:
    output_trials = 11 + 2 * xbinning * ybinning
elif histFlag == 1:
    output_trials = num_trials * 3

output_array        = np.zeros(output_trials, dtype=np.double)
#par_bestfit         = [0.09459, 0.765, 19., 0.2535, 19.75, 1., 0., 0., 0.,]
par_bestfit         = np.loadtxt(PATH+'bestfit.dat')[1].T
nuisance_par_array = np.asarray(par_bestfit, dtype=np.float32)
print(nuisance_par_array)

# important step, need to push (or copy) it to GPU memory so that GPU function can use it
# this step take time so shall minimize the number of times calling it
tArgs               = [
    drv.In(GPUSeed),
    drv.In(input_array),
    drv.In(nuisance_par_array),
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
if histFlag == 0:
    np.savetxt(PATH+'2dband.dat',output_array)
elif histFlag == 1:
    np.savetxt(PATH+'mcoutputs.dat',output_array)
