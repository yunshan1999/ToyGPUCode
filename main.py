##############################################
## Simple code for Yunshan
##############################################
import numpy as np
import time
import matplotlib.pyplot as plt

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
num_trials          = 2**19
GPUSeed             = np.asarray(gpu_seed, dtype=np.int32)
input_array         = np.asarray([num_trials, 100, 0., 50., 100., 0.5, 4.], dtype=np.float32)
output_array        = np.zeros(1+9+100*100*2, dtype=np.float32)
nuisance_par_array = np.asarray([0.09997, 0.3, 0.2, 28., 5.09017 * 4, 0.72717, 7., 600.], dtype=np.float32)

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
start_time = time.time()
start.record()
GPUFunc(*tArgs, **d_gpu_scale)
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
#gpuContext.pop() #finish
run_time = time.time() - start_time
print("GPU run time %f seconds " % secs)
np.savetxt("./outputs/outputs_gpu.dat",output_array)
#time.sleep(5)
# decode the output array to get shape hist

#hist_array = output_array[0:-1]
#xbin = int(input_array[1])
#xmin = input_array[2]
#xmax = input_array[3]
#ybin = int(input_array[4])
#ymin = input_array[5]
#ymax = input_array[6]
#hist_array = hist_array.reshape(ybin, xbin)
#plt.pcolor(hist_array)
#plt.savefig("./outputs/hist.pdf")
