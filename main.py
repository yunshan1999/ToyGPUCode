##############################################
## Simple code for Yunshan
##############################################
import numpy as np
import time

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


# dev         = drv.Device(0) # hard coded, because we only have 1 GPU
# gpuContext  = dev.make_context()

GPUFunc     = SourceModule(pandax4t_signal_sim, no_extern_c=True).get_function("signal_simulation")



########################################
## Run the GPU sim
########################################

# gpuContext.push() # start

# define a input array under cpu level
gpu_seed            = int(time.time()*100)
num_trials          = 1024 * 1024  # for example we print out 1024 * 1024 GPU nodes

GPUSeed             = np.asarray(gpu_seed, dtype=np.int32)
input_array         = np.asarray([num_trials], dtype=np.int32)
output_array        = np.zeros(num_trials * 2 + 1, dtype=np.float32)

# important step, need to push (or copy) it to GPU memory so that GPU function can use it
# this step take time so shall minimize the number of times calling it
tArgs               = [
    drv.In(GPUSeed),
    drv.In(input_array),
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


# to get running time
start_time = time.time()

# Run the GPU code
GPUFunc(*tArgs, **d_gpu_scale)

run_time = time.time() - start_time

print("GPU run time %f seconds " % run_time)

output_file = open("./outputs/outputs.txt", "w")
np.savetxt(output_file, output_array)
output_file.close()
# gpuContext.pop() # finish


