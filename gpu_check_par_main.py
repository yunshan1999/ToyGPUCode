import numpy as np
import time
import json
#import matplotlib.pyplot as plt

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


from gpu_modules.TestReweightSignalSimCuda import pandax4t_signal_sim

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
#some initial values
num_trials          = 2**23
branch              = 40 #branch for output data

GPUSeed             = np.asarray(gpu_seed, dtype=np.int32)

#ouput "tree" initialize
output_array        = np.zeros((num_trials,branch), dtype=np.float32)
N_output_array      = np.int32(output_array.shape[0])
stride_output_array = np.int32(output_array.strides[0])

file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)

#elife info
elife = np.asarray(config['elife']['lifetime_b'],dtype=np.float32)
elife_duration = np.asarray(config['elife']['duration'],dtype=np.float32)
n_elife = np.int32(elife.shape[0])
print("lengh of elife ",n_elife)
print("lengh of duration ",elife_duration.shape[0])

#varying variables
v_variables = np.array([],dtype=np.float32)
rowNo = 0
for attri,luRange in config['python3.6']['nuissanceParSpace'].items():
  rowNo = rowNo + 1
  a = np.asarray(luRange,dtype=np.float32)
  v_variables = np.append(v_variables,a)
for attri,luRange in config['python3.6']['fittingParSpace'].items():
  rowNo = rowNo + 1
  a = np.asarray(luRange,dtype=np.float32)
  v_variables = np.append(v_variables,a)

best_variable = np.asarray([0.0923185, 3.59445, 6.696541, 0.22419741 , 0.22511, -0.25262 , 1.13235351 , 29.046818],dtype = np.float32)
v_variables = best_variable 
print("variable input: ",best_variable)


# important step, need to push (or copy) it to GPU memory so that GPU function can use it
# this step take time so shall minimize the number of times calling it
minE = config['E_eeTh']
maxE = config['E_eeMaxSimu']
const_pars = np.asarray([num_trials,n_elife,minE,maxE],dtype=np.float32)
print(const_pars)
mode=np.uint(1)
tArgs               = [
    drv.In(GPUSeed),
    drv.In(const_pars),
    drv.In(v_variables.T),
    drv.In(elife),drv.In(elife_duration),
    drv.InOut(output_array),stride_output_array,N_output_array,mode
]

# define the block & grid
# this is some GPU specific parameter
# don't need to change
# HARDCODE WARNING
d_gpu_scale = {}
#Maximum grid dimensions: 	2147483647, 65535, 65535
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
print(output_array[1])
outputfile = config['python3.6']['files']['oriBestSimuNpz']
np.savez(outputfile,oriTree=output_array)
#np.savetxt("./outputs/outputs_gpu.txt",output_array)
#time.sleep(5)

