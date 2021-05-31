import numpy as np
import time
import json
import sys
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


from gpu_modules.ReweightSignalSimCuda import pandax4t_signal_sim


if __name__ == "__main__":

  jsonFileName = sys.argv[1]
  jFile = jsonFileName 
  with open(jFile) as json_data:
    config = json.load(json_data)

  # define a input array under cpu level
  gpu_seed            = int(time.time()*100)
  num_trials          = 2**24 #simulation event number
  branch              = 25 #branch for output data
  GPUSeed             = np.asarray(gpu_seed, dtype=np.int32)
  
  #########################################
  ##constant parameters 
  #########################################

  #simuTypeNR: true for NR; false for ER
  simuTypeNRStr = np.str(config['NR'])
  simuTypeNR = np.int(0)
  if (simuTypeNRStr == 'True ' or simuTypeNRStr == 'true '):
    simuTypeNR = 1
  
  #energyRange
  #minE = config['E_eeTh']
  minE = 0 
  maxE = config['E_eeMaxSimu']
  energyType = config['python3.6']['type']

  #efficiencies
  v_s1eff = np.asarray(config['eff']['s1par'],dtype=np.float32)
  v_s1eff_tag = np.asarray(config['eff']['s1par_tag'],dtype=np.float32)
  v_s2eff = np.asarray(config['eff']['s2par'],dtype=np.float32)
  
  #detector related parameters
  sPEres = np.float(config['pmt_resolution'])
  P_dphe = np.float(config['dpe'])
  E_drift = np.float(config['field'])
  driftvelocity = np.float(config['v_e'])
  dt_min = np.float(config['dt_l'])
  dt_max = np.float(config['dt_u'])
  zDrift = np.float(config['zDrift'])

  const_pars = np.asarray([num_trials,simuTypeNR,minE,maxE,
              v_s1eff[0],v_s1eff[1],v_s1eff_tag[0],v_s1eff_tag[1],
              v_s2eff[0],v_s2eff[1],
              energyType, sPEres, P_dphe, E_drift,
              driftvelocity,dt_min,dt_max,zDrift],dtype=np.float32)
  print(const_pars)

  #elife info
  elife = np.asarray(config['elife']['lifetime_b'],dtype=np.float32)
  elife_duration = np.asarray(config['elife']['duration'],dtype=np.float32)
  n_elife = np.float32(elife.shape[0])
  elife = np.append([n_elife],elife)
  elife_duration = np.append([n_elife],elife_duration)
  print("lengh of elife ",n_elife)
  print("lengh of duration ",elife_duration.shape[0])

  #########################################
  ## Initialization
  ########################################
  
  #dev         = drv.Device(0) # hard coded, because we only have 1 GPU
  #gpuContext  = dev.make_context()
  start = drv.Event()
  end = drv.Event()
  GPUFunc     = SourceModule(pandax4t_signal_sim, no_extern_c=True).get_function("signal_simulation")
  
  #gpuContext.push() # start
  
  #ouput "tree" initialize
  output_array        = np.zeros((num_trials,branch), dtype=np.float32)
  N_output_array      = np.int32(output_array.shape[0])
  stride_output_array = np.int32(output_array.strides[0])
  
  
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
  
  v_variables = v_variables.reshape((rowNo,2))
  print(v_variables)
  
  
  # important step, need to push (or copy) it to GPU memory so that GPU function can use it
  # this step take time so shall minimize the number of times calling it
  mode=np.uint(0)

  tArgs               = [
      drv.In(GPUSeed),
      drv.In(const_pars),
      drv.In(v_variables),
      drv.In(elife),drv.In(elife_duration),
      drv.InOut(output_array),
      stride_output_array,N_output_array,mode
  ]
  
  # define the block & grid
  # this is some GPU specific parameter
  # don't need to change
  # HARDCODE WARNING
  d_gpu_scale = {}
  #Maximum grid dimensions: 	2147483647, 65535, 65535
  nThreads = 256
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
  outputfile = config['python3.6']['files']['oriSimuNpz']
  #outputfile = 'outputs/ori_test.npy.npz' 
  np.savez(outputfile,oriTree=output_array)
  #np.savetxt("./outputs/outputs_gpu.txt",output_array)
  #time.sleep(5)
  
