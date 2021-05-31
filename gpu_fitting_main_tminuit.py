###################################
## DEMO version for GPU fitting
###################################
## Created @ Apr. 12th, 2021
## By Yunshan Cheng and Dan Zhang
## contact: yunshancheng1@gmail.com
###################################
import numpy as np
import sys
import time
import math
import uproot
import emcee
#import corner
import json
from ROOT import TMinuit
import ROOT
import ctypes
from array import array as arr

try:
  from pycuda.compiler import SourceModule
  import pycuda.driver as drv
  import pycuda.tools
  import pycuda.gpuarray as gpuarray
  import pycuda.autoinit
except ImportError:
  print(
    'Warning: Cuda libraries not found - RunManager.init() not available!'
   )

from gpu_modules.reweight import pandax4t_reweight_fitting 


#three kernels needed
GPUFunc   = SourceModule(pandax4t_reweight_fitting, no_extern_c=True).get_function("main_reweight_fitting")

GPUFuncRenormalization = SourceModule(pandax4t_reweight_fitting, no_extern_c=True).get_function("renormalization_reweight")

GPUlnLikelihood = SourceModule(pandax4t_reweight_fitting, no_extern_c=True).get_function("ln_likelihood")
  
def main(jsonFile):
  
  global config
  with open(jFile) as json_data:
    config = json.load(json_data)
  
  #parameters of interest
  pde = np.float32(config['pde'])
  g2b = np.float32(config['g2b'])
  segb = np.float32(config['segb'])

  global pde_g2b_theta,nuissancePars
  nuissancePars = np.asarray([pde,g2b,segb],dtype = np.float32)
  pde_g2b_theta = np.float32(config['python3.6']['nuissanceParRelativeUncertainty']['pde_g2b_theta'])
  global nuissanceParsReUnc
  nuissanceParsReUnc = np.asarray([config['python3.6']['nuissanceParRelativeUncertainty']['pde'],
  config['python3.6']['nuissanceParRelativeUncertainty']['g2b'],
  config['python3.6']['nuissanceParRelativeUncertainty']['segb'] ] ,dtype = np.float32)
  
  global fittingParSpace
  fittingParSpace = np.array([],dtype=np.float32)
  rowNo = 0
  for attri,luRange in config['python3.6']['nuissanceParSpace'].items():
    rowNo = rowNo + 1
    a = np.asarray(luRange,dtype=np.float32)
    fittingParSpace = np.append(fittingParSpace,a)
  for attri,luRange in config['python3.6']['fittingParSpace'].items():
    rowNo = rowNo + 1
    a = np.asarray(luRange,dtype=np.float32)
    fittingParSpace = np.append(fittingParSpace,a)
    fittingParSpace = fittingParSpace.reshape((rowNo,2))
  if (debug):
    print('nuissancePars', nuissancePars)
  
  #energy spectrum array initialize
  minE = np.float32(config['E_eeTh']) - 0.1
  #minE = np.float32(0.)
  maxE = np.float32(config['E_eeMaxSimu']) + 0.1
  stepE = 0.2;
  v_Erange = np.arange(minE,maxE,stepE)
  global v_E
  v_E = v_Erange[:-1] + 0.5 * stepE
  
  
  #simulation data loading
  oriTreeFile  = config['python3.6']['files']['oriSimuNpz']
  oriTree = np.load(oriTreeFile)
  simu_tree = oriTree['oriTree']
  global simu_tree_gpu, weight_gpu, normalization_gpu, eff_gpu, cS1_gpu, cS2_gpu, const_par_gpu
  simu_tree_gpu = gpuarray.to_gpu(simu_tree)
  weight_gpu = drv.mem_alloc(simu_tree[:,0].nbytes)
  normalization_gpu = drv.mem_alloc(simu_tree[:,0].nbytes)
  eff_gpu = drv.mem_alloc(simu_tree[:,0].nbytes)
  global num_trials,simu_tree_par
  num_trials = simu_tree.shape[0]
  simu_tree_par = np.asarray([num_trials,simu_tree.strides[0]],dtype=np.float32)
  simu_tree_bytes = simu_tree.size * simu_tree.dtype.itemsize
  print('input simu_tree_par shape : ',simu_tree_par)

  #detector parameters

  #simuTypeNR: true for NR; false for ER
  simuTypeNRStr = np.str(config['NR'])
  simuTypeNR = np.int(0)
  if (simuTypeNRStr == 'True ' or simuTypeNRStr == 'true '):
    simuTypeNR = 1

  #energeType: 3 for tritium, 220 for th228/rn220
  energyType = np.int(config['python3.6']['type'])

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

  const_pars = np.asarray([stepE,simuTypeNR,minE,maxE,
        v_s1eff[0],v_s1eff[1],v_s1eff_tag[0],v_s1eff_tag[1],
        v_s2eff[0],v_s2eff[1],
        energyType, sPEres, P_dphe, E_drift,
        driftvelocity,dt_min,dt_max,zDrift],dtype=np.float32)
  if (debug):
    print(const_pars)
  const_par_gpu = gpuarray.to_gpu(const_pars)

  #boundary conditions
  global s1min,s1u_c,s2min,s2minRaw,s2max,s2u_c,nHitsS1min
  s1min = np.float32(config['s1min'])
  s1u_c = np.float32(config['s1u_c'])
  s2min = np.float32(config['s2min'])
  s2minRaw = np.float32(config['s2minRaw'])
  s2max = np.float32(config['s2max'])
  s2u_c = np.float32(config['s2u_c'])
  nHitsS1min = np.float32(config['nHitsS1min'])
  

  cS1,cS2 = get_2dData()
  global Ndata
  Ndata = np.uint([cS1.shape[0]])
  print("survived data number: ",Ndata)
  cS1_gpu = gpuarray.to_gpu(cS1)
  cS2_gpu = gpuarray.to_gpu(cS2)
  
  
  #for main_reweight_fitting
  global d_gpu_scale
  d_gpu_scale = {}
  nThreads = 128
  d_gpu_scale['block'] = (nThreads, 1, 1)
  numBlocks = np.floor(float(num_trials) / float(nThreads))
  d_gpu_scale['grid'] = (int(numBlocks), 1)


  #binning setting
  xbin = config['Bin_s1']
  ybin = config['Bin_s2']
  xmin=s1min
  xmax=s1u_c
  ymin=np.log10(s2min)
  ymax=np.log10(s2u_c)
  global binning
  binning = np.asarray([xbin,xmin,xmax,ybin,ymin,ymax],dtype=np.double)
  
  histogram2d_0 = np.zeros((xbin,ybin),dtype = np.double)
  global histogram2d_0_gpu
  histogram2d_0_gpu = drv.mem_alloc(histogram2d_0.nbytes)
  
  #for ln_likelihood
  global d_gpu_scale1
  d_gpu_scale1 = {}
  d_gpu_scale1['block'] = (nThreads, 1, 1)
  d_gpu_scale1['grid'] = ( int(np.floor(float(Ndata)/float(nThreads) ) ),1)




def get_ln_nuissance(nuissanceTheta):
  ln = np.double(0.);
  pde_i,g2b_i,segb_i = nuissanceTheta
  if (g2b_i > segb_i ):
    return -1000000
  re_pde,re_g2b,re_segb= np.divide(nuissanceTheta,nuissancePars)
  #print(nuissanceParsReUnc,"test nuissance")
  #re_pde_rotate = (re_pde - 1.) * np.cos(pde_g2b_theta) + (re_g2b - 1.) * np.sin(pde_g2b_theta)
  #re_g2b_rotate = -(re_pde - 1.) * np.sin(pde_g2b_theta) + (re_g2b - 1.) * np.cos(pde_g2b_theta)
  #print(re_pde_rotate,re_g2b_rotate,re_segb,re_s2b_fraction,"test nuissance")
  re_pde_rotate = re_pde
  re_g2b_rotate = re_g2b
  ln += -0.5*np.power( (re_pde_rotate - 1.0)/nuissanceParsReUnc[0],2)
  ln += -0.5*np.power( (re_g2b_rotate - 1.0)/nuissanceParsReUnc[1],2)
  ln += -0.5*np.power( (re_segb - 1.0)/ nuissanceParsReUnc[2],2)
  if(debug):
    print('ln : ',ln,re_pde_rotate,re_g2b_rotate,re_segb)
  return ln


def get_fitting_ln_prior(fittingTheta):
  ln = np.double(0.);
  for i in range(fittingTheta.shape[0]):
    if (fittingTheta[i] < fittingParSpace[i][0] or fittingTheta[i] > fittingParSpace[i][1]):
      ln = -1000000
  return ln
#print("test lnproir ", get_ln_nuissance([0.094,0.77,19,0.25]))


#calibration data loading
def get_2dData():
  DATA_PATH = config['calDataRootFile']; 
  data_tree = uproot.open(DATA_PATH)["out_tree"]
  cS1_0= np.asarray(data_tree["qS1C_max"].array(),dtype=np.float32)
  cS2_0 = np.asarray(data_tree["qS2BC_max"].array(),dtype=np.float32)
  nPMTS1_max = np.asarray(data_tree["nPMTS1_max"].array(),dtype=np.float32) 
  cS1_0_raw = np.asarray(data_tree["qS1_max"].array(),dtype=np.float32)
  cS2_0_raw = np.asarray(data_tree["qS2_max"].array(),dtype=np.float32)
  Ndata_0 = cS1_0.shape[0]
  cS1 = np.array([],dtype=np.float32)
  cS2 = np.array([],dtype=np.float32)

  for i in range(Ndata_0):
    if (nPMTS1_max[i] > nHitsS1min) and (cS1_0[i] > s1min ) and (cS1_0[i] < s1u_c) and (cS2_0_raw[i] > s2minRaw) and (cS2_0_raw[i] < s2max) and (cS2_0[i] > s2min) and (cS2_0[i] < s2u_c ):
      cS1 = np.append(cS1,[cS1_0[i]])
      cS2 = np.append(cS2,[cS2_0[i]])
  return cS1,cS2



def get_lnLikelihood(theta,modeIn):

  #########################################
  ## Initialization
  ########################################
  histogram2d_par   = np.asarray(np.append([0.0],binning),dtype=np.double)
  global v_E 
  v_E_dRdE = np.zeros(v_E.shape,dtype=np.float32)
  if (debug):
    print('v_E shape: ',v_E_dRdE.shape[0])
  
  #histogram2d_par contains 1.summation of sth, 2. binning, 3. energy spectrum before normalization
  histogram2d_par   = np.append( histogram2d_par, v_E_dRdE)


  #print("dtype check ",histogram2d_par.dtype)
  histogram2d = np.zeros((int(binning[0]),int(binning[3])),dtype = np.double)

  fitting_par_array = np.asarray(theta, dtype=np.float32)
  if (debug):
    global out_weight, out_norm_energy, out_eff
  out_weight = np.zeros(num_trials,dtype = np.float32)
  out_norm_energy = np.zeros(num_trials,dtype = np.float32)
  out_eff = np.zeros(num_trials,dtype = np.float32)
  drv.memcpy_htod(histogram2d_0_gpu,histogram2d)
  drv.memcpy_htod(weight_gpu,out_weight)
  drv.memcpy_htod(normalization_gpu,out_norm_energy)
  drv.memcpy_htod(eff_gpu,out_eff)

  # this step take time so shall minimize the number of times calling it
  #start.record()
  mode = np.uint(modeIn)
  tArgs         = [
    drv.In(fitting_par_array),
    simu_tree_gpu, drv.In(simu_tree_par),
    const_par_gpu, drv.InOut(histogram2d_par),
    weight_gpu,normalization_gpu,eff_gpu,
    mode
  ]

  ###############################
  # 1st reweighting kernel 
  ###############################
  GPUFunc(*tArgs, **d_gpu_scale)
  #end.record()
  drv.Context.synchronize()

  effectiveNo = histogram2d_par[0] 
  if(debug):
    print("effective integral ", effectiveNo)
  if(effectiveNo < (num_trials * 0.0005)):
    return -1000000.
  elif (math.isnan(effectiveNo)):
    return -1000000
   
  #the before normalization energy spectrum is prepared to be like [#of points, content]
  v_E_dRdE = histogram2d_par[7:]
  v_E_dRdE = np.asarray(np.append([v_E_dRdE.shape[0]],v_E_dRdE), dtype = np.float32)
  v_E_0 = np.asarray(np.append([v_E.shape[0]],v_E),dtype = np.float32)
  histogram2d_par = histogram2d_par[:8]
  histogram2d_par[0] = 0.
  histogram2d_par[7] = 0.

  tArgsR = [
    drv.In(v_E_0),drv.In(v_E_dRdE),
    simu_tree_gpu, drv.In(simu_tree_par),
    weight_gpu,normalization_gpu,eff_gpu,
    histogram2d_0_gpu, drv.InOut(histogram2d_par),
  ]
  
  ###############################
  #2nd normalization kernel
  ###############################
  GPUFuncRenormalization(*tArgsR , **d_gpu_scale)
  if(debug):
    print(effectiveNo)
    print("Nmc : ",histogram2d_par[0] )
    print("eff : ",histogram2d_par[0]/histogram2d_par[7] )
    #np.save('outputs/hist2dEx.npy',histogram2d)

  drv.Context.synchronize()

  #[#ofMC,binningInfo]
  likeli_par = histogram2d_par[:7]

  lnLikelihood_total = np.array([0.0],dtype=np.double)
  tArgs1 = [cS1_gpu, cS2_gpu, histogram2d_0_gpu, drv.In(likeli_par),
       drv.InOut(lnLikelihood_total),Ndata
  ]

  ###############################
  #3rd likelihood calculation
  ###############################
  GPUlnLikelihood(*tArgs1,**d_gpu_scale1)
  drv.Context.synchronize()



  if math.isnan(lnLikelihood_total[0]):
    return -1000000
  return lnLikelihood_total[0]


def get_lnprob(theta):
  NoNui=nuissancePars.shape[0]
  nuissancePar_i = theta[:NoNui]
  lp = get_ln_nuissance(nuissancePar_i)
  if not np.isfinite(lp):
    return -1000000
  lp1 = get_fitting_ln_prior(theta)
  if not np.isfinite(lp1):
    return -1000000
  lnprob =  get_lnLikelihood(theta,1)
  return  lp + lnprob

def test_get_likehood(theta):
  likelihood = get_lnprob(theta)
  ###############################
  #copy info back from GPU
  ###############################
  drv.memcpy_dtoh(out_weight,weight_gpu)
  drv.memcpy_dtoh(out_norm_energy,normalization_gpu)
  drv.memcpy_dtoh(out_eff,eff_gpu)
  print('newWeightSum',sum(out_weight))
  print('normSum',sum(out_norm_energy))
  outFile = np.str(config['python3.6']['files']['bestNpz'])
  np.savez(outFile,pars=theta,new_weight=out_weight,normalization = out_norm_energy)
  #likelihood = get_lnLikelihood(theta)
  print("likelihood_output: ",likelihood)


def fitting_with_emcee(theta):
  ndim = theta.shape[0]
  p0_stater= np.asarray(theta,dtype = np.float32)
  nwalkers = 100
  #becareful
  p0 = np.asarray([p0_stater * (1 + 0.2*np.random.randn(ndim)) for i in range(nwalkers)])
  sampler = emcee.EnsembleSampler(nwalkers, ndim, get_lnprob, args=())
  sampler.run_mcmc(p0,300,progress=True)
  samples = sampler.get_chain()
  lnls = sampler.lnprobability
  acceptance = sampler.acceptance_fraction
  outFileName = np.str(config['python3.6']['files']['fittingNpz'])
  np.savez(outFileName,lnls=lnls,acceptance = acceptance,samples = samples)



class functor_min(ROOT.Math.IMultiGenFunction):
  def NDim( self ):
      print('PYTHON NDim called')
      return 2

  def DoEval( self, args ):
    theta_i = np.asarray([],dtype=np.float32)
    for i in range(8):
      theta_i = np.append(theta_i,[args[i]])
    minLike = np.double(-1.) * get_lnprob(theta_i)
    #print('test',theta_i,'  minlike:',minLike)
    return minLike

  def Clone( self ):
      x = functor_min()
      ROOT.SetOwnership(x, False)
      return x

def fitting_with_tminuit2(theta):
  minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2","Migrad")
  minimizer.SetMaxFunctionCalls(10000)
  minimizer.SetMaxIterations(100000)
  minimizer.SetTolerance(0.01)
  minimizer.SetPrintLevel(1)
  npar = theta.shape[0] 
  f = ROOT.Math.Functor(functor_min(),npar)
  minimizer.SetFunction(f)
  name=np.asarray([],dtype=np.str)
  for i in range(npar): # Define the parameters for the fit
    name_i = "par_%d" % i
    np.append(name,[name_i])
    #minimizer.SetVariable(i,name_i,theta[i], np.abs(0.01*theta[i]))
    #print("limit ",i,fittingParSpace[i][0],fittingParSpace[i][1])
    minimizer.SetLimitedVariable(i,name_i,theta[i],0.001,fittingParSpace[i][0],fittingParSpace[i][1])

  minimizer.Minimize()
  finalPar = np.zeros(npar,dtype=np.float32)
  finalParErr = np.zeros(npar,dtype=np.float32)
  for i in range(npar):
    finalPar[i] = minimizer.X()[i] # retrieve parameters and errors
    finalParErr[i] = minimizer.Errors()[i]
  
  buf = arr('d', npar*npar*[0.])
  
  minimizer.GetCovMatrix( buf ) # retrieve error matrix
  emat=np . array ( buf ) . reshape ( npar , npar )
  emat_norm = np.zeros(emat.shape,dtype=np.float32)

  icstat = minimizer.Status()
  amin = minimizer.MinValue()

  print("\n")
  print("*==* MINUIT fit completed:")
  print(' fcn@minimum = %.2f' %(amin), "status =",icstat)
  print(finalPar)
  print(finalParErr)
  print(emat)
  print(" Results: \t value error corr. mat.")
  for i in range(npar):
    #print(' %s: \t%10.3e +/- %.1e '%(name[i] ,finalPar[i] ,finalParErr[i]))
    emat_norm[i,i ] = 1.
    for j in range (0,i):
      emat_norm[i,j] = emat[i][j]/np.sqrt(emat[i][i])/np.sqrt(emat[j][j])
      emat_norm[j,i] = emat_norm[i,j]
      print('i%d,j%d: xi %.4f, yj %.4f, cov3 %.4f'%(i,j,np.sqrt(emat[i][i]),np.sqrt(emat[j][j]),emat_norm[i,j]))
  print(emat_norm) 
  outFileName = np.str(config['python3.6']['files']['fittingNpz_tminuit'])
  np.savez(outFileName,error_matrix = emat,error_matrix_norm = emat_norm, mean = finalPar,)


if __name__ == "__main__":
  #file = open('./parameters/p4_run1_tritium_5kV.json')
  jsonFileName = sys.argv[1]
  jFile = np.str(jsonFileName)
  global debug
  debug=False
  main(jFile)
  #theta0 = np.asarray([0.0923185, 3.59445, 6.696541, 0.22419741 , 0.22511, -0.25262 , 1.13235351 , 29.046818],dtype=np.float32)
  #theta0 = np.asarray([0.1, 3.6, 5.65, 0.22419741 , 0.22511, -0.25262 , 1.13235351 , 29.046818],dtype=np.float32)
  theta0 = np.asarray([0.1195, 4.156, 6, 0.1 , 0.1, 0.1, 0.7 , 40],dtype=np.float32)
  if (debug):
    test_get_likehood(theta0)
  else :
    fitting_with_tminuit2(theta0)
    #fitting_with_emcee(theta0)
