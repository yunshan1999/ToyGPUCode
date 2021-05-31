###################################
## DEMO version for GPU fitting
###################################
## Created @ Apr. 12th, 2021
## By Yunshan Cheng and Dan Zhang
## contact: yunshancheng1@gmail.com
###################################
import numpy as np
import time
import math
import uproot
import emcee
#import corner
import json

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

debug=False

start = drv.Event()
end = drv.Event()
GPUFunc     = SourceModule(pandax4t_reweight_fitting, no_extern_c=True).get_function("main_reweight_fitting")

GPUFuncRenormalization = SourceModule(pandax4t_reweight_fitting, no_extern_c=True).get_function("renormalization_reweight")

GPUlnLikelihood = SourceModule(pandax4t_reweight_fitting, no_extern_c=True).get_function("ln_likelihood")

file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)

#parameters of interest
pde = np.float32(config['pde'])
g2b = np.float32(config['g2b'])
segb = np.float32(config['segb'])
nuissancePars = np.asarray([pde,g2b,segb],dtype = np.float32)

pde_g2b_theta = np.float32(config['python3.6']['nuissanceParRelativeUncertainty']['pde_g2b_theta'])
nuissanceParsReUnc = np.asarray([config['python3.6']['nuissanceParRelativeUncertainty']['pde'],
  config['python3.6']['nuissanceParRelativeUncertainty']['g2b'],
  config['python3.6']['nuissanceParRelativeUncertainty']['segb'] ] ,dtype = np.float32)

def get_ln_nuissance(nuissanceTheta):
  ln = np.double(0.);
  pde_i,g2b_i,segb_i = nuissanceTheta
  if (g2b_i > segb_i ):
    return -np.inf
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
  #if (ln < -100. * nuissanceTheta.shape[0]):
  #  return -np.inf
  return ln

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

def get_fitting_ln_prior(fittingTheta):
  ln = np.double(0.);
  for i in range(rowNo):
    if (fittingTheta[i] < fittingParSpace[i][0] or fittingTheta[i] > fittingParSpace[i][1]):
      ln = -np.inf
  return ln
#print("test lnproir ", get_ln_nuissance([0.094,0.77,19,0.25]))

fittingPars = np.asarray([0.05,0.05,0.05,1.5,20.],dtype = np.float32)#p0Recomb,p1Recomb,p0FlucRecombStretch

totalPars = np.append(nuissancePars,fittingPars,axis=0)
print('totalParameters ',totalPars)
print('nuissancePars', nuissancePars)
print('fittingPars',fittingPars)


#energy spectrum array initialize
minE = np.float32(config['E_eeTh'])
#minE = np.float32(0.)
maxE = np.float32(config['E_eeMaxSimu']) + 0.1
stepE = 0.5;
v_Erange = np.arange(minE,maxE,stepE)


#simulation data loading
oriTreeFile  = config['python3.6']['files']['oriSimuNpz']
oriTree = np.load(oriTreeFile)
simu_tree = oriTree['oriTree']
simu_tree_gpu = gpuarray.to_gpu(simu_tree)

weight_gpu = drv.mem_alloc(simu_tree[:,0].nbytes)
normalization_gpu = drv.mem_alloc(simu_tree[:,0].nbytes)

simu_tree_par = np.asarray([simu_tree.shape[0],simu_tree.strides[0]],dtype=np.float32)
simu_tree_bytes = simu_tree.size * simu_tree.dtype.itemsize
simu_tree_par     = np.asarray(np.append(simu_tree_par,[stepE,minE,maxE]),dtype=np.float32)
print('input simu_tree_par shape : ',simu_tree_par)

#boundary conditions
s1min = np.float32(config['s1min'])
s1u_c = np.float32(config['s1u_c'])
s2min = np.float32(config['s2min'])
s2minRaw = np.float32(config['s2minRaw'])
s2max = np.float32(config['s2max'])
s2u_c = np.float32(config['s2u_c'])
nHitsS1min = np.float32(config['nHitsS1min'])


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

cS1,cS2 = get_2dData()
Ndata = np.uint([cS1.shape[0]])
print("survived data: ",Ndata)
cS1_gpu = gpuarray.to_gpu(cS1)
cS2_gpu = gpuarray.to_gpu(cS2)



#for main_reweight_fitting
gpu_seed            = int(time.time()*100)
num_trials          = simu_tree.shape[0]
#num_trials          = 1000
GPUSeed             = np.asarray(gpu_seed, dtype=np.int32)
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
binning = np.asarray([xbin,xmin,xmax,ybin,ymin,ymax],dtype=np.double)

histogram2d_0 = np.zeros((xbin,ybin),dtype = np.double)
histogram2d_0_gpu = drv.mem_alloc(histogram2d_0.nbytes)

#for ln_likelihood
d_gpu_scale1 = {}
d_gpu_scale1['block'] = (nThreads, 1, 1)
d_gpu_scale1['grid'] = ( int(np.floor(float(Ndata)/float(nThreads) ) ),1)


def get_lnLikelihood(theta):

    #########################################
    ## Initialization
    ########################################
    histogram2d_par     = np.asarray(np.append([0.0],binning),dtype=np.double)
    v_E = v_Erange[:-1] + 0.5 * stepE
    
    v_E_dRdE = np.zeros(v_E.shape,dtype=np.float32)
    if (debug):
      print('v_E shape: ',v_E_dRdE.shape[0])
    histogram2d_par     = np.append( histogram2d_par, v_E_dRdE)

    #print("dtype check ",histogram2d_par.dtype)
    #histogram2d_par = np.append(histogram2d_par, binning)
    histogram2d = np.zeros((xbin,ybin),dtype = np.double)

    par_array = np.asarray(theta, dtype=np.float32)
    out_weight = np.zeros(simu_tree.shape[0],dtype = np.float32)
    out_normalization = np.zeros(simu_tree.shape[0],dtype = np.float32)
    drv.memcpy_htod(histogram2d_0_gpu,histogram2d)
    drv.memcpy_htod(weight_gpu,out_weight)
    drv.memcpy_htod(normalization_gpu,out_normalization)

    # important step, need to push (or copy) it to GPU memory so that GPU function can use it
    # this step take time so shall minimize the number of times calling it
    start.record()
    #simu_tree_gpu = drv.mem_alloc(simu_tree_bytes)
    #drv.memcpy_htod(simu_tree_gpu,simu_tree)
    mode = np.uint(1)
    tArgs               = [
        drv.In(par_array),
        simu_tree_gpu, drv.In(simu_tree_par),
        drv.InOut(histogram2d_par),
        weight_gpu,normalization_gpu,
        mode
    ]

    # Run the GPU code
    GPUFunc(*tArgs, **d_gpu_scale)
    #end.record()
    drv.Context.synchronize()

    effectiveNo = histogram2d_par[0] 
    if(debug):
      print("effective integral ", effectiveNo)
    if(effectiveNo<5000.):
      return -10000.
    elif (math.isnan(effectiveNo)):
      return -np.inf
    v_E_dRdE = histogram2d_par[7:]
    v_E_dRdE = np.asarray(np.append([v_E_dRdE.shape[0]],v_E_dRdE), dtype = np.float32)
    v_E = np.asarray(np.append([v_E.shape[0]],v_E),dtype = np.float32)
    histogram2d_par = histogram2d_par[:7]
    histogram2d_par[0] = 0.

    tArgsR = [
      drv.In(v_E),drv.In(v_E_dRdE),
      simu_tree_gpu, drv.In(simu_tree_par),
      weight_gpu,normalization_gpu,
      histogram2d_0_gpu, drv.InOut(histogram2d_par),
    ]
    GPUFuncRenormalization(*tArgsR , **d_gpu_scale)
    if(debug):
      print(effectiveNo)
      print("check 2 : ",histogram2d_par[0] )
      #np.save('outputs/hist2dEx.npy',histogram2d)

    drv.Context.synchronize()
    #return histogram2d
    #if histogram2d_par[0] == 0. : 
    #  return -np.inf
    likeli_par = histogram2d_par[:7]
    lnLikelihood_total = np.array([0.0],dtype=np.double)
    #global cS1, cS2
    tArgs1 = [cS1_gpu, cS2_gpu, histogram2d_0_gpu, drv.In(likeli_par),
             drv.InOut(lnLikelihood_total),Ndata
            ]
    GPUlnLikelihood(*tArgs1,**d_gpu_scale1)
    drv.Context.synchronize()
    #print("Nmc ",histogram2d_par[0])
    if math.isnan(lnLikelihood_total[0]):
      return -np.inf
    return lnLikelihood_total[0]


def get_lnprob(theta):
    NoNui=nuissancePars.shape[0]
    nuissancePar_i = theta[:NoNui]
    lp = get_ln_nuissance(nuissancePar_i)
    if not np.isfinite(lp):
      return -np.inf
    fittingTheta = theta[:]
    lp1 = get_fitting_ln_prior(fittingTheta)
    if not np.isfinite(lp1):
      return -np.inf
    lnprob =  get_lnLikelihood(theta)
    return lp + lnprob

def test_get_likehood(theta):
    likelihood = get_lnprob(theta)
    #likelihood = get_lnLikelihood(theta)
    #print("sum2: ",histogram2d_par[0])
    x0 = np.linspace(xmin,xmax,num=np.int(xbin))
    y0 = np.linspace(ymin,ymax,num=np.int(ybin))
    xv, yv = np.meshgrid(x0,y0)
    #c =plt.pcolor(xv,yv,histogram2d,cmap='magma',vmin=0, vmax = histogram2d.max())
    #plt.axis([xmin,xmax,ymin,ymax])
    #plt.colorbar(c)
    print("likelihood_output: ",likelihood)
    #plt.show()
    #plt.savefig('testh2fitting.png')


def fitting_with_emcee():
    #g1_starter, EEE_starter, SEG_starter = 0.090954, 0.6, 19.
    ndim = totalPars.shape[0]
    p0_stater= np.asarray([9.69480824e-02,  4.04646044e+00,  6.07677144e+00,  0.1,0.1, -0.1,0.8,2.09005314e+01],dtype = np.float32)
    nwalkers = 100
    #becareful
    p0 = np.asarray([p0_stater * (1 + 0.2*np.random.randn(ndim)) for i in range(nwalkers)])
    #print(p0)
    #p0 = np.random.randn(nwalkers, ndim)
    #p0 = np.asarray(np.asarray([0.094,0.76,19,0.25,0,0,2,100]) * (1 + 0.1*np.random.randn(ndim)) for i in range(nwalkers)])
    #print("ndim ",ndim,totalPars,p0.shape)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, get_lnprob, args=())
    sampler.run_mcmc(p0,1000,progress=True)
    #fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    flat_samples = sampler.get_chain(discard=50,thin=20,flat =True)
    samples = sampler.get_chain()
    lnls = sampler.lnprobability
    acceptance = sampler.acceptance_fraction
    np.savez("./outputs/fitting_result_1.npy",lnls=lnls,acceptance = acceptance,samples = samples)

#    labelsG = ["g1", "g2b", "segb","seb_frac","p0Recomb","p1Recomb","p0FlucRecomb","flatE"]
#    for i in range(ndim):
#        ax = axes[i]
#        ax.plot(samples[:, :, i], "k", alpha=0.3)
#        ax.set_xlim(0, len(samples))
#        ax.set_ylabel(labelsG[i])
#        ax.yaxis.set_label_coords(-0.1, 0.5)
#    
#    axes[-1].set_xlabel("step number");
#    
#    plt.savefig('test_fitting.png')
    
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[-100, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(mcmc[1],q[0],q[1])
   # fig1 = corner.corner(flat_samples,labels=labelsG);
   # plt.savefig('test_fitting_cor.png')

fitting_with_emcee()
#test_get_likehood(np.asarray([9.69480824e-02,  4.04646044e+00,  6.07677144e+00,  0.43, 0.28, -0.28,0.75,2.09005314e+01],dtype=np.float32))

