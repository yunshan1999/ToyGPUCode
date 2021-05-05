import corner
import numpy as np

ndim = 7
nconv = 2500
samples = np.loadtxt("../FittingGPU/outputs/samples.dat")
samples_orig = samples.reshape(samples.shape[0], samples.shape[1]//ndim, ndim)

##add acceptance ratio cut

acceptances = np.loadtxt("../FittingGPU/outputs/acceptances.dat")
lnls = np.loadtxt("../FittingGPU/outputs/lnls.dat")
#inds = np.where((acceptances>0.01))[0]
inds = np.where((-7544.12861184<=lnls[:,-1])&(lnls[:,-1]<=-6709.57301148))[0]
samples_plot = samples_orig[inds,nconv:,:].reshape((-1,ndim))

fig = corner.corner(samples_plot, labels=["$g1$", "$EEE$", "$SEG$","$b frac$", "$scale$ $of$ $deltaR$", "$p0$","$p1$"],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 10})

fig.savefig("../FittingGPU/outputs/triangle.pdf")

parsBestFit = np.percentile(samples_plot, [16, 50, 84], axis = 0)
np.savetxt("../FittingGPU/outputs/bestfit.dat", parsBestFit)
