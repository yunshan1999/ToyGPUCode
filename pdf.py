import corner
import numpy as np

PATH = '../FittingGPU_9p/outputs/'

ndim = 9
nconv = 2500
samples = np.loadtxt(PATH+'samples.dat')
samples_orig = samples.reshape(samples.shape[0], samples.shape[1]//ndim, ndim)

##add acceptance ratio cut

acceptances = np.loadtxt(PATH+'acceptances.dat')
lnls = np.loadtxt(PATH+'lnls.dat')
#inds = np.where((acceptances>0.01))[0]
inds = np.where((-7476.757684<=lnls[:,-1])&(-2000>=lnls[:,-1]))[0]
samples_plot = samples_orig[inds,nconv:,:].reshape((-1,ndim))

fig = corner.corner(samples_plot, labels=["$g1$", "$EEE$", "$SEG$","$b frac$","flatER", "$scale$ $of$ $deltaR$", "$p0$","$p1$", "$p2$"],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 10})

fig.savefig(PATH+'triangle.pdf')

parsBestFit = np.percentile(samples_plot, [16, 50, 84], axis = 0)
np.savetxt(PATH+"bestfit.dat", parsBestFit)
