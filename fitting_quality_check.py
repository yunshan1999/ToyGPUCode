import numpy as np
import uproot
import json
import matplotlib.pyplot as plt
import corner
from normal_corner import normal_corner
#from KDEpy import FFTKDE
import scipy.stats as st
import datetime
from matplotlib.backends.backend_pdf import PdfPages

file = open('./parameters/p4_run1_tritium_5kV.json')
config = json.load(file)
inFileName = np.str(config['python3.6']['files']['fittingNpz'])
fitting_data = np.load(inFileName)

samples = fitting_data['samples']
flat_samples = samples[-200:,:,:]
flat_samples = flat_samples.reshape(flat_samples.shape[0] * flat_samples.shape[1],flat_samples.shape[2])
ndim = flat_samples.shape[1]
print('check ndim ',ndim)

inFileName_t = np.str(config['python3.6']['files']['fittingNpz_tminuit'])
data_tminuit = np.load(inFileName_t)
error_matrix_norm = data_tminuit['error_matrix_norm']
error_matrix = data_tminuit['error_matrix']
mean_t = np.asarray(data_tminuit['mean'])
error_matrix_norm = np.asarray(error_matrix_norm.reshape(mean_t.shape[0],mean_t.shape[0]))
error_matrix = np.asarray(error_matrix.reshape(mean_t.shape[0],mean_t.shape[0]))
print('emat:',error_matrix)
print('mean_t:',mean_t)
grid_points = 150


v_min=np.zeros(ndim,dtype = np.float32)
v_max=np.zeros(ndim,dtype = np.float32)
zRange=5.;
for i in range(ndim):
  mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
  q = np.diff(mcmc)
  #v_min[i]=max(mcmc[1]-zRange*q[0],min(flat_samples[:, i]))
  #v_max[i]=min(mcmc[1]+zRange*q[1],max(flat_samples[:, i]))
  v_min[i]=mean_t[i] - zRange*np.sqrt(error_matrix[i][i])
  v_max[i]=mean_t[i] + zRange*np.sqrt(error_matrix[i][i])
  print(mcmc[1],q[0],q[1])

#exit()
#quality check
nGrid = np.uint(100)
mydict={}
labelsG=["$g1$", "$g2b$", "$seg_b$","$p2Recomb$","$p0Recomb$","$p1Recomb$","$p0FlucRecomb$","$flatE$"]
outPutFile=np.str(config['python3.6']['files']['kdeParNpz'])

#z_total_cor = np.zeros((ndim,z_frame.shape[0],z_frame.shape[1]),dtype=np.float32)

#print('total shape',total_cor.shape)
#exit()
v_rv = []


with PdfPages('fitting_quality.pdf') as pdf:

    fig0,ax0 = plt.subplots(ndim-1,ndim-1,figsize = (30,30))
    for i in range(ndim):
      x = np.asarray(flat_samples[:, i])
      xmin,xmax = v_min[i],v_max[i]
      for j in range(ndim-1,i,-1):
        y = np.asarray(flat_samples[:, j])
        ymin,ymax = v_min[j],v_max[j]
        xx,yy = np.meshgrid(np.linspace(xmin,xmax,nGrid),np.linspace(ymin,ymax,nGrid))
        name='%d_%d'%(i,j)
        print(name)
        #xx,yy = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
        pos = np.vstack([xx.ravel(),yy.ravel()])
        values = np.vstack([x,y])
        #print(test1,test2)
        #test_data = np.append(test1,test2).reshape(2,test1.shape[0]).T
        #test_data = np.vstack([test1,test2]).T
        #print(test_data)
        #grid, points = kde.fit(test_data).evaluate(grid_points)
        #x, y = np.unique(grid[:,0]),np.unique(grid[:,1])
        #z = points.reshape(grid_points,grid_points).T
        #plt.contour(x,y,z,10,linewidths=0.8,colors='k')
        kernel = st.gaussian_kde(values)

        z = np.reshape(kernel(pos).T,xx.shape)
        mydict_temp={'x':xx,'y':yy,'z':z}
        #mydict.append(name)
        mydict[name]=mydict_temp
        #print(z)
        ax0[j-1,i].contourf(xx,yy,z,cmap="RdBu_r")
        ax0[j-1,i].set_xlabel(labelsG[i])
        ax0[j-1,i].set_ylabel(labelsG[j])
    
    np.savez(outPutFile,mydict=mydict)
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize = (35,30))
    fig0,ax0 = plt.subplots(ndim-1,ndim-1,figsize = (30,30))
    index = 0
    for i in range(ndim):
      #sigma_i = np.sqrt(error_matrix[i,i])
      #mean_i = mean_t[i]
      v_x = np.linspace(v_min[i], v_max[i], grid_points, endpoint=False)
      for j in range(ndim-1,i,-1):
        v_y = np.linspace(v_min[j], v_max[j], grid_points, endpoint=False)
        xx,yy = np.meshgrid(v_x,v_y)
        pos = np.dstack((xx,yy))
        cov_i_j= np.asarray([[error_matrix[i][i],error_matrix[i][j]],[error_matrix[i][j],error_matrix[j][j]]])
        rv = st.multivariate_normal([mean_t[i],mean_t[j]],cov_i_j)
        z = rv.pdf(pos)
        ax0[j-1,i].contourf(xx,yy,z,cmap="RdBu_r")
        ax0[j-1,i].set_xlabel(labelsG[i])
        ax0[j-1,i].set_ylabel(labelsG[j])
        v_rv.append(rv)
        mydict[tuple([i,j])]=index
        index = index + 1
    print(mydict)
    #print('emat shape:',error_matrix.shape)
    #print('mean shape:',mean_t.shape)
    #figCorner_t = normal_corner.normal_corner(error_matrix,mean_t,labelsG)
    pdf.savefig()
    plt.close()

    plt.figure(figsize = (35,30))
    figCorner = corner.corner(flat_samples,labels=labelsG,quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 10});
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize = (6,5))
    lnls = fitting_data['lnls']
    plt.plot(lnls.T, linewidth = 0.1, color = "black")
    plt.title('likelihood')
    pdf.savefig()
    plt.close()

    plt.figure(figsize = (6,5))
    acceptance = fitting_data['acceptance']
    plt.hist(acceptance, bins = 40,color = "black")
    plt.title('acceptance')
    pdf.savefig()
    plt.close()
 

    #plt.figure(figsize = (20,20))
    fig, axes = plt.subplots(ndim, figsize=(15, 15), sharex=True)
    for i in range(ndim):
      ax = axes[i]
      ax.plot(samples[:, :, i], "k", alpha=0.3)
      ax.set_xlim(0, len(samples))
      ax.set_ylabel(labelsG[i])
      ax.yaxis.set_label_coords(-0.1, 0.5)
      axes[-1].set_xlabel("step number");
    pdf.savefig()
    plt.close()



    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'
    d['Author'] = 'dan'
    d['Subject'] = 'fitting check'
    d['Keywords'] = 'PdfPages multipage keywords author title subject'
    d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['ModDate'] = datetime.datetime.today()


