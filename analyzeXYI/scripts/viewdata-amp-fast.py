# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 17:07:23 2022

@author: darro
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
import init

# The default data class
# Contains data from the XY stage and the electrometer
class load_data:
    def __init__(self, loc, fname):
        self.XY = np.genfromtxt(os.path.join(loc,fname + 'XY.csv'), delimiter = ',', comments='#', names = True, skip_header=1)
        self.IV = np.genfromtxt(os.path.join(loc,fname + 'IV.csv'), delimiter = ',', comments='#', names = True, skip_header=2)
        self.Iproc = {}

# Main loop for aligning and analysing the data
# Do all of the analysis tasks necessary for finding the sensor
def main(i,dt):
    timecorr(i,dt)
    parsescan(i)
    rastersnake(i)
    cluster(i)   
        
# Correct for the offset between electrometer time and unix time  
# The parameter dt can be used as a fitting paramter
# The correct value of dt will 'focus' the plots from lightmap and histplot
def timecorr(i,dt):
    #dt = -85
    data.IV['TsCH_%i_s'%i] += dt
  
# Reorganize data into ordered structures
# This function makes assumptions about the structure of the input data
def parsescan(i):
    data.coordinates = []
    for j in range(int(len(data.XY)/2)):
        data.coordinates.append([data.XY['xpos'][2*j],data.XY['ypos'][2*j],data.XY['time'][2*j],data.XY['time'][2*j+1]])
    #data.coordinates.append([data.XY['xpos'][-1],data.XY['ypos'][-1],data.XY['time'][-1],data.XY['time'][-1]+param[6]])
    #print(data.coordinates)
    data.XYIr = []
    data.XYI = []
    data.X, data.Y, data.I = [],[],[]
    f, g = open(os.path.join(PROC,bname+'XYI_rich.csv'),'w+'), open(os.path.join(PROC,bname+'XYI.csv'),'w+')
    f.write('X,Y,I');g.write('X,Y,I,Ierr')
    data.index = []
    for row in data.coordinates:
        I = []
        for k in range(len(data.IV['TsCH_%i_s'%i])):
            if row[2] < data.IV['TsCH_%i_s'%i][k] < row[3]:
                I.append(data.IV['IsCH_%i_A'%i][k])
                data.XYIr.append([row[0],row[1],data.IV['IsCH_%i_A'%i][k]])
                f.write('\n%s,%s,%s'%(row[0],row[1],I)) 
        Iavg = np.average(I)
        #if Iavg != Iavg: # Check for missing data
        #    print(data.IV['CH1_Current'][l:m])
        #    print(l,m)
        #    print(row[2],row[3])
        Ierr = np.std(I)
        data.XYI.append([row[0],row[1],Iavg])
        data.X.append(row[0])
        data.Y.append(row[1])
        data.I.append(Iavg)
        g.write('\n%s,%s,%s,%s'%(row[0],row[1],Iavg,Ierr))
    data.X = np.asarray(data.X)
    data.Y = np.asarray(data.Y)
    data.I = np.asarray(data.I)
    f.close();g.close()
    data.Xr, data.Yr, data.Ir = [],[],[]
    for row in data.XYIr:
        data.Xr.append(row[0])
        data.Yr.append(row[1])
        data.Ir.append(row[2])

# Convert the shape of the data from bidirectional to unidirectional        
def rastersnake(i):
    X = np.unique(data.X)
    Y = np.unique(data.Y)
    data.Xproc, data.Yproc, data.Iproc['%s'%i] = [],[],[]
    for x in X:
        for y in Y:
            for j in range(len(data.XYI)):
                if x == data.XYI[j][0]:
                    if y == data.XYI[j][1]:
                        data.Xproc.append(data.XYI[j][0])
                        data.Yproc.append(data.XYI[j][1])
                        data.Iproc['%s'%i].append(data.XYI[j][2])
    data.Xproc = np.asarray(data.Xproc)
    data.Yproc = np.asarray(data.Yproc)
    data.Iproc['%s'%i] = np.asarray(data.Iproc['%s'%i])
    data.Iproc['%s'%i] = np.nan_to_num(data.Iproc['%s'%i],nan=min(data.Iproc['%s'%i])) # Replace missing data with 0

# Clump the data and set thresholds
def cluster(i):           
    # Determine which K-Means cluster each point belongs to
    cluster_id = KMeans(10).fit_predict(data.Iproc['%s'%i].reshape(-1, 1))
    # Determine densities by cluster assignment and plot
    figure, axis = plt.subplots(dpi=200)
    bins = np.linspace(data.Iproc['%s'%i].min(), data.Iproc['%s'%i].max(), 40)
    data.Kmean, data.Kstd = [],[]
    axis.set_xlabel('Current [A]')
    axis.set_ylabel('Counts')
    # Average the clusters
    for ii in np.unique(cluster_id):
        subset = data.Iproc['%s'%i][cluster_id==ii]
        axis.hist(subset, bins=bins, alpha=0.5, label=f"Cluster {ii}")
        data.Kmean.append(np.mean(subset))
        data.Kstd.append(np.std(subset))
    # Find thresholds relative to the clusters
    data.Imax = data.Kmean[np.where(data.Kmean == max(data.Kmean))[0][0]]
    data.Imin = data.Kmean[np.where(data.Kmean == min(data.Kmean))[0][0]]
    data.Ihalf = (data.Imax - data.Imin)/2+data.Imin
    data.Ie2 = (data.Imax - data.Imin)/np.exp(2)+data.Imin
    #data.Imaxx = np.max(data.Iproc['%s'%i])
    data.Icluster = [data.Ihalf]
    # Plot the thresholds
    for th in data.Icluster:
        axis.axvline(th)
    axis.legend()
    
# Find the centroid of the X and Y data above some photosensor current threshold
def findsensor(thresh,i):
    data.Xtemp, data.Ytemp, data.Itemp = [],[],[]
    for j in range(len(data.Iproc['%s'%i])):
        if data.Iproc['%s'%i][j] >= thresh:
            data.Xtemp.append(data.Xproc[j])
            data.Ytemp.append(data.Yproc[j])
            data.Itemp.append(data.Iproc['%s'%i][j]) 
    data.Isum = np.sum(data.Itemp)
    data.Xmean = np.dot(data.Xtemp,data.Itemp)/data.Isum
    data.Ymean = np.dot(data.Ytemp,data.Itemp)/data.Isum
    return data.Xmean, data.Ymean

# Plots go here
# Plot 2D histograms of position and signal    
def histplot(i):
    figure, axis = plt.subplots(2,1,dpi=200,figsize=(12,12))  
    axis[0].hist2d(data.Xr,np.abs(data.Ir),bins=30,norm=LogNorm()) 
    axis[1].hist2d(data.Yr,np.abs(data.Ir),bins=30,norm=LogNorm()) 
    for th in data.Icluster:
        for j in range(len(findsensor(th,i))):
            print(findsensor(th,i))
            axis[j].axvline(findsensor(th,i)[j])
            axis[j].axhline(th)
    for ax in axis:
        ax.set_ylabel('PD Current\n'+r'I = 10$^y$ [A]')
    axis[0].set_xlabel('X position [mm]')
    axis[1].set_xlabel('Y position [mm]')
    #figure.savefig(os.path.join(PLOTS,'%s_histplot.svg'))

# Plot a 2D reconstruction of the sensor lightmap
def lightmap(i):
    x=np.unique(data.Xproc)
    y=np.unique(data.Yproc)
    X,Y = np.meshgrid(y,x)
    I=data.Iproc['%s'%i].reshape(len(x),len(y))
    figure, axis = plt.subplots(1,1,dpi=200,figsize=(5,4))
    im = axis.pcolormesh(Y,X,I)#,norm=LogNorm(vmin=I.min(), vmax=I.max()))
    axis.set_xlabel('X position [mm]')
    axis.set_ylabel('Y position [mm]')
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    figure.colorbar(im,cax=cax,orientation='vertical',label='Current [A]')
    axis.axis('square')
    for th in data.Icluster:
        axis.scatter(findsensor(th,i)[0],findsensor(th,i)[1])
        axis.contour(Y,X,I,[th],colors='k')
    #figure.savefig(os.path.join(PLOTS,'%s_lightmap.svg'))

# Plot a 2D reconstruction of the sensor lightmap
def tilemap():
    x=np.unique(data.Xproc)
    y=np.unique(data.Yproc)
    X,Y = np.meshgrid(y,x)
    I=(data.Iproc['sum'].reshape(len(x),len(y)))/max(data.Iproc['sum'])
    figure, axis = plt.subplots(1,1,dpi=200,figsize=(5,4))
    im = axis.pcolormesh(Y,X,I)#,norm=LogNorm(vmin=I.min(), vmax=I.max())) 
    axis.set_xlabel('X position [mm]')
    axis.set_ylabel('Y position [mm]')
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    figure.colorbar(im,cax=cax,orientation='vertical',label='Current [a.u]')
    axis.axis('square')
    for th in data.Icluster:
        for i in a:
            axis.scatter(findsensor(th,i)[0],findsensor(th,i)[1])
            I=data.Iproc['%s'%i].reshape(len(x),len(y))
            axis.contour(Y,X,I,[th],colors='k')
    #figure.savefig(os.path.join(PLOTS,'%s_lightmap.svg'))
    
# Plot the position and signal data as a function of time    
def plot(i):
    figure, axis = plt.subplots(3,1,dpi=200,figsize=(12,12))
    axis[0].plot(data.IV['TsCH_%i_s'%i],data.IV['IsCH_%i_A'%i]*1E6,label='Channel %i'%i)
    axis[1].plot(data.XY['time'],data.XY['xpos'])
    axis[2].plot(data.XY['time'],data.XY['ypos'])
    axis[0].set_yscale('log')
    for ax in axis:
        ax.set_xlim(data.XY['time'][0],data.XY['time'][-1])
    axis[0].set_ylabel('PD Current [uA]')
    axis[1].set_ylabel('X-position [mm]')
    axis[2].set_ylabel('Y-position [mm]')
    axis[2].set_xlabel('Time [s]')
    axis[0].legend()
    #figure.savefig(os.path.join(PLOTS,'%s_XYI.svg'))

# Debugging plots go here
# Plot the position and signal data, highlight the averaged data
def stepplot():
    figure, axis = plt.subplots(3,1,dpi=200,figsize=(12,12))
    axis[0].plot(data.IV['TsCH_2_s'],data.IV['IsCH_2_A']*1E6)
    axis[1].plot(data.XY['time'],data.XY['xpos'])
    axis[2].plot(data.XY['time'],data.XY['ypos'])
    axis[0].set_yscale('log')
    for index in data.coordinates[0:100]:
        axis[0].axvspan(index[2],index[3],color='g',alpha=0.2)
        axis[1].axvspan(index[2],index[3],color='g',alpha=0.2)
        axis[2].axvspan(index[2],index[3],color='g',alpha=0.2)
    for ax in axis:
        ax.set_xlim(data.XY['time'][0],data.XY['time'][100])
        #ax.set_xlim(100,200)
    axis[0].set_ylabel('PD Current [pA]')
    axis[1].set_ylabel('X-position [mm]')
    axis[2].set_ylabel('Y-position [mm]')
    figure.savefig(os.path.join(PLOTS,'%s_stepplot.svg'))

# Plot the time each measurement is made, useful for finding range issues with instrument
def checkgap(tvals):
    figure, axis = plt.subplots(1,1,dpi=200)
    for tval in tvals:
        data.plot(tval,data.IV['TsCH_2_s'][tval],'ro')
    data.set_xlabel('Measurement')
    data.set_ylabel('Time [s]')    
    
def plots(i):
    plot(i)
    #stepplot()
    histplot(i)
    lightmap(i)
    
if __name__ == "__main__":
    SCRIPTS,HOME,DATA,ARCHIVE,TEMP,DEV,PROC,PLOTS,REPORTS = init.envr() # Setup the local environment
    bname = os.listdir(DEV)[0][:-6] # Find the basename for the data files
    data = load_data(DEV,bname) # Create the data class
    a = [6,7,8,9,10,11,12,14,15,16]
    b = [];c=[]
    data.Iproc['sum'] = []
    for i in a:
        main(i,-84.3) # Align the data and do analysis
        plots(i) # What plots to draw
        if i == a[0]:
            data.Iproc['sum'] = data.Iproc['%s'%i]**3
        else:
            data.Iproc['sum'] += data.Iproc['%s'%i]**3
        b.append(i)
        c.append([i,findsensor(data.Icluster[0],i)[0],findsensor(data.Icluster[0],i)[1]])
    tilemap()
    print(b)
    print(c)
    
# -85 works for 1,5,6,7,'8',9,10
# -84 works for 2,3
# -83 works for 4


