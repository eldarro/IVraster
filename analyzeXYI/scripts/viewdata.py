# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 17:07:23 2022

@author: darro
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
import init

class load_data:
    def __init__(self, loc, fname):
        self.XY = np.genfromtxt(os.path.join(loc,fname + 'XY.csv'), delimiter = ',', comments='#', names = True, skip_header=1)
        self.IV = np.genfromtxt(os.path.join(loc,fname + 'IV.csv'), delimiter = ',', comments='#', names = True, skip_header=0)

def plotlk(xylk,ivlk,param):
    figure, axis = plt.subplots(3,1,dpi=200,figsize=(12,12))
    axis[0].plot(data.IV['CH1_Time'],data.IV['CH1_Current']*1E12)
    axis[1].plot(data.XY['time']-data.XY['time'][0],data.XY['xpos'])
    axis[2].plot(data.XY['time']-data.XY['time'][0],data.XY['ypos'])
    
    axis[0].set_yscale('log')
    for ax in axis:
        ax.set_xlim(param[0],param[1])
    axis[0].set_ylabel('PD Current [pA]')
    axis[1].set_ylabel('X-position [mm]')
    axis[2].set_ylabel('Y-position [mm]')
    
    axis[0].axvline(ivlk,c='k',ls='--')
    axis[1].axvline(xylk,c='k',ls='--')
    axis[2].axvline(xylk,c='k',ls='--')
    
    axis[0].axhline(15,c='r')
    axis[0].axvline(param[0],c='r')
    axis[0].axvline(param[1],c='r')
    axis[0].add_patch(Rectangle((param[0], -5), param[1]-param[0], 20 ,facecolor ='red',alpha=0.5))
    
def histplot():
    figure, axis = plt.subplots(2,1,dpi=200,figsize=(12,12))  
    axis[0].hist2d(data.Xr,np.abs(data.Ir),bins=30,norm=LogNorm()) 
    axis[1].hist2d(data.Yr,np.abs(data.Ir),bins=30,norm=LogNorm()) 
    axis[0].axvline(data.Xmean)
    axis[1].axvline(data.Ymean)
    axis[0].axhline(data.Ihalf)
    axis[1].axhline(data.Ihalf)
    axis[0].axhline(data.Imaxx*1E-1)
    axis[1].axhline(data.Imaxx*1E-1)
    for ax in axis:
        ax.set_ylabel('PD Current\n'+r'I = 10$^y$ [A]')
    axis[0].set_xlabel('X position [mm]')
    axis[1].set_xlabel('Y position [mm]')
    
def lightmap():
    x=np.unique(data.Xproc)
    y=np.unique(data.Yproc)
    X,Y = np.meshgrid(x,y)
    I=data.Iproc.reshape(len(x),len(y))
    figure, axis = plt.subplots(1,1,dpi=200,figsize=(6,6))
    im = axis.pcolormesh(Y,X,I) 
    axis.scatter(data.Xmean,data.Ymean)
    axis.set_xlabel('X position [mm]')
    axis.set_ylabel('Y position [mm]')
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    figure.colorbar(im,cax=cax,orientation='vertical',label='Current [A]')
    axis.contour(np.transpose(I),[data.Imaxx*1E-1],colors='k')
    axis.contour(np.transpose(I),[data.Ihalf],colors='k')
    
def plot():
    figure, axis = plt.subplots(3,1,dpi=200,figsize=(12,12))
    axis[0].plot(data.IV['CH1_Time'],data.IV['CH1_Current']*1E12)
    axis[1].plot(data.XY['time']-data.XY['time'][0],data.XY['xpos'])
    axis[2].plot(data.XY['time']-data.XY['time'][0],data.XY['ypos'])
    axis[0].set_yscale('log')
    for ax in axis:
        ax.set_xlim(data.IV['CH1_Time'][0],data.IV['CH1_Time'][-1])
    axis[0].set_ylabel('PD Current [pA]')
    axis[1].set_ylabel('X-position [mm]')
    axis[2].set_ylabel('Y-position [mm]')
    axis[2].set_xlabel('Time [s]')
    
def stepplot():
    figure, axis = plt.subplots(3,1,dpi=200,figsize=(12,12))
    axis[0].plot(data.IV['CH1_Time'],data.IV['CH1_Current']*1E12)
    axis[1].plot(data.XY['time']-data.XY['time'][0],data.XY['xpos'])
    axis[2].plot(data.XY['time']-data.XY['time'][0],data.XY['ypos'])
    axis[0].set_yscale('log')
    for index in data.coordinates:
        axis[0].axvspan(index[2],index[3],color='g',alpha=0.2)
        axis[1].axvspan(index[2],index[3],color='g',alpha=0.2)
        axis[2].axvspan(index[2],index[3],color='g',alpha=0.2)
    for ax in axis:
        #ax.set_xlim(msmt.IV['CH1_Time'][0],msmt.IV['CH1_Time'][-1])
        ax.set_xlim(100,200)
    axis[0].set_ylabel('PD Current [pA]')
    axis[1].set_ylabel('X-position [mm]')
    axis[2].set_ylabel('Y-position [mm]')
    figure.savefig('file.svg')
    
def rangefinder(tvals):
    figure, axis = plt.subplots(1,1,dpi=200)
    for tval in tvals:
        data.plot(tval,data.IV['CH1_Time'][tval],'ro')
    data.set_xlabel('Measurement')
    data.set_ylabel('Time [s]')

def linktime(param):
    XYlktime = np.average([data.XY['time'][param[2]],data.XY['time'][param[3]]])-data.XY['time'][0]
    minI = set(np.where(data.IV['CH1_Current'] < param[4])[0])
    mint = set(np.where(data.IV['CH1_Time'] > param[0])[0])
    maxt = set(np.where(data.IV['CH1_Time'] < param[1])[0])
    tvals = minI&mint&maxt
    t, n = 0, 0
    for tval in tvals:
        t += data.IV['CH1_Time'][tval]
        n += 1
    avtime = t/n
    IVlktime = data.IV['CH1_Time'][np.where(data.IV['CH1_Time'] >= avtime)[0][0]]
    return XYlktime, IVlktime, tvals

def timecorr(param):
    if param[5] == 0:
        dt = linktime(param)[0]-linktime(param)[1]+4
    else:
        dt = param[5]
    data.IV['CH1_Time'] = data.IV['CH1_Time']+dt    
    
def parsescan():
    data.coordinates = []
    for i in range(int(len(data.XY)/2)):
        data.coordinates.append([data.XY['xpos'][2*i],data.XY['ypos'][2*i],data.XY['time'][2*i]-data.XY['time'][0],data.XY['time'][2*i+1]-data.XY['time'][0]])
    data.coordinates.append([data.XY['xpos'][-1],data.XY['ypos'][-1],data.XY['time'][-1]-data.XY['time'][0],data.XY['time'][-1]-data.XY['time'][0]+5])
    #print(data.coordinates)
    data.XYIr = []
    data.XYI = []
    data.X, data.Y, data.I = [],[],[]
    f, g = open(os.path.join(PROC,'XYI_rich'+bname),'w+'), open(os.path.join(PROC,'XYI'+bname),'w+')
    f.write('X,Y,I');g.write('X,Y,I,Ierr')
    data.index = []
    for row in data.coordinates:
        l = np.where(data.IV['CH1_Time'] >= row[2])[0][0]
        m = np.where(data.IV['CH1_Time'] <= row[3])[0][-1]
        data.index.append([l,m])
        for i in range(l,m):
            I = data.IV['CH1_Current'][i]
            data.XYIr.append([row[0],row[1],I])
            f.write('\n%s,%s,%s'%(row[0],row[1],I))
        Iavg = np.average(data.IV['CH1_Current'][l:m])
        #if Iavg != Iavg:
        #    print(data.IV['CH1_Current'][l:m])
        #    print(l,m)
        #    print(row[2],row[3])
        Ierr = np.std(data.IV['CH1_Current'][l:m])
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
        
def rastersnake():
    X = np.unique(data.X)
    Y = np.unique(data.Y)
    data.Xproc, data.Yproc, data.Iproc = [],[],[]
    for x in X:
        for y in Y:
            for i in range(len(data.XYI)):
                if x == data.XYI[i][0]:
                    if y == data.XYI[i][1]:
                        data.Xproc.append(data.XYI[i][0])
                        data.Yproc.append(data.XYI[i][1])
                        data.Iproc.append(data.XYI[i][2])
                        #if x == 8.0:
                        #    print(data.XYI[i][0])
                        #    print(data.XYI[i][1])
                        #    print(data.XYI[i][2])
    data.Xproc = np.asarray(data.Xproc)
    data.Yproc = np.asarray(data.Yproc)
    data.Iproc = np.asarray(data.Iproc)
    
def findsensor():
    data.Iproc = np.nan_to_num(data.Iproc,nan=1E-13)
    data.Isum = np.sum(data.Iproc)
    data.Xmean = np.dot(data.Xproc,data.Iproc)/data.Isum
    data.Ymean = np.dot(data.Yproc,data.Iproc)/data.Isum
    
def cluster():
    # determine which K-Means cluster each point belongs to
    cluster_id = KMeans(10).fit_predict(data.Iproc.reshape(-1, 1))
    # determine densities by cluster assignment and plot
    fig, ax = plt.subplots(dpi=200)
    bins = np.linspace(data.Iproc.min(), data.Iproc.max(), 40)
    data.Kmean, data.Kstd = [],[]
    ax.set_xlabel('Current [A]')
    ax.set_ylabel('Counts')
    for ii in np.unique(cluster_id):
        subset = data.Iproc[cluster_id==ii]
        ax.hist(subset, bins=bins, alpha=0.5, label=f"Cluster {ii}")
        data.Kmean.append(np.mean(subset))
        data.Kstd.append(np.std(subset))
    data.Imax = data.Kmean[np.where(data.Kmean == max(data.Kmean))[0][0]]
    data.Imin = data.Kmean[np.where(data.Kmean == min(data.Kmean))[0][0]]
    data.Ihalf = (data.Imax - data.Imin)/2
    data.Imaxx = np.max(data.Iproc)
    ax.axvline(data.Ihalf)
    ax.axvline(data.Imaxx*1E-1)
    ax.legend()

    
if __name__ == "__main__":
    param = [3250,3750,1177,1178,1.5E-11,0]
    #param = [210,250,56,57,1.5E-11,0]
    #timestamp = 1658176417.202768
    #print(__name__)
    SCRIPTS,HOME,DATA,ARCHIVE,TEMP,DEV,PROC,PLOTS,REPORTS = init.envr() # Setup the local environment
    bname = os.listdir(DEV)[0][:-6]
    data = load_data(DEV,bname)
    #plot()
    #rangefinder(linktime(param)[2])
    #plotlk(linktime(param)[0],linktime(param)[1],param)
    timecorr(param)
    plot()
    parsescan()
    rastersnake()
    #stepplot()
    findsensor()
    cluster()
    histplot()
    lightmap()



