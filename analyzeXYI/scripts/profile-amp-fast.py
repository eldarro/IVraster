# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 17:07:23 2022

@author: darro
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

from scipy import special
from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
        self.Iproc, self.Ieproc, self.Ithresh, self.Xmean, self.Ymean = {},{},{},{},{}

# Main loop for aligning and analysing the data
# Do all of the analysis tasks necessary for finding the sensor
def main(i,dt):
    timecorr(i,dt)
    parsescan(i)
    rastersnake(i) 
        
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
        Ierr = np.std(I)
        #if Iavg != Iavg: # Check for missing data
        #    print(data.IV['CH1_Current'][l:m])
        #    print(l,m)
        #    print(row[2],row[3])
        data.XYI.append([row[0],row[1],Iavg,Ierr])
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
    data.Xproc, data.Yproc, data.Iproc['%s'%i], data.Ieproc['%s'%i] = [],[],[],[]
    for x in X:
        for y in Y:
            for j in range(len(data.XYI)):
                if x == data.XYI[j][0]:
                    if y == data.XYI[j][1]:
                        data.Xproc.append(data.XYI[j][0])
                        data.Yproc.append(data.XYI[j][1])
                        data.Iproc['%s'%i].append(data.XYI[j][2])
                        data.Ieproc['%s'%i].append(data.XYI[j][3])
    data.Xproc = np.asarray(data.Xproc)
    data.Yproc = np.asarray(data.Yproc)
    data.Iproc['%s'%i] = np.asarray(data.Iproc['%s'%i])
    data.Ieproc['%s'%i] = np.asarray(data.Ieproc['%s'%i])
    
# Clump the data and set thresholds
def cluster(i):           
    # Determine which K-Means cluster each point belongs to
    cluster_id = KMeans(10).fit_predict(data.Iproc['%s'%i].reshape(-1, 1))
    # Determine densities by cluster assignment and plot
    figure, axis = plt.subplots(dpi=200)
    bins = np.linspace(data.Iproc['%s'%i].min(), data.Iproc['%s'%i].max(), 40)
    data.Kmean, data.Kstd = [],[]
    axis.set_xlabel('SiPM Current [A]')
    axis.set_ylabel('Counts')
    axis.set_yscale('log')
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
    data.Ithresh['%s'%i] = [data.Ihalf]
    # Plot the thresholds
    for th in data.Ithresh['%s'%i]:
        findsensor(th,i)
        axis.axvline(th)
    axis.legend()

# Find the centroid of the X and Y data above some threshold
def findsensor(thresh,i):
    data.Xtemp, data.Ytemp, data.Itemp = [],[],[]
    for j in range(len(data.Iproc['%s'%i])):
        if data.Iproc['%s'%i][j] >= thresh:
            data.Xtemp.append(data.Xproc[j])
            data.Ytemp.append(data.Yproc[j])
            data.Itemp.append(data.Iproc['%s'%i][j]) 
    data.Inorm = np.sum(data.Itemp)
    data.Xmean['%s'%i] = np.dot(data.Xtemp,data.Itemp)/data.Inorm
    data.Ymean['%s'%i] = np.dot(data.Ytemp,data.Itemp)/data.Inorm
    return data.Xmean['%s'%i], data.Ymean['%s'%i]
    
# Plot the position and signal data as a function of time    
def plot(i):
    figure, axis = plt.subplots(3,1,dpi=200,figsize=(12,12))
    axis[0].plot(data.IV['TsCH_%i_s'%i],data.IV['IsCH_%i_A'%i],label='Channel %i'%i)
    axis[0].plot(data.IV['TsCH_%i_s'%i],data.IV['IsCH_%i_A'%i],label='Channel %i'%i)
    axis[1].plot(data.XY['time'],data.XY['xpos'])
    axis[2].plot(data.XY['time'],data.XY['ypos'])
    axis[0].set_yscale('log')
    for ax in axis:
        ax.set_xlim(data.XY['time'][0],data.XY['time'][-1])
    axis[0].set_ylabel('SiPM Current [uA]')
    axis[1].set_ylabel('X-position [mm]')
    axis[2].set_ylabel('Y-position [mm]')
    axis[2].set_xlabel('Time [s]')
    axis[0].legend()
    #figure.savefig(os.path.join(PLOTS,'%s_XYI.svg'))
    
def topbeam(x,*p):
    return p[0]/2*(1-special.erf(np.sqrt(2)*(p[1]-x)/p[2]))+p[3]
def botbeam(x,*p):
    return p[0]/2*(1-special.erf(np.sqrt(2)*(x-p[1])/p[2]))+p[3]

# Fit the beam profile and plot the results
def profile(i):
    # Check to see what profile is in the data
    if len(np.unique(data.Xproc)) > 1:
        posdata = data.Xproc
        prof = 'x'
    if len(np.unique(data.Yproc)) > 1:
        posdata = data.Yproc
        prof = 'y'
        
    # Find the parameters for the ERF fit
    # Locate the three regions about the two inflection points (sensor edges)
    #i0 = np.where(np.gradient(data.Iproc['%s'%i],posdata)==max(np.gradient(data.Iproc['%s'%i],posdata)))[0][0]
    #i1 = np.where(np.gradient(data.Iproc['%s'%i],posdata)==min(np.gradient(data.Iproc['%s'%i],posdata)))[0][0]
    i0 = np.where(data.Iproc['%s'%i]>=data.Ithresh['%s'%i])[0][0]
    i1 = np.where(data.Iproc['%s'%i]>=data.Ithresh['%s'%i])[0][-1]
    # Locate the FWHM from the scan
    plt.plot()
    # Find the location of the edges
    s0 = posdata[i0]
    s1 = posdata[i1]
    print(s0,s1)
    # Locate the index corresponding to the center of the beam
    i2 = np.where(posdata >= s0+(s1-s0)/2)[0][0]
    # Estimate the max power and offset
    print(data.Iproc['%s'%i][i0:i1])
    I0 = np.mean([x for x in data.Iproc['%s'%i][i0:i1] if str(x) != 'nan']) # Max
    I1 = np.mean([x for x in data.Iproc['%s'%i][:i0] if str(x) != 'nan']) # Offset
    I2 = np.mean([x for x in data.Iproc['%s'%i][i1:] if str(x) != 'nan']) # Offset
    print(I0,I1,I2)
    # Estimate the indices corresponding the beam radius
    i3 = np.where(data.Iproc['%s'%i] >= I1)[0][0]
    i4 = np.where(data.Iproc['%s'%i] >= I0)[0][0]
    i5 = np.where(data.Iproc['%s'%i] >= I0)[0][-1]
    i6 = np.where(data.Iproc['%s'%i] >= I2)[0][-1]
    # Estimate the beam radius
    w0 = (posdata[i4]-posdata[i3])/2
    w1 = (posdata[i6]-posdata[i5])/2
    # Organise the best-guess parameters into an array
    P0 = [I0,s0,w0,I1]
    P1 = [I0,s1,w1,I2]
    
    # Split the data into sections corresponding to each side of the sensor and remove nan
    toppos, topsignal, toperror, botpos, botsignal, boterror = [],[],[],[],[],[]
    for j in range(len(data.Iproc['%s'%i][:i2])):
        if str(data.Iproc['%s'%i][:i2][j]) != 'nan':
            toppos.append(posdata[:i2][j])
            topsignal.append(data.Iproc['%s'%i][:i2][j])
            toperror.append(data.Ieproc['%s'%i][:i2][j])
    for j in range(len(data.Iproc['%s'%i][i2:])):
        if str(data.Iproc['%s'%i][i2:][j]) != 'nan':
            botpos.append(posdata[i2:][j])
            botsignal.append(data.Iproc['%s'%i][i2:][j])  
            boterror.append(data.Ieproc['%s'%i][i2:][j])
    toppos, topsignal, toperror, botpos, botsignal, boterror = np.array(toppos), np.array(topsignal), np.array(toperror), np.array(botpos), np.array(botsignal), np.array(boterror)

    # Fit the data
    top_popt, top_pcov = curve_fit(topbeam,toppos,topsignal,p0=P0)
    bot_popt, bot_pcov = curve_fit(botbeam,botpos,botsignal,p0=P1)
    
    # Calculate the residuals
    topres = (topsignal-topbeam(toppos,*top_popt))/toperror
    botres = (botsignal-botbeam(botpos,*bot_popt))/boterror

    # Calculate the chi2
    top_X2 = np.sum(topres**2)/(len(topsignal)-len(P0))
    bot_X2 = np.sum(botres**2)/(len(botsignal)-len(P1))

    # Plot the data and fit
    figure, axis = plt.subplots(2,1,dpi=200,figsize=(12,12))
    axis[1].errorbar(posdata,data.Iproc['%s'%i],yerr=data.Ieproc['%s'%i])

    axis[1].plot(toppos,topbeam(toppos,*top_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm\nE0 = %.2e A'%(top_popt[0],top_popt[1],top_popt[2],top_popt[3]))
    axis[1].plot(botpos,botbeam(botpos,*bot_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm\nE0 = %.2e A'%(bot_popt[0],bot_popt[1],bot_popt[2],bot_popt[3]))
    # Plot the residuals from the fit
    axis[0].plot(toppos,topres,'o',label=r'$\chi^2$ = '+'%.2f'%top_X2)
    axis[0].plot(botpos,botres,'o',label=r'$\chi^2$ = '+'%.2f'%bot_X2)

    # Plot the best-guess parameters
    #axis[1].plot(toppos,topbeam(toppos,P0))
    #axis[1].plot(botpos,botbeam(botpos,P1))  
    #axis[0].plot(posdata,np.gradient(data.Iproc,posdata))
    
    # Format the plots
    for ax in axis:
        if prof == 'x':
            ax.set_xlabel('X-position [mm]')
        if prof == 'y':
            ax.set_xlabel('Y-position [mm]')
        ax.set_xlim(posdata[0],posdata[-1])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
    axis[0].set_ylabel('Normalised Residuals')
    axis[1].set_ylabel('SiPM Current [pA]')
    axis[0].legend()
    axis[1].legend()
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
    axis[0].set_ylabel('SiPM Current [pA]')
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
    cluster(i)
    #stepplot()
    profile(i)
    
if __name__ == "__main__":
    SCRIPTS,HOME,DATA,ARCHIVE,TEMP,DEV,PROC,PLOTS,REPORTS = init.envr() # Setup the local environment
    bname = os.listdir(DEV)[0][:-6] # Find the basename for the data files
    data = load_data(DEV,bname) # Create the data class
    a = [16]
    b = [];c=[]
    data.Iproc['sum'] = []
    for i in a:
        main(i,-84.3) # Align the data and do analysis
        plots(i) # What plots to draw

        



