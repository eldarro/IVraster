# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 17:07:23 2022

@author: darro
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import curve_fit
import init

# The default data class
# Contains data from the XY stage and the electrometer
class load_data:
    def __init__(self, loc, fname):
        self.XY = np.genfromtxt(os.path.join(loc,fname + 'XY.csv'), delimiter = ',', comments='#', names = True, skip_header=1)
        self.IV = np.genfromtxt(os.path.join(loc,fname + 'IV.csv'), delimiter = ',', comments='#', names = True, skip_header=0)

# Main loop for aligning and analysing the data
# Do all of the analysis tasks necessary for finding the sensor
def main():
    timecorr()
    parsescan()
    rastersnake() 
        
# Correct for the offset between electrometer time and unix time  
# The parameter dt can be used as a fitting paramter
# The correct value of dt will 'focus' the plots from lightmap and histplot
def timecorr():
    t0 = 1659568610.6043196 
    if t0 == 0:
        t0 = data.XY['time'][0]
        dt = -13
    else:
        dt = -3
    t = t0 + dt
    data.IV['CH1_Time'] = data.IV['CH1_Time']+t
  
# Reorganize data into ordered structures
# This function makes assumptions about the structure of the input data
def parsescan():
    settle = 10 # number of indices to trim while electrometer is settling
    data.coordinates = []
    for i in range(int(len(data.XY)/2)):
        data.coordinates.append([data.XY['xpos'][2*i],data.XY['ypos'][2*i],data.XY['time'][2*i],data.XY['time'][2*i+1]])
    #data.coordinates.append([data.XY['xpos'][-1],data.XY['ypos'][-1],data.XY['time'][-1],data.XY['time'][-1]+param[6]])
    #print(data.coordinates)
    data.XYIr = []
    data.XYI = []
    data.X, data.Y, data.I = [],[],[]
    f, g = open(os.path.join(PROC,bname+'XYI_rich.csv'),'w+'), open(os.path.join(PROC,bname+'XYI.csv'),'w+')
    f.write('X,Y,I');g.write('X,Y,I,Ierr')
    data.index = []
    for row in data.coordinates:
        l = np.where(data.IV['CH1_Time'] >= row[2])[0][0]+settle
        m = np.where(data.IV['CH1_Time'] <= row[3])[0][-1]-settle
        data.index.append([l,m])
        for i in range(l,m):
            I = data.IV['CH1_Current'][i]
            data.XYIr.append([row[0],row[1],I])
            f.write('\n%s,%s,%s'%(row[0],row[1],I))
        Iavg = np.average(data.IV['CH1_Current'][l:m])
        Ierr = np.std(data.IV['CH1_Current'][l:m])/np.sqrt(len(data.IV['CH1_Current'][l:m]))
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
def rastersnake():
    X = np.unique(data.X)
    Y = np.unique(data.Y)
    data.Xproc, data.Yproc, data.Iproc, data.Ieproc = [],[],[],[]
    for x in X:
        for y in Y:
            for i in range(len(data.XYI)):
                if x == data.XYI[i][0]:
                    if y == data.XYI[i][1]:
                        data.Xproc.append(data.XYI[i][0])
                        data.Yproc.append(data.XYI[i][1])
                        data.Iproc.append(data.XYI[i][2])
                        data.Ieproc.append(data.XYI[i][3])
    data.Xproc = np.asarray(data.Xproc)
    data.Yproc = np.asarray(data.Yproc)
    data.Iproc = np.asarray(data.Iproc)
    data.Ieproc = np.asarray(data.Ieproc)
    
# Find the centroid of the X and Y data above some photosensor current threshold
def findsensor(thresh):
    data.Xtemp, data.Ytemp, data.Itemp = [],[],[]
    for i in range(len(data.Iproc)):
        if data.Iproc[i] >= thresh:
            data.Xtemp.append(data.Xproc[i])
            data.Ytemp.append(data.Yproc[i])
            data.Itemp.append(data.Iproc[i]) 
    data.Isum = np.sum(data.Itemp)
    data.Xmean = np.dot(data.Xtemp,data.Itemp)/data.Isum
    data.Ymean = np.dot(data.Ytemp,data.Itemp)/data.Isum
    return data.Xmean, data.Ymean

# Plot the position and signal data as a function of time    
def plot():
    figure, axis = plt.subplots(3,1,dpi=200,figsize=(12,12))
    axis[0].plot(data.IV['CH1_Time'],data.IV['CH1_Current']*1E12)
    axis[1].plot(data.XY['time'],data.XY['xpos'])
    axis[2].plot(data.XY['time'],data.XY['ypos'])
    axis[0].set_yscale('log')
    for ax in axis:
        ax.set_xlim(data.IV['CH1_Time'][0],data.IV['CH1_Time'][-1])
    axis[0].set_ylabel('PD Current [pA]')
    axis[1].set_ylabel('X-position [mm]')
    axis[2].set_ylabel('Y-position [mm]')
    axis[2].set_xlabel('Time [s]')
    #figure.savefig(os.path.join(PLOTS,'%s_XYI.svg'))

def topbeam(x,*p):
    return p[0]/2*(1-special.erf(np.sqrt(2)*(p[1]-x)/p[2]))+p[3]
def botbeam(x,*p):
    return p[0]/2*(1-special.erf(np.sqrt(2)*(x-p[1])/p[2]))+p[3]

# Fit the beam profile and plot the results
def profile():
    # Check to see what profile is in the data
    if len(np.unique(data.Xproc)) > 1:
        posdata = data.Xproc
        prof = 'x'
    if len(np.unique(data.Yproc)) > 1:
        posdata = data.Yproc
        prof = 'y'
        
    # Find the parameters for the ERF fit
    # Locate the three regions about the two inflection points (sensor edges)
    i0 = np.where(np.gradient(data.Iproc,posdata)==max(np.gradient(data.Iproc,posdata)))[0][0]
    i1 = np.where(np.gradient(data.Iproc,posdata)==min(np.gradient(data.Iproc,posdata)))[0][0]
    # Find the location of the edges
    s0 = posdata[i0]
    s1 = posdata[i1]
    print(s0,s1)
    # Locate the index corresponding to the center of the beam
    i2 = np.where(posdata >= s0+(s1-s0)/2)[0][0]
    # Estimate the max power and offset

    I0 = np.mean([x for x in data.Iproc[i0:i1] if str(x) != 'nan']) # Max
    I1 = np.mean([x for x in data.Iproc[:i0] if str(x) != 'nan']) # Offset
    I2 = np.mean([x for x in data.Iproc[i1:] if str(x) != 'nan']) # Offset
    # Estimate the indices corresponding the beam radius
    i3 = np.where(data.Iproc >= I1)[0][0]
    i4 = np.where(data.Iproc >= I0)[0][0]
    i5 = np.where(data.Iproc >= I0)[0][-1]
    i6 = np.where(data.Iproc >= I2)[0][-1]
    # Estimate the beam radius
    w0 = (posdata[i4]-posdata[i3])/2
    w1 = (posdata[i6]-posdata[i5])/2
    # Organise the best-guess parameters into an array
    P0 = [I0,s0,w0,I1]
    P1 = [I0,s1,w1,I2]
    
    # Split the data into sections corresponding to each side of the sensor and remove nan
    toppos, topsignal, toperror, botpos, botsignal, boterror = [],[],[],[],[],[]
    for i in range(len(data.Iproc[:i2])):
        if str(data.Iproc[:i2][i]) != 'nan':
            toppos.append(posdata[:i2][i])
            topsignal.append(data.Iproc[:i2][i])
            toperror.append(data.Ieproc[:i2][i])
    for i in range(len(data.Iproc[i2:])):
        if str(data.Iproc[i2:][i]) != 'nan':
            botpos.append(posdata[i2:][i])
            botsignal.append(data.Iproc[i2:][i])  
            boterror.append(data.Ieproc[i2:][i])
    toppos, topsignal, toperror, botpos, botsignal, boterror = np.array(toppos), np.array(topsignal), np.array(toperror), np.array(botpos), np.array(botsignal), np.array(boterror)

    # Fit the data
    top_popt, top_pcov = curve_fit(topbeam,toppos,topsignal,p0=P0)
    bot_popt, bot_pcov = curve_fit(botbeam,botpos,botsignal,p0=P1)
    
    # Calculate the residuals
    topres = (topbeam(toppos,*top_popt)-topsignal)/toperror
    botres = (botbeam(botpos,*bot_popt)-botsignal)/boterror

    # Calculate the chi2
    top_X2 = np.sum(topres**2/(len(topsignal)-len(P0)))
    bot_X2 = np.sum(botres**2/(len(botsignal)-len(P1)))

    # Plot the data and fit
    figure, axis = plt.subplots(2,1,dpi=200,figsize=(12,12))
    axis[1].errorbar(posdata,data.Iproc,yerr=data.Ieproc)

    axis[1].plot(toppos,topbeam(toppos,*top_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm\nE0 = %.2e A'%(top_popt[0],top_popt[1],top_popt[2],top_popt[3]))
    axis[1].plot(botpos,botbeam(botpos,*bot_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm\nE0 = %.2e A'%(bot_popt[0],bot_popt[1],bot_popt[2],bot_popt[3]))
    # Plot the residuals from the fit
    axis[0].plot(toppos,(topbeam(toppos,*top_popt)-topsignal)/toperror)
    axis[0].plot(botpos,(botbeam(botpos,*bot_popt)-botsignal)/boterror)

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
    axis[1].set_ylabel('PD Current [pA]')
    axis[1].legend()
    #figure.savefig(os.path.join(PLOTS,'%s_XYI.svg'))    

# Debugging plots go here
# Plot the position and signal data, highlight the averaged data
def stepplot():
    figure, axis = plt.subplots(1,1,dpi=200,figsize=(12,12))
    axis.plot(data.IV['CH1_Time'],data.IV['CH1_Current']*1E12)
    #axis[1].plot(data.XY['time'],data.XY['xpos'])
    #axis[2].plot(data.XY['time'],data.XY['ypos'])
    #axis[0].set_yscale('log')

    for index in data.index:
        axis.axvspan(data.IV['CH1_Time'][index[0]],data.IV['CH1_Time'][index[1]],color='g',alpha=0.2)
        #axis.axvspan(index[2],index[3],color='g',alpha=0.2)
        #axis[1].axvspan(index[2],index[3],color='g',alpha=0.2)
        #axis[2].axvspan(index[2],index[3],color='g',alpha=0.2)

    limit = [3000,5500,0,-1]
    axis.set_xlim(data.IV['CH1_Time'][limit[0]],data.IV['CH1_Time'][limit[1]])
    axis.axvspan(data.IV['CH1_Time'][limit[0]],data.IV['CH1_Time'][limit[1]],alpha=0.2)
    axis.set_ylabel('PD Current [pA]')
    #axis[1].set_ylabel('X-position [mm]')
    #axis[2].set_ylabel('Y-position [mm]')
    axis.set_xlabel('Time [s]')
    figure.savefig(os.path.join(PLOTS,'%s_stepplot.svg'))

# Plot the time each measurement is made, useful for finding range issues with instrument
def checkgap(tvals):
    figure, axis = plt.subplots(1,1,dpi=200)
    for tval in tvals:
        data.plot(tval,data.IV['CH1_Time'][tval],'ro')
    data.set_xlabel('Measurement')
    data.set_ylabel('Time [s]')    
    
def plots():
    plot()
    #stepplot()
    profile()
    
if __name__ == "__main__":
    SCRIPTS,HOME,DATA,ARCHIVE,TEMP,DEV,PROC,PLOTS,REPORTS = init.envr() # Setup the local environment
    bname = os.listdir(DEV)[6][:-6] # Find the basename for the data files
    data = load_data(DEV,bname) # Create the data class
    main() # Align the data and do analysis
    plots() # What plots to draw



