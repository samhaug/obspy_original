#!/usr/bin/env python

'''
seis_plot.py includes all functions for plotting seismic data
'''

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import scipy
import obspy
import h5py
import numpy as np
import copy

from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,
                                                  DictFormatter)
from seispy.data import phase_window
from matplotlib import pyplot as plt
import obspy.signal.filter
import obspy.signal
from obspy.taup import TauPyModel
model = TauPyModel(model="prem50")
from matplotlib import colors, ticker, cm
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import os
from scipy.optimize import curve_fit
import math
from matplotlib.colors import LightSource
from mpl_toolkits.basemap import Basemap
from cycler import cycler
import multiprocessing
import shutil
from seispy import mapplot
from seispy import data
from seispy import convert

ppt_font =  {'family' : 'sans-serif',
             'style' : 'normal',
             'variant' : 'normal',
             'weight' : 'bold',
             'size' : 'xx-large'}
paper_font =  {'family' : 'serif',
             'style' : 'normal',
             'variant' : 'normal',
             'weight' : 'medium',
             'size' : 'large'}


def mapplot(self):

    width = 28000000
    fig = plt.figure(figsize=(10,10))
    lat_0 = st[0].stats.sac['evla']
    lon_0 = st[0].stats.sac['evlo']
    m = Basemap(projection='ortho',lat_0=lat_0,lon_0=lon_0)
    xpt, ypt = m(lon_0, lat_0)
    m.scatter(xpt,ypt,s=99,c='red',marker='o',lw=1)
    # fill background.
    #m.drawmapboundary(fill_color='aqua')
    # draw coasts and fill continents.
    m.drawcoastlines(linewidth=0.5)
    #m.fillcontinents(color='coral',lake_color='aqua')
    # 20 degree graticule.
    m.drawparallels(np.arange(-80,81,20))
    m.drawmeridians(np.arange(-180,180,20))
    # draw a black dot at the center.
    for tr in st:
        lat_0 = tr.stats.sac['stla']
        lon_0 = tr.stats.sac['stlo']
        xpt, ypt = m(lon_0, lat_0)
        m.scatter(xpt,ypt,s=5,c='green',marker='o',lw=0)
    plt.show()

def vespagram(self,**kwargs):
    '''
    make vespagram of stream object. Recommended to pass stream object
    through remove_dirty
    '''

    window_tuple = kwargs.get('x_lim',(-10,230))
    window_phase = kwargs.get('window_phase',['P'])
    window_slowness = kwargs.get('p_lim',(-1.5,1.5))
    slowness_tick = kwargs.get('p_tick',-0.1)
    phase_list = kwargs.get('phase_list',False)
    plot_line = kwargs.get('plot_line',False)
    n_root = kwargs.get('n_root',2.0)
    save = kwargs.get('save',False)
    clim = kwargs.get('clim',(-3,0))
    cmap = kwargs.get('cmap','gnuplot')
    font = kwargs.get('font',paper_font)
    plot = kwargs.get('plot',True)
    title = kwargs.get('title',False)
    ax_grab = kwargs.get('ax_grab',False)

    st = obspy.core.Stream()
    for idx, tr in enumerate(st_in):
        st += phase_window(tr,window_phase,window=window_tuple)

    def mean_range(st):
        a = []
        for tr in st:
            a.append(tr.stats.sac['gcarc'])
        mn_r = np.mean(a)
        return mn_r

    def roll_zero(array,n):
        if n < 0:
            array = np.roll(array,n)
            array[n::] = 0
        else:
            array = np.roll(array,n)
            array[0:n] = 0
        return array

    def slant_stack(st,mn_range,slowness,n):
        d = st[0].stats.delta
        R = np.zeros(st[0].data.shape[0])
        for tr in st:
            az = tr.stats.sac['gcarc']-mn_range
            shift_in_sec = slowness*az
            shift_in_bin = int(shift_in_sec/d)
            x = roll_zero(tr.data,shift_in_bin)
            R += np.sign(x)*pow(np.abs(x),1./n)
        R = R/float(len(st))
        yi = R*pow(abs(R),n-1)
        hil = scipy.fftpack.hilbert(yi)
        yi = pow(hil**2+R**2,1/2.)
        return yi,R

    def phase_plot(ax,evdp,degree,phases,text_color):
        P_arrivals = model.get_travel_times(distance_in_degree = degree,
        source_depth_in_km = evdp,
        phase_list = ['P'])

        P_slowness = P_arrivals[0].ray_param_sec_degree
        P_time = P_arrivals[0].time

        arrivals = model.get_travel_times(distance_in_degree=degree,
        source_depth_in_km=evdp,
        phase_list = phases)
        if len(arrivals) != 0:
            colors = ['b','g','r','c','m','y','k']
            name = []
            for idx, ii in enumerate(arrivals):
                if ii.name in name:
                    continue
                else:
                    name.append(ii.name)
                    p = ii.ray_param_sec_degree-P_slowness
                    time = ii.time-P_time
                    ax.scatter(time,p,s=300,marker='D',zorder=20,
                           facecolors='None',lw=1,edgecolor=text_color)
                #ax.text(time,p,ii.name,fontsize=16,color=text_color)

    mn_r = mean_range(st)
    evdp = st[0].stats.sac['evdp']

    vesp_y = np.linspace(0,0,num=st[0].data.shape[0])
    vesp_R = np.linspace(0,0,num=st[0].data.shape[0])
    for ii in np.arange(window_slowness[0],window_slowness[1],-1*slowness_tick):
        yi,R = slant_stack(st,mn_r,ii,1.0)
        vesp_y= np.vstack((vesp_y,yi))
        vesp_R= np.vstack((vesp_R,R))
    vesp_y = vesp_y[1::,:]
    vesp_R1 = vesp_R[1::,:]*2
    if plot == False:
        vesp_R = vesp_R-vesp_R.mean(axis=1,keepdims=True)
        return vesp_R
    #vesp_y = vesp_y/ vesp_y.max()

    if ax_grab != False:
        ax = ax_grab
        image_0 = ax.imshow(np.log10(vesp_y),aspect='auto',interpolation='lanczos',
           extent=[window_tuple[0],window_tuple[1],window_slowness[0],window_slowness[1]],
           cmap=cmap,vmin=clim[0],vmax=clim[1])
        return image_0

    fig, ax = plt.subplots(2,sharex=True,figsize=(15,10))

    image_0 = ax[0].imshow(np.log10(vesp_y),aspect='auto',interpolation='lanczos',
           extent=[window_tuple[0],window_tuple[1],window_slowness[0],window_slowness[1]],
           cmap=cmap,vmin=clim[0],vmax=clim[1])
    one_stack = vesp_y

    vesp_y = np.linspace(0,0,num=st[0].data.shape[0])
    vesp_R = np.linspace(0,0,num=st[0].data.shape[0])
    for ii in np.arange(window_slowness[0],window_slowness[1],-1*slowness_tick):
        yi,R = slant_stack(st,mn_r,ii,n_root)
        vesp_y= np.vstack((vesp_y,yi))
        vesp_R= np.vstack((vesp_R,R))
    vesp_y = vesp_y[1::,:]
    vesp_R = vesp_R[1::,:]
    #vesp_y = vesp_y/ vesp_y.max()

    image_1 = ax[1].imshow(np.log10(vesp_y), aspect='auto',
         interpolation='lanczos', extent=[window_tuple[0],
         window_tuple[1],window_slowness[0],window_slowness[1]],cmap=cmap, vmin=clim[0],vmax=clim[1])

    two_stack = vesp_y

    cbar_0 = fig.colorbar(image_0,ax=ax[0])
    cbar_0.set_label('Log(Seismic energy)',fontdict=font)
    cbar_1 = fig.colorbar(image_1,ax=ax[1])
    cbar_1.set_label('Log(Seismic energy)',fontdict=font)
    ax[0].set_xlim(window_tuple)
    ax[1].set_xlim(window_tuple)
    ax[1].xaxis.set(ticks=range(window_tuple[0],window_tuple[1],10))
    ax[0].xaxis.set(ticks=range(window_tuple[0],window_tuple[1],10))
    ax[0].set_ylim([window_slowness[0],window_slowness[1]])
    ax[1].set_ylim([window_slowness[0],window_slowness[1]])
    ax[0].grid(color='w',lw=2,alpha=0.9)
    ax[1].grid(color='w',lw=2,alpha=0.9)
    ax[1].set_ylabel('Slowness (s/deg)',fontdict=font)
    ax[0].set_ylabel('Slowness (s/deg)',fontdict=font)
    ax[1].set_xlabel('Seconds after {}'.format(window_phase[0]),fontdict=font)
    if title == True:
        ax[0].set_title('Start: {} \n Source Depth: {} km, Ref_dist: {} deg, {} \
                     \n Bottom : N-root = {} Top: N-root = 1'
                  .format(st[0].stats.starttime,
                  round(st[0].stats.sac['evdp'],3),
                  round(mn_r,3), os.getcwd().split('-')[3],
                  str(n_root)))

    if phase_list:
        phase_plot(ax[0],evdp,mn_r,phase_list,text_color='white')
        phase_plot(ax[1],evdp,mn_r,phase_list,text_color='white')

    if save != False:
        plt.savefig(save+'/vespagram.pdf',format='pdf')

    time_vec = np.linspace(window_tuple[0], window_tuple[1],
               num=vesp_R.shape[1])

    figR, axR= plt.subplots(1,figsize=(14,7))

    #vesp_R *= 2
    for idx, ii in enumerate(np.arange(window_slowness[1],window_slowness[0],
                             slowness_tick)):
        vesp_R1[idx,:] += ii
        axR.fill_between(time_vec,ii,vesp_R1[idx,:],where=vesp_R1[idx,:] >= ii,
                         facecolor='goldenrod',alpha=0.8,lw=0.5)
        axR.fill_between(time_vec,ii,vesp_R1[idx,:],where=vesp_R1[idx,:] <= ii,
                         facecolor='blue',alpha=0.8,lw=0.5)
        #axR.plot(time_vec,vesp_R[idx,:])
        if phase_list:
            phase_plot(axR,evdp,mn_r,phase_list,text_color='black')

    axR.set_xlim(window_tuple)
    axR.set_ylim([window_slowness[0],window_slowness[1]])
    axR.set_ylabel('Slowness (s/deg)')
    axR.set_xlabel('Seconds after P')
    if title == True:
        axR.set_title('Start: {} \n Source Depth: {} km, Ref_dist: {} deg, {}'
                  .format(st[0].stats.starttime,
                   round(st[0].stats.sac['evdp'],3),round(mn_r,3),
                   os.getcwd().split('-')[3]))
    axR.set_xticks(np.arange(window_tuple[0],window_tuple[1],10))
    axR.grid()


    if save != False:
        plt.savefig(save+'/wave.pdf',format='pdf')
    else:
        plt.show()

    return vesp_R1


def plot(self,**kwargs):
    '''
    plot trace object
    phase = optional list of phases. list of strings corresponding
            to taup names
    '''
    phases = kwargs.get('phase_list',False)
    window = kwargs.get('window',False)
    t_model = kwargs.get('model',model)

    fig, ax = plt.subplots(figsize=(23,9))
    if phases != False:
        arrivals = t_model.get_travel_times(
                   distance_in_degree=tr.stats.sac['gcarc'],
                   source_depth_in_km=tr.stats.sac['evdp'],
                   phase_list = phases)
        window_list = []
        colors = ['b','g','r','c','m','y','k']
        if len(arrivals) != 0:
            for idx, ii in enumerate(arrivals):
                ax.axvline(ii.time,label=ii.purist_name,c=np.random.rand(3,1))
                window_list.append(ii.time)
    ax.legend()

    time = np.linspace(-1*tr.stats.sac['o'],
           (tr.stats.delta*tr.stats.npts)-tr.stats.sac['o'],
           num=tr.stats.npts)
    ax.plot(time,tr.data,c='k')
    ax.grid()
    ax.set_title('Network: {}, Station: {}, Channel: {},\
    Dist (deg): {}, Depth (km): {} \nStart: {} \nEnd: {}'.format(
                  tr.stats.network,
                  tr.stats.station,
                  tr.stats.channel,
                  round(tr.stats.sac['gcarc'],3),
                  round(tr.stats.sac['evdp'],3),
                  tr.stats.starttime,
                  tr.stats.endtime))
    ax.set_xlabel('Time (s), sampling_rate: {}, npts: {}'
                 .format(tr.stats.sampling_rate,tr.stats.npts))

    if window != False:
        ax.set_xlim([min(window_list)+window[0],max(window_list)+window[1]])
        #ax.set_xlim([min(window_list)-300,max(window_list)+300])

    plt.show()

def component_plot(tr_list,**kwargs):
    '''
    Plot three component section
    '''
    if tr_list[0].stats.station != tr_list[1].stats.station:
        print 'Components from different stations'
    separate = kwargs.get('separate',True)
    phase_list = kwargs.get('phase_list',[])
    window_tuple = kwargs.get('window_tuple',False)

    arrivals = model.get_travel_times(
               distance_in_degree=tr_list[0].stats.sac['gcarc'],
               source_depth_in_km=tr_list[0].stats.sac['evdp'],
               phase_list = phase_list)
    colors = ['b','g','r','c','m','y','k','b','g','r','c','m','y']
    trace_c = ['k','m','goldenrod']
    title = kwargs.get('title',False)


    if separate:
        fig, ax = plt.subplots(len(tr_list), sharey=True,sharex=True,
                               figsize=(23,15))
        for idx, tr in enumerate(tr_list):
            t_len = tr.stats.endtime - tr.stats.starttime
            start = -1*tr.stats.sac['o']
            end = t_len-tr.stats.sac['o']
            time = np.linspace(start,end,num=tr.stats.npts)
            data = tr.data
            trace = ax[idx].plot(time,data,color='k',zorder=99)
            ax[idx].text(1, 1, tr.stats.channel,
                           horizontalalignment='right',
                           verticalalignment='bottom',
                           transform=ax[idx].transAxes,
                           size='x-large')
            ax[idx].grid()
            ax[idx].legend(loc=2)
            ax[idx].set_xticks(np.arange(time[0],time[-1],25))
            #add arrivals
            for jdx, ii in enumerate(arrivals):
                ax[idx].axvline(ii.time,label=ii.name,c=colors[jdx])
                ax[idx].legend(loc=2)

        t_len = tr_list[0].stats.endtime - tr_list[0].stats.starttime
        start = -1*tr_list[0].stats.sac['o']
        end = t_len-tr_list[0].stats.sac['o']
        if window_tuple:
            ax[0].set_xlim((arrivals[0].time+window_tuple[0],
                        arrivals[0].time+window_tuple[1]))
            ax[1].set_xlim((arrivals[0].time+window_tuple[0],
                        arrivals[0].time+window_tuple[1]))
        ax[-1].set_xlabel('Seconds after event')
        if title == True:
            ax[0].set_title('{} \n Depth (km): {} Dist (deg): {}, Az (deg): {}, {}, {}'.format(
              tr_list[0].stats.starttime,
              str(round(tr_list[0].stats.sac['evdp'],3)),
              str(round(tr_list[0].stats.sac['gcarc'],3)),
              str(round(tr_list[0].stats.sac['az'],3)),
              tr_list[0].stats.network+tr_list[0].stats.station,
              os.getcwd().split('-')[3]))
        for ii in ax:
            ii.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.show()



    elif separate != True:
        fig, ax = plt.subplots(figsize=(23,9))
        for idx, tr in enumerate(tr_list):
            t_len = tr.stats.endtime - tr.stats.starttime
            start = -1*tr.stats.sac['o']
            end = t_len-tr.stats.sac['o']
            time = np.linspace(start,end,num=tr.stats.npts)
            data = tr.data
            trace = ax.plot(time,data,label=tr.stats.channel,color=trace_c[idx])
            ax.legend(loc=2)
            #add arrivals
        for jdx, ii in enumerate(arrivals):
            ax.axvline(ii.time,label=ii.name,c=colors[jdx])
            ax.legend(loc=2)

        t_len = tr[0].stats.endtime - tr[0].stats.starttime
        start = -1*tr[0].stats.sac['o']
        end = t_len-tr[0].stats.sac['o']
        if window_tuple:
            ax.set_xlim((start+arrivals[0].time+window_tuple[0],
                        start+arrivals[0].time+window_tuple[1]))
        ax.grid()
        ax.set_xlabel('Seconds after P')
        ax.set_xticks(np.arange(start-50,end,25))
        ax.set_title('{} \n Depth (km): {} Dist (deg): {}, Az (deg): {}, {}'.format(
            tr_list[0].stats.starttime,
            str(round(tr_list[0].stats.sac['evdp'],3)),
            str(round(tr_list[0].stats.sac['gcarc'],3)),
            str(round(tr_list[0].stats.sac['baz'],3)),
            os.getcwd().split('-')[3]))
        plt.show()

def compare_section(self,sts,**kwargs):
    '''
    compare two streams, std is data and sts is synthetic
    '''
    a_list = kwargs.get('a_list',True)
    fig = kwargs.get('fig',None)
    ax = kwargs.get('ax',None)

    if fig == None and ax == None:
        fig,ax = plt.subplots(figsize=(10,15))

    for ii in range(0,len(std)):
        try:
            ds = std[ii].stats.starttime
            ss = sts[ii].stats.starttime
            ddelt = std[ii].stats.delta
            sdelt = sts[ii].stats.delta
            dpts = std[ii].stats.npts
            spts = sts[ii].stats.npts
            t0 = min(ss,ds)
            P_time = 0

            if a_list != True:
                evdp = std[ii].stats.sac['evdp']
                gcarc = std[ii].stats.sac['gcarc']
                P = model.get_travel_times(distance_in_degree=gcarc,
                    source_depth_in_km=evdp,
                    phase_list = a_list)
                P_time += P[0].time

            t_dat = np.linspace(ds-t0,ds-t0+dpts*ddelt,num=dpts)
            t_syn = np.linspace(ss-t0,ss-t0+spts*sdelt,num=spts)
            ax.plot(t_dat-P_time,std[ii].data+std[ii].stats.sac['gcarc'],alpha=0.5,color='k')
            ax.plot(t_syn-P_time,sts[ii].data+sts[ii].stats.sac['gcarc'],alpha=0.5,color='r',label='sim')
        except IndexError:
            plt.show()

    if fig == None and ax == None:
        plt.show()

def simple_section(self,**kwargs):
    '''
    Simpler section plotter for obspy stream object
    '''
    a_list = kwargs.get('a_list',True)
    fig = kwargs.get('fig',None)
    ax = kwargs.get('ax',None)
    color = kwargs.get('color','k')

    if fig == None and ax == None:
        fig,ax = plt.subplots(figsize=(10,15))
    else:
        print('using outside figure')

    def plot(tr,o,ax):
        e = tr.stats.npts/tr.stats.sampling_rate
        t = np.linspace(o,o+e,num=tr.stats.npts)
        ax.plot(t,tr.data+tr.stats.sac['gcarc'],alpha=0.5,color=color)

    if a_list == True:
        for tr in st:
            plot(tr,0,ax)

    elif type(a_list) == list:
        if len(a_list) != 1:
            print('Must have phase identifier string of len = 1')
            return
        else:
            for tr in st:
                evdp = tr.stats.sac['evdp']
                gcarc = tr.stats.sac['gcarc']
                P = model.get_travel_times(distance_in_degree=gcarc,
                    source_depth_in_km=evdp,
                    phase_list = a_list)
                P_time = P[0].time
                plot(tr,-1*(P_time+tr.stats.sac['o']),ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Epicentral Distance (deg)')
    plt.show()

def section(st,**kwargs):
    '''
    Plot record section of obspy stream object

    labels = kwargs.get('labels',False)
    phases = kwargs.get('phase_list',False)
    fill = kwargs.get('fill',False)
    shift = kwargs.get('shift',False)
    save = kwargs.get('save',False)
    title = kwargs.get('title',True)
    x_lim = kwargs.get('x_lim',(-50,1000))
    color = kwargs.get('color',False)
    picker = kwargs.get('picker',False)
    '''
    labels = kwargs.get('labels',False)
    phases = kwargs.get('phase_list',False)
    fill = kwargs.get('fill',False)
    shift = kwargs.get('shift',False)
    save = kwargs.get('save',False)
    title = kwargs.get('title',False)
    x_lim = kwargs.get('x_lim',(-50,1000))
    y_lim = kwargs.get('y_lim',False)
    color = kwargs.get('color',False)
    picker = kwargs.get('picker',False)
    align_phase = kwargs.get('align_phase',['P','Pdiff'])
    name = kwargs.get('name_plot',False)
    az_color = kwargs.get('az_color',False)
    ax_grab = kwargs.get('ax_grab',False)

    def main():
        p_list,name_list,dist_list = p_list_maker(st)
        lim_tuple = ax_limits(p_list)

        if ax_grab != False:
            ax = ax_grab
        else:
            fig, ax = plt.subplots(figsize =(10,15))
        ax.set_xlim((x_lim[0],x_lim[1]))
        if y_lim != False:
            ax.set_ylim((y_lim[0],y_lim[1]))
        for idx,ii in enumerate(p_list):
            add_to_axes(ii,ax)

        if phases != False:
            range_list = []
            for tr in st:
                tr.stats.location = tr.stats.sac['gcarc']
            st.sort(['location'])
            ymin = st[0].stats.location
            ymax = st[-1].stats.location
            t = st[len(st)/2]
            phase_plot(lim_tuple,70.,t.stats.sac['evdp'],phases,ax,
                       t.stats.sac['o'])
            ax.set_ylim((ymin-2,ymax+2))

        ax.set_ylabel('Distance (deg)')
        ax.set_xlabel('Seconds After P')

        if shift:
            ax.set_xlabel('Seconds')
        if title != False:
           ax.set_title(title)
        if labels:
            y1, y2 = ax.get_ylim()
            ax_n = ax.twinx()
            ax_n.set_yticks(dist_list)
            ax_n.set_yticklabels(name_list)
            ax_n.set_ylim(y1,y2)
            for ii in dist_list:
                ax.axhline(ii,alpha=0.5,c='k')

        if picker == True:
            rmfile = file('./REMOVE_LIST','w')
            remove_list = []
            def on_pick(event):
                artist = event.artist
                artist.set_c('white')
                artist.set_alpha(0.0)
                remove_list.append(artist.get_label())
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', on_pick)
            plt.show()
            for tr in st:
                if tr.stats.network+'.'+tr.stats.station in remove_list:
                    st.remove(tr)
            for item in remove_list:
                rmfile.write("%s\n" % item)
            rmfile.close()

        if save == False and ax_grab == True:
            print "added_axis"
        if save != False:
            plt.savefig(save+'/section.pdf',format='pdf')
        if save == False and ax_grab == False:
            plt.show()

    def phase_plot(lim_tuple,ref_degree,evdp,phases,ax,o):
        arrivals = model.get_travel_times(distance_in_degree=ref_degree,
                   source_depth_in_km=evdp,
                   phase_list = phases)
        P = model.get_travel_times(distance_in_degree=ref_degree,
                   source_depth_in_km=evdp,
                   phase_list = phases)
        P_slow = P[0].ray_param_sec_degree
        P_time = P[0].time
        ax.axvline(0,c='b',alpha=0.7,lw=9.0)
        if len(arrivals) != 0:
            colors = ['b','r','g','b','r','g','b','r','g','b']
            name_list = []
            for idx, ii in enumerate(arrivals):
                if ii.purist_name in name_list:
                    continue
                else:
                    name_list.append(ii.purist_name)
                    p =  ii.ray_param_sec_degree - P_slow
                    if ii.name == 'PKIKPPKIKP':
                        p *= 2.0
                    time = ii.time-P_time
                    x = np.linspace(time-500,time+500)
                    y = (1/p)*(x-time)+ref_degree
                    ax.plot(x,y,alpha=0.7,label=ii.purist_name,lw=9.0)
            ax.legend()

    def add_to_axes(trace_tuple,ax):
        data = trace_tuple[0]
        time = trace_tuple[1]
        dist = trace_tuple[2]
        name = trace_tuple[3]
        az = trace_tuple[4]
        if color == True and picker == True:
            ax.plot(time,data+dist,alpha=0.7,lw=0.5,picker=5,label=name)
        if color == True and picker != True:
            ax.plot(time,data+dist,alpha=0.7,lw=0.5)
        if color != True and picker == True:
            ax.plot(time,data+dist,alpha=0.7,lw=0.5,picker=5,label=name)
        if color != True and picker != True:
            if az_color != False:
                if az_color > az:
                    ax.plot(time,data+dist,alpha=0.7,lw=0.5,c='k')
                if az_color < az:
                    ax.plot(time,data+dist,alpha=0.7,lw=0.5,c='darkgreen')
            else:
                ax.plot(time,data+dist,alpha=0.7,lw=1,c='k')
        if fill:
            ax.fill_between(time, dist, data+dist, where=data+dist <= dist,
                            facecolor='r', alpha=0.7, interpolate=True)
            ax.fill_between(time, dist, data+dist, where=data+dist >= dist,
                            facecolor='g', alpha=0.7, interpolate=True)

    def p_list_maker(st):
        p_list = []
        name_list = []
        dist_list = []
        for tr in st:
            o = tr.stats.sac['o']
            data = tr.data
            start = tr.stats.starttime+o
            end = tr.stats.endtime+o
            if align_phase == None:
                align_phase == ['Pdiff']
                p_time = o
            else:
                arrivals = model.get_travel_times(
                   distance_in_degree=tr.stats.sac['gcarc'],
                   source_depth_in_km=tr.stats.sac['evdp'],
                   phase_list = align_phase)
                p = arrivals[0]
                p_time = p.time+o
            time = np.linspace(-1*p_time,end-start-p_time,num=tr.stats.npts)
            name = (str(tr.stats.network)+'.'+str(tr.stats.station))
            name_list.append(name)
            dist_list.append(tr.stats.sac['gcarc']+tr.data[0])
            p_list.append((data,time,tr.stats.sac['gcarc'],name,
                           tr.stats.sac['az']))
        return p_list,name_list,dist_list

    def ax_limits(p_list):
        range_list = []
        for ii in p_list:
            range_list.append([ii[2],ii[1][0],ii[1][-1]])
        range_list = np.array(range_list)
        min_range = min(range_list[:,0])
        max_range = max(range_list[:,0])
        min_time = min(range_list[:,1])
        max_time = max(range_list[:,2])
        return ([min_time,max_time],[min_range-3,max_range+3])

    main()

def fft(self,**kwargs):
    '''
    plot fast fourier transform of trace object
    '''
    freqmin = kwargs.get('freqmin',0.0)
    freqmax = kwargs.get('freqmax',2.0)
    plot = kwargs.get('plot',True)

    Fs = tr.stats.sampling_rate  # sampling rate
    Ts = tr.stats.delta # sampling interval
    time = tr.stats.endtime - tr.stats.starttime
    t = np.arange(0,time,Ts) # time vector

    n = len(tr.data) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = np.fft.fft(tr.data)/n # fft computing and normalization
    Y = Y[range(n/2)]
    if tr.data.shape[0] - t.shape[0] == 1:
        t = np.hstack((t,0))

    if plot == False:
        return frq, abs(Y)
    else:

        fig, ax = plt.subplots(2, 1,figsize=(15,8))
        ax[0].plot(t,tr.data,'k')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude')
        ax[0].grid()
        ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
        ax[1].set_xlim([freqmin,freqmax])
        ax[1].set_xticks(np.arange(freqmin,freqmax,0.25))
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('|Y(freq)|')
        ax[1].grid()
        plt.show()


def slowness_stack(self,slowness):
    '''
    Stack on slowness relative to P
    '''
    def roll_zero(array,n):
        if n < 0:
            array = np.roll(array,n)
            array[n::] = 0
        else:
            array = np.roll(array,n)
            array[0:n] = 0
        return array

    slowness *= -1

    range_list = []
    for tr in st:
        range_list.append(tr.stats.sac['gcarc'])
    mn_range = np.mean(range_list)
    print 'Mean range', mn_range

    arrivals = model.get_travel_times(
               distance_in_degree=mn_range,
               source_depth_in_km=st[0].stats.sac['evdp'],
               phase_list = ['P'])

    P_slow = arrivals[0].ray_param_sec_degree

    d = st[0].stats.delta
    stack = []
    for tr in st:
        az = tr.stats.sac['gcarc']-mn_range
        shift_in_sec = (-1*P_slow+slowness)*az
        shift_in_bin = int(shift_in_sec/d)
        stack.append(roll_zero(tr.data,shift_in_bin))

    stack = np.mean(stack,axis=0)
    t = np.linspace(0,st[0].stats.endtime-st[0].stats.starttime,
                    num=len(st[0].data))
    plt.plot(t,stack)
    plt.show()
    return stack

