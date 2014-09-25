from tools import *

import sys
import csv
import nmrglue as ng
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LSQUnivariateSpline
from multiprocessing import cpu_count, Pool
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import patches, rc
from matplotlib.font_manager import FontProperties
from matplotlib.pylab import show, get_current_fig_manager, figure, cm, text, plt
import matplotlib.ticker as ticker
from multiprocessing import Pool, cpu_count
try:
    import pywt
except:
    #print 'module pywt is not installed, wavelet smoothing capabilities are disabled'
    pass


#configure fonts
rc('text.latex', preamble = \
    '\usepackage{sfmath},' \
    '\usepackage{amsmath},' )
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],'weight' : 'normal'}




def fid_from_path(path='.', varian=False, bruker=False):
    """ imports and creates a Varian (Agilent) or Bruker FID and returns an FID_array instance. """
    if path:
        if varian:
            procpar, data = ng.varian.read(path)
        if bruker:
            procpar, data = ng.bruker.read(path)
            procpar = procpar['acqus']
        fid = FID_array(data=data, procpar=procpar, path=path, varian=varian, bruker=bruker)
        return fid


class FID_array(object):
    def __init__(self, data=None, procpar=None, path=None, varian=False, bruker=False):
        """Instantiate the FID class."""
        self._data = None
        self.data = data
        self._procpar = None
        self.procpar = procpar
        self.filename = path
        self._varian = varian
        self._bruker = bruker
        if varian:
            self._f_extract_proc_varian()
        if bruker:
            self._f_extract_proc_bruker()
            #self.data = self.data[:,::-1]
        self._peaks = None
        self.peaks = []
        self.ranges = []
        self.fits = []
        self.integrals = []
        self._flags = {
            "ft"    : False,
            "gl"    : 0
        }

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if type(value) is list:
            value = np.array(value)
        #if len(value) <= 1:
        #    raise ValueError('FID data must be an iterable: %s'%str(value))
        if len(value.shape) == 1:
            value = np.array([value])
        self._data = value

    @property
    def procpar(self):
        return self._procpar

    @procpar.setter
    def procpar(self, value):
        if not value:
            raise ValueError('Parameter dictionary cannot be empty.')
        if type(value) != dict:
            raise ValueError('Paramters must be in a dictionary object, not type %s.'%type(value).__name__)
        self._procpar = value

    @property
    def peaks(self):
        return self._peaks

    @peaks.setter
    def peaks(self,value):
        #value = [np.array(i) for i in value]
        value = [i for i in value]
        self._peaks = value

    def shift_sw(self, shift):
        self.params['sw_left'] += shift

    def _f_extract_proc_varian(self):
        """Extract NMR parameters (using Varian denotations) and create a parameter dictionary 'params'."""
        at      = float(self._procpar['procpar']['at']['values'][0])
        d1      = float(self._procpar['procpar']['d1']['values'][0])
        sfrq    = float(self._procpar['procpar']['sfrq']['values'][0])
        reffrq  = float(self._procpar['procpar']['reffrq']['values'][0])
        rfp     = float(self._procpar['procpar']['rfp']['values'][0])
        rfl     = float(self._procpar['procpar']['rfl']['values'][0])
        tof     = float(self._procpar['procpar']['tof']['values'][0])
        rt      = at+d1
        nt      = np.array([self._procpar['procpar']['nt']['values']],dtype=float)
        acqtime = (nt*rt).cumsum()/60. #convert to mins.
        sw      = round(float(self._procpar['procpar']['sw']['values'][0])/float(self._procpar['procpar']['sfrq']['values'][0]),2)
        sw_hz   = float(self._procpar['procpar']['sw']['values'][0])
        sw_left = (0.5+1e6*(sfrq-reffrq)/sw_hz)*sw_hz/sfrq
        self.params = dict(
            at=at,
            d1=d1,
            rt=rt,
            nt=nt,
            acqtime=acqtime,
            sw=sw,
            sw_hz=sw_hz,
            sfrq=sfrq,
            reffrq=reffrq,
            rfp=rfp,
            rfl=rfl,
            tof=tof,
            sw_left=sw_left)
        self.t = acqtime[:len(self.data)]

    def _f_extract_proc_bruker(self):
        """Extract NMR parameters (using Varian denotations) and create a parameter dictionary 'params'."""
        d1 = self._procpar['RD']
        sfrq    = self._procpar['SFO1']
        nt = self._procpar['NS']
        sw_hz = self._procpar['SW_h']
        sw = self._procpar['SW']
        #lefthand offset of the processed data in ppm
        for i in open(self.filename+'/pdata/1/procs').readlines():
            if 'OFFSET' in i:
                sw_left = float(i.split(' ')[1])
        at = self._procpar['TD']/(2*sw_hz)
        rt = at+d1
        acqtime = (nt*rt)/60. #convert to mins.
        self.params = dict(
            at=at,
            d1=d1,
            sfrq=sfrq,
            rt=rt,
            nt=nt,
            acqtime=acqtime,
            sw_left=sw_left,
            sw=sw,
            sw_hz=sw_hz)
        self.t = np.array([acqtime]*len(self.data))

    def ui(self):
        fidplot = DataPlotter(self)
        fidplot.configure_traits()

    def zf_2(self):
        """Apply a single degree of zero-filling.

        Note: returns double-length FID array.

        """
        dz = []
        if len(self.data.shape) == 2:
            for i in self.data:
                dz.append(i.tolist()+np.zeros_like(i).tolist())
        self.data = np.array(dz)

    def zf(self):
        """Apply a single degree of zero-filling.

        Note: returns FID array extended in length to the next highest power of 2.

        """
        dz = []
        if len(self.data.shape) == 2:
            for i in self.data:
                dz.append(i.tolist()+np.zeros(2**(1+np.floor(np.log(len(i))/np.log(2)))-len(i)).tolist())#_like(i).tolist())
        self.data = np.array(dz)


    def emhz(self, lb=10.0):
        """Apply exponential line-broadening.

        lb -- degree of line-broadening in Hz.

        """
        self.data = np.exp(-np.pi*np.arange(len(self.data[0]))*(lb/self.params['sw_hz']))*self.data

    def ft(self):
        """Fourier Transform the FID array.

        Note: calculates the Discrete Fourier Transform using the Fast Fourier Transform algorithm as implemented in NumPy [1].

        [1] Cooley, James W., and John W. Tukey, 1965, 'An algorithm for the machine calculation of complex Fourier series,' Math. Comput. 19: 297-301.

        """
        if self._flags['ft']:
            return
        self._flags['ft'] = True
        data =  np.array(np.fft.fft(self.data),dtype=self.data.dtype)
        s = data.shape[-1]
        if self._varian:
            self.data = np.append(data[:,int(s/2.0)::-1], data[:,s:int(s/2.0):-1], axis=1)[:,::-1]
        if self._bruker:
            self.data = np.append(data[:,int(s/2.0)::-1], data[:,s:int(s/2.0):-1], axis=1)


    def savefids(self,filename=None):
        """Save FID array to a binary file in NumPy '.npy' format."""
        if len(self.data.shape) == 2:
            s0,s1 = self.data.shape
            data = self.data.real
            new_array = np.array([s0,s1]+data.flatten().tolist())
        if filename==None:
            filename=self.filename[:-4]
        np.save(filename,new_array)

    def loadfids(self,filename=None):
        """Load FID array from binary file in NumPy '.npy' format."""
        if filename==None:
            filename=self.filename[:-3]+'npy'
        new_array = np.load(filename)
        if np.isnan(new_array[1]):
            self.data = new_array[2:]
        else:
            self.data = new_array[2:].reshape([int(new_array[0]),int(new_array[1])])


    def wsmooth(self,level=20,thresh=2,filt='db8'):
        """Perform wavelet smoothing on FID array.

        Keyword Arguments:
        level -- decomposition level
        thresh -- smoothing threshold
        filt -- filter function

        """
        if 'pywt' not in sys.modules:
            print 'Wavelet smoothing has been disabled as the pywt module is not installed.'
            return
        data = self.data
        d = []
        if len(self.data.shape) == 2:
            for i in data:
                y = pywt.wavedec(i,wavelet=filt,level=level)
                for t in range(thresh):
                    y[-(t+1)] = y[-(t+1)]*0
                ys = pywt.waverec(y,'db8')
                d.append(ys)
        self.data = np.array(d)



    def plot_array_ppm(self, plotrange=None, index=None, sw_left=None, lw=0.5, amp=0.7, azm=-90, elv=45, bh=0.05, filled=False, plotratio=[1,1], labels=None, fs=10, x_label=r'$\deltaup$ ($^{31}$P), ppm', y_label='min', y_space=None, filename=None):
        """
        Plot FID array in 3D using the Matplotlib mplot3d toolkit.

        Keyword Arguments:
        index -- range of FID array to plot
        sw_left -- upfield boundary of spectral width
        lw -- linewidth of the FIDs in the plot
        amp -- normalised amplitude of the plot for scaling purposes
        azm -- azimuth of the plot
        elv -- elevation of the plot
        bh -- base box height in fraction of normalised data
        filled -- FIDs as filled polygons (True or False)
        plotratio -- list of XY scaling factors for plot window
        labels -- dictionary of peak labels
                e.g. {'foo': 5, 'bar': -4}

        fs -- label font size
        x_label -- x-axis label
        y_label -- y-axis label
        y_space -- y-axial displacement of x-axis label, defaults to: -0.4*acqtime[-1] where acqtime is an array of acquisition times for each FID in minutes representing the y-axis
        filename -- save file to filename (default 'None' will not save)

        """
        if len(self.data) == 1:
            print 'Error: 1D FID not suitable for 3D plotting.'
            return

        if sw_left is None:
            sw_left = self.params['sw']

        if plotrange is None:
            if sw_left is not None:
                plotrange = [sw_left,sw_left-self.params['sw']]
            else:
                plotrange = [self.params['sw'],0]

        left,right = plotrange

        if index is None:
            data_ft1 = self.data
        else:
            data_ft1 = self.data[index[0]:index[1]]

        acqtime = np.array([self.params['acqtime'][0]]*len(data_ft1)).cumsum()
        acqtime -= acqtime[0]
        sw = [sw_left, sw_left-self.params['sw']]
        if left>sw[0]:
            return 'Up-field limit of '+str(left)+' exceeds spectral width of '+str(sw[0])
        if right<sw[1]:
            return 'Down-field limit of '+str(right)+' exceeds spectral width of '+str(sw[1])
        sw = sw[::-1]
        ppm = np.mgrid[sw[0]:sw[1]:complex(0,data_ft1.shape[1])][::-1]
        left = np.where(abs(left-ppm)==abs(left-ppm).min())[0][0]
        right= np.where(abs(right-ppm)==abs(right-ppm).min())[0][0]
        data_ft = data_ft1.transpose()[left:right].transpose()
        #data_ft = data_ft/data_ft.max()
        ppm = ppm[left:right].copy()
        x = np.arange(0,data_ft.shape[1],1)
        y = np.ones_like(x)
        fig_all = figure(figsize=[18,6])
        ax_all = Axes3D(fig_all,azim=azm,elev=elv)
        if filled == False:
            for i in range(data_ft.shape[0]):
                ax_all.plot(x,(y*acqtime[i]),data_ft[i],'k',linewidth=lw,zorder=0)
        if filled == True:
            verts = []
            box_h = bh*data_ft.max()
            for i in (data_ft+box_h):#range(data_ft.shape[0]):
                #zs = i#data_ft[i]
                i[0],i[-1] = 0,0
                verts.append(zip(x, i))#zs))
            poly = PolyCollection(verts,linewidths=lw,edgecolor='k',facecolors='w')
            ax_all.add_collection3d(poly, zs=acqtime, zdir='y')

        #labelling ------------------
        if labels is not None:
            for i in labels:
                if ppm[0]>labels[i]>ppm[-1]:
                    xlbl = np.where(abs(labels[i]-ppm)==abs(labels[i]-ppm).min())[0][0]
                    ax_all.plot((xlbl,xlbl),(acqtime[-1],acqtime[-1]),(data_ft[-1][xlbl]+0.1,data_ft[-1][xlbl]+0.2),'k',linewidth=1)
                    ax_all.text(xlbl-0.009*abs(right-left),acqtime[-1], data_ft[-1][xlbl] + 0.2*max(data_ft[-1]), i, zdir=(1,0,0),rotation_mode='anchor',fontsize=fs)
        #----------------------------

        ax_all.set_xlim3d(x[0],x[-1])
        ax_all.set_ylim3d(0, acqtime[data_ft.shape[0]-1])
        ax_all.set_zlim3d(0, (1/amp)*data_ft.max())

        if np.ceil(plotrange[0]) == plotrange[0]:
            nlbls = np.arange(np.ceil(plotrange[1]),np.ceil(plotrange[0])+1)[::-1]#[1:-1]
        else:
            nlbls = np.arange(np.ceil(plotrange[1]),np.ceil(plotrange[0]))[::-1]#[1:-1]
        lbl_ind = []
        for i in nlbls:
            lbl_ind.append(np.where(abs(ppm-i) == abs(ppm-i).min())[0][0])

        lbl_ind = np.array(lbl_ind)

        nlbls = np.array(nlbls,dtype=int)
        maxlbls = 20
        minlbls = 4
        if len(nlbls)>2*maxlbls:
            ind_red = np.array(np.mgrid[0:len(nlbls)-1:complex(10)],dtype=int)
            nlbls = nlbls[ind_red]#np.array([0,int(np.ceil(len(nlbls)/2)),-1])]
            lbl_ind = lbl_ind[ind_red]#np.array([0,int(np.ceil(len(nlbls)/2)),-1])]

        if len(nlbls)>maxlbls and len(nlbls)<2*maxlbls:
            ind_red = np.array(np.mgrid[0:len(nlbls)-1:2],dtype=int)
            nlbls = nlbls[ind_red]#np.array([0,int(np.ceil(len(nlbls)/2)),-1])]
            lbl_ind = lbl_ind[ind_red]#np.array([0,int(np.ceil(len(nlbls)/2)),-1])]

        if len(nlbls)<minlbls:
            nlbls = np.array([plotrange[0],np.mean(plotrange),plotrange[1]])#
            lbl_ind = [0,data_ft.shape[1]/2,data_ft.shape[1]]

        #print nlbls,lbl_ind,right,left,plotrange#,ind_red
        #sys.stdout.flush()

        ax_all.w_xaxis.set_major_locator(ticker.FixedLocator(lbl_ind))
        ax_all.w_yaxis.set_ticks_position('top')
        ax_all.w_xaxis.set_ticklabels(nlbls)
        ax_all.w_zaxis.set_ticklabels([])
        ytcks = np.array([acqtime[0],acqtime[data_ft.shape[0]-1]/2,acqtime[data_ft.shape[0]-1]])
        ylbls = []
        for i in ytcks: ylbls.append(str(np.round(i,0)))
        ax_all.w_yaxis.set_major_locator(ticker.FixedLocator(ytcks))
        ax_all.w_yaxis.set_ticklabels(ylbls,fontsize=11)
        #ax_all.set_ylabel(y_label)
        #ax_all.set_xlabel(x_label)
        ax_all.grid(on=False)

        xts = ax_all.w_xaxis.get_tick_positions()
        xcor = (xts[1][1]-xts[1][0])/2
        for i in zip(xts[0],xts[1]):
            ax_all.plot((i[1],i[1]),(acqtime[0],acqtime[0]),(-0.01,-0.05),'k',linewidth=1)
            ax_all.text(i[1],-0.2*acqtime[-1], -0.2*ax_all.get_ylim()[-1], i[0], zdir=(1,0,0),ha='center',rotation_mode='anchor',fontsize=10)

        yts = acqtime[np.array([0,-1])].round(1)
        for i in yts:
            ax_all.text(-0.05*data_ft.shape[1],i,0,i,zdir=(1,0,0),ha='right',rotation_mode='anchor',fontsize=10)

        ax_all.text(-0.05*data_ft.shape[1],acqtime[-1]/2,0,y_label, zdir=(1,0,0),ha='center',rotation_mode='anchor',fontsize=10)
        if y_space == None:     y_space = -0.4*acqtime[-1]
        ax_all.text(0.5*data_ft.shape[1],y_space,0,x_label,zdir=(1,0,0),ha='center',rotation_mode='anchor',fontsize=10)
        ax_all.set_xlim3d(x[0],x[-1])
        ax_all.set_ylim3d(0, acqtime[data_ft.shape[0]-1])
        ax_all.set_zlim3d(0, (1/amp)*data_ft.max())
        ax_all.set_axis_off()
        if filename is not None: fig_all.savefig(filename,format='pdf')
        show()

    def plot_fid(self,index=0, sw_left=None, lw=0.5,x_label='ppm', y_label=None, labels='peaks', label_distance_frac=0.07, filename=None):
        """Plot an FID.

        Keyword arguments:
        index -- index of FID array to plot, can be an integer or a list or simply 'all'
        sw_left -- upfield boundary of spectral width
        lw -- plot linewidth
        x_label -- x-axis label
        y_label -- y-axis label
        lables -- a dictionary of peak labels of the form: {'name': ppm, ... }, or simply 'peaks' to plot the stored peak values, or None
        filename -- save file to filename (default 'None' will not save)
        """
        if index == 'all':
            index = range(len(self.data))
        if labels == 'peaks':
            if self.peaks:
                peaks = [round(i, 3) for i in self.peaks]
                labels = dict(zip([str(i) for i in peaks], peaks))
            else:
                labels = {}
        if not sw_left:
            sw_left = self.params['sw_left']
        fig = figure(figsize=[15,6])
        ax1 = fig.add_subplot(111)
        label_index = 0
        if len(self.data.shape) == 2:
            ppm = np.mgrid[sw_left-self.params['sw']:sw_left:complex(len(self.data[0]))]
            if type(index) == int:
                ax1.plot(ppm[::-1], self.data[index],'k',lw=lw)
            if type(index) == list:
                label_index = index[0]
                cl = cm.RdBu_r(np.linspace(0,1,len(index)))
                for i in np.arange(len(index))[::-1]:
                    ax1.plot(ppm[::-1], self.data[index[i]], color=cl[i], lw=lw)

        #labelling ------------------
        if labels:
            lbl_gap = ax1.get_ylim()[1]
            for i in labels:
                if ppm[0]<labels[i]<ppm[-1]:
                    xlbl = len(self.data[label_index])-np.where(abs(labels[i]-ppm)==abs(labels[i]-ppm).min())[0][0]
                    ax1.plot((labels[i], labels[i]), (self.data[label_index][xlbl]+label_distance_frac/3.0*lbl_gap, self.data[label_index][xlbl]+2/3.*label_distance_frac*lbl_gap), color='0.5')
                    ax1.text(labels[i], self.data[label_index][xlbl]+0.2*lbl_gap, i, ha='center')
        #----------------------------
        ax1.set_xlim([ppm[-1], ppm[0]])
        ax1.set_xlabel(x_label)
        ax1.set_yticklabels([])
        if y_label is not None: ax1.set_ylabel(y_label)
        if filename is not None: fig.savefig(filename,format='pdf')
        show()

    def phasespace(self,index=0, inc=50, lims=180):
        """Evaluate total integral area of specified FID over phase range of p0: [-180,180] and p1: [-180,180] in degrees. Used for deriving rough phasing parameters for spectra that are difficult to phase.

        Keyword arguments:
        index -- index of specified FID
        inc -- number of increments over which to evaluate phases
        lims -- limit to the range and domain for phase searching in degrees

        Returns a dictionary (self.ph_space) containing a 2D array of total integrated area of FID and associated parameters.
        """
        lims = [-lims, lims]
        data = self.data[index]
        phases = np.mgrid[lims[0]:lims[1]:complex(inc),lims[0]:lims[1]:complex(inc)]
        phases = np.transpose([phases[0].flatten(), phases[1].flatten()])
        d_p = np.array([sum(abs(self.ps(data,i[0],i[1]).real)) for i in phases])
        d = d_p.reshape(np.sqrt(len(d_p)),-1)
        d = d/d.max()
        self.ph_space = {'mat':d,
                        'inc':inc,
                        'lims':lims,
                        'min':list(phases[np.where(d_p==d_p.min())][0]),
                        'max':list(phases[np.where(d_p==d_p.max())][0]),
                        'index':index}
        return self.ph_space

    def plot_phasespace(self):
        """Produce a heat map plot of the phase space evaluated for a specified FID generated by self.phasepace(). Minima are indicates on the map with a 'x'.
        """
        lims = self.ph_space['lims']
        fig = figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.ph_space['mat'],interpolation='nearest',cmap=cm.RdBu_r)
        ph_min = np.array(np.where(self.ph_space['mat']==self.ph_space['mat'].min())).transpose()
        ph_r = np.mgrid[lims[0]:lims[1]:complex(self.ph_space['inc'])]
        for i in ph_min:
            ax.plot(i[1],i[0],'xk')
            ax.text(i[1]+1,i[0],str(round(ph_r[i[1]],3))+'\n'+str(round(ph_r[i[0]],3)),color='k',size=10,va='center')
        ftick = self.ph_space['mat'].shape[0]-1
        ax.set_yticks([0,ftick])
        ax.set_xticks([0,ftick])
        ax.set_yticklabels(lims)
        ax.set_xticklabels(lims)
        ax.set_ylabel('p1')
        ax.set_xlabel('p0')
        ax.set_ylim([-.5,ftick+0.5])
        ax.set_xlim([-.5,ftick+0.5])
        ax.set_title('phasespace for FID '+str(self.ph_space['index']))
        show()


    def phase_manual(self, index=0, universal=False, norm=False, discard_imaginary=True):
        """Instantiate a widget for manual phasing of specified FID.

        Keyword arguments:
        index -- index of FID to phase
        universal -- phase a single spectrum and apply parameters to all spectra
        norm -- normalise data

        Note: left-click - phase p0, right-click - phase p1.
        """
        if np.sum(np.iscomplex(self.data) == False) > 0:
            for i in self.data[np.iscomplex(self.data) == False]: #as np.iscomplex returns False for 0+0j, we need to check manually
                if type(i) != np.complex128:
                    print "Cannot perform phase correction on non-imaginary data."
                    return
        if norm:
            self.data = self.data/self.data.max()
        self.phases = []
        if len(self.data.shape) == 2:
            if universal is False:
                for data in self.data:
                    self.phaser = Phaser(data)
                    self.phases.append(list(self.phaser.phases))
            else:
                data = self.data[index]
                self.phaser = Phaser(data)
                self.phases = self.phaser.phases
                self.phases = [list(self.phases)]*len(self.data)

        self.phaser = None #this instance must be destroyed to be able to pickle the FID instance for multiprocessing
        self._phase_all_data_using_phases()
        if discard_imaginary:
            self.real()

    def real(self):
        """Discard imaginary component of data."""
        self.data = np.real(self.data)

    def _phase_all_data_using_phases(self):
        self.data = np.array([self.ps(i[0], p0=i[1][0], p1=i[1][1]) for i in zip(self.data, self.phases)])

    def phase_auto(self, method='area', thresh=0.0, mp=True, discard_imaginary=True):
        """ Automatically phase array of spectra.


            Keyword arguments:
            method -- phasing method, the available options are:
                        area     - minimise total integral of spectrum
                        neg      - minimise negative area of spectrum
                        neg_area - a combination of the previous two methods
            thresh -- threshold below which to consider data as signal and not noise (typically negative or 0), used by the 'neg' method
            mp     -- multiprocessing, parallelise the phasing process over multiple processors, significantly reduces computation time
            discard_imaginary -- discards imaginary component of complex values after phasing
        """
        if np.sum(np.iscomplex(self.data) == False) > 0:
            for i in self.data[np.iscomplex(self.data) == False]: #as np.iscomplex returns False for 0+0j, we need to check manually
                if type(i) != np.complex128:
                    print "Cannot perform phase correction on non-imaginary data."
                    return

        if method == 'area':
            if mp:
                self._phase_area_mp()
            else:
                self._phase_area()

        if method == 'neg':
            if mp:
                self._phase_neg_mp(thresh=thresh, )
            else:
                self._phase_neg(thresh=thresh, )

        if method == 'neg_area':
            if mp:
                self._phase_neg_area_mp(thresh=thresh, )
            else:
                self._phase_neg_area(thresh=thresh, )
        if discard_imaginary:
            self.real()

    def _phase_area_single(self, n):
        def err_ps(pars, data):
            err = self.ps(data, pars[0], pars[1], inv=False).real
            return np.array([abs(err).sum()]*2)

        phase = leastsq(err_ps, [1.0, 0.0], args=(self.data[n]), maxfev=10000)[0]
        self.data[n] = self.ps(self.data[n], phase[0], phase[1])
        for i in range(len(self.data)):
            if abs(self.data[i].min()) > abs(self.data[i].max()):
                self.data[i] *= -1
        print '%i\t%d\t%d'%(n, phase[0], phase[1])
        return self.data[n]

    def _phase_neg_single(self, n):
        def err_ps(pars, data):
            err = self.ps(data, pars[0], pars[1], inv=False).real
            return np.array([err[err<self._thresh].sum()]*2)

        phase = leastsq(err_ps, [1.0, 0.0], args=(self.data[n]), maxfev=10000)[0]
        self.data[n] = self.ps(self.data[n], phase[0], phase[1])
        for i in range(len(self.data)):
            if abs(self.data[i].min()) > abs(self.data[i].max()):
                self.data[i] *= -1
        print '%i\t%d\t%d'%(n, phase[0], phase[1])
        return self.data[n]

    def _phase_neg_area_single(self, n):
        def err_ps(pars, data):
            err = self.ps(data, pars[0], pars[1], inv=False).real
            err = np.array(2*[abs(err).sum() + abs(err[err<self._thresh]).sum()])
            return err
        phase = leastsq(err_ps, [1.0, 0.0], args=(self.data[n]), maxfev=10000)[0]
        self.data[n] = self.ps(self.data[n], phase[0], phase[1])
        for i in range(len(self.data)):
            if abs(self.data[i].min()) > abs(self.data[i].max()):
                self.data[i] *= -1
        print '%i\t%d\t%d'%(n, phase[0], phase[1])
        return self.data[n]



    """
    Note that the following function has to use a top-level global function '_unwrap_fid' to parallelise as it is a class method
    """
    def _phase_area_mp(self, ):
        print 'fid\tp0\tp1'
        proc_pool = Pool()
        self.data = np.array(proc_pool.map(_unwrap_fid_area, zip([self]*len(self.data), range(len(self.data)))))

    def _phase_neg_mp(self, thresh=0.0, ):
        print 'fid\tp0\tp1'
        self._thresh = thresh
        proc_pool = Pool()
        self.data = np.array(proc_pool.map(_unwrap_fid_neg, zip([self]*len(self.data), range(len(self.data)))))

    def _phase_neg_area_mp(self, thresh=0.0, ):
        print 'fid\tp0\tp1'
        self._thresh = thresh
        proc_pool = Pool()
        self.data = np.array(proc_pool.map(_unwrap_fid_neg_area, zip([self]*len(self.data), range(len(self.data)))))


    def _phase_area(self):
        print 'fid\tp0\tp1'
        data = self.data
        data_ph = np.ones_like(data)
        self.phases = []
        def err_ps(p0,data):
            err = self.ps(data,p0[0],p0[1],inv=False).real
            return np.array([abs(err).sum()]*2)
        for i in range(data.shape[0]):
            p0 = [1.0,0.0]
            phase = leastsq(err_ps,p0,args=(data[i]),maxfev=10000)[0]
            self.phases.append(phase)
            data_ph[i] = self.ps(data[i],phase[0],phase[1])
            print '%i\t%d\t%d'%(i, phase[0], phase[1])
            sys.stdout.flush()
            #if data_ph[i].mean() < 0:
            #    data_ph[i] = -data_ph[i]
        self.data = data_ph

    def _phase_neg(self, thresh=0.0, ):
        print 'fid\tp0\tp1'
        data = self.data
        data_ph = np.ones_like(data)
        self.phases = []
        def err_ps(p0,data):
            err = self.ps(data,p0[0],p0[1],inv=False).real
            return np.array([err[err<thresh].sum()]*2)
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                p0 = [0,0]
                phase = leastsq(err_ps,p0,args=(data[i]),maxfev=10000)[0]
                self.phases.append(phase)
                data_ph[i] = self.ps(data[i],phase[0],phase[1])
                print '%i\t%d\t%d'%(i, phase[0], phase[1])
                sys.stdout.flush()
                if data_ph[i].mean() < 0:       data_ph[i] = -data_ph[i]

    def _phase_neg_area(self, thresh=0.0, ):
        print 'fid\tp0\tp1'
        data = self.data
        data_ph = np.ones_like(data)
        self.phases = []
        def err_ps(p0,data):
            err = self.ps(data,p0[0],p0[1],inv=False).real
            err = np.array(2*[abs(err).sum() + abs(err[err<thresh]).sum()])
            return err #np.array([abs(err).sum(),abs(err[err<thresh]).sum()])
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                p0 = [0,0]
                phase = leastsq(err_ps,p0,args=(data[i]),maxfev=10000)[0]
                self.phases.append(phase)
                data_ph[i] = self.ps(data[i],phase[0],phase[1])
                print '%i\t%d\t%d'%(i, phase[0], phase[1])
                sys.stdout.flush()
                #if data_ph[i].mean() < 0:      data_ph[i] = -data_ph[i]
                if abs(data_ph[i].min()) > abs(data_ph[i].max()):       data_ph[i] = -data_ph[i]



    def baseline_correct(self,index=0, deg=2):
        """Instantiate a widget to select points for baseline correction which are stored in self.bl_points.

        Keyword arguments:
        index -- index of FID array to use for point selection.

        Note: left-click selects point, right-click deselects point
        """
        if len(self.data.shape) == 2:
            data = self.data[index]
        self.baseliner = Baseliner(data)
        self.bl_points = np.array(self.baseliner.xs,dtype='int32')
        self.bl_fit(deg=deg)
        self.baseliner = None #this instance must be destroyed to be able to pickle the FID instance for multiprocessing

    def bl_fit(self,deg=2):
        """Perform baseline correction by fitting specified baseline points (stored in self.bl_points) with polynomials of specified degree (stored in self.bl_polys) and subtract these polynomials from the respective FIDs.

        Keyword arguments:
        deg -- degree of fitted polynomial
        """
        data = self.data.real
        if len(self.data.shape) == 2:
            x = np.arange(len(data[0]),dtype='f')
        m = np.ones_like(x)
        m[self.bl_points] = 0
        d = []
        self.bl_polys = []
        if len(self.data.shape) == 2:
            for i in data:
                ym = np.ma.masked_array(i,m)
                xm = np.ma.masked_array(x,m)
                p = np.ma.polyfit(xm,ym,deg)
                yp = sp.polyval(p, x)
                self.bl_polys.append(yp)
                d.append(i-yp)
        self.data = np.array(d)

    def peakpicker(self,index=0):
        """Instantiate a peak-picker widget.

        Keyword arguments:
        index -- index of FID array to use for peak selection.

        Note: left-click - select a point, right-click - begin a dragging selection of a fitting range. Only peaks included in selected fitting ranges will be retained.
        """
        data = self.data[index]
        self.picker = SpanSelector(data, self.params)
        self.ranges = self.picker.ranges
        if self.ranges == []:
            self.ranges = [[self.params['sw_left']-self.params['sw'], self.params['sw_left']]]
        xs = np.array(self.picker.peaks)
        xs.sort()
        peaks = []
        for i in self.ranges:
            x = xs[xs>=i[0]]
            #peaks.append(np.array(x[x<=i[1]]))
            peaks.append(list(x[x<=i[1]]))
        self.peaks = peaks
        self.picker = None #this instance must be destroyed to be able to pickle the FID instance for multiprocessing

    def _convert_peaklist_to_ppm(self):
        """ Convert a peaklist of array indices to ppm"""
        if not self.peaks:
            print "No peaks to convert."
            return
        self.peaks = self.conv_to_ppm(self.data[0], self.peaks, self.params['sw_left'], self.params['sw'])
        self.ranges= [self.conv_to_ppm(self.data[0], i, self.params['sw_left'], self.params['sw'])  for i in self.ranges]


    def _convert_peaklist_to_index(self):
        """ Convert a peaklist of ppm generated by peakpicker() to array indices"""
        if not self.peaks:
            print "No peaks to convert."
            return
        self.peaks = self.conv_to_index(self.data[0], self.peaks, self.params['sw_left'], self.params['sw'])
        self.ranges= [self.conv_to_index(self.data[0], i, self.params['sw_left'], self.params['sw'])  for i in self.ranges]

    def grouped_peaklist(self):
        return [[peak for peak in self.peaks if peak > peak_range[0] and peak < peak_range[1]] for peak_range in self.ranges] 


    def deconv(self, gl=None, mp=True):
        """Deconvolute array of spectra (self.data) using specified peak positions (self.peaks) and ranges (self.ranges) by fitting the data with combined Gaussian/Lorentzian functions. Uses the Levenberg-Marquardt least squares algorithm [1] as implemented in SciPy.optimize.leastsq.

            Keyword arguments:
            gl -- ratio of peak function to be Gaussian (1 -- pure Gaussian, 0 -- pure Lorentzian)
            mp     -- multiprocessing, parallelise the deconvlution process over multiple processors, significantly reduces computation time


            [1] Marquardt, Donald W. 'An algorithm for least-squares estimation of nonlinear parameters.' Journal of the Society for Industrial & Applied Mathematics 11.2 (1963): 431-441.
        """
        self.real()
        self._convert_peaklist_to_index()
        self.data = self.data[:, ::-1]
        if mp:
            self._deconv_mp(gl=gl)
        else:
            self._deconv(gl=gl)
        self.data = self.data[:, ::-1]
        self._convert_peaklist_to_ppm()
        print 'done!'


    def _deconv_single(self, n):
        fit = self._deconv_datum(self.data[n], self.grouped_peaklist(), self.ranges, self._flags['gl'])
        print 'fit %i/%i'%(n+1, len(self.data))
        return fit
         

    def _deconv_mp(self, gl=None):
        self._flags['gl'] = gl
        proc_pool = Pool()
        data_zip = zip([self]*len(self.data), range(len(self.data)))
        fits = proc_pool.map(_unwrap_fid_deconv, data_zip)
        self.fits = np.array(fits)
        self.integrals = f_integrals_array(self.data,self.fits)
        #self.integrals = np.array([list(i) for i in self.integrals])
        #return f


    def _deconv(self, gl=None):
        data = self.data
        peaks = self.grouped_peaklist()
        ranges = self.ranges
        fits = []
        if len(data.shape)==2:
            for i in data:
                fits.append(self._deconv_datum(i, peaks, ranges, gl))
                print 'fit %i/%i'%(len(fits), len(self.data))
            self.fits = np.array(fits)
            self.integrals = f_integrals_array(self.data,self.fits)

    @staticmethod
    def _deconv_datum(i, peaks , ranges, gl):
        fit = []
        for j in zip(peaks,ranges):
            d_slice = i[j[1][0]:j[1][1]]
            p_slice = j[0]-j[1][0]
            f=f_fitp(d_slice, p_slice, gl)[0]
            f = np.array(f).transpose()
            f[0] += j[1][0]
            f = f.transpose()
            fit.append(f)
        return fit

    def plot_deconv(self, index=0, txt=True, sw_left=None, x_label='ppm', filename=None):
        """Generate a plot of data with fitted peaks.

        Keyword arguments:
        index -- index of data array to plot
        txt -- print peak number on plot (True/False)
        sw_left -- upfield boundary of the spectral width
        x_label -- x-axis label
        filename -- filename to save image under

        Returns:
        Plot of data (black), fitted Gaussian/Lorentzian peakshapes (blue), residual (red).

        """

        if sw_left == None:
            sw_left = self.params['sw_left']

        data = self.data[index][::-1]
        paramarray = self.fits[index]

        def i2ppm(index_value):
            return np.mgrid[sw_left-self.params['sw']:sw_left:complex(len(data))][index_value]

        def peaknum(paramarray,peak):
            pkr = []
            for i in paramarray:
                for j in i:
                    pkr.append(np.array(j-peak).sum())
            return np.where(np.array(pkr)==0.)[0][0]
        cl = ['#336699','#999966','#990000']
        x = np.arange(len(data))
        ppm = np.mgrid[sw_left-self.params['sw']:sw_left:complex(len(x))]#[::-1]
        fig = figure(figsize=[15,6])
        ax = fig.add_subplot(111)
        peakplots = 0
        for irange in paramarray:
            for ipeak in irange:
                pk = f_pk(ipeak,x)
                #ax.plot(ppm, pk/max(data), color='0.5', linewidth=1)
                ax.plot(ppm, pk, color='0.5', linewidth=1)
                if txt:
                    text(i2ppm(int(ipeak[0])), 0.1+pk.max(), str(peaknum(paramarray,ipeak)), color='#336699')
                peakplots += f_pk(ipeak,x)

        #these plots are all normalised
        ax.plot(ppm,(data-peakplots), color=cl[2], linewidth=1)
        ax.plot(ppm,data, color='k', linewidth=1)
        ax.plot(ppm,peakplots, '-', color=cl[0], linewidth=1)
        ax.invert_xaxis()
        ax.set_xlabel(x_label)
        ax.set_xlim([ppm[-1], ppm[0]])
        box1 = ax.get_position()
        ax.set_position([box1.x0, box1.y0+0.05*box1.height, box1.width, box1.height * 0.95])
        while ax.get_yticks()[0] < 0.0:
            ax.set_yticks(ax.get_yticks()[1:])
        if filename is not None: fig_all.savefig(filename,format='pdf')
        show()

    @staticmethod
    def plot_dict(x, d, index='all', title=None, xlabel='', ylabel='', ylim=None, fmt=None, cl=None, filename=None, text=None):
        if index == 'all':
            index = d.keys()
        if not cl:
            cl = dict(zip(d, plt.cm.Set1(np.linspace(0,1,len(d)))))
        if not fmt:
            fmt = {i:'o' for i in index}
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i in index:
            ax1.plot(x, d[i], fmt[i], color=cl[i], lw=2, label=i)
    
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.grid()
        if title:
            ax1.set_title(title)
        if text:
            ax1.text(0.6*ax1.get_xlim()[1], 0.05*ax1.get_ylim()[1], text)
        if ylim:
            ax1.set_ylim([0,ylim])
        if filename:
            fig.savefig(filename, format='pdf')
        plt.show()

    def plot_integrals(self, index='all'):
        self.integral_dict = dict(zip(['%0.3f'%i for i in self.peaks], self.integrals.transpose()))
        x = self.t[:len(self.integrals.transpose()[0])]
        self.plot_dict(x, self.integral_dict, index=index, title='peak integrals', xlabel='time')
        




    def integrate(self,index=None,ranges=None):
        """
        Perform a simple 'box integration' over a specified index using the ranges stored in self.ranges.

        Keyword arguments:
        index -- indices of spectra to integrate, can be a single value or a list: [low,high]

        Returns an array of integrals by spectrum.
        """
        if ranges is None:
            if len(self.data.shape) == 2 and index is None:
                return np.array([[sum(j[i[0]:i[1]]) for i in self.ranges] for j in self.data])
            if len(self.data.shape) == 2 and index is not None:
                if isinstance(index,int):               return np.array([sum(self.data[index][i[0]:i[1]]) for i in self.ranges])
                if isinstance(index,list):      return np.array([[sum(self.data[j][i[0]:i[1]]) for i in self.ranges] for j in range(index[0],index[1])])

        if ranges is not None:
            if len(self.data.shape) == 2 and index is None:
                return np.array([[sum(j[i[0]:i[1]]) for i in [ranges]] for j in self.data])
            if len(self.data.shape) == 2 and index is not None:
                if isinstance(index,int):               return np.array([sum(self.data[index][i[0]:i[1]]) for i in [ranges]])
                if isinstance(index,list):      return np.array([[sum(self.data[j][i[0]:i[1]]) for i in [ranges]] for j in range(index[0],index[1])])




    def f_integral_sums(self,names,peak_index):
        """Generate integrals for species by summing together individual NMR peaks (stored as self.integrals_sum).

        Keyword arguments:
        names -- list of names of species e.g. ['species1','species2','species3']
        peak_index -- list of peaks to sum per species e.g. [[0,2],[1],[3,4,5]]


        """
        ints = self.integrals
        if len(ints.shape) > 1:
            integrals_sum = []
            for fid in ints:
                integral = []
                for i in peak_index:
                    integral.append(fid[i].sum())
                integrals_sum.append(integral)
            self.integrals_sum = np.array(integrals_sum).transpose()
            self.integral_names = names
        elif len(ints.shape) == 1:
            integrals_sum = []
            for i in peak_index:
                integrals_sum.append(ints[i].sum())
            self.integrals_sum = np.array(integrals_sum)
            self.integral_names = names


    def f_splines(self,s=1000,k=3):
        """
        Use B-splines to approximate the summed integral time series (self.integrals_sum) as implemented in the SciPy.interpolate module.

        Keyword arguments:
        s -- The smoothing factor of the splines.
        k -- The order of the splines (can be a list to specify individual orders for species)
        d -- The order of derivative of the splines to compute (must be less than or equal to k)

        """
        if len(self.data.shape) == 1:
            print 'Error: 1D data not suitable for spline-fitting.'
            sys.stdout.flush()
            return
        if len(self.t) is not len(self.data):   self.t = self.t[:len(self.data)]
        x = self.t
        splines = []
        rates = []
        concs = []
        for i in range(len(self.integrals_sum)):
            if isinstance(k,int) is True:   spln = UnivariateSpline(x, self.integrals_sum[i], s=s,k=k)
            elif isinstance(k,int) is False:        spln = UnivariateSpline(x, self.integrals_sum[i], s=s,k=k[i])
            ys = spln(x)
            ds = []
            for j in x:
                ds.append(spln.derivatives(j)[1])
            splines.append(spln)
            rates.append(ds)
            concs.append(ys)
        self.s_splines = np.array(splines)
        self.s_rates = np.array(rates)
        self.s_concs = np.array(concs)


    def f_plot_splines(self,x_label1='min',x_label2='min',y_label1='[mM]',y_label2='r'):
        """Plot spline fits generated using self.f_splines().

        """
        if len(self.data.shape) == 1:
            print 'Error: 1D data not suitable for spline-fitting.'
            sys.stdout.flush()
            return
        integrals_sum = self.integrals_sum
        integral_names = np.array(self.integral_names)
        s_concs = self.s_concs
        s_rates = self.s_rates
        cl = cm.Paired(np.mgrid[0:1:np.complex(len(self.integrals_sum))])

        fig = figure(figsize=(10,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for i in range(len(integrals_sum)):
            ax1.plot(self.t,integrals_sum[i],'s',color=cl[i],label=integral_names[i])
            ax1.plot(self.t,s_concs[i],'-',lw=2,color=cl[i])
            ax2.plot(self.t,s_rates[i],'-',lw=2,color=cl[i],label=integral_names[i])
        ax1.grid()
        ax2.grid()
        ax2.set_xlim([self.t[0],self.t[-1]])
        # Shink current axis by 20%
        box1 = ax1.get_position()
        box2 = ax2.get_position()
        ax1.set_position([box1.x0, box2.y0+0.05*box1.height, box1.width * 0.85, box1.height * 0.95])
        ax2.set_position([box1.x0+1.1*box1.width, box2.y0+0.05*box1.height, box2.width * 0.85, box2.height * 0.95])
        # Put a legend to the right of the current axis
        leg = ax2.legend(loc='center left', bbox_to_anchor=(1., 0.5),fancybox=True,shadow=True)
        for i in range(len(leg.legendHandles)):
            leg.legendHandles[i].set_linewidth(2)

        ax1.set_xlabel(x_label1)
        ax1.set_ylabel(y_label1)
        ax2.set_xlabel(x_label2)
        ax2.set_ylabel(y_label2)
        #fig.subplots_adjust(bottom=0.15)

        show()

    def f_polys(self,deg=5):
        """Fit integral time series with polynomial of specified degree.

        Keyword arguments:
        deg -- degree of polynomial to fit

        """
        data = self.integrals_sum
        x = np.arange(data.shape[1],dtype='f')
        d = []
        for y in data:
            p = np.polyfit(x,y,deg)
            d.append(sp.polyval(p, x))
        self.polys = np.array(d)


    def f_save(self,concs,rate,names,filename=None):
        """Save spline-approximated concentration/rate data as text file.

        Keyword arguments:
        concs -- concentration time series
        rate -- rate data
        names -- names of species
        filename -- filename to save data under

        """
        if len(self.data.shape) == 1:
            print 'Error: 1D data not suitable for spline-fitting.'
            sys.stdout.flush()
            return
        concs = np.array(concs)
        concs[np.where(concs<0)] = 1e-6
        concs = concs.tolist()
        concs.append(rate)
        a = np.array(concs)
        print a
        if filename==None:
            filename=self.filename
            f = open(filename[:-3]+'txt','wb')
        elif filename is not None:
            f = open(filename,'wb')
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(names)
        writer.writerows(a.transpose())
        f.close()




    #GENERAL FUNCTIONS
    @staticmethod
    def ps(data,p0=0.0,p1=0.0,inv=False):

        """
        Linear Phase Correction

        Parameters:

        * data  Array of spectral data.
        * p0    Zero order phase in degrees.
        * p1    First order phase in degrees.

        """
        p0 = p0*np.pi/180. # convert to radians
        p1 = p1*np.pi/180.
        #size = data.shape[-1]
        if len(data.shape)==2:
            size = float(len(data[0]))
        if len(data.shape)==1:
            size = float(len(data))
        ph = np.exp(1.0j*(p0+(p1*np.arange(size)/size)))
        return ph*data

    @staticmethod
    def conv_to_ppm(data, index, sw_left, sw):
        if type(index) is list:
            index = np.array(index)
        frc_sw = index/float(len(data))
        return list(sw_left-sw+frc_sw*sw)

    @staticmethod
    def conv_to_index(data, ppm, sw_left, sw):
        if type(ppm) is list:
            ppm = np.array(ppm)
        frc_sw = (ppm+(sw-sw_left))/sw
        return list(np.array(frc_sw*len(data), int))



def f_pk(p,x):
    """Return the evaluation of a combined Gaussian/3-parameter Lorentzian function for deconvolution.

    Keyword arguments:
    p -- parameter list: [spectral offset (x), gauss: 2*sigma**2, gauss: amplitude, lorentz: scale (HWHM), lorentz: amplitude, fraction of function to be Gaussian (0 -> 1)]
    x -- array of equal length to FID
    Note: specifying a Gaussian fraction of 0 will produce a pure Lorentzian and vice versa.
    """
    fgss  = lambda p,x: p[2]*np.exp(-(p[0]-x)**2/p[1])
    flor = lambda p,x: p[2]*p[1]**2/(p[1]**2+4*(p[0]-x)**2)
    f = p[-1]*fgss(p[np.array([0,1,2])],x)+(1-p[-1])*flor(p[np.array([0,3,4])],x)
    return f

def f_pks(p,x):
    """Return the sum of a series of peak evaluations for deconvolution. See f_pk().

    Keyword arguments:
    p -- parameter list: [spectral offset (x), gauss: 2*sigma**2, gauss: amplitude, lorentz: scale (HWHM), lorentz: amplitude, fraction of function to be Gaussian (0 -> 1)]
    x -- array of equal length to FID
    """
    peaks = x*0
    for i in p:
        f = f_pk(i,x)
        peaks+=f
    return peaks

def f_res(p,data,gl):
    """Objective function for deconvolution. Return error of the devonvolution fit.

    Keyword arguments:
    p -- parameter list: [spectral offset (x), gauss: 2*sigma**2, gauss: amplitude, lorentz: scale (HWHM), lorentz: amplitude, fraction of function to be Gaussian (0 -> 1)]


    """
    if len(p.shape) < 2: p = p.reshape([-1,6])
    p = abs(p)      # forces positive parameter values
#===================================
    if gl is not None:
        p = p.transpose()#constrain p
        p[-1] = p[-1]*0+gl
        p = p.transpose()
#===================================
    x = np.arange(len(data),dtype='f8')
    res = data-f_pks(p,x)
    err = 0
    for i in p:
        if i[-1] > 1:   err += (i[-1]-1)        # this constrains the lor/gauss ratio <= 1
    return res*(1+err)

def f_makep(data,peaks):
    """Make a set of initial peak parameters for deconvolution.

    Keyword arguments:
    data -- data to be fitted
    peaks -- selected peak positions (see peakpicker())


    """
    p = []
    for i in peaks:
        SINGLE_PEAK = [i,10,data.max()/2,10,data.max()/2,0.5]
        p.append(SINGLE_PEAK)
    return np.array(p)

def f_fitp(data,peaks,gl):
    """Fit a section of spectral data with a combination of Gaussian/Lorentzian peak for deconvolution.

    Keyword arguments:
    data -- data to be fitted, 1D array
    peaks -- selected peak positions (see peakpicker())
    gl -- fraction of fitted function to be Gaussian (1 - Guassian, 0 - Lorentzian)

    Note: peaks are fitted using the Levenberg-Marquardt algorithm as implemented in SciPy.optimize [1].

    [1] Marquardt, Donald W. 'An algorithm for least-squares estimation of nonlinear parameters.' Journal of the Society for Industrial & Applied Mathematics 11.2 (1963): 431-441.
    """
    p               = f_makep(data,peaks)
    init_ref                = f_conv(p,data)
    p               = f_makep(data,peaks+init_ref)
    p               = p.flatten()
    fit             = leastsq(f_res,p,args=(data,gl),full_output=1)
    fits            = np.array(abs(fit[0].reshape([-1,6])))#.transpose()
    #===================================
    if gl is not None:
        fits = fits.transpose()
        fits[-1] = fits[-1]*0+gl
        fits = fits.transpose()
    #===================================
    return fits,fit[1]

def f_conv(p,data):
    """Returns the maximum of a convolution of an initial set of lineshapes and the data to be fitted.

    Keyword arguments:
    p -- parameter list: [spectral offset (x), gauss: 2*sigma**2, gauss: amplitude, lorentz: scale (HWHM), lorentz: amplitude, fraction of function to be Gaussian (0 -> 1)]
    data -- data to be fitted

    """
    data[data==0.]  = 1e-6
    x = np.arange(len(data),dtype='f8')
    peaks_init = f_pks(p,x)
    data_convolution = np.convolve(data,peaks_init[::-1])
    auto_convolution = np.convolve(peaks_init,peaks_init[::-1])
#       init_ref = np.where(data_convolution == data_convolution_max)[0][0]/-len(data)
    init_ref = np.where(data_convolution == data_convolution.max())[0][0]-np.where(auto_convolution == auto_convolution.max())[0][0]
    return init_ref

def f_integrals(data,params):
    """Returns the integrals of a series of lineshapes.

    Keyword arguments:
    data -- data to be fitted
    params -- fitted peak parameters

    """
    x = np.arange(len(data))
    ints = []
    for irange in params:
        for ipeak in irange:
            ints.append(f_pk(ipeak,x).sum())
    return np.array(ints)

def f_integrals_array(data_array,param_array):
    """Returns the integrals of a series of lineshapes for a whole array of spectra.

    Keyword arguments:
    data_array -- array of data to be fitted
    param_array -- array fitted peak parameters

    """

    ints = []
    if len(data_array.shape)==2:
        for idata_params in zip(data_array,param_array):
            ints.append(f_integrals(idata_params[0],idata_params[1]))
    return np.array(ints)


#CLASSES
from matplotlib.mlab import dist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Cursor

class Baseliner(object):
    def __init__(self, data):
        fig = figure(figsize=[15,6])
        self.data = data
        self.x = np.array([])
        self.ax = fig.add_subplot(111)
        self.ax.plot(self.data,color='0.3',lw=1)
        self.xs = [0,len(self.data)-1]
        self.ax.plot(self.xs,self.data[np.array(self.xs)],'o',color='#CC0000')
        self.ax.hlines(0,0,len(self.data)-1)
        self.visible = True
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.buttonDown = False
        cursor = Cursor(self.ax, useblit=True,color='k', linewidth=0.5 )
        cursor.horizOn = False
        self.ax.set_xlim([0,len(self.data)])
        self.ax.text(0.05*self.ax.get_xlim()[1],0.9*self.ax.get_ylim()[1],'Baseline correction\nLeft - select points')
        show()


    def press(self, event):
        tb = get_current_fig_manager().toolbar
        if tb.mode == '':
            self.xs.sort()
            x,y = event.xdata,event.ydata
            self.buttonDown = True
            self.button = event.button
            self.x = x
            if event.inaxes is not None and (x>=0) and (x<=len(self.data)-1) and event.button==1:
                if 1*(np.array(self.xs)[np.array(self.xs)==x]).sum() == 0:      #test for previous inclusion of current datum
                    self.xs.append(int(x))
            self.ax.lines[1].set_data(np.array(self.xs),self.data[np.array(self.xs,dtype='int32')])
            self.canvas.draw_idle()

    def release(self, event):
        self.buttonDown = False
        self.xs.sort()
        return False

    def onmove(self, event):
        if self.buttonDown is False or event.inaxes is None: return
        x = event.xdata
        inc = len(self.data)/200
        if event.inaxes is not None and (x>=0) and (x<=len(self.data)-1) and self.button==1:
            if 1*(np.array(self.xs)[np.array(self.xs)==x]).sum() == 0:      #test for previous inclusion of current datum
                self.xs.append(int(x))
                self.ax.lines[1].set_data(np.array(self.xs),self.data[np.array(self.xs,dtype='int32')])
        if event.inaxes is not None and (x>=inc) and (x<=len(self.data)-inc) and self.button==3:
            VPRES_low = np.where((x-inc)<np.array(self.xs))[0]
            VPRES = VPRES_low[np.where(np.array(self.xs)[VPRES_low]<(x+inc))[0]]
            for i in VPRES: self.xs.pop(i)
            self.ax.lines[1].set_data(np.array(self.xs),self.data[np.array(self.xs,dtype='int32')])
        self.canvas.draw_idle()
        return False

class Phaser(object):
    def __init__(self, data):
        self.ps = FID_array.ps
        fig = figure(figsize=[15,6])
        self.data = data.copy()
        self.datanew = data.copy()
        self.phases = np.array([0.,0.],dtype='f')
        self.y = 0
        self.ax = fig.add_subplot(111)
        self.ax.plot(self.datanew,color='0.3',linewidth=1.0)
        self.ax.hlines(0,0,len(self.datanew)-1)
        self.visible = True
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.pressv = None
        self.buttonDown = False
        self.prev = (0, 0)
        self.ranges = []
        ylims = np.array([-1,1])*np.array([max(self.ax.get_ylim())]*2)
        self.ax.set_ylim(ylims)
        self.ax.set_xlim([0,len(self.data)])
        self.ax.text(0.05*self.ax.get_xlim()[1],0.8*self.ax.get_ylim()[1],'phasing\nleft - zero-order\nright - first order')
        cursor = Cursor(self.ax, useblit=True,color='k', linewidth=0.5 )
        cursor.horizOn = False
        show()


    def press(self, event):
        tb = get_current_fig_manager().toolbar
        if tb.mode == '':
            x,y = event.xdata, event.ydata
            if event.inaxes is not None:
                self.buttonDown = True
                self.button = event.button
                self.y = y
                print y

    def release(self, event):
        self.buttonDown = False
        #self.data = self.datanew
        return False

    def onmove(self, event):
        if self.buttonDown is False or event.inaxes is None: return
        x = event.xdata
        y = event.ydata
        dy = y-self.y
        self.y = y
        if self.button == 1:
            self.phases[0] = self.phases[0] + 50*dy/self.ax.get_ylim()[1]
        if self.button == 3:
            self.phases[1] = self.phases[1] + 50*dy/self.ax.get_ylim()[1]
        self.datanew = self.ps(self.data,p0=self.phases[0],p1=self.phases[1])
        self.ax.lines[0].set_data(np.array([np.arange(len(self.datanew)),self.datanew]))
        self.canvas.draw()#_idle()
        print 'p0: '+str(self.phases[0])+'\t'+'p1: '+str(self.phases[1])
        return False

class SpanSelector:
    def __init__(self, data, params):
        fig = figure(figsize=[15,6])
        self.data = data
        self.ax = fig.add_subplot(111)
        ppm = np.mgrid[params['sw_left']-params['sw']:params['sw_left']:complex(len(data))]
        self.ax.plot(ppm[::-1], data, color='0.3', lw=1)#color='#3D3D99', lw=1)
        self.ax.set_xlabel('ppm')
        self.rectprops = dict(facecolor='0.5', alpha=0.2)
        self.visible = True
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.minspan = 0
        self.rect = None
        self.pressv = None
        self.buttonDown = False
        self.prev = (0, 0)
        trans = blended_transform_factory(self.ax.transData, self.ax.transAxes)
        w,h = 0,1
        self.rect = Rectangle( [0,0], w, h,
                                           transform=trans,
                                           visible=False,
                                           **self.rectprops
                                           )
        self.ax.add_patch(self.rect)
        self.ranges = []
        self.peaks = []
        self.ylims = np.array([self.ax.get_ylim()[0],self.data.max()+abs(self.ax.get_ylim()[0])])
        self.ax.set_ylim([self.ax.get_ylim()[0],self.data.max()*1.1])
        self.ax_lims = self.ax.get_ylim()
        self.xlims = [ppm[-1], ppm[0]]
        self.ax.set_xlim(self.xlims)
        self.ax.text(0.95*self.ax.get_xlim()[0], 0.7*self.ax.get_ylim()[1],'Peak picking\nLeft - select peak\nMiddle - delete last peak\nDrag Right - select range')
        cursor = Cursor(self.ax, useblit=True,color='k', linewidth=0.5 )
        cursor.horizOn = False
        show()

    def press(self, event):
        tb = get_current_fig_manager().toolbar
        if tb.mode == '':
            x = round(event.xdata, 3)
            if event.button == 2:
                self.peaks = self.peaks[:-1]
                self.ax.lines = self.ax.lines[:-1]
            if event.button == 3:
                self.buttonDown = True
                self.pressv = event.xdata
            if event.button == 1 and (x >= self.xlims[1]) and (x <= self.xlims[0]):
                self.peaks.append(x)
                self.ax.plot([x,x],self.ax_lims,color='#CC0000',lw=0.5)
                print x
                self.peaks = sorted(self.peaks)[::-1]
            self.canvas.draw()

    def release(self, event):
        if self.pressv is None or not self.buttonDown:
            return
        self.buttonDown = False
        self.rect.set_visible(False)
        vmin = self.pressv
        vmax = event.xdata or self.prev[0]
        if vmin>vmax: vmin, vmax = vmax, vmin
        span = vmax - vmin
        self.pressv = None
        spantest = False
        if len(self.ranges) > 0:
            for i in self.ranges:
                if (vmin>=i[0]) and (vmin<=i[1]): spantest = True
                if (vmax>=i[0]) and (vmax<=i[1]): spantest = True
        if span > self.minspan and spantest is False:
            self.ranges.append([round(vmin,3), round(vmax,3)])
            self.ax.bar(left=vmin,height=sum(abs(self.ylims)),width=span,bottom=self.ylims[0],alpha=0.2,color='0.5',edgecolor='k')
        self.canvas.draw()
        self.ranges.sort()
        return False

    def onmove(self, event):
        if self.pressv is None or self.buttonDown is False or event.inaxes is None:
            return
        self.rect.set_visible(self.visible)
        x, y = event.xdata, event.ydata
        self.prev = x, y
        v = x
        minv, maxv = v, self.pressv
        if minv>maxv:
            minv, maxv = maxv, minv
        self.rect.set_xy([minv,self.rect.xy[1]])
        self.rect.set_width(maxv-minv)
        vmin = self.pressv
        vmax = event.xdata #or self.prev[0]
        if vmin>vmax:
            vmin, vmax = vmax, vmin
        self.canvas.draw_idle()
        return False


def _unwrap_fid_area(arg, **kwarg):
    return FID_array._phase_area_single(*arg, **kwarg)

def _unwrap_fid_neg(arg, **kwarg):
    return FID_array._phase_neg_single(*arg, **kwarg)

def _unwrap_fid_neg_area(arg, **kwarg):
    return FID_array._phase_neg_area_single(*arg, **kwarg)

def _unwrap_fid_deconv(arg, **kwarg):
    return FID_array._deconv_single(*arg, **kwarg)

if __name__ == '__main__':
    print 'NMRPy must be imported as a module.'
    pass
