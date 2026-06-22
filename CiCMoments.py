from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import fitsio
import healpy
from scipy.spatial import cKDTree
from astropy.cosmology import Planck18 as cosmo
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u
import astropy.constants as const
import matplotlib
import matplotlib.patheffects as pe
from math import log10, floor
import scipy.stats


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class CiCHistnZ:
    def __init__(self,tables,elgname = "N_elgCiC",lrgname="N_lrgCiC",zname = 'Z',binmaxelg = 24,binmaxlrg = 24,
        redshiftbins = np.linspace(0.80695,0.9919,num=20), elgweights = None, lrgweights = None):
        if elgweights is None:
            elgweights = np.repeat(1,binmaxelg)
        else: binmaxelg = len(elgweights)
        if lrgweights is None:
            lrgweights = np.repeat(1,binmaxlrg)
        else: binmaxlrg = len(lrgweights)
        bins = [np.arange(-0.5,binmaxelg+0.5,1),np.arange(-0.5,binmaxlrg+0.5,1),redshiftbins]
        incompletenessweights = np.outer(elgweights,lrgweights)
        self.hists = [np.histogramdd((table[elgname],table[lrgname],table[zname]),bins = bins)[0]*incompletenessweights[:,:,np.newaxis] 
            for table in tables]
        alltab = vstack(tables)
        self.histsum = np.histogramdd((alltab[elgname],alltab[lrgname],alltab[zname]),bins = bins)[0]*incompletenessweights[:,:,np.newaxis]
        self.zbins = bins[2]
        self.zcenters = [(self.zbins[i]+self.zbins[i+1])/2 for i in range(len(self.zbins)-1)]
        self.binmax_elg = bins[0][-1]
        self.binmax_lrg = bins[0][-1]
        self.nbins_elg = len(bins[0]) - 1
        self.nbins_lrg = len(bins[1]) - 1
        self.ntables = len(tables)
        self.weights = [len(table) for table in tables]

    def getMoment(self,hist,momentfunction,order,otherargs = []):
        momentsum = 0
        histsum = 0
        for i in range(self.nbins_elg - 1):
            for j in range(self.nbins_lrg - 1):
                for k in range(len(self.zbins)-1):
                    momentsum += momentfunction(order,i,j,k,*otherargs)*hist[i][j][k]
                    histsum += hist[i][j][k] #for normalization
        return momentsum/histsum

    def makeMomentTable(self, maxorder,momentfunction,extraargs=[],filename = '',savesubsamples = False):
        hists = self.hists
        momentlist = []
        orderlist = []
        stderrlist = []
        if savesubsamples:
            ssamples = []
        ntables = len(hists)
        for i in range(maxorder + 1):
            j = 0
            while (i + j < maxorder + 1):
                if i == 0 and j == 0: 
                    j += 1
                    continue
                moments = [self.getMoment(hists[k],momentfunction,[i,j],otherargs=extraargs) for k in range(self.ntables)]
                stderrlist.append(np.std(moments)/np.sqrt(ntables))
                momentlist.append(np.average(moments,weights=self.weights))
                orderlist.append([i,j])
                if savesubsamples:
                    ssamples.append(moments)
                j += 1
        if savesubsamples:
                moments = Table([orderlist,momentlist,stderrlist,ssamples],
                   names=("ELG Order, LRG Order","Moment","Standard Error","SubsampleMoments"),
                   meta={'name': "Moments of ELG-centered bivariate counts in cylinders distribution"})
        else:
            moments = Table([orderlist,momentlist,stderrlist],
                   names=("ELG Order, LRG Order","Moment","Standard Error"),
                   meta={'name': "Moments of ELG-centered bivariate counts in cylinders distribution"})
        if len(filename) > 0:
            moments.write(filename, format = "ascii.ecsv", overwrite=True)
            print("saved file "+filename)
        #moments['Moment'].format = '%.4e'
        #moments['Standard Error'].format = '%.4e'
        return moments

        
def getnZ(table,solidangle,zlim,nbins,zname = "Z"):
    bins = np.linspace(*zlim,num=nbins)
    hist, bins = np.histogram(table[zname],bins)
    nz = []
    for i in range(len(bins)-1):
        d1 = cosmo.comoving_distance(bins[i+1])/u.Mpc
        d2 = cosmo.comoving_distance(bins[i])/u.Mpc
        v = 1/3*solidangle*(d1**3-d2**3)
        nz.append(hist[i]/v) #NO factors of little h
    return nz,bins

def getnZfromNCiC(table,zlim,nbins,Vcyl,zname='Z',cicname = "N_CiC"):
    bins = np.linspace(*zlim,num=nbins)
    nz = []
    for i in range(len(bins)-1):
        nz.append(np.average(table[(table[zname] < bins[i+1]) & (table[zname] > bins[i])][cicname])/Vcyl)
    return nz, bins


def momentSimple(order,n_elg,n_lrg,z):
    n = order[0]
    m = order[1]
    return n_elg**n*n_lrg**m

def factorialProduct(N,m):
    product = 1
    for i in range(m):
        product = product*(N-i)
    return product

def factorialProductCombinations(N,m,nVcyl):
    Ntilde = factorialProduct(N,m)
    for l in range(1,m+1):
        Ntilde += (-nVcyl)**l*np.math.comb(m,l)*factorialProduct(N,m-l)
    return Ntilde

def factorialMoment(order,n_elg,n_lrg,zbin):
    n = order[0]
    m = order[1]
    return factorialProduct(n_elg,n)*factorialProduct(n_lrg,m)

def Ntilde(order,n_elg,n_lrg,zbin,Vcyl,nz1,nz2):
    n = order[0]
    m = order[1]
    density1 = nz1[zbin]
    density2 = nz2[zbin]
    center1 = Vcyl*density1
    center2 = Vcyl*density2
    return factorialProductCombinations(n_elg,n,center1)*factorialProductCombinations(n_lrg,m,center2)

def nZintegral(order,n_elg,n_lrg,zbin,nz1,nz2):
    n = order[0]
    m = order[1]
    return nz1[zbin]**n*nz2[zbin]**m #I thiiiink this is correct?

def vCylPower(order,n_elg,n_lrg,zbin,V_cyl):
    return V_cyl**(order[0] + order[1])


def showMomentTable(table,title="",lrg = False,scientific = True,filename = "momentfigs/test"):
    fig , ax = plt.subplots()
    if lrg:
        ax.set_ylabel("ELG Order")
        ax.set_xlabel("LRG Order")
    else:
        ax.set_ylabel("LRG Order")
        ax.set_xlabel("ELG Order")
    def textpos(order):
        return 0.1 + 0.2*np.array(order)
    plt.rcParams['font.size'] = 16
    plt.plot([-0.5,-0.5],[0.5,3.5],color='black')
    plt.plot([0.5,0.5],[-0.5,3.5],color='black')
    plt.plot([1.5,1.5],[-0.5,2.5],color='black')
    plt.plot([2.5,2.5],[-0.5,1.5],color='black')
    plt.plot([3.5,3.5],[-0.5,0.5],color='black')
    plt.plot([0.5,3.5],[-0.5,-0.5],color='black')
    plt.plot([-0.5,3.5],[0.5,0.5],color='black')
    plt.plot([-0.5,2.5],[1.5,1.5],color='black')
    plt.plot([-0.5,1.5],[2.5,2.5],color='black')
    plt.plot([-0.5,0.5],[3.5,3.5],color='black')
    ax.set_xticks([0,1,2,3])
    ax.set_yticks([0,1,2,3])
    ax.set_title(title)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    for moment in table:
        ordertxt = "ELG Order, LRG Order"
        m = (moment["Moment"])
        err = (moment["Standard Error"])
        if lrg:
            if scientific:
                ax.text(moment[ordertxt][1],moment[ordertxt][0],f"{m:.2e} \n "+r"$\pm$"+f" {err:.1e}",horizontalalignment = "center",
                   verticalalignment = 'center',size = 12)
            else:
                ax.text(moment[ordertxt][1],moment[ordertxt][0],f"{m:.3f} \n "+r"$\pm$"+f" {err:.3f}",horizontalalignment = "center",
                   verticalalignment = 'center',size = 12)
        else:
            if scientific:
                ax.text(*moment[ordertxt],f"{m:.2e} \n "+r"$\pm$"+f" {err:.1e}",horizontalalignment = "center",
                   verticalalignment = 'center',size = 12)
            else:
                ax.text(*moment[ordertxt],f"{m:.3f} \n "+r"$\pm$"+f" {err:.3f}",horizontalalignment = "center",
                   verticalalignment = 'center',size = 12)
    for spine in ax.spines.values():
         spine.set_visible(False)
    plt.tight_layout()
    plt.gcf().set_size_inches(7, 5)
    plt.savefig(filename +".png",dpi=199)


#def printMTLaTeX(table,title = ''):
def momentRatio(table1,table2):
    comp = Table()
    comp["ELG Order, LRG Order"] = table1["ELG Order, LRG Order"]
    comp["Moment"] = table1["Moment"]/table2["Moment"]
    comp["Standard Error"] = np.sqrt((table1["Standard Error"]/table2["Moment"])**2 \
                           + (table1["Moment"]/table2["Moment"]**2*table2['Standard Error'])**2)
    return comp


def avgnpcf(tab_ntilde_dat,tab_ntilde_ran,tab_nz,tab_vcyl):
    tab_npcf = Table()
    tab_npcf["ELG Order, LRG Order"] = tab_ntilde_dat["ELG Order, LRG Order"]
    tab_npcf["Moment"] = (tab_ntilde_dat["Moment"]-tab_ntilde_ran["Moment"])/(tab_nz["Moment"]*tab_vcyl["Moment"])
    tab_npcf["Standard Error"] = np.sqrt((tab_ntilde_dat["Standard Error"]/(tab_nz["Moment"]*tab_vcyl["Moment"]))**2 \
                           + (tab_ntilde_ran["Standard Error"]/(tab_nz["Moment"]*tab_vcyl["Moment"]))**2 \
                                         +(tab_nz["Standard Error"]/tab_nz["Moment"])**2)
    return tab_npcf

def formatMoment(table, index, scientific = False):
    m = table["Moment"][index]
    err = table["Standard Error"][index]
    string = ''
    if scientific:
        string += f"{m:.2e} "+r"$\pm$"+f" {err:.1e}"
    else:
        lastorder = int(np.log10(m))-2
        errfloat = np.round(err, -lastorder)
        errstring = str(errfloat).lstrip('0.')
        if lastorder > -4 and lastorder < 0:
            string += f"{m:.3g} ("+errstring+")"
        elif lastorder >= 0:
            string += f"{m:.3g} ("+errstring[:3]+")"
        else:
            string += f"{m:.2e} ("+errstring+")"
    return string

def printMomentTableLaTeX(table,captiontxt,lrg = False,scientific = False):
    #so hacky but whatever
    if lrg:
        ix = [3,6,8,0,4,7,1,5,2]
    else:
        ix = np.arange(9)
    string = r'''
    \begin{table}
    \caption{''' + captiontxt + r'''}
    \begin{tabular}{ c  c  c  c c c}
    \multicolumn{2}{c}{}&\multicolumn{4}{c}{'''+("ELG Order" if lrg else "LRG Order")+r'''}\\
     \multicolumn{1}{c}{}& &  0&1&  2 &3\\ \cline{3-6}
    
    \multirow[|l|]{4}{*}{\rotatebox[origin=c]{90}{'''+("LRG Order" if lrg else "ELG Order")+r'''}}& \multicolumn{1}{c|}{0} && ''' \
    + formatMoment(table,ix[0]) + r''' & ''' + formatMoment(table,ix[1]) + r"&" + formatMoment(table,ix[2]) \
    + r"\\" \
    + r'''
     & \multicolumn{1}{c|}{1} &''' +formatMoment(table,ix[3]) +"&" + formatMoment(table,ix[4]) +"&" + formatMoment(table,ix[5]) + r'''&\multicolumn{1}{c}{}\\
    
     &\multicolumn{1}{c|}{2} & ''' + formatMoment(table,ix[6]) + "&" +formatMoment(table,ix[7]) + r'''& \multicolumn{2}{c}{}\\
     &\multicolumn{1}{c|}{3} &''' +formatMoment(table,ix[8]) + r'''& \multicolumn{2}{c}{}\\
    
    \end{tabular}
    
    \end{table}'''
    return string


def printCiCTableLaTeX(hist,captiontxt,lrg = False,scientific = False):
    string = r'''
    \begin{table}
    \caption{''' + captiontxt + r'''}
    \begin{tabular}{c  c  c  c c c c c c }
    \multicolumn{2}{c}{}&\multicolumn{4}{c}{'''+"LRG Counts"+r'''}\\
     \multicolumn{1}{c}{}& &  0&1&  2 &3\\[2pt] \hline
    
    \multirow[|l|]{4}{*}{\rotatebox[origin=c]{90}{'''+("LRG Order" if lrg else "ELG Order")+r'''}}& \multicolumn{1}{c}{0} && ''' \
    + formatMoment(table,ix[0]) + r''' & ''' + formatMoment(table,ix[1]) + r"&" + formatMoment(table,ix[2]) \
    + r"\\[2pt]" \
    + r'''
     & \multicolumn{1}{c}{1} &''' +formatMoment(table,ix[3]) +"&" + formatMoment(table,ix[4]) +"&" + formatMoment(table,ix[5]) + r'''&\multicolumn{1}{c}{}\\[2pt]
    
     &\multicolumn{1}{c}{2} & ''' + formatMoment(table,ix[6]) + "&" +formatMoment(table,ix[7]) + r'''& \multicolumn{2}{c}{}\\[2pt]
     &\multicolumn{1}{c}{3} &''' +formatMoment(table,ix[8]) + r'''& \multicolumn{2}{c}{}\\[2pt]
    
    \end{tabular}
    
    \end{table}'''
    return string
    
        

        