import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from math import floor, log10

class bootstrapCiCError:
    #makes histograms
    #bootstraps CiC error
    def __init__(self,elgtables, lrgtables, binmax, weights = None, weights_err = None):
        '''
        Initializes bootstrapCiCerror class. Makes histograms of counts in cylinders for each of several samples.
        Combines these into one histogram and bootstraps error.
        Does so for each of four tracer combinations (ELG counts around ELGs, LRG counts around LRGs, ELG counts around LRGs, and LRG counts around ELGs)
        as well as for the bivariate distribution of ELG and LRG counts around ELGs and around LRGs.
        Arguments:
            elgtables: list of tables with ELG CiC counts
            lrgtables: list of tables with LRG CiC counts
            binmax: maximum number of counts in cylinders to consider
            weights: list of [elgelgweights, lrglrgweights, elglrgweights, lrgelgweights]
            weights_err: list of [elgelgweights_err, lrglrgweights_err, elglrgweights_err, lrgelgweights_err],
                uncertainties on each weight array. If provided, these are propagated into the
                error arrays in quadrature with the sample variance. Defaults to zero arrays
                (no weight uncertainty). Each weight error v_i affects only histogram bin i via
                sigma_w[i] = histcomb[i] * (weights_err[i] / weights[i]).
                For the bivariate histograms two independent weight errors contribute:
                sigma_w[i,j] = histcomb[i,j] * sqrt((err_row[i]/w_row[i])^2 + (err_col[j]/w_col[j])^2).
        '''
        #make sure binmax is not greater than the length of the weights
        if weights is not None:
            weightmaxes = [len(weigh) for weigh in weights]
            minweightlength = np.min(weightmaxes)
            binmax = np.min([binmax, minweightlength])
        self.binmax = binmax
        #1D histograms
        ntables = len(elgtables)
        #print(ntables)
        bins = np.arange(-0.5,binmax+0.5,step=1)

        #make completeness weights
        elgweights, lrgweights, elglrgweights, lrgelgweights = weights if weights is not None else (None, None, None, None)
        if elgweights is None:
            elgweights = np.repeat(1,binmax)
        if lrgweights is None:
            lrgweights = np.repeat(1,binmax)
        if elglrgweights is None:
            elglrgweights = np.repeat(1,binmax)
        if lrgelgweights is None:
            lrgelgweights = np.repeat(1,binmax)

        elgweights_err, lrgweights_err, elglrgweights_err, lrgelgweights_err = \
            weights_err if weights_err is not None else (None, None, None, None)
        if elgweights_err is None:
            elgweights_err = np.zeros(binmax)
        if lrgweights_err is None:
            lrgweights_err = np.zeros(binmax)
        if elglrgweights_err is None:
            elglrgweights_err = np.zeros(binmax)
        if lrgelgweights_err is None:
            lrgelgweights_err = np.zeros(binmax)

        multitracerweights_elg = np.outer(elglrgweights,elgweights)
        multitracerweights_lrg = np.outer(lrgweights,lrgelgweights)

        elghist = np.array([np.histogram(elgtables[i]["N_CiC"], bins = bins,
                                                density = True)[0]*elgweights for i in range(ntables)])
        lrghist = np.array([np.histogram(lrgtables[i]["N_CiC"], bins = bins,
                                                density = True)[0]*lrgweights for i in range(ntables)],)
        
        elglrghist = np.array([np.histogram(elgtables[i]["N_lrgCiC"], bins = bins,
                                                   density = True)[0]*elglrgweights for i in range(ntables)])
        lrgelghist = np.array([np.histogram(lrgtables[i]["N_elgCiC"], bins = bins,
                                                   density = True)[0]*lrgelgweights for i in range(ntables)])

        self.elghist_nodens = np.array([np.histogram(elgtables[i]["N_CiC"], bins = bins,
                                                density = False)[0]*elgweights for i in range(ntables)])
        self.lrghist_nodens = np.array([np.histogram(lrgtables[i]["N_CiC"], bins = bins,
                                                density = False)[0]*lrgweights for i in range(ntables)])
        
        self.elglrghist_nodens = np.array([np.histogram(elgtables[i]["N_lrgCiC"], bins = bins,
                                                   density = False)[0]*elglrgweights for i in range(ntables)])
        self.lrgelghist_nodens = np.array([np.histogram(lrgtables[i]["N_elgCiC"], bins = bins,
                                                   density = False)[0]*lrgelgweights for i in range(ntables)])

        #first index: rosette number
        #second index: 
        
        self.elg_CiCerr = [np.std([elghist[j][i] for j in np.arange(ntables)])/np.sqrt(ntables) for i in np.arange(binmax)]
        self.lrg_CiCerr = [np.std([lrghist[j][i] for j in np.arange(ntables)])/np.sqrt(ntables) for i in np.arange(binmax)]
        
        self.elglrg_CiCerr = [np.std([elglrghist[j][i] for j in np.arange(ntables)])/np.sqrt(ntables) for i in np.arange(binmax)]
        self.lrgelg_CiCerr = [np.std([lrgelghist[j][i] for j in np.arange(ntables)])/np.sqrt(ntables) for i in np.arange(binmax)]
        
        #2D histograms
        self.elgmultitracerhist = np.array([np.histogram2d(elgtables[i]["N_lrgCiC"],elgtables[i]["N_CiC"], 
                                                  density=True,bins=bins)[0]*multitracerweights_elg for i in range(ntables)])
        self.lrgmultitracerhist = np.array([np.histogram2d(lrgtables[i]["N_CiC"],lrgtables[i]["N_elgCiC"], 
                                                  density=True,bins=bins)[0]*multitracerweights_lrg for i in range(ntables)])
        #first index: rosette number
        #second index: 0 to get histogram data
        #third index: lrg CiC
        #fourth index: elg CiC
        self.elgmultitracer_CiCerr = [[np.std([self.elgmultitracerhist[i][j,k] for i in range(ntables)])/np.sqrt(ntables) for k in range(binmax)] \
                                 for j in range(binmax)]
        self.lrgmultitracer_CiCerr = [[np.std([self.lrgmultitracerhist[i][j,k] for i in range(ntables)])/np.sqrt(ntables) for k in range(binmax)] \
                                 for j in range(binmax)]
    

        elghistsum = np.sum(elghist, axis=0)
        lrghistsum = np.sum(lrghist, axis=0)
        elgbivariatesum = np.sum(self.elgmultitracerhist, axis=0)
        lrgbivariatesum = np.sum(self.lrgmultitracerhist, axis=0)
        elglrghistsum = np.sum(elglrghist, axis=0)
        lrgelghistsum = np.sum(lrgelghist, axis=0)
        '''
        elghistsum = elghist[0][0]
        lrghistsum = lrghist[0][0]
        elglrghistsum = elglrghist[0][0]
        lrgelghistsum = lrgelghist[0][0]
        elgbivariatesum = self.elgmultitracerhist[0][0]
        lrgbivariatesum = self.lrgmultitracerhist[0][0]
        for i in np.arange(1,ntables):
            elghistsum += elghist[i][0]
            lrghistsum += lrghist[i][0]
            elglrghistsum += elglrghist[i][0]
            lrgelghistsum += lrgelghist[i][0]
            elgbivariatesum += self.elgmultitracerhist[i][0]
            lrgbivariatesum += self.lrgmultitracerhist[i][0]
        '''

        self.elghistcomb = elghistsum/ntables
        self.lrghistcomb = lrghistsum/ntables
        self.elglrghistcomb = elglrghistsum/ntables
        self.lrgelghistcomb = lrgelghistsum/ntables
        self.elgbivariate = elgbivariatesum/ntables
        self.lrgbivariate = lrgbivariatesum/ntables

        # Propagate weight uncertainties into histogram errors.
        # For a histogram bin weighted by w[i], sigma_hist = hist * (sigma_w / w).
        # For a 2D bin weighted by w_row[j] * w_col[k], relative errors add in quadrature.
        elg_rel_err = np.where(elgweights != 0, elgweights_err / elgweights, 0.0)
        lrg_rel_err = np.where(lrgweights != 0, lrgweights_err / lrgweights, 0.0)
        elglrg_rel_err = np.where(elglrgweights != 0, elglrgweights_err / elglrgweights, 0.0)
        lrgelg_rel_err = np.where(lrgelgweights != 0, lrgelgweights_err / lrgelgweights, 0.0)

        self.elg_CiCerr = np.sqrt(np.array(self.elg_CiCerr)**2 + (self.elghistcomb * elg_rel_err)**2)
        self.lrg_CiCerr = np.sqrt(np.array(self.lrg_CiCerr)**2 + (self.lrghistcomb * lrg_rel_err)**2)
        self.elglrg_CiCerr = np.sqrt(np.array(self.elglrg_CiCerr)**2 + (self.elglrghistcomb * elglrg_rel_err)**2)
        self.lrgelg_CiCerr = np.sqrt(np.array(self.lrgelg_CiCerr)**2 + (self.lrgelghistcomb * lrgelg_rel_err)**2)

        elgbivariate_weight_err = self.elgbivariate * np.sqrt(
            elglrg_rel_err[:, np.newaxis]**2 + elg_rel_err[np.newaxis, :]**2)
        lrgbivariate_weight_err = self.lrgbivariate * np.sqrt(
            lrg_rel_err[:, np.newaxis]**2 + lrgelg_rel_err[np.newaxis, :]**2)
        self.elgmultitracer_CiCerr = np.sqrt(np.array(self.elgmultitracer_CiCerr)**2 + elgbivariate_weight_err**2)
        self.lrgmultitracer_CiCerr = np.sqrt(np.array(self.lrgmultitracer_CiCerr)**2 + lrgbivariate_weight_err**2)

        self.elghist = elghist
        self.lrghist = lrghist
        self.elglrghist = elglrghist
        self.lrgelghist = lrgelghist

        '''def makenZhistogram(self, elgtables, lrgtables,zbins):
            bins = [np.arange(self.binmax),np.arange(self.binmax),zbins]
            self.elgnZhist = np.array([np.histogram2d(elgtables[i]["N_lrgCiC"],elgtables[i]["N_CiC"],elgtables['Z']
                                                  density=True,bins=bins) for i in range(ntables)],dtype='object')
            self.lrgnZhist = np.array([np.histogram2d(lrgtables[i]["N_CiC"],lrgtables[i]["N_elgCiC"],lrgtables['Z']
                                                  density=True,bins=bins) for i in range(ntables)],dtype='object')'''

def compare2CatalogsCiC(cat1, cat2, name1, name2, binmax, figdir, figtag, xranges = [5,5,5,5],weights=[None,None],plot_unweighted=False,staroffset=0):
    """Plot and save side-by-side comparison of two CiC distributions.

    Builds bootstrapCiCError objects for both catalogs, then produces two figures:
    a 1D figure comparing all four 1D CiC histograms with ratio panels, and a
    bivariate figure showing ELG- and LRG-centered 2D distributions as log-ratio
    color maps with per-cell uncertainty annotations.

    Parameters
    ----------
    cat1, cat2 : tuple
        Argument tuples forwarded to bootstrapCiCError for each catalog.
    name1, name2 : str
        Legend labels for the two catalogs.
    binmax : int
        Maximum count bin passed to bootstrapCiCError.
    figdir : str
        Directory where output figures are saved.
    figtag : str
        String appended to output figure filenames.
    xranges : list of int, optional
        Display x-range for each of the four 1D histogram panels.
    weights : list of two weight tuples, optional
        Completeness weights for each catalog, passed to bootstrapCiCError.
    plot_unweighted : bool, optional
        If True, also build unweighted catalogs and overlay them on the ratio panels.
    staroffset : float, optional
        Coordinate offset for significance-star annotations in bivariate panels.
    """
    fontsize = 15
    titlesize = 20
    catalog1 = bootstrapCiCError(*cat1,binmax,weights=weights[0])
    catalog2 = bootstrapCiCError(*cat2,binmax,weights=weights[1])

    if plot_unweighted:
        catalog_unweighted1 = bootstrapCiCError(*cat1,binmax)
        catalog_unweighted2 = bootstrapCiCError(*cat2,binmax)
    cmap = 'RdYlBu_r'
    bins = np.arange(-0.5,binmax+0.5,step=1)
    
    def make1DCiCHist(ax, index, tracers=["ELG","ELG"],xrange = 5):
        def getHist(catalog):
            match tracers:
                case ["ELG","ELG"]:
                    hist = catalog.elghistcomb
                    err = catalog.elg_CiCerr
                case ["LRG","LRG"]:
                    hist = catalog.lrghistcomb
                    err = catalog.lrg_CiCerr
                case ["ELG","LRG"]:
                    hist = catalog.elglrghistcomb
                    err = catalog.elglrg_CiCerr
                case ["LRG","ELG"]:
                    hist = catalog.lrgelghistcomb
                    err = catalog.lrgelg_CiCerr
            return hist, err
        hist1, err1 = getHist(catalog1)
        hist2, err2 = getHist(catalog2)
        #fig,ax = plt.subplots(2,gridspec_kw={'height_ratios': [2, 1]},figsize=(7,7))
        ax[1,index].set_xlabel("Counts in Cylinders",fontsize=fontsize)
        ax[0,index].set_title(tracers[0]+" Counts Around "+tracers[1]+"s",fontsize=fontsize)
        ax[0,index].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax[1,index].set_xticks(np.arange(xrange + 1))
        ax[0,index].set_xlim(-0.5,xrange + 0.5)
        ax[0,index].set_ylim(1e-6,1.5)
        ax[1,index].set_xlim(-0.5,xrange + 0.5)
        ax[0,index].bar(bins[:-1], hist1, label=name1, yerr = err1,align='edge',
                edgecolor='purple',ecolor='purple',capsize=2,color="none",hatch='///')
        ax[0,index].bar(bins[:-1], hist2, label=name2,alpha=0.6, yerr = err2,align='edge',
                edgecolor='green',ecolor='green',capsize=2,color='green')
        ax[0,index].set_yscale("log")
    
        hist_unweighted1, err_unweighted1 = getHist(catalog_unweighted1)
        hist_unweighted2, err_unweighted2 = getHist(catalog_unweighted2)
        #ratio of 2 catalogs
        ratio = hist1/hist2

        ratio_err = np.sqrt((err1/hist2)**2 + (hist1/hist2**2*err2)**2)
        if plot_unweighted:
            ratio_unweighted = hist_unweighted1/hist_unweighted2
            ratio_unweighted_err = np.sqrt((err_unweighted1/hist_unweighted2)**2 + (hist_unweighted1/hist_unweighted2**2*err_unweighted2)**2)
            ax[1,index].errorbar(bins[:-1]+0.5,ratio_unweighted,yerr=ratio_unweighted_err,color='gray',capsize=2,label='Unweighted')
        ax[1,index].errorbar(bins[:-1]+0.5,ratio,yerr=ratio_err,color='black',capsize=2,label='Weighted')
        ax[1,index].plot([-0.5,xrange+0.5],[1,1],color='black',linestyle='dotted')
        #ax[1].set_ylim(0,2)
        ax[1,index].set_ylim(0,2)
        plt.subplots_adjust(hspace=0.04)
        if index == 3:
            ax[0,3].legend()
            ax[1,3].legend()
        
        #plt.savefig(figdir+tracers[0]+"_"+tracers[1]+"_"+figtag+".png")

    fig,ax = plt.subplots(2,4,gridspec_kw={'height_ratios': [2, 1]},figsize=(14,7))
    plt.suptitle("Counts in Cylinders in SV3 and Mock Catalogs",fontsize=titlesize)
    
    make1DCiCHist(ax,0,tracers=['LRG','LRG'],xrange=xranges[0])
    ax[1,0].set_ylabel("Ratio of \n Mocks to SV3",fontsize = fontsize)
    ax[0,0].set_ylabel("Probability",fontsize=fontsize)
    make1DCiCHist(ax,1,tracers=['ELG','ELG'],xrange=xranges[1])
    make1DCiCHist(ax,2,tracers=['LRG','ELG'],xrange=xranges[2])
    make1DCiCHist(ax,3,tracers=['ELG','LRG'],xrange=xranges[3])
    fig.savefig(figdir+"CiC1D"+figtag+".png")

    def round_to_2(x):
        return round(np.round(x,decimals=2), -int(floor(log10(x))-2))
    maxcounts = 7

    binsx = bins
    binsy = bins
    #ELG bivariate distribution
    hist1 = catalog1.elgbivariate
    hist2 = catalog2.elgbivariate
    diff = np.log(hist1)-np.log(hist2)
    fig, ax = plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios': [1, 1.2]})
    plt.suptitle("Bivariate Counts in Cylinders Distribution in SV3 and Mock Catalogs",fontsize=titlesize)
    #plt.tight_layout()
    pc = ax[0].pcolormesh(binsx,binsy,diff,vmax=np.log(8),vmin=-np.log(8),cmap=cmap)
    #cbar = plt.colorbar(pc,ticks = np.log([1/8,1/4,1/2,1,2,4,8]))
    #cbar.ax.set_yticklabels(['1/8','1/4',"1/2","1",'2','4','8'])
    ax[0].set_title("ELG-Centered Counts",fontsize=fontsize)
    ax[0].set_xlabel("Counts in Cylinders, ELG Secondaries",fontsize=fontsize)
    ax[0].set_ylabel("Counts in Cylinders, LRG Secondaries",fontsize=fontsize)
    ax[0].set_xlim(-0.5,maxcounts+0.5)
    ax[0].set_ylim(-0.5,maxcounts+0.5)
    ax[0].set_xticks(np.arange(0,6),labels=[str(i) for i in np.arange(0,6)])

    #elgerr = np.ndenumerate([[catalog1.elgmultitracer_CiCerr[i][j]/hist2[i][j] for j in range(maxcounts+1)]
                                  #for i in range(maxcounts+1)])
    for (i, j), z in np.ndenumerate(np.exp(diff[:maxcounts+1,:maxcounts+1])):
        if z > 0 and (hist2[i][j] != 0):
            errtoprint = np.sqrt((catalog1.elgmultitracer_CiCerr[i][j]/hist1[i][j])**2 + (catalog2.elgmultitracer_CiCerr[i][j]/hist2[i][j])**2)*hist1[i][j]/hist2[i][j]
            if np.abs(1 - z) > 2*errtoprint:
                ax[0].text(j+staroffset,i+staroffset,"*",color='black',size=14,path_effects=[pe.withStroke(linewidth=2, foreground="white")],weight = 600)
            ax[0].text((j+0.1)-0.5, (i+0.1)-0.5, "{:.2g}".format(z) +"\n" +r" $\pm$ "+ "{:g}".format(round_to_2(errtoprint)), ha='left', va='bottom',
                    color='black',path_effects=[pe.withStroke(linewidth=2, foreground="white")],weight=600)
            '''if z < errtoprint:
                ax[0].text(j+0.8-0.5,i+0.6-0.5,"*",color='black',path_effects=[pe.withStroke(linewidth=2, foreground="white")],weight = 600)'''

    ax[0].set_xlim(-0.5,5.5)
    ax[0].set_ylim(-0.5,5.5)
    #plt.savefig(figdir+"ELGbivariate"+figtag+".png")

    #LRG bivariate distribution
    hist1 = catalog1.lrgbivariate
    hist2 = catalog2.lrgbivariate
    diff = np.log(hist1)-np.log(hist2)
    #fig, ax = plt.subplots(1)
    pc = ax[1].pcolormesh(binsx,binsy,diff,vmax=np.log(8),vmin=-np.log(8),cmap=cmap)
    cbar = plt.colorbar(pc,ticks = np.log([1/8,1/4,1/2,1,2,4,8]))
    cbar.ax.set_yticklabels(['1/8','1/4',"1/2","1",'2','4','8'])
    ax[1].set_title("LRG-Centered Counts",fontsize=fontsize)
    ax[1].set_xlabel("Counts in Cylinders, ELG Secondaries",fontsize=fontsize)
    ax[1].set_ylabel("Counts in Cylinders, LRG Secondaries",fontsize=fontsize)
    ax[1].set_xlim(-0.5,maxcounts+0.5)
    ax[1].set_ylim(-0.5,maxcounts+0.5)
    ax[1].set_xticks(np.arange(0,6),labels=[str(i) for i in np.arange(0,6)])

    #lrgerr = np.ndenumerate([[catalog1.lrgmultitracer_CiCerr[i][j]/hist2[i][j] for j in range(maxcounts+1)]
                                  #for i in range(maxcounts+1)])
    for (i, j), z in np.ndenumerate(np.exp(diff[:maxcounts+1,:maxcounts+1])):
        if z > 0 and (hist2[i][j] != 0) and (hist1[i][j] != 0):
            errtoprint = np.sqrt((catalog1.lrgmultitracer_CiCerr[i][j]/hist1[i][j])**2 + (catalog2.lrgmultitracer_CiCerr[i][j]/hist2[i][j])**2)*hist1[i][j]/hist2[i][j]
            if np.abs(1 - z) > 2*errtoprint:
                ax[1].text(j+staroffset,i+staroffset,"*",color='black',size=14,path_effects=[pe.withStroke(linewidth=2, foreground="white")],weight = 600)
            ax[1].text((j+0.1)-0.5, (i+0.1)-0.5, '{:.2g}'.format(z)+" \n"+r" $\pm$ "+'{:g}'.format(round_to_2(errtoprint)), ha='left', va='bottom',
                    color='black',path_effects=[pe.withStroke(linewidth=2, foreground="white")],weight=600)
            '''if z < errtoprint:
                ax[1].text(j+staroffset,i+staroffset,"*",color='black',size = 14,path_effects=[pe.withStroke(linewidth=2, foreground="white")],weight = 600)'''

    ax[1].text(7, 0, name1+" / "+name2+" Count Probability",rotation=90, rotation_mode='anchor',transform_rotates_text=True,fontsize=fontsize)

    ax[1].set_xlim(-0.5,5.5)
    ax[1].set_ylim(-0.5,5.5)
    fig.savefig(figdir+"Bivariate"+figtag+".png")
    

def plotCiCRatio(cat1, cat2, name="", title='', plot=True, saveratios = True):
    """Plot the bin-by-bin ratio of two CiC distributions using jackknife errors.

    Computes the ratio of catalog2 to catalog1 for each of the four 1D tracer
    combinations (ELG-ELG, LRG-LRG, ELG-LRG, LRG-ELG). Errors are estimated
    via a leave-one-out jackknife over the subsample tables. The ratio plotted
    is catalog2 / catalog1.

    Parameters
    ----------
    cat1, cat2 : bootstrapCiCError
        Pre-built catalog objects.
    name : str, optional
        Label appended to the plot title.
    title : str, optional
        Additional plot title text.
    plot : bool, optional
        If True, draw the ratio curves on the current axes.
    """
    titlesize=15
    textsize=15
    def getHist(catalog,tracers):
        match tracers:
            case ["ELG","ELG"]:
                hist = catalog.elghist_nodens
                histcomb = catalog.elghistcomb
                #err = catalog.elg_CiCerr
            case ["LRG","LRG"]:
                hist = catalog.lrghist_nodens
                histcomb = catalog.lrghistcomb
                #err = catalog.lrg_CiCerr
            case ["ELG","LRG"]:
                hist = catalog.elglrghist_nodens
                histcomb = catalog.elglrghistcomb
                #err = catalog.elglrg_CiCerr
            case ["LRG","ELG"]:
                hist = catalog.lrgelghist_nodens
                histcomb = catalog.lrgelghistcomb
                #err = catalog.lrgelg_CiCerr
        return hist, histcomb
    def histRatio(cat1,cat2,tracers,plot=True, name = "sv3incompletenessratios"):
        def histratiojacks(h1,h2):
            #first index: jackknife slice
            #second index: N_CiC
            h1 = np.transpose(np.array(h1))
            h2 = np.transpose(np.array(h2))
            #now, first index is N_CiC, second is jackknife slice
            hist1counts = np.sum(h1)
            hist2counts = np.sum(h2)
            print("hist1counts: ", hist1counts)
            print("hist2counts: ", hist2counts)
            def getjack(list1, list2):
                nsamples = len(list1)
                sum1 = np.sum(list1)
                sum2 = np.sum(list2)
                def notzero(n):
                    if n == 0: return 1
                    else: return n
                squares = [(((sum1-list1[i])/notzero(sum2-list2[i]) - sum1/notzero(sum2))*hist2counts/hist1counts)**2 for i in range(nsamples)]
                sumofsquares = np.sum(squares)
                SE = np.sqrt((nsamples-1)/nsamples*sumofsquares)
                print(squares)
                return SE
            jackstoplot = [getjack(h2[i],h1[i]) for i in range(len(h1))]
            jacksforcorrection = [getjack(h1[i],h2[i]) for i in range(len(h1))]
            return jackstoplot,jacksforcorrection
        hist1,hist1comb = getHist(cat1,tracers)
        hist2,hist2comb = getHist(cat2,tracers)
        hist1list = [hist1[i] for i in range(len(hist1))]
        hist2list = [hist2[i] for i in range(len(hist1))]
        jackstoplot,jacksforcorrection = histratiojacks(hist1list,hist2list)
        with np.errstate(divide='ignore', invalid='ignore'):
            histavg = hist2comb/hist1comb
        if plot:
            plt.errorbar(np.arange(len(histavg)),histavg,yerr=jackstoplot,label=tracers[0]+"-"+tracers[1],capsize=3)
        return 1./histavg, jacksforcorrection
        '''
        #print(len(hist1[0][0]))
        #print(np.shape(histratios))
        for i in range(len(hist1[0][0])):
            ratios = [histratios[j][i] for j in range(len(hist1))]
            #j: rosette number
            #i: N_CiC
            histerrs.append(np.std(ratios)*np.sqrt(len(hist1)-1))
            #histavg.append(np.average(ratios))
        #error = np.std(histratio)/np.sqrt(len(histratio))
        '''

    if plot:
        plt.figure()
    lrglrgrat, lrglrgerr = histRatio(cat1,cat2,["LRG","LRG"],plot=plot)
    elgelgrat, elgelgerr = histRatio(cat1,cat2,["ELG","ELG"],plot=plot)
    lrgelgrat, lrgelgerr = histRatio(cat1,cat2,["LRG","ELG"],plot=plot)
    elglrgrat, elglrgerr = histRatio(cat1,cat2,["ELG","LRG"],plot=plot)
    if plot:
        plt.legend()
        plt.title(title,fontsize=titlesize)
        plt.xlabel("Counts in Cylinders",fontsize=textsize)
        plt.ylabel(r"Ratio of $P_\mathrm{CiC,incomplete}$ to $P_\mathrm{CiC,complete}$",fontsize=textsize)
        #plt.yscale('log')
        plt.xlim(-0.5,6.5)
        plt.ylim(0,1.1)
        plt.savefig("forpaper/ratio"+name+".png",dpi=300)
    ratios = np.array([elgelgrat, lrglrgrat, elglrgrat, lrgelgrat])
    ratioerrs = np.array([elgelgerr, lrglrgerr, elglrgerr, lrgelgerr])
    if saveratios:
        #write to a file
        np.save("datafiles/"+name+".csv",ratios)
        np.save("datafiles/"+name+"error.csv",ratioerrs)
    return ratios, ratioerrs