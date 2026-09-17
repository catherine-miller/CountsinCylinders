import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from math import floor, log10


def jackknifeSE(replicates,fullvalue):
    '''
    Jackknife standard error from a set of leave-one-out replicates.

    Follows the convention already used by getjack: squared deviations of each
    replicate from the FULL-SAMPLE value (not from the mean of the replicates),
    with prefactor (nsamples-1)/nsamples. Operates elementwise on any trailing
    shape, so 1D and bivariate histograms both use it.
    '''
    replicates = np.asarray(replicates)
    nsamples = len(replicates)
    deviations = replicates - fullvalue
    return np.sqrt((nsamples-1)/nsamples*np.sum(deviations**2,axis = 0))


def jackknifeHistogram(counts,weights = 1,fullinverse = None,jackinverses = None,
                       relweighterr = None,primaryaxis = 0):
    '''
    Pooled central value and separated jackknife error components for one histogram.

    Arguments:
        counts: per-rosette RAW count histograms, shape (nrosettes, ...). Works
            for 1D histograms and for 2D (bivariate) ones.
        weights: per-bin incompleteness weights (the diagonal, separable
            correction). Applied as a plain multiplier without renormalizing,
            preserving the pre-existing convention.
        fullinverse: full-sample incompleteness correction matrix from
            IncompletenessMatrix, or None for no matrix correction.
        jackinverses: array of leave-one-out correction matrices, ordered to
            match `counts`. Supplying these is what produces a nonzero
            incompleteness error component.
        relweighterr: relative uncertainty on `weights` -- the diagonal
            alternative to jackinverses for the incompleteness component.
        primaryaxis: for a 2D histogram, which axis holds the PRIMARY tracer.
            IncompletenessMatrix flattens primary-slow (flat = i*n_sec + k), so
            an axis order of (secondary, primary) needs a transpose. 0 means the
            histogram is already primary-slow.

    Returns (central, err_total, err_sample, err_inc).

    The two error components are deliberately computed by holding one input
    fixed while the other varies -- that is what makes each attributable:
        err_sample: correction held at the full sample, data resampled
        err_inc:    data held at the full sample, correction resampled
    They are combined in quadrature, which drops the cross-term between them.
    That is valid when the correction is calibrated on a different sample than
    the histograms (e.g. a no-fiberassign/fiberassign pair versus the observed
    catalog); it would NOT hold if the correction and the data were built from
    the same objects over the same rosettes.
    '''
    counts = np.asarray(counts)
    nrosettes = len(counts)
    total = np.sum(counts,axis = 0)

    def transform(pooledcounts,inverse):
        distribution = pooledcounts/np.sum(pooledcounts)
        if inverse is None:
            return distribution*weights
        oriented = distribution.T if primaryaxis == 1 else distribution
        corrected = inverse @ np.ravel(oriented)
        corrected = corrected/np.sum(corrected)
        corrected = np.reshape(corrected,np.shape(oriented))
        return corrected.T if primaryaxis == 1 else corrected

    central = transform(total,fullinverse)

    samplereplicates = [transform(total - counts[k],fullinverse) for k in np.arange(nrosettes)]
    err_sample = jackknifeSE(samplereplicates,central)

    if jackinverses is not None:
        increplicates = [transform(total,jackinverses[k]) for k in np.arange(len(jackinverses))]
        err_inc = jackknifeSE(increplicates,central)
    elif relweighterr is not None:
        err_inc = np.abs(central*relweighterr)
    else:
        err_inc = np.zeros_like(central)

    return central, np.sqrt(err_sample**2 + err_inc**2), err_sample, err_inc


class jackknifeCiCError:
    #makes histograms
    #computes CiC error by leave-one-out rosette jackknife
    def __init__(self,elgtables, lrgtables, binmax, weights = None, weights_err = None,
                 matrices = None):
        '''
        Initializes jackknifeCiCError class. Makes histograms of counts in cylinders for each of several samples.
        Pools these into one histogram and computes a leave-one-out rosette jackknife error.
        Does so for each of four tracer combinations (ELG counts around ELGs, LRG counts around LRGs, ELG counts around LRGs, and LRG counts around ELGs)
        as well as for the bivariate distribution of ELG and LRG counts around ELGs and around LRGs.

        The error is reported as two components combined in quadrature:
        a sample variance from rosette scatter (*_CiCerr_sample) and an
        incompleteness-correction error (*_CiCerr_inc). See jackknifeHistogram
        for how each is isolated and what independence assumption that makes.

        Arguments:
            elgtables: list of tables with ELG CiC counts, one per rosette
            lrgtables: list of tables with LRG CiC counts, one per rosette
            binmax: maximum number of counts in cylinders to consider
            weights: list of [elgelgweights, lrglrgweights, elglrgweights, lrgelgweights].
                The diagonal, separable incompleteness correction. Mutually
                exclusive with `matrices`.
            weights_err: list of [elgelgweights_err, lrglrgweights_err, elglrgweights_err, lrgelgweights_err],
                uncertainties on each weight array. If provided, these become the
                incompleteness error component, added in quadrature with the
                sample variance. Defaults to zero arrays (no weight uncertainty).
                Each weight error v_i affects only histogram bin i via
                sigma_w[i] = histcomb[i] * (weights_err[i] / weights[i]).
                For the bivariate histograms two independent weight errors contribute:
                sigma_w[i,j] = histcomb[i,j] * sqrt((err_row[i]/w_row[i])^2 + (err_col[j]/w_col[j])^2).
            matrices: full incompleteness correction matrices from
                IncompletenessMatrix, as six entries in the same order as
                `weights` extended by the two bivariate cases:
                [elgelg, lrglrg, elglrg, lrgelg, elgbivariate, lrgbivariate].
                Each entry is what IncompletenessMatrix.leaveOneOutMatrices
                returns; the full-sample inverse and the leave-one-out inverse
                array are taken from it. Unlike `weights` these capture bin
                migration and (bivariately) primary-secondary correlation.
                Mutually exclusive with `weights`, since both correct for
                incompleteness and applying both would double-correct.

                The rosette ordering of the leave-one-out matrices MUST match
                the ordering of elgtables/lrgtables. That cannot be checked
                here and is the caller's responsibility.
        '''
        if matrices is not None and weights is not None:
            raise ValueError("weights and matrices both correct for incompleteness; "
                             "supplying both would double-correct. Pass only one.")
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

        def matrixargs(index):
            '''
            Pull the full-sample and leave-one-out inverses for one histogram out of
            `matrices`, checking the shape.

            Entries 0-3 are the 1D tracer combinations and must be (binmax, binmax);
            entries 4-5 are the bivariate ones and must be (binmax**2, binmax**2).
            Unlike the weights path, binmax is NOT clamped to fit: truncating a
            deconvolution matrix is not a valid operation on it, so a mismatch is an
            error the caller has to fix.
            '''
            if matrices is None:
                return {}
            entry = matrices[index]
            jackinverses, fullinverse = entry[1], entry[3]
            expected = binmax**2 if index >= 4 else binmax
            if np.shape(fullinverse) != (expected,expected):
                raise ValueError("matrices[%d] has shape %s but binmax=%d requires (%d, %d)"
                                 %(index,np.shape(fullinverse),binmax,expected,expected))
            if len(jackinverses) != ntables:
                raise ValueError("matrices[%d] has %d leave-one-out matrices but %d rosette "
                                 "tables were supplied; they must correspond one to one"
                                 %(index,len(jackinverses),ntables))
            return {'fullinverse': fullinverse, 'jackinverses': jackinverses}

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

        # Relative weight uncertainties. These are the diagonal way incompleteness
        # error can arrive; they feed the same *_CiCerr_inc slot that a matrix
        # jackknife would, so the quadrature combination is identical either way.
        elg_rel_err = np.where(elgweights != 0, elgweights_err / elgweights, 0.0)
        lrg_rel_err = np.where(lrgweights != 0, lrgweights_err / lrgweights, 0.0)
        elglrg_rel_err = np.where(elglrgweights != 0, elglrgweights_err / elglrgweights, 0.0)
        lrgelg_rel_err = np.where(lrgelgweights != 0, lrgelgweights_err / lrgelgweights, 0.0)

        elgbivariate_rel_err = np.sqrt(elglrg_rel_err[:, np.newaxis]**2 + elg_rel_err[np.newaxis, :]**2)
        lrgbivariate_rel_err = np.sqrt(lrg_rel_err[:, np.newaxis]**2 + lrgelg_rel_err[np.newaxis, :]**2)

        #per-rosette RAW counts. Kept unweighted here; the *_nodens attributes
        #below apply the weights, preserving what plotCiCRatio has always consumed.
        elgcounts = np.array([np.histogram(elgtables[i]["N_CiC"], bins = bins,
                                           density = False)[0] for i in range(ntables)])
        lrgcounts = np.array([np.histogram(lrgtables[i]["N_CiC"], bins = bins,
                                           density = False)[0] for i in range(ntables)])
        elglrgcounts = np.array([np.histogram(elgtables[i]["N_lrgCiC"], bins = bins,
                                              density = False)[0] for i in range(ntables)])
        lrgelgcounts = np.array([np.histogram(lrgtables[i]["N_elgCiC"], bins = bins,
                                              density = False)[0] for i in range(ntables)])

        self.elghist_nodens = elgcounts*elgweights
        self.lrghist_nodens = lrgcounts*lrgweights
        self.elglrghist_nodens = elglrgcounts*elglrgweights
        self.lrgelghist_nodens = lrgelgcounts*lrgelgweights

        #2D raw counts. NOTE the two axis orders differ, and deliberately so:
        #both bivariate figures plot ELG secondaries on x and LRG secondaries on y,
        #which puts the PRIMARY tracer on axis 1 for the ELG-centered histogram and
        #on axis 0 for the LRG-centered one. primaryaxis below tracks this.
        elgbivariatecounts = np.array([np.histogram2d(elgtables[i]["N_lrgCiC"],elgtables[i]["N_CiC"],
                                                      density = False,bins = bins)[0] for i in range(ntables)])
        lrgbivariatecounts = np.array([np.histogram2d(lrgtables[i]["N_CiC"],lrgtables[i]["N_elgCiC"],
                                                      density = False,bins = bins)[0] for i in range(ntables)])

        #first index: rosette number
        #third index: lrg CiC
        #fourth index: elg CiC
        #kept per-rosette and density-normalized, as before, since these are read externally
        self.elgmultitracerhist = np.array([h/np.sum(h) for h in elgbivariatecounts])*multitracerweights_elg
        self.lrgmultitracerhist = np.array([h/np.sum(h) for h in lrgbivariatecounts])*multitracerweights_lrg

        #pooled central values with the two jackknife error components
        self.elghistcomb, self.elg_CiCerr, self.elg_CiCerr_sample, self.elg_CiCerr_inc = \
            jackknifeHistogram(elgcounts,weights = elgweights,relweighterr = elg_rel_err,
                               **matrixargs(0))
        self.lrghistcomb, self.lrg_CiCerr, self.lrg_CiCerr_sample, self.lrg_CiCerr_inc = \
            jackknifeHistogram(lrgcounts,weights = lrgweights,relweighterr = lrg_rel_err,
                               **matrixargs(1))
        self.elglrghistcomb, self.elglrg_CiCerr, self.elglrg_CiCerr_sample, self.elglrg_CiCerr_inc = \
            jackknifeHistogram(elglrgcounts,weights = elglrgweights,relweighterr = elglrg_rel_err,
                               **matrixargs(2))
        self.lrgelghistcomb, self.lrgelg_CiCerr, self.lrgelg_CiCerr_sample, self.lrgelg_CiCerr_inc = \
            jackknifeHistogram(lrgelgcounts,weights = lrgelgweights,relweighterr = lrgelg_rel_err,
                               **matrixargs(3))

        self.elgbivariate, self.elgmultitracer_CiCerr, self.elgmultitracer_CiCerr_sample, \
            self.elgmultitracer_CiCerr_inc = \
            jackknifeHistogram(elgbivariatecounts,weights = multitracerweights_elg,
                               relweighterr = elgbivariate_rel_err,primaryaxis = 1,
                               **matrixargs(4))
        self.lrgbivariate, self.lrgmultitracer_CiCerr, self.lrgmultitracer_CiCerr_sample, \
            self.lrgmultitracer_CiCerr_inc = \
            jackknifeHistogram(lrgbivariatecounts,weights = multitracerweights_lrg,
                               relweighterr = lrgbivariate_rel_err,primaryaxis = 0,
                               **matrixargs(5))

        #uncorrected pooled histograms, so a corrected/uncorrected comparison does
        #not require building a second object
        self.elghistcomb_raw = np.sum(elgcounts,axis = 0)/np.sum(elgcounts)
        self.lrghistcomb_raw = np.sum(lrgcounts,axis = 0)/np.sum(lrgcounts)
        self.elglrghistcomb_raw = np.sum(elglrgcounts,axis = 0)/np.sum(elglrgcounts)
        self.lrgelghistcomb_raw = np.sum(lrgelgcounts,axis = 0)/np.sum(lrgelgcounts)
        self.elgbivariate_raw = np.sum(elgbivariatecounts,axis = 0)/np.sum(elgbivariatecounts)
        self.lrgbivariate_raw = np.sum(lrgbivariatecounts,axis = 0)/np.sum(lrgbivariatecounts)

        '''def makenZhistogram(self, elgtables, lrgtables,zbins):
            bins = [np.arange(self.binmax),np.arange(self.binmax),zbins]
            self.elgnZhist = np.array([np.histogram2d(elgtables[i]["N_lrgCiC"],elgtables[i]["N_CiC"],elgtables['Z']
                                                  density=True,bins=bins) for i in range(ntables)],dtype='object')
            self.lrgnZhist = np.array([np.histogram2d(lrgtables[i]["N_CiC"],lrgtables[i]["N_elgCiC"],lrgtables['Z']
                                                  density=True,bins=bins) for i in range(ntables)],dtype='object')'''

def compare2CatalogsCiC(cat1, cat2, name1, name2, binmax, figdir, figtag, xranges = [5,5,5,5],weights=[None,None],weightserr = [None,None],matrices = [None,None],plot_unweighted=False,staroffset=0):
    """Plot and save side-by-side comparison of two CiC distributions.

    Builds jackknifeCiCError objects for both catalogs, then produces two figures:
    a 1D figure comparing all four 1D CiC histograms with ratio panels, and a
    bivariate figure showing ELG- and LRG-centered 2D distributions as log-ratio
    color maps with per-cell uncertainty annotations.

    Error bars show the total error (sample variance and incompleteness
    correction combined in quadrature); the components are available separately
    on the catalog objects as *_CiCerr_sample and *_CiCerr_inc.

    Parameters
    ----------
    cat1, cat2 : tuple
        Argument tuples forwarded to jackknifeCiCError for each catalog.
    name1, name2 : str
        Legend labels for the two catalogs.
    binmax : int
        Maximum count bin passed to jackknifeCiCError.
    figdir : str
        Directory where output figures are saved.
    figtag : str
        String appended to output figure filenames.
    xranges : list of int, optional
        Display x-range for each of the four 1D histogram panels.
    weights : list of two weight tuples, optional
        Completeness weights for each catalog, passed to jackknifeCiCError.
    plot_unweighted : bool, optional
        If True, also build unweighted catalogs and overlay them on the ratio panels.
    staroffset : float, optional
        Coordinate offset for significance-star annotations in bivariate panels.
    """
    fontsize = 15
    titlesize = 20
    catalog1 = jackknifeCiCError(*cat1,binmax,weights=weights[0],weights_err = weightserr[0],
                                 matrices = matrices[0])
    catalog2 = jackknifeCiCError(*cat2,binmax,weights=weights[1],weights_err = weightserr[1],
                                 matrices = matrices[1])

    if plot_unweighted:
        catalog_unweighted1 = jackknifeCiCError(*cat1,binmax)
        catalog_unweighted2 = jackknifeCiCError(*cat2,binmax)
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
    
        #ratio of 2 catalogs
        ratio = hist1/hist2

        ratio_err = np.sqrt((err1/hist2)**2 + (hist1/hist2**2*err2)**2)
        if plot_unweighted:
            #only built when plot_unweighted is set, so these must stay inside the guard
            hist_unweighted1, err_unweighted1 = getHist(catalog_unweighted1)
            hist_unweighted2, err_unweighted2 = getHist(catalog_unweighted2)
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

    These ratios are deliberately NOT incompleteness-matrix corrected. This
    function consumes the raw per-rosette count arrays (*hist_nodens), and a
    matrix correction is not a per-bin multiplier so it cannot be folded into
    counts. That is the right behavior: this is the function that *derives*
    corrections, so it must see the uncorrected distributions.

    Parameters
    ----------
    cat1, cat2 : jackknifeCiCError
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
            #per-rosette grand totals over all bins. These must be per-rosette, not
            #scalars: each leave-one-out replicate has to be normalized by its own
            #footprint's totals, or its numerator excludes a rosette while its
            #normalization still includes it.
            totals1 = np.sum(h1,axis = 0)
            totals2 = np.sum(h2,axis = 0)
            print("hist1counts: ", np.sum(totals1))
            print("hist2counts: ", np.sum(totals2))
            def notzero(n):
                if n == 0: return 1
                else: return n
            def getjack(counts1, counts2, totals1, totals2):
                '''
                Jackknife SE of the probability ratio P1/P2 for one count bin.

                counts1/counts2: per-rosette raw counts in this bin
                totals1/totals2: per-rosette raw totals over all bins
                '''
                nsamples = len(counts1)
                def probabilityratio(c1,c2,t1,t2):
                    return (c1/notzero(t1))/notzero(c2/notzero(t2))
                full = probabilityratio(np.sum(counts1),np.sum(counts2),
                                        np.sum(totals1),np.sum(totals2))
                replicates = [probabilityratio(np.sum(counts1)-counts1[i],np.sum(counts2)-counts2[i],
                                               np.sum(totals1)-totals1[i],np.sum(totals2)-totals2[i])
                              for i in range(nsamples)]
                return jackknifeSE(replicates,full)
            jackstoplot = [getjack(h2[i],h1[i], totals2, totals1) for i in range(len(h1))]
            jacksforcorrection = [getjack(h1[i],h2[i], totals1, totals2) for i in range(len(h1))]
            return jackstoplot,jacksforcorrection
        def histratiopoissonerrors(h1,h2,tracers,plot=True,name = "sv3incompletenessratios_poissonerror"):
            '''
            arguments
                h1: histogram of counts in cylinders (NOT divided into jackknife samples)
                h2: a second hist of CiC
                tracers: format ["ELG," "ELG"], ['ELG','LRG'], etc
            '''
            
        hist1,hist1comb = getHist(cat1,tracers)
        hist2,hist2comb = getHist(cat2,tracers)
        hist1list = [hist1[i] for i in range(len(hist1))]
        hist2list = [hist2[i] for i in range(len(hist1))]
        jackstoplot,jacksforcorrection = histratiojacks(hist1list,hist2list)
        with np.errstate(divide='ignore', invalid='ignore'):
            histavg = hist2comb/hist1comb
        if plot:
            plt.errorbar(np.arange(len(histavg)),histavg,yerr=jackstoplot,label=tracers[0]+"-"+tracers[1],capsize=3)
        print("jacks to plot: ",jackstoplot)
        print("jacks for correction: ",jacksforcorrection)
        correctionratios = 1./histavg
        for i in range(len(correctionratios)):
            #correct to avoid infinities. Just say there is no correction to the incompleteness.
            if np.isinf(correctionratios[i]): correctionratios[i] = 1
        return correctionratios, jacksforcorrection
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
        np.save("datafiles/"+name,ratios)
        np.save("datafiles/"+name+"error",ratioerrs)
    return ratios, ratioerrs