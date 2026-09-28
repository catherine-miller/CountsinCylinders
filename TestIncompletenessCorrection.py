"""Split-half validation of the incompleteness matrix correction.

Derives the correction from one half of the rosettes and applies it to the other
half, then plots how well the corrected distribution recovers the complete one.
Produces, in figs/:

    figs/inccorrection_1d<tag>.png    four 1D tracer combinations
    figs/inccorrection_2d<tag>.png    the two bivariate distributions
    figs/inccorrection_moments<tag>.png
                                      simple moments <N_ELG^m N_LRG^n> of the
                                      two bivariate distributions

Each panel compares two ratios against the complete (no-fiberassign) catalog of
the TEST half:

    incomplete / complete    how much fiber assignment distorts the distribution
    corrected  / complete    what is left after the matrix correction

A correction that works drives the second curve to 1 while the first deviates.

The moment test computes the moments with CiCMoments.CiCHistnZ -- the code path
the real measurement uses -- so it also checks the correction's orientation and
count range there, not just in this script.

Run with:
    python TestIncompletenessCorrection.py            calibrate on the first half
    python TestIncompletenessCorrection.py --swap     calibrate on the second half

Importing IncompletenessMatrix does the catalog loading and the TARGETID
cross-match that produces the inc_counts columns, so nothing else is needed.

Note on rcond: the regularization strength is tuned by a split *inside the
calibration half*, never on the test half. Tuning it against the test half would
be fitting the knob on the data it is then judged against, which would make the
validation meaningless.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import CiCPlot
import CiCMoments as CM
import IncompletenessMatrix as IM

FIGDIR = "figs"
MAXCOUNTS = IM.MAXCOUNTS            #counts 0..MAXCOUNTS, so binmax = MAXCOUNTS+1
CMAP = 'RdYlBu_r'
RATIOLIMIT = 2.0                    #colour scale spans 1/RATIOLIMIT to RATIOLIMIT

#Bins holding fewer than this many objects in the complete TEST half are excluded
#from the summary statistics and left blank in the 2D maps. Zero means show
#everything -- the sparse high-count cells are exactly where the correction is
#worth scrutinising, so they are not hidden by default. Note that they will
#dominate the RMS: dividing two nearly-empty bins produces large ratios, so read
#the RMS as "including the worst cells" rather than as a typical deviation.
#Raise this if you want the summary restricted to well-populated bins.
MINCOUNTS = 0

#Moments up to total order MAXORDER, matching the paper's moment tables.
MAXORDER = 3

#Simple moments do not depend on redshift, and the two catalogs name their
#redshift columns differently (RSDZ vs Z), so the moment test gives CiCHistnZ a
#constant dummy column and a single redshift bin instead.
ZCOLUMN = "ZDUMMY"
ZBINS = np.array([-1.,1.])


def rosetteSubsets(table,rosettes):
    """Split a table into one sub-table per rosette, in the given order."""
    column = np.asarray(table['rosette'])
    return [table[column == r] for r in rosettes]


def perRosetteHistograms(tables,column,bins):
    return np.array([np.histogram(np.asarray(t[column]),bins = bins)[0] for t in tables])


def perRosetteHistograms2D(tables,secondarycolumn,bins):
    """Bivariate histograms in IncompletenessMatrix's primary-slow orientation.

    Axis 0 is the primary tracer (N_CiC), axis 1 the secondary, matching
    countMatrixBivariate's flat = i*n_sec + k convention. Flattened so the
    correction matrix can be applied directly.
    """
    return np.array([np.histogram2d(np.asarray(t["N_CiC"]),np.asarray(t[secondarycolumn]),
                                    bins = [bins,bins])[0].ravel() for t in tables])


def pooledDistribution(perrosette):
    total = np.sum(perrosette,axis = 0)
    return total/np.sum(total)


def ratioToComplete(observedperrosette,completeperrosette,inverse = None,shape = None):
    """Ratio of the (optionally corrected) incomplete distribution to the complete one.

    The error is a leave-one-out jackknife over the TEST rosettes with the
    correction matrix held fixed -- the matrix came from the other half, so it
    contributes no scatter here. Both numerator and denominator are resampled
    together, since they describe the same footprint.
    """
    observedperrosette = np.asarray(observedperrosette)
    completeperrosette = np.asarray(completeperrosette)

    def value(observed,complete):
        incomplete = pooledDistribution(observed)
        truth = pooledDistribution(complete)
        if inverse is not None:
            corrected = inverse @ np.ravel(incomplete)
            incomplete = corrected/np.sum(corrected)
        with np.errstate(divide = 'ignore',invalid = 'ignore'):
            ratio = np.ravel(incomplete)/np.ravel(truth)
        return np.reshape(ratio,shape) if shape is not None else ratio

    full = value(observedperrosette,completeperrosette)
    nrosettes = len(observedperrosette)
    replicates = [value(np.delete(observedperrosette,k,axis = 0),
                        np.delete(completeperrosette,k,axis = 0))
                  for k in np.arange(nrosettes)]
    return full, CiCPlot.jackknifeSE(replicates,full)


def calibrate1D(complete,incomplete,truecolumn,observedcolumn,rosettes,bins,label):
    """Build the 1D correction from the calibration rosettes only."""
    completesubsets = rosetteSubsets(complete,rosettes)
    incompletesubsets = rosetteSubsets(incomplete,rosettes)
    countmatrices = [IM.countMatrix(s,truecolumn,observedcolumn,MAXCOUNTS)
                     for s in completesubsets]
    result, rcond, deviations = IM.chooseRcondAndBuild(
        countmatrices,
        perRosetteHistograms(incompletesubsets,truecolumn,bins),
        perRosetteHistograms(completesubsets,truecolumn,bins),
        IM.makeNormalizer(),label = label)
    #result[3] is the full-sample inverse over the calibration rosettes
    return result[3], rcond


def calibrateBivariate(complete,incomplete,secondarycolumn,rosettes,bins,label):
    """Build the bivariate correction from the calibration rosettes only.

    Returns (inverse, leaveoneoutinverses, rcond): the full inverse over the
    calibration rosettes, and the leave-one-out inverses over those same
    rosettes, which the moment test uses for its incompleteness error.
    """
    completesubsets = rosetteSubsets(complete,rosettes)
    incompletesubsets = rosetteSubsets(incomplete,rosettes)
    countmatrices = [IM.countMatrixBivariate(s,"N_CiC",secondarycolumn,MAXCOUNTS,MAXCOUNTS)
                     for s in completesubsets]
    result, rcond, deviations = IM.chooseRcondAndBuild(
        countmatrices,
        perRosetteHistograms2D(incompletesubsets,secondarycolumn,bins),
        perRosetteHistograms2D(completesubsets,secondarycolumn,bins),
        IM.makeNormalizerBivariate(MAXCOUNTS),label = label)
    return result[3], result[1], rcond


def occupancyMask(completecounts):
    """Bins with enough objects in the complete test half to say anything about."""
    return np.asarray(completecounts) >= MINCOUNTS


def summarize(name,uncorrected,uncorrectederr,corrected,correctederr,completecounts):
    """Print how far each ratio sits from 1.

    The worst-sigma figure is reported together with how many objects sit in the
    bin that produced it: a huge sigma from a bin holding a handful of objects is
    an artifact of a vanishing error bar, not evidence the correction failed
    there. Reporting the occupancy makes that visible without hiding the bin.
    """
    completecounts = np.asarray(completecounts)
    mask = occupancyMask(completecounts)

    def stats(ratio,err):
        good = mask & np.isfinite(ratio)
        if not np.any(good):
            return np.nan, np.nan, np.nan
        rms = np.sqrt(np.mean((ratio[good] - 1)**2))
        finiteerr = good & np.isfinite(err) & (err > 0)
        if not np.any(finiteerr):
            return rms, np.nan, np.nan
        sigma = np.where(finiteerr,np.abs(ratio - 1)/np.where(err > 0,err,1),-np.inf)
        worstindex = np.unravel_index(np.argmax(sigma),np.shape(sigma))
        return rms, sigma[worstindex], completecounts[worstindex]

    rmsuncorrected, worstuncorrected, _ = stats(uncorrected,uncorrectederr)
    rmscorrected, worstcorrected, worstn = stats(corrected,correctederr)
    excluded = np.sum(~mask)
    negative = np.sum(mask & np.isfinite(corrected) & (np.asarray(corrected) < 0))
    notes = []
    if excluded:
        notes.append("%d bin(s) below %d objects excluded"%(excluded,MINCOUNTS))
    if negative:
        notes.append("*** %d NEGATIVE probability bin(s) after correction ***"%negative)
    print("  %-28s uncorrected RMS %.4f   corrected RMS %.4f "
          "(worst %.3g sigma, in a bin holding %g objects)%s"
          %(name,rmsuncorrected,rmscorrected,worstcorrected,worstn,
            "   ["+"; ".join(notes)+"]" if notes else ""))


def plot1D(results,tag):
    """Four panels, one per tracer combination, each with both ratios."""
    #ncic is the x axis (the counts-in-cylinders value); completecounts is the
    #number of OBJECTS in each of those bins. Keep the names distinct -- calling
    #both of them "counts" is how they got swapped once already.
    ncic = np.arange(MAXCOUNTS+1)
    fig, axes = plt.subplots(2,2,figsize = (11,8),sharex = True)
    for ax, entry in zip(axes.ravel(),results):
        name, uncorrected, uncorrectederr, corrected, correctederr, completecounts = entry
        mask = occupancyMask(completecounts)
        ax.axhline(1.0,color = 'gray',linestyle = 'dotted')
        ax.errorbar(ncic[mask],np.asarray(uncorrected)[mask],
                    yerr = np.asarray(uncorrectederr)[mask],capsize = 3,marker = 'o',
                    color = 'tab:red',label = 'incomplete / complete')
        ax.errorbar(ncic[mask],np.asarray(corrected)[mask],
                    yerr = np.asarray(correctederr)[mask],capsize = 3,marker = 's',
                    color = 'tab:blue',label = 'corrected / complete')
        if np.any(~mask) and np.any(mask):
            #shade the sparse tail rather than dropping it silently
            ax.axvspan(ncic[mask].max()+0.5,ncic[-1]+0.5,color = 'gray',alpha = 0.12)
            ax.text(ncic[mask].max()+0.6,ax.get_ylim()[1],
                    "< %d objects"%MINCOUNTS,fontsize = 8,va = 'top',color = 'gray')
        ax.set_title(name,fontsize = 13)
        ax.set_xticks(ncic)
    for ax in axes[1]:
        ax.set_xlabel(r"$N_\mathrm{CiC}$",fontsize = 13)
    for ax in axes[:,0]:
        ax.set_ylabel("Ratio to complete catalog",fontsize = 13)
    axes[0,0].legend(fontsize = 11)
    fig.suptitle("Incompleteness correction validated on held-out rosettes",fontsize = 15)
    fig.tight_layout()
    path = os.path.join(FIGDIR,"inccorrection_1d%s.png"%tag)
    fig.savefig(path,dpi = 200)
    plt.close(fig)
    print("wrote %s"%path)


def plot2D(results,tag):
    """Two rows (ELG- and LRG-centered), two columns (uncorrected, corrected)."""
    edges = np.arange(-0.5,MAXCOUNTS+1.5)
    fig, axes = plt.subplots(2,2,figsize = (11,10),constrained_layout = True)
    limit = np.log(RATIOLIMIT)
    mesh = None
    for row, entry in enumerate(results):
        name, primarylabel, secondarylabel, uncorrected, corrected, mask = entry
        for col, panel in enumerate([("incomplete / complete",uncorrected),
                                     ("corrected / complete",corrected)]):
            title, ratio = panel
            ax = axes[row,col]
            with np.errstate(divide = 'ignore',invalid = 'ignore'):
                logratio = np.log(np.where(mask,ratio,np.nan))
            logratio = np.ma.masked_invalid(logratio)
            #axis 0 is the primary tracer, so it becomes the y axis
            mesh = ax.pcolormesh(edges,edges,logratio,cmap = CMAP,vmin = -limit,vmax = limit)
            ax.set_title("%s\n%s"%(name,title),fontsize = 12)
            ax.set_xlabel(secondarylabel,fontsize = 12)
            ax.set_ylabel(primarylabel,fontsize = 12)
            ax.set_xticks(np.arange(MAXCOUNTS+1))
            ax.set_yticks(np.arange(MAXCOUNTS+1))
            for index, value in np.ndenumerate(ratio):
                if not mask[index] or not np.isfinite(value):
                    continue
                #a negative ratio means the deconvolution produced a negative
                #probability -- a hard failure, not a small error, so flag it in red
                negative = value < 0
                ax.text(index[1],index[0],"%.2g"%value,ha = 'center',va = 'center',
                        fontsize = 8,color = 'crimson' if negative else 'black',
                        fontweight = 'bold' if negative else 'normal')
    ticks = np.log([1/RATIOLIMIT,1/np.sqrt(RATIOLIMIT),1,np.sqrt(RATIOLIMIT),RATIOLIMIT])
    cbar = fig.colorbar(mesh,ax = axes,ticks = ticks,shrink = 0.85)
    cbar.ax.set_yticklabels(["%.2g"%np.exp(t) for t in ticks])
    cbar.set_label("Ratio to complete catalog",fontsize = 12)
    subtitle = "blank = ratio undefined (complete bin empty); red = negative probability"
    if MINCOUNTS > 0:
        subtitle = ("blank = undefined or fewer than %d objects; "
                    "red = negative probability"%MINCOUNTS)
    fig.suptitle("Bivariate incompleteness correction validated on held-out rosettes\n"
                 + subtitle,fontsize = 14)
    path = os.path.join(FIGDIR,"inccorrection_2d%s.png"%tag)
    fig.savefig(path,dpi = 200)
    plt.close(fig)
    print("wrote %s"%path)


def momentOrders():
    """[ELG order, LRG order] pairs in the order CiCMoments.makeMomentTable uses."""
    return [[i,j] for i in range(MAXORDER+1) for j in range(MAXORDER+1-i) if i+j > 0]


def momentHistogram(tables,secondarycolumn,inverse = None):
    """CiCHistnZ over the count range the matrices cover, with a dummy redshift.

    The primary is always N_CiC; secondarycolumn says which tracer the secondary
    is, and so whether the tables are ELG- or LRG-centered. If inverse is given
    it is passed in as a fixed correction (the same matrix for every "leave-one-
    out" slot), since the leave-one-out matrices here come from the calibration
    rosettes and do not correspond one to one with the test tables.
    """
    if secondarycolumn == "N_lrgCiC":
        elgname, lrgname = "N_CiC", "N_lrgCiC"
    else:
        elgname, lrgname = "N_elgCiC", "N_CiC"
    counttables = [Table({"N_CiC": np.asarray(t["N_CiC"]),
                          secondarycolumn: np.asarray(t[secondarycolumn]),
                          ZCOLUMN: np.zeros(len(t))}) for t in tables]
    matrices = None
    if inverse is not None:
        matrices = (None,[inverse]*len(tables),None,inverse)
    return CM.CiCHistnZ(counttables,elgname = elgname,lrgname = lrgname,zname = ZCOLUMN,
                        binmaxelg = MAXCOUNTS+1,binmaxlrg = MAXCOUNTS+1,
                        redshiftbins = ZBINS,matrices = matrices)


def momentRatios(incompletetables,completetables,secondarycolumn,inverse,calibrationinverses):
    """Simple moments of the incomplete test half, with and without correction, over complete.

    incompletetables and completetables must be per-rosette lists over the SAME
    test rosettes in the same order. As in ratioToComplete, numerator and
    denominator are resampled together in the jackknife, since they describe the
    same footprint.

    Two error components, isolated in the same way as CiCMoments.makeMomentTable:
        sample: correction held at the calibration-half matrix, one test rosette
            removed from both numerator and denominator at a time
        inc:    test data held fixed, the correction replaced by each
            leave-one-out matrix of the calibration half in turn
    Because the matrix comes from different rosettes than the data, the cross
    term dropped by adding these in quadrature really is zero here. The
    uncorrected ratio has only the sample component.

    Returns a dict of per-order arrays.
    """
    incomplete = momentHistogram(incompletetables,secondarycolumn,inverse)
    complete = momentHistogram(completetables,secondarycolumn)
    orders = momentOrders()
    grids = [incomplete.momentGrid(CM.momentSimple,order) for order in orders]

    def moments(hist):
        return np.array([np.sum(grid*hist)/np.sum(hist) for grid in grids])

    def ratio(rawincomplete,rawcomplete,correction):
        return moments(incomplete.correctHist(rawincomplete,correction))/moments(rawcomplete)

    totalincomplete = incomplete.histsum_raw
    totalcomplete = complete.histsum_raw
    nrosettes = len(incomplete.hists_raw)

    def sampleerror(correction,full):
        replicates = [ratio(totalincomplete - incomplete.hists_raw[k],
                            totalcomplete - complete.hists_raw[k],correction)
                      for k in np.arange(nrosettes)]
        return CM.jackknifeSE(replicates,full)

    uncorrected = ratio(totalincomplete,totalcomplete,None)
    uncorrectederr = sampleerror(None,uncorrected)

    corrected = ratio(totalincomplete,totalcomplete,inverse)
    correctederr_sample = sampleerror(inverse,corrected)
    correctederr_inc = CM.jackknifeSE([ratio(totalincomplete,totalcomplete,correction)
                                       for correction in calibrationinverses],corrected)

    return {'orders': orders,
            'complete': moments(totalcomplete),
            'uncorrected': uncorrected,'uncorrectederr': uncorrectederr,
            'corrected': corrected,
            'correctederr': np.sqrt(correctederr_sample**2 + correctederr_inc**2),
            'correctederr_sample': correctederr_sample,
            'correctederr_inc': correctederr_inc}


def summarizeMoments(name,result):
    """Print each moment's ratio to the complete catalog, and how many sigma from 1."""
    print("  %s"%name)
    print("    %-9s %12s %20s %30s %8s"%("ELG,LRG","complete","incomplete/complete",
                                           "corrected/complete (sample, inc)","sigma"))
    for index, order in enumerate(result['orders']):
        err = result['correctederr'][index]
        sigma = (result['corrected'][index] - 1)/err if err > 0 else np.nan
        print("    %-9s %12.4g %11.4f +- %.4f %11.4f +- %.4f (%.4f, %.4f) %8.2f"
              %("%d,%d"%tuple(order),result['complete'][index],
                result['uncorrected'][index],result['uncorrectederr'][index],
                result['corrected'][index],err,result['correctederr_sample'][index],
                result['correctederr_inc'][index],sigma))
    rmsuncorrected = np.sqrt(np.mean((result['uncorrected'] - 1)**2))
    rmscorrected = np.sqrt(np.mean((result['corrected'] - 1)**2))
    print("    RMS from 1: uncorrected %.4f   corrected %.4f"%(rmsuncorrected,rmscorrected))


def plotMoments(results,tag):
    """One panel per primary tracer: each moment's ratio to the complete catalog."""
    fig, axes = plt.subplots(2,1,figsize = (10,8),sharex = True)
    for ax, entry in zip(axes,results):
        name, result = entry
        x = np.arange(len(result['orders']))
        ax.axhline(1.0,color = 'gray',linestyle = 'dotted')
        ax.errorbar(x - 0.08,result['uncorrected'],yerr = result['uncorrectederr'],
                    capsize = 3,marker = 'o',linestyle = 'none',color = 'tab:red',
                    label = 'incomplete / complete')
        ax.errorbar(x + 0.08,result['corrected'],yerr = result['correctederr'],
                    capsize = 3,marker = 's',linestyle = 'none',color = 'tab:blue',
                    label = 'corrected / complete')
        ax.set_title(name,fontsize = 13)
        ax.set_ylabel("Moment ratio to complete catalog",fontsize = 12)
        ax.set_xticks(x)
        ax.set_xticklabels([r"$\langle N_\mathrm{ELG}^{%d} N_\mathrm{LRG}^{%d}\rangle$"%tuple(order)
                            for order in result['orders']],fontsize = 10)
    axes[0].legend(fontsize = 11)
    axes[-1].set_xlabel("Simple moment",fontsize = 13)
    fig.suptitle("Incompleteness correction of bivariate CiC moments, validated on "
                 "held-out rosettes\n(counts 0-%d only; corrected error includes the "
                 "calibration matrix's jackknife)"%MAXCOUNTS,fontsize = 13)
    fig.tight_layout()
    path = os.path.join(FIGDIR,"inccorrection_moments%s.png"%tag)
    fig.savefig(path,dpi = 200)
    plt.close(fig)
    print("wrote %s"%path)


def run(swap = False):
    os.makedirs(FIGDIR,exist_ok = True)
    bins = np.arange(MAXCOUNTS+2)
    shape2d = (MAXCOUNTS+1,MAXCOUNTS+1)

    #rosettes common to both catalogs of both tracers, so every split is usable
    rosettes = IM.rosetteList(IM.sv3elgnofiberassign,IM.sv3elgfiberassign,
                              IM.sv3lrgnofiberassign,IM.sv3lrgfiberassign)
    half = len(rosettes)//2
    first, second = rosettes[:half], rosettes[half:]
    calibration, test = (second,first) if swap else (first,second)
    tag = "_swapped" if swap else ""

    print("rosettes available: %s"%list(rosettes))
    print("calibrating on:     %s"%list(calibration))
    print("testing on:         %s\n"%list(test))

    combinations1d = [
        ("ELG counts around ELGs",IM.sv3elgnofiberassign,IM.sv3elgfiberassign,
         "N_CiC",'inc_counts'),
        ("LRG counts around LRGs",IM.sv3lrgnofiberassign,IM.sv3lrgfiberassign,
         "N_CiC",'inc_counts'),
        ("LRG counts around ELGs",IM.sv3elgnofiberassign,IM.sv3elgfiberassign,
         "N_lrgCiC",'inc_counts_sec'),
        ("ELG counts around LRGs",IM.sv3lrgnofiberassign,IM.sv3lrgfiberassign,
         "N_elgCiC",'inc_counts_sec'),
    ]

    print("choosing rcond within the calibration half (1D):")
    results1d = []
    for entry in combinations1d:
        name, complete, incomplete, truecolumn, observedcolumn = entry
        inverse, rcond = calibrate1D(complete,incomplete,truecolumn,observedcolumn,
                                     calibration,bins,name)
        completetest = perRosetteHistograms(rosetteSubsets(complete,test),truecolumn,bins)
        incompletetest = perRosetteHistograms(rosetteSubsets(incomplete,test),truecolumn,bins)
        uncorrected, uncorrectederr = ratioToComplete(incompletetest,completetest)
        corrected, correctederr = ratioToComplete(incompletetest,completetest,inverse = inverse)
        completecounts = np.sum(completetest,axis = 0)
        results1d.append((name,uncorrected,uncorrectederr,corrected,correctederr,completecounts))

    print("\nchoosing rcond within the calibration half (bivariate):")
    combinations2d = [
        ("ELG-centered",IM.sv3elgnofiberassign,IM.sv3elgfiberassign,"N_lrgCiC",
         "ELG counts","LRG counts"),
        ("LRG-centered",IM.sv3lrgnofiberassign,IM.sv3lrgfiberassign,"N_elgCiC",
         "LRG counts","ELG counts"),
    ]
    results2d = []
    summaries2d = []
    resultsmoments = []
    for entry in combinations2d:
        name, complete, incomplete, secondarycolumn, primarylabel, secondarylabel = entry
        inverse, calibrationinverses, rcond = calibrateBivariate(complete,incomplete,secondarycolumn,
                                                                 calibration,bins,name)
        completetables = rosetteSubsets(complete,test)
        incompletetables = rosetteSubsets(incomplete,test)
        completetest = perRosetteHistograms2D(completetables,secondarycolumn,bins)
        incompletetest = perRosetteHistograms2D(incompletetables,secondarycolumn,bins)
        uncorrected, uncorrectederr = ratioToComplete(incompletetest,completetest,shape = shape2d)
        corrected, correctederr = ratioToComplete(incompletetest,completetest,
                                                  inverse = inverse,shape = shape2d)
        completecounts = np.reshape(np.sum(completetest,axis = 0),shape2d)
        results2d.append((name,primarylabel,secondarylabel,uncorrected,corrected,
                          occupancyMask(completecounts)))
        summaries2d.append((name,uncorrected,uncorrectederr,corrected,correctederr,
                            completecounts))
        resultsmoments.append((name,momentRatios(incompletetables,completetables,secondarycolumn,
                                                 inverse,calibrationinverses)))

    print("\nhow far each ratio sits from 1 on the held-out half:")
    for entry in results1d:
        summarize(*entry)
    for entry in summaries2d:
        name, uncorrected, uncorrectederr, corrected, correctederr, completecounts = entry
        summarize(name+" (bivariate)",uncorrected,uncorrectederr,corrected,correctederr,
                  completecounts)

    print("\nsimple moments of the bivariate distribution on the held-out half "
          "(counts 0-%d only):"%MAXCOUNTS)
    for entry in resultsmoments:
        summarizeMoments(*entry)

    print()
    plot1D(results1d,tag)
    plot2D(results2d,tag)
    plotMoments(resultsmoments,tag)
    print("\nA working correction drives 'corrected / complete' to 1 where "
          "'incomplete / complete' deviates. Blank bivariate cells are where the "
          "complete catalog is empty, so the ratio is undefined rather than zero.")


if __name__ == "__main__":
    run(swap = "--swap" in sys.argv)
