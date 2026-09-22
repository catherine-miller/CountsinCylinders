"""Split-half validation of the incompleteness matrix correction.

Derives the correction from one half of the rosettes and applies it to the other
half, then plots how well the corrected distribution recovers the complete one.
Produces, in figs/:

    figs/inccorrection_1d<tag>.png    four 1D tracer combinations
    figs/inccorrection_2d<tag>.png    the two bivariate distributions

Each panel compares two ratios against the complete (no-fiberassign) catalog of
the TEST half:

    incomplete / complete    how much fiber assignment distorts the distribution
    corrected  / complete    what is left after the matrix correction

A correction that works drives the second curve to 1 while the first deviates.

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

import CiCPlot
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
    """Build the bivariate correction from the calibration rosettes only."""
    completesubsets = rosetteSubsets(complete,rosettes)
    incompletesubsets = rosetteSubsets(incomplete,rosettes)
    countmatrices = [IM.countMatrixBivariate(s,"N_CiC",secondarycolumn,MAXCOUNTS,MAXCOUNTS)
                     for s in completesubsets]
    result, rcond, deviations = IM.chooseRcondAndBuild(
        countmatrices,
        perRosetteHistograms2D(incompletesubsets,secondarycolumn,bins),
        perRosetteHistograms2D(completesubsets,secondarycolumn,bins),
        IM.makeNormalizerBivariate(MAXCOUNTS),label = label)
    return result[3], rcond


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
    for entry in combinations2d:
        name, complete, incomplete, secondarycolumn, primarylabel, secondarylabel = entry
        inverse, rcond = calibrateBivariate(complete,incomplete,secondarycolumn,
                                            calibration,bins,name)
        completetest = perRosetteHistograms2D(rosetteSubsets(complete,test),secondarycolumn,bins)
        incompletetest = perRosetteHistograms2D(rosetteSubsets(incomplete,test),secondarycolumn,bins)
        uncorrected, uncorrectederr = ratioToComplete(incompletetest,completetest,shape = shape2d)
        corrected, correctederr = ratioToComplete(incompletetest,completetest,
                                                  inverse = inverse,shape = shape2d)
        completecounts = np.reshape(np.sum(completetest,axis = 0),shape2d)
        results2d.append((name,primarylabel,secondarylabel,uncorrected,corrected,
                          occupancyMask(completecounts)))
        summaries2d.append((name,uncorrected,uncorrectederr,corrected,correctederr,
                            completecounts))

    print("\nhow far each ratio sits from 1 on the held-out half:")
    for entry in results1d:
        summarize(*entry)
    for entry in summaries2d:
        name, uncorrected, uncorrectederr, corrected, correctederr, completecounts = entry
        summarize(name+" (bivariate)",uncorrected,uncorrectederr,corrected,correctederr,
                  completecounts)

    print()
    plot1D(results1d,tag)
    plot2D(results2d,tag)
    print("\nA working correction drives 'corrected / complete' to 1 where "
          "'incomplete / complete' deviates. Blank bivariate cells are where the "
          "complete catalog is empty, so the ratio is undefined rather than zero.")


if __name__ == "__main__":
    run(swap = "--swap" in sys.argv)
