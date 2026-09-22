from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import log10, floor
import CiCPlot

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from importlib import reload

sv3elgfiberassign= Table.read("datafiles/SV3_fiberassign_elg.csv") #Table.read("datafiles/SV3_HIPnotqso_cleaned_R1L40_elg.csv")
sv3lrgfiberassign = Table.read("datafiles/SV3_fiberassign_lrg.csv") #Table.read("datafiles/SV3_HIPnotqso_cleaned_R1L40_lrg.csv")

sv3elgnofiberassign= Table.read("datafiles/SV3_nofiberassign_elg.csv") #Table.read("datafiles/SV3_HIPnotqso_cleaned_R1L40_elg.csv")
sv3lrgnofiberassign = Table.read("datafiles/SV3_nofiberassign_lrg.csv") #Table.read("datafiles/SV3_HIPnotqso_cleaned_R1L40_lrg.csv")

#sv3elgnofiberassign_ros = [sv3elgnofiberassign[(sv3elgnofiberassign['rosette'] == i)] for i in np.arange(1,21)]
#sv3lrgnofiberassign_ros = [sv3lrgnofiberassign[(sv3lrgnofiberassign['rosette'] == i)] for i in np.arange(1,21)]


def countsAfterFiberassign(CiCtable_incomplete,targetids,CiCname):
    """Look up post-fiberassignment CiC counts for a list of targets.

    Builds a TARGETID -> counts hash table from CiCtable_incomplete once, then
    queries it for each entry of targetids. Targets absent from the
    fiberassigned catalog (i.e. never assigned a fiber) get -1.
    """
    ids = np.asarray(CiCtable_incomplete['TARGETID'])
    lookup = dict(zip(ids,np.asarray(CiCtable_incomplete[CiCname])))
    if len(lookup) != len(ids):
        print("Warning: duplicate TARGETIDs in fiberassigned table; keeping last occurrence")
    return np.array([lookup.get(ID,-1) for ID in targetids])

sv3elgnofiberassign['inc_counts'] = countsAfterFiberassign(sv3elgfiberassign,sv3elgnofiberassign['TARGETID'], "N_CiC")
sv3elgnofiberassign['inc_counts_sec'] = countsAfterFiberassign(sv3elgfiberassign,sv3elgnofiberassign['TARGETID'], "N_lrgCiC")


sv3lrgnofiberassign['inc_counts'] = countsAfterFiberassign(sv3lrgfiberassign,sv3lrgnofiberassign['TARGETID'], "N_CiC")
sv3lrgnofiberassign['inc_counts_sec'] = countsAfterFiberassign(sv3lrgfiberassign,sv3lrgnofiberassign['TARGETID'], "N_elgCiC")


sv3elgfiberassign_ros = [sv3elgfiberassign[(sv3elgfiberassign['rosette'] == i)] for i in np.arange(1,21)]
sv3lrgfiberassign_ros = [sv3lrgfiberassign[(sv3lrgfiberassign['rosette'] == i)] for i in np.arange(1,21)]

sv3elgfiberassign_jack = [sv3elgfiberassign[(sv3elgfiberassign['rosette'] != i)] for i in np.arange(1,21)]
sv3lrgfiberassign_jack = [sv3lrgfiberassign[(sv3lrgfiberassign['rosette'] != i)] for i in np.arange(1,21)]


#-----------------------------------------------------------------------------
# Unnormalized count matrices
#
# These are the raw (integer) building blocks of every incompleteness matrix.
# They are kept unnormalized so that a leave-one-out jackknife replicate can be
# formed by subtracting one rosette's counts from the total, matching the
# convention in CiCPlot.histratiojacks rather than re-histogramming.
#
# Index 0 along each *observed* axis is the "never assigned a fiber" (-1) bin.
# It is retained through normalization -- so rows do not sum to 1, and the
# deficit is the fraction of targets that never got a fiber -- then dropped.
#-----------------------------------------------------------------------------

def countMatrix(CiCtable_complete,truecolumn,observedcolumn,maxcounts):
    """Unnormalized count matrix for a single (1D) count distribution.

    Returns C with shape (maxcounts+1, maxcounts+2), where C[i] is the
    histogram of observed counts for objects whose true count is i. Column 0
    of C is the -1 "no fiber" bin; columns 1: are observed counts 0..maxcounts.

    maxcounts must be supplied explicitly rather than taken from the data, so
    that jackknife subsamples all produce matrices of the same shape.
    """
    bins = np.arange(-1,maxcounts+2)
    C = np.zeros((maxcounts+1,maxcounts+2))
    truevals = np.asarray(CiCtable_complete[truecolumn])
    obsvals = np.asarray(CiCtable_complete[observedcolumn])
    for i in np.arange(0,maxcounts+1):
        hist, bins = np.histogram(obsvals[truevals == i],bins = bins)
        C[i] = hist
    return C


def countMatrixBivariate(CiCtable_complete,primarycolumn,secondarycolumn,maxcounts_primaries,
                         maxcounts_secondaries):
    """Unnormalized count matrix for the bivariate (primary, secondary) distribution.

    Returns C with shape (n_prim*n_sec, n_prim+1, n_sec+1), where
    C[i*n_sec + k] is the 2D histogram of observed (primary, secondary) counts
    for objects whose true joint state is (i, k), and n_prim/n_sec are
    maxcounts_primaries+1 / maxcounts_secondaries+1.

    The true joint state is flattened C-order with the PRIMARY as the slow
    axis: flat = i*n_sec + k == np.ravel_multi_index((i,k),(n_prim,n_sec)).
    This is the convention that makes the separable case exactly
    np.kron(primary_matrix, secondary_matrix) -- see kroneckerBivariate.

    Index 0 along each observed axis is the -1 "no fiber" bin.
    """
    n_prim = maxcounts_primaries + 1
    n_sec = maxcounts_secondaries + 1
    binsprim = np.arange(-1,maxcounts_primaries+2)
    binssec = np.arange(-1,maxcounts_secondaries+2)

    C = np.zeros((n_prim*n_sec,n_prim+1,n_sec+1))
    trueprim = np.asarray(CiCtable_complete[primarycolumn])
    truesec = np.asarray(CiCtable_complete[secondarycolumn])
    obsprim = np.asarray(CiCtable_complete['inc_counts'])
    obssec = np.asarray(CiCtable_complete['inc_counts_sec'])
    for i in np.arange(0,n_prim):
        for k in np.arange(0,n_sec):
            sel = (trueprim == i) & (truesec == k)
            hist, binsprim, binssec = np.histogram2d(obsprim[sel],obssec[sel],bins = [binsprim,binssec])
            C[i*n_sec + k] = hist
    return C


def padDiagonal(C):
    """Add one count to each true bin's own observed bin (`addextracount`).

    ON BY DEFAULT. Empirically the inversion is unstable on the real SV3
    catalogs without it: a true bin whose observed distribution is sparse or
    empty leaves the matrix singular or nearly so, and the single count on the
    diagonal is enough to keep it invertible. The cost is a small bias toward
    "this bin was observed correctly", which is negligible wherever the bin is
    well populated and is doing necessary work wherever it is not.

    Apply this to a *summed* count matrix, never per-rosette: padding each of
    20 rosettes and then summing would pad the total 20 times over.
    """
    C = C.copy()
    ntrue = C.shape[0]
    C[np.arange(ntrue),np.arange(ntrue)+1] += 1
    return C


def padDiagonalBivariate(C,maxcounts_secondaries):
    """Bivariate counterpart of padDiagonal: pads the (i,k)->(i,k) cell."""
    C = C.copy()
    n_sec = maxcounts_secondaries + 1
    flat = np.arange(C.shape[0])
    C[flat,flat//n_sec + 1,flat % n_sec + 1] += 1
    return C


#-----------------------------------------------------------------------------
# Normalization and regularized inversion
#-----------------------------------------------------------------------------

def normalizeCountMatrix(C):
    """Normalize a 1D count matrix into M[true, observed] and drop the -1 column.

    Each row is divided by its own total *including* the -1 "no fiber" bin, so
    rows sum to the fraction of objects at that true count that were observed
    at all. True-count bins with no objects normalize to a row of zeros (not
    NaN): a zero row means that true count is unconstrained by the data, which
    the regularized inverse handles by putting no weight on that direction.
    """
    totals = np.sum(C,axis = 1,keepdims = True)
    M = np.divide(C,totals,out = np.zeros_like(C,dtype = float),where = totals > 0)
    return M[:,1:]


def normalizeCountMatrixBivariate(C):
    """Normalize a bivariate count matrix into M[flat_true, flat_observed].

    Normalizes each true cell over its whole 2D observed histogram (including
    the -1 bins), drops the -1 row and column, then flattens the observed axes
    C-order so the observed index matches the true index convention. Empty true
    cells normalize to zero rows, as in normalizeCountMatrix.
    """
    totals = np.sum(C,axis = (1,2),keepdims = True)
    M = np.divide(C,totals,out = np.zeros_like(C,dtype = float),where = totals > 0)
    M = M[:,1:,1:]
    return M.reshape(M.shape[0],-1)


def regularizedInverse(M,rcond):
    """Return the forward operator M.T and its regularized (pseudo-)inverse.

    M is indexed [true, observed]; the transpose is the forward operator
    P(observed | true) that acts on a true distribution to give an observed
    one, and its pseudo-inverse is the correction matrix.

    rcond is passed to np.linalg.pinv and must be chosen deliberately -- see
    scanRcond. numpy's default (~1e-15) only removes numerically-zero singular
    values and will happily amplify near-empty rows into noise. The same rcond
    must be used for every jackknife replicate, or the regularization's own
    variation leaks into the jackknife error.
    """
    forward = M.T
    return forward, np.linalg.pinv(forward,rcond = rcond)


def emptyTrueBins(C):
    """Indices of true-count bins (or flattened true cells) with no objects."""
    axes = tuple(np.arange(1,C.ndim))
    return np.flatnonzero(np.sum(C,axis = axes) == 0)


def makeNormalizer(addextracount = True):
    """Build the count-matrix -> M[true, observed] callable used by the jackknife.

    Returned as a closure so that padDiagonal is applied to each summed
    jackknife replicate exactly once.
    """
    def normalizer(C):
        if addextracount:
            C = padDiagonal(C)
        return normalizeCountMatrix(C)
    return normalizer


def makeNormalizerBivariate(maxcounts_secondaries,addextracount = True):
    """Bivariate counterpart of makeNormalizer."""
    def normalizer(C):
        if addextracount:
            C = padDiagonalBivariate(C,maxcounts_secondaries)
        return normalizeCountMatrixBivariate(C)
    return normalizer


#-----------------------------------------------------------------------------
# Matrix builders
#-----------------------------------------------------------------------------

def incompletenessMatrix(CiCtable_complete,secondarytracerCiC,maxcounts_primaries = None,
                         maxcounts_secondaries = None,rcond = 1e-15,addextracount = True):
    """Build the 1D primary and secondary incompleteness matrices and their inverses.

    Returns (inc_prim, inc_sec, inc_prim_inv, inc_sec_inv): the two forward
    matrices P(observed | true) followed by their regularized inverses. Both
    are returned so that call sites never have to guess whether a matrix has
    already been inverted.

    maxcounts_primaries / maxcounts_secondaries default to the maxima in
    CiCtable_complete, but must be passed explicitly when comparing or
    combining matrices built from different subsamples.
    """
    if maxcounts_primaries is None:
        maxcounts_primaries = np.max(CiCtable_complete["N_CiC"])
    if maxcounts_secondaries is None:
        maxcounts_secondaries = np.max(CiCtable_complete[secondarytracerCiC])

    normalizer = makeNormalizer(addextracount = addextracount)
    Cprim = countMatrix(CiCtable_complete,"N_CiC",'inc_counts',maxcounts_primaries)
    Csec = countMatrix(CiCtable_complete,secondarytracerCiC,'inc_counts_sec',maxcounts_secondaries)

    inc_prim, inc_prim_inv = regularizedInverse(normalizer(Cprim),rcond)
    inc_sec, inc_sec_inv = regularizedInverse(normalizer(Csec),rcond)

    return inc_prim, inc_sec, inc_prim_inv, inc_sec_inv


def bivariateIncompletenessMatrix(CiCtable_complete,secondarytracerCiC,maxcounts_primaries = None,
                                  maxcounts_secondaries = None,rcond = 1e-15,addextracount = True):
    """Build the bivariate incompleteness matrix and its regularized inverse.

    Returns (inc_biv, inc_biv_inv), both of shape (n_prim*n_sec, n_prim*n_sec),
    with the flattening convention documented in countMatrixBivariate.

    This relaxes the two approximations made by the production correction,
    which applies np.outer(elgweights, lrgweights) (CiCMoments.py) -- i.e. the
    incompleteness matrix reduced to its diagonal (no bin migration) and
    assumed separable in primary and secondary.
    """
    if maxcounts_primaries is None:
        maxcounts_primaries = np.max(CiCtable_complete["N_CiC"])
    if maxcounts_secondaries is None:
        maxcounts_secondaries = np.max(CiCtable_complete[secondarytracerCiC])

    normalizer = makeNormalizerBivariate(maxcounts_secondaries,addextracount = addextracount)
    C = countMatrixBivariate(CiCtable_complete,"N_CiC",secondarytracerCiC,maxcounts_primaries,
                             maxcounts_secondaries)
    return regularizedInverse(normalizer(C),rcond)


def columnCompleteness(inc):
    """Fraction of objects in each true bin that were observed at all.

    The forward matrices are indexed [observed, true] and deliberately do not
    have unit column sums: the deficit is the fraction of targets in that true
    bin which never got a fiber.
    """
    return np.sum(inc,axis = 0)


def kroneckerBivariate(inc_prim,inc_sec,completeness = None):
    """The bivariate matrix implied by separable primary/secondary incompleteness.

    With the C-order, primary-slow flattening of countMatrixBivariate, the
    separable hypothesis P(j,l | i,k) = P(j|i)*P(l|k) is a Kronecker product --
    an identity that holds only for this ordering.

    The normalization needs care, though. "Never assigned a fiber" is a single
    event for the whole target, so it is perfectly correlated between primary
    and secondary and does NOT factorize: a bare np.kron(inc_prim, inc_sec) has
    column sums f_i*f_k where the measured bivariate matrix has f_ik. So the
    two 1D matrices are made column-stochastic before the product, and the
    result is rescaled by `completeness` -- the per-true-cell observed fraction,
    i.e. columnCompleteness(inc_biv) of the measured bivariate matrix. Pass it
    to compare shapes on equal footing; leave it None to get the
    column-stochastic separable prediction.

    The difference between the measured bivariate matrix and this prediction is
    the correlated part of the incompleteness -- precisely what the np.outer
    weights in CiCMoments/CiCPlot cannot represent.
    """
    fprim = columnCompleteness(inc_prim)
    fsec = columnCompleteness(inc_sec)
    stochastic_prim = np.divide(inc_prim,fprim,out = np.zeros_like(inc_prim),where = fprim > 0)
    stochastic_sec = np.divide(inc_sec,fsec,out = np.zeros_like(inc_sec),where = fsec > 0)

    separable = np.kron(stochastic_prim,stochastic_sec)
    if completeness is None:
        return separable
    return separable*np.asarray(completeness)


#-----------------------------------------------------------------------------
# Jackknife matrices
#
# These build the ARRAY of leave-one-out incompleteness matrices, one per
# rosette, to be applied to correspondingly-jackknifed data downstream (in
# CiCPlot / CiCMoments), where the error is then computed from the scatter of
# the corrected quantities. No error is computed here.
#
# Replicate k is formed by subtracting rosette k's raw counts from the total
# rather than re-histogramming, matching the convention in CiCPlot.getjack.
#-----------------------------------------------------------------------------

def rosetteList(*tables):
    """Sorted rosette IDs present in every supplied table.

    Uses the intersection rather than np.arange(1,21) so that a rosette which
    is missing or empty in one catalog does not produce an unusable subsample.
    """
    common = set.intersection(*[set(np.unique(np.asarray(t['rosette']))) for t in tables])
    return np.array(sorted(common))


def correctHistogram(C,h,normalizer,rcond):
    """Apply the correction implied by count matrix C to observed histogram h.

    h is an unnormalized observed histogram; it is converted to a distribution
    here so that jackknife replicates can be formed by subtracting raw counts.
    The corrected distribution is renormalized to unit sum.
    """
    forward, inv = regularizedInverse(normalizer(C),rcond)
    corrected = inv @ (h/np.sum(h))
    return corrected/np.sum(corrected)


def leaveOneOutMatrices(countmatrices,normalizer,rcond):
    """Turn per-region count matrices into leave-one-out matrices and inverses.

    Returns (matrices, inverses, full, fullinverse): `matrices[k]` is built
    from every region except k, and `full` from all of them. All are forward
    operators indexed [observed, true]; the inverses are the corrections to
    apply to data.

    Regions are not dropped even if some true bin is empty without them -- such
    a bin normalizes to a zero row and the regularized inverse simply puts no
    weight on it. Any region that leaves a true bin unconstrained is reported,
    since the correction there rests entirely on the regularization.
    """
    countmatrices = np.asarray(countmatrices)
    Ctotal = np.sum(countmatrices,axis = 0)

    full, fullinverse = regularizedInverse(normalizer(Ctotal),rcond)

    matrices = []
    inverses = []
    unconstrained = []
    for k in np.arange(len(countmatrices)):
        Cjack = Ctotal - countmatrices[k]
        empty = emptyTrueBins(Cjack)
        if len(empty) > 0:
            unconstrained.append((k,list(empty)))
        forward, inverse = regularizedInverse(normalizer(Cjack),rcond)
        matrices.append(forward)
        inverses.append(inverse)

    if unconstrained:
        allbins = sorted(set(b for _,bins in unconstrained for b in bins))
        if all(bins == unconstrained[0][1] for _,bins in unconstrained):
            print("Warning: true bins %s are empty in every leave-one-out sample; "
                  "the correction there rests entirely on the regularization"%allbins)
        else:
            print("Warning: %d of %d regions leave some true bin empty (bins seen: %s); "
                  "the correction there rests entirely on the regularization"
                  %(len(unconstrained),len(countmatrices),allbins))

    return np.array(matrices), np.array(inverses), full, fullinverse


def jackknifeMatrices(CiCtable_complete,secondarytracerCiC,maxcounts_primaries,
                      maxcounts_secondaries,rcond,addextracount = True,rosettes = None):
    """Array of leave-one-out 1D incompleteness matrices, one per rosette.

    Returns (rosettes, prim, sec). Each of prim and sec is the 4-tuple
    (matrices, inverses, full, fullinverse) described in leaveOneOutMatrices,
    with `matrices[k]` built from every rosette except rosettes[k].

    Apply inverses[k] to the data jackknife sample that also excludes
    rosettes[k], so the matrix and the data describe the same footprint; the
    error then comes from the scatter of the corrected quantities.

    maxcounts_primaries / maxcounts_secondaries are required here rather than
    taken from the data: a subsample can have a lower maximum, which would give
    replicates of different shapes.
    """
    if rosettes is None:
        rosettes = rosetteList(CiCtable_complete)

    rosettecolumn = np.asarray(CiCtable_complete['rosette'])
    Cprim = []
    Csec = []
    for r in rosettes:
        complete_r = CiCtable_complete[rosettecolumn == r]
        Cprim.append(countMatrix(complete_r,"N_CiC",'inc_counts',maxcounts_primaries))
        Csec.append(countMatrix(complete_r,secondarytracerCiC,'inc_counts_sec',maxcounts_secondaries))

    normalizer = makeNormalizer(addextracount = addextracount)
    return (rosettes,
            leaveOneOutMatrices(Cprim,normalizer,rcond),
            leaveOneOutMatrices(Csec,normalizer,rcond))


def jackknifeMatricesBivariate(CiCtable_complete,secondarytracerCiC,maxcounts_primaries,
                               maxcounts_secondaries,rcond,addextracount = True,rosettes = None):
    """Array of leave-one-out bivariate incompleteness matrices, one per rosette.

    Returns (rosettes, (matrices, inverses, full, fullinverse)), with matrices
    of shape (n_prim*n_sec, n_prim*n_sec) flattened primary-slow as documented
    in countMatrixBivariate. A corrected vector reshapes back to the 2D
    distribution with reshape(maxcounts_primaries+1, maxcounts_secondaries+1).
    """
    if rosettes is None:
        rosettes = rosetteList(CiCtable_complete)

    rosettecolumn = np.asarray(CiCtable_complete['rosette'])
    C = []
    for r in rosettes:
        complete_r = CiCtable_complete[rosettecolumn == r]
        C.append(countMatrixBivariate(complete_r,"N_CiC",secondarytracerCiC,
                                      maxcounts_primaries,maxcounts_secondaries))

    normalizer = makeNormalizerBivariate(maxcounts_secondaries,addextracount = addextracount)
    return rosettes, leaveOneOutMatrices(C,normalizer,rcond)


#-----------------------------------------------------------------------------
# Choosing the regularization strength
#-----------------------------------------------------------------------------

def scanRcond(CiCtable_complete,CiCtable_incomplete,truecolumn,observedcolumn,maxcounts,
              rcondvalues,addextracount = True):
    """Pick rcond by how well the corrected histogram recovers the known truth.

    The no-fiberassign catalog IS the complete distribution, so the bias of the
    regularized correction can be measured directly rather than guessed: for
    each rcond, correct the observed histogram and compare to the true one.
    Returns (rcondvalues, rmsdeviations) where the deviation is the RMS of
    corrected/complete - 1, the same ratio the notebook already plots.

    The jackknife measures the variance of the regularized estimator, not its
    distance from truth, so this scan is the separate check on bias. Use the
    chosen value for every replicate.
    """
    bins = np.arange(maxcounts+2)
    complete, edges = np.histogram(np.asarray(CiCtable_complete[truecolumn]),bins = bins)
    observed, edges = np.histogram(np.asarray(CiCtable_incomplete[truecolumn]),bins = bins)

    C = countMatrix(CiCtable_complete,truecolumn,observedcolumn,maxcounts)
    normalizer = makeNormalizer(addextracount = addextracount)
    return np.asarray(rcondvalues), rcondDeviations(C,observed,complete,normalizer,rcondvalues)


def rcondDeviations(C,observedhist,truehist,normalizer,rcondvalues):
    """RMS of corrected/true - 1 for each rcond. Works for 1D and flattened bivariate.

    observedhist and truehist are unnormalized; both are converted to
    distributions here. Bins where the truth is empty are skipped, since the
    ratio is undefined there.
    """
    truth = np.ravel(truehist)/np.sum(truehist)
    observed = np.ravel(observedhist)
    nonzero = truth > 0
    deviations = []
    for rcond in rcondvalues:
        corrected = np.ravel(correctHistogram(C,observed,normalizer,rcond))
        deviations.append(np.sqrt(np.mean((corrected[nonzero]/truth[nonzero] - 1)**2)))
    return np.asarray(deviations)


#-----------------------------------------------------------------------------
# Building and saving the matrices for a whole catalog
#-----------------------------------------------------------------------------

#CiCPlot uses a single binmax for all four 1D tracer combinations, so every
#matrix has to be built on the same grid. binmax=6 in PlotCiC2.ipynb means
#counts 0..5, hence maxcounts=5 and 6x6 (36x36 bivariate) matrices.
MAXCOUNTS = 5
RCONDVALUES = np.logspace(-15,-0.5,25)


def chooseRcondAndBuild(percountmatrices,perrosetteobserved,perrosettetrue,normalizer,
                        rcondvalues = RCONDVALUES,label = ""):
    """Pick rcond by SPLIT-SAMPLE validation, then build the leave-one-out matrices.

    The scan must be split-sample or it is meaningless. The matrix is derived
    from the same objects that make up the observed histogram (the fiberassign
    catalog is the subset of the no-fiberassign catalog that got fibers), so a
    matrix tested against its own sample reproduces the truth algebraically at
    any rcond -- the scan would then always pick the smallest value offered and
    tell you nothing.

    So: build the trial matrix from half the rosettes, and judge it against the
    complete and observed histograms of the OTHER half. That measures how well
    the correction generalizes, which is what rcond actually trades against.

    The final matrices are then built from ALL rosettes at the chosen rcond, and
    the same rcond is used for every jackknife replicate so the regularization
    adds no spurious scatter to the incompleteness error.

    Returns (leaveOneOutMatrices result, chosen rcond, deviation curve).
    """
    counts = np.asarray(percountmatrices)
    observed = np.asarray(perrosetteobserved)
    truth = np.asarray(perrosettetrue)

    calibration = np.arange(len(counts)) % 2 == 0
    Ccalibration = np.sum(counts[calibration],axis = 0)
    observedtest = np.sum(observed[~calibration],axis = 0)
    truthtest = np.sum(truth[~calibration],axis = 0)

    deviations = rcondDeviations(Ccalibration,observedtest,truthtest,normalizer,rcondvalues)
    bestindex = np.nanargmin(deviations)
    best = rcondvalues[bestindex]
    print("  %-14s rcond = %.3e  (split-sample RMS corrected/true - 1 = %.4f; "
          "worst in scan %.4f)"%(label,best,deviations[bestindex],np.nanmax(deviations)))
    if bestindex in (0,len(rcondvalues)-1):
        print("    NOTE: the optimum is at %s end of the scanned range. Either no "
              "regularization is needed (%s end) or the range does not reach far "
              "enough -- widen RCONDVALUES and rerun to be sure."
              %("the low" if bestindex == 0 else "the high",
                "low" if bestindex == 0 else "high"))
    return leaveOneOutMatrices(counts,normalizer,best), best, deviations


def buildMatricesForCatalog(CiCtable_complete,CiCtable_incomplete,secondarytracerCiC,
                            maxcounts = MAXCOUNTS,rcondvalues = RCONDVALUES,
                            addextracount = True,rosettes = None):
    """Build primary, secondary and bivariate matrix sets for one primary tracer.

    CiCtable_complete is the no-fiberassign catalog carrying the inc_counts
    columns; CiCtable_incomplete is the fiberassign catalog, used for the
    observed histograms that the rcond scan is judged against.

    Returns a dict ready to hand to saveMatrices.
    """
    if rosettes is None:
        rosettes = rosetteList(CiCtable_complete,CiCtable_incomplete)
    rosettecolumn = np.asarray(CiCtable_complete['rosette'])
    subsets = [CiCtable_complete[rosettecolumn == r] for r in rosettes]

    bins = np.arange(maxcounts+2)
    normalizer1d = makeNormalizer(addextracount = addextracount)
    normalizerbiv = makeNormalizerBivariate(maxcounts,addextracount = addextracount)

    #the rcond scan needs PER-ROSETTE histograms so it can be split-sample
    incompleterosette = np.asarray(CiCtable_incomplete['rosette'])
    incompletesubsets = [CiCtable_incomplete[incompleterosette == r] for r in rosettes]

    def perrosette1d(tablelist,column):
        return [np.histogram(np.asarray(t[column]),bins = bins)[0] for t in tablelist]

    def perrosettebivariate(tablelist,column):
        return [np.histogram2d(np.asarray(t["N_CiC"]),np.asarray(t[column]),
                               bins = [bins,bins])[0].ravel() for t in tablelist]

    #primary: true N_CiC vs observed inc_counts
    Cprim = [countMatrix(s,"N_CiC",'inc_counts',maxcounts) for s in subsets]
    prim, rcondprim, devprim = chooseRcondAndBuild(
        Cprim,perrosette1d(incompletesubsets,"N_CiC"),perrosette1d(subsets,"N_CiC"),
        normalizer1d,rcondvalues,label = "primary")

    #secondary
    Csec = [countMatrix(s,secondarytracerCiC,'inc_counts_sec',maxcounts) for s in subsets]
    sec, rcondsec, devsec = chooseRcondAndBuild(
        Csec,perrosette1d(incompletesubsets,secondarytracerCiC),
        perrosette1d(subsets,secondarytracerCiC),
        normalizer1d,rcondvalues,label = "secondary")

    #bivariate, in this module's primary-slow convention: axis 0 is the primary.
    #CiCPlot transposes for its ELG-centered histogram via primaryaxis=1.
    Cbiv = [countMatrixBivariate(s,"N_CiC",secondarytracerCiC,maxcounts,maxcounts) for s in subsets]
    biv, rcondbiv, devbiv = chooseRcondAndBuild(
        Cbiv,perrosettebivariate(incompletesubsets,secondarytracerCiC),
        perrosettebivariate(subsets,secondarytracerCiC),
        normalizerbiv,rcondvalues,label = "bivariate")

    return {'rosettes': np.asarray(rosettes),'maxcounts': maxcounts,
            'rcondvalues': np.asarray(rcondvalues),
            'rcond_prim': rcondprim,'rcond_sec': rcondsec,'rcond_biv': rcondbiv,
            'deviations_prim': devprim,'deviations_sec': devsec,'deviations_biv': devbiv,
            'prim_matrices': prim[0],'prim_inverses': prim[1],
            'prim_full': prim[2],'prim_fullinverse': prim[3],
            'sec_matrices': sec[0],'sec_inverses': sec[1],
            'sec_full': sec[2],'sec_fullinverse': sec[3],
            'biv_matrices': biv[0],'biv_inverses': biv[1],
            'biv_full': biv[2],'biv_fullinverse': biv[3]}


def saveMatrices(filename,matrixset):
    """Write one primary tracer's matrix set to a .npz bundle."""
    np.savez(filename,**matrixset)
    print("wrote %s"%filename)


def loadMatrices(elgfile = "datafiles/sv3incmatrix_elg.npz",
                 lrgfile = "datafiles/sv3incmatrix_lrg.npz"):
    """Load saved matrices as the six-entry list CiCPlot.jackknifeCiCError wants.

    Order is [elgelg, lrglrg, elglrg, lrgelg, elgbivariate, lrgbivariate]: the
    two "primary" sets are counts of a tracer around itself, and the two
    "secondary" sets are the cross-tracer counts.

    Each entry is the (matrices, inverses, full, fullinverse) tuple that
    leaveOneOutMatrices produces, so it can also be unpacked directly.
    """
    elg = np.load(elgfile)
    lrg = np.load(lrgfile)
    def entry(bundle,prefix):
        return (bundle[prefix+'_matrices'],bundle[prefix+'_inverses'],
                bundle[prefix+'_full'],bundle[prefix+'_fullinverse'])
    return [entry(elg,'prim'),entry(lrg,'prim'),
            entry(elg,'sec'),entry(lrg,'sec'),
            entry(elg,'biv'),entry(lrg,'biv')]


if __name__ == "__main__":
    print("ELG primaries (secondary tracer: N_lrgCiC)")
    elgset = buildMatricesForCatalog(sv3elgnofiberassign,sv3elgfiberassign,"N_lrgCiC",addextracount=True)
    saveMatrices("datafiles/sv3incmatrix_elg_plusone.npz",elgset)

    print("LRG primaries (secondary tracer: N_elgCiC)")
    lrgset = buildMatricesForCatalog(sv3lrgnofiberassign,sv3lrgfiberassign,"N_elgCiC",addextracount=True)
    saveMatrices("datafiles/sv3incmatrix_lrg_plusone.npz",lrgset)

    print("\nLoad into CiCPlot with:")
    print("    import IncompletenessMatrix")
    print("    matrices = IncompletenessMatrix.loadMatrices()")
    print("    cat = CiCPlot.jackknifeCiCError(elgtables,lrgtables,%d,matrices=matrices)"
          %(MAXCOUNTS+1))
    print("NOTE: elgtables/lrgtables must be ordered to match rosettes =",elgset['rosettes'])


