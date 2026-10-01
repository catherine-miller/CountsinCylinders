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
from math import log10, floor, comb
import scipy.stats


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def jackknifeSE(replicates,fullvalue):
    """Jackknife standard error from a set of leave-one-out replicates.

    Same convention as CiCPlot.jackknifeSE: squared deviations of each replicate
    from the FULL-SAMPLE value (not from the mean of the replicates), with
    prefactor (nsamples-1)/nsamples.
    """
    replicates = np.asarray(replicates)
    nsamples = len(replicates)
    deviations = replicates - fullvalue
    return np.sqrt((nsamples-1)/nsamples*np.sum(deviations**2,axis = 0))


class CiCHistnZ:
    def __init__(self,tables,elgname = "N_elgCiC",lrgname="N_lrgCiC",zname = 'Z',binmaxelg = 24,binmaxlrg = 24,
        redshiftbins = np.linspace(0.80695,0.9919,num=20), matrices = None):
        """Build a joint (N_ELG, N_LRG, redshift) histogram from a list of CiC tables.

        Args:
            tables: List of astropy Tables, one per rosette (or per subsample or mock
                catalog). Each table must contain per-primary-galaxy ELG counts, LRG
                counts, and redshift columns. Moment errors are a leave-one-out jackknife
                over these tables.
            elgname: Column name for ELG counts-in-cylinders.
            lrgname: Column name for LRG counts-in-cylinders.
            zname: Column name for redshift.
            binmaxelg: Number of ELG count bins (covers N=0 to N=binmaxelg-1). Ignored if
                matrices is provided.
            binmaxlrg: Number of LRG count bins (covers N=0 to N=binmaxlrg-1). Ignored if
                matrices is provided.
            redshiftbins: Edges of the redshift bins (length = n_zbins + 1).
            matrices: Bivariate incompleteness correction from IncompletenessMatrix: one
                entry of IncompletenessMatrix.loadMatrices(), i.e. index 4 for
                ELG-centered and index 5 for LRG-centered tables. Each entry is the
                (matrices, inverses, full, fullinverse) tuple from leaveOneOutMatrices.
                Defaults to None, meaning no incompleteness correction is applied and the
                incompleteness error is zero.

                The matrix fixes the count range, and binmaxelg/binmaxlrg are ignored. The
                matrices carry an OVERFLOW state (counts above the range) as the last
                state along each tracer's axis, so an ((n+1)^2, (n+1)^2) matrix gives
                both tracers n in-range count bins, 0..n-1. The raw histograms
                (hists_raw, histsum_raw) include the overflow bin, because the
                correction needs it to correct the in-range bins properly; correctHist
                drops it, so the corrected histograms, the moments and their errors only
                ever see counts 0..n-1. Rebuild the matrices with a larger MAXCOUNTS to
                extend that range.

                The same matrix is applied to every redshift slice, to raw counts, with
                no per-slice renormalization.

                The primary tracer is whichever of elgname/lrgname is "N_CiC". The
                matrices flatten primary-slow (flat = i*n_sec + k), so the LRG-centered
                histogram is transposed before the correction and back afterwards.

                The rosette ordering of the leave-one-out matrices MUST match the
                ordering of tables. Only the number of each is checked here.
        """
        self.fullinverse = None
        self.jackinverses = None
        self.primaryaxis = None
        if matrices is not None:
            if (elgname == "N_CiC") == (lrgname == "N_CiC"):
                raise ValueError("cannot tell the primary tracer: exactly one of elgname (%s) and "
                                 "lrgname (%s) must be 'N_CiC' when matrices is given"%(elgname,lrgname))
            self.primaryaxis = 0 if elgname == "N_CiC" else 1
            jackinverses, fullinverse = np.asarray(matrices[1]), np.asarray(matrices[3])
            side = fullinverse.shape[0]
            nstates = int(round(np.sqrt(side)))
            if fullinverse.ndim != 2 or fullinverse.shape[1] != side or nstates*nstates != side:
                raise ValueError("fullinverse has shape %s; expected ((n+1)^2, (n+1)^2) for a bivariate "
                                 "matrix with an overflow state"%(fullinverse.shape,))
            n = nstates - 1
            if len(jackinverses) != len(tables):
                raise ValueError("matrices has %d leave-one-out matrices but %d tables were supplied; "
                                 "they must correspond one to one"%(len(jackinverses),len(tables)))
            binmaxelg = binmaxlrg = n
            self.fullinverse = fullinverse
            self.jackinverses = jackinverses
        #half-integer edges, so bin N holds exactly count N
        bins = [np.arange(-0.5,binmaxelg+0.5,1),np.arange(-0.5,binmaxlrg+0.5,1),redshiftbins]
        self.overflow = matrices is not None
        if self.overflow:
            #counts above the range are clipped into one extra (overflow) bin per tracer
            rawbins = [np.arange(-0.5,binmaxelg+1.5,1),np.arange(-0.5,binmaxlrg+1.5,1),redshiftbins]
            self.hists_raw = np.array([np.histogramdd((np.minimum(np.asarray(table[elgname]),binmaxelg),
                                                       np.minimum(np.asarray(table[lrgname]),binmaxlrg),
                                                       table[zname]),bins = rawbins)[0]
                for table in tables])
        else:
            self.hists_raw = np.array([np.histogramdd((table[elgname],table[lrgname],table[zname]),bins = bins)[0]
                for table in tables])
        self.histsum_raw = np.sum(self.hists_raw,axis = 0)
        if self.overflow:
            alltab = vstack(tables)
            overflow = np.sum((np.asarray(alltab[elgname]) >= binmaxelg) | (np.asarray(alltab[lrgname]) >= binmaxlrg))
            if overflow > 0:
                print("Note: %d of %d primaries (%.3g%%) have counts above the matrix range (0..%d); "
                      "they are used only to correct the in-range bins and never enter a moment"
                      %(overflow,len(alltab),100*overflow/len(alltab),binmaxelg-1))
        self.hists = [self.correctHist(h,self.fullinverse) for h in self.hists_raw]
        self.histsum = self.correctHist(self.histsum_raw,self.fullinverse)
        self.zbins = bins[2]
        self.zcenters = [(self.zbins[i]+self.zbins[i+1])/2 for i in range(len(self.zbins)-1)]
        self.binmax_elg = bins[0][-1]
        self.binmax_lrg = bins[1][-1]
        self.nbins_elg = len(bins[0]) - 1
        self.nbins_lrg = len(bins[1]) - 1
        self.ntables = len(tables)

    def correctHist(self,hist,inverse):
        """Apply an incompleteness correction matrix to a raw (N_ELG, N_LRG, z) histogram.

        Each redshift slice is corrected with the same matrix. The correction is linear
        and does not renormalize, so correcting per-table histograms and summing gives
        the same result as correcting the pooled histogram. If inverse is None the
        histogram is not corrected.

        If this object was built with matrices, hist must be a raw histogram including
        the overflow bins, and the overflow row and column are dropped from the result
        -- whether or not a correction is applied -- so what comes back always covers
        the in-range counts only.
        """
        if inverse is not None:
            oriented = hist.transpose(1,0,2) if self.primaryaxis == 1 else hist
            n0, n1, nz = oriented.shape
            corrected = (inverse @ oriented.reshape(n0*n1,nz)).reshape(n0,n1,nz)
            hist = corrected.transpose(1,0,2) if self.primaryaxis == 1 else corrected
        if self.overflow:
            hist = hist[:-1,:-1,:]
        return hist

    def momentGrid(self,momentfunction,order,otherargs = []):
        """Evaluate momentfunction(order, i, j, k, *otherargs) on every histogram cell.

        Returns an array of shape (n_elg_bins, n_lrg_bins, n_zbins). i and j are the ELG
        and LRG count values (0, 1, 2, ...) and k is the redshift bin index.
        """
        grid = np.zeros((self.nbins_elg,self.nbins_lrg,len(self.zbins)-1))
        for i in range(self.nbins_elg):
            for j in range(self.nbins_lrg):
                for k in range(len(self.zbins)-1):
                    grid[i,j,k] = momentfunction(order,i,j,k,*otherargs)
        return grid

    def getMoment(self,hist,momentfunction,order,otherargs = []):
        """Average momentfunction over the joint (N_ELG, N_LRG, redshift) histogram.

        Returns the histogram-weighted mean of momentfunction(order, i, j, k, *otherargs)
        over all cells (see momentGrid).

        Args:
            hist: 3-D array of shape (n_elg_bins, n_lrg_bins, n_zbins).
            momentfunction: Callable f(order, n_elg, n_lrg, zbin, *otherargs) -> scalar.
                Must follow the signature expected by makeMomentTable (see momentSimple,
                factorialMoment, Ntilde, nZintegral, vCylPower for examples).
            order: List [elg_order, lrg_order] passed as first argument to momentfunction.
            otherargs: Additional arguments forwarded to momentfunction.

        Returns:
            Histogram-weighted average of momentfunction as a scalar.
        """
        grid = self.momentGrid(momentfunction,order,otherargs)
        return np.sum(grid*hist)/np.sum(hist)

    def makeMomentTable(self, maxorder,momentfunction,extraargs=[],filename = '',savesubsamples = False):
        """Compute a table of moments for all orders up to maxorder.

        Iterates over all [elg_order, lrg_order] pairs with elg_order + lrg_order <= maxorder,
        excluding [0, 0]. The moment is computed from the pooled, corrected histogram, and
        its error is a leave-one-out jackknife over the tables, in the same way as
        CiCPlot.jackknifeHistogram. There are two components:
            sample error: correction held at the full-sample matrix, one table removed
                from the data at a time
            incompleteness error: data held at the full sample, correction replaced by
                each leave-one-out matrix in turn (zero if no matrices were given)
        These are combined in quadrature, which drops the cross-term between them. That
        is valid when the correction is calibrated on different objects from the ones
        being corrected (e.g. SV3 mocks with and without fiberassign, applied to SV3
        data); it would NOT hold if the correction and the data were the same objects.

        Args:
            maxorder: Maximum total order (elg_order + lrg_order) to compute.
            momentfunction: Per-primary integrand function passed to getMoment. See getMoment
                for the required signature.
            extraargs: Additional arguments forwarded to momentfunction via getMoment.
            filename: If non-empty, saves the table to this path in ascii.ecsv format.
            savesubsamples: If True, adds a "SubsampleMoments" column containing the
                moment of each table's own (corrected) histogram.

        Returns:
            Astropy Table with columns "ELG Order, LRG Order", "Moment", "Standard Error"
            (the total error), "Sample Error", "Incompleteness Error", and optionally
            "SubsampleMoments".
        """
        def moment(grid,hist):
            return np.sum(grid*hist)/np.sum(hist)

        total = self.histsum_raw
        samplehists = [self.correctHist(total - h,self.fullinverse) for h in self.hists_raw]
        if self.jackinverses is not None:
            inchists = [self.correctHist(total,inverse) for inverse in self.jackinverses]
        momentlist = []
        orderlist = []
        stderrlist = []
        sampleerrlist = []
        incerrlist = []
        if savesubsamples:
            ssamples = []
        for i in range(maxorder + 1):
            j = 0
            while (i + j < maxorder + 1):
                if i == 0 and j == 0: 
                    j += 1
                    continue
                grid = self.momentGrid(momentfunction,[i,j],otherargs=extraargs)
                central = moment(grid,self.histsum)
                sampleerr = jackknifeSE([moment(grid,h) for h in samplehists],central)
                if self.jackinverses is not None:
                    incerr = jackknifeSE([moment(grid,h) for h in inchists],central)
                else:
                    incerr = 0.
                momentlist.append(central)
                sampleerrlist.append(sampleerr)
                incerrlist.append(incerr)
                stderrlist.append(np.sqrt(sampleerr**2 + incerr**2))
                orderlist.append([i,j])
                if savesubsamples:
                    ssamples.append([moment(grid,h) for h in self.hists])
                j += 1
        columns = [orderlist,momentlist,stderrlist,sampleerrlist,incerrlist]
        names = ["ELG Order, LRG Order","Moment","Standard Error","Sample Error","Incompleteness Error"]
        if savesubsamples:
            columns.append(ssamples)
            names.append("SubsampleMoments")
        moments = Table(columns,names=names,
                   meta={'name': "Moments of ELG-centered bivariate counts in cylinders distribution"})
        if len(filename) > 0:
            moments.write(filename, format = "ascii.ecsv", overwrite=True)
            print("saved file "+filename)
        #moments['Moment'].format = '%.4e'
        #moments['Standard Error'].format = '%.4e'
        return moments

        
def getnZ(table,solidangle,zlim,nbins,zname = "Z"):
    """Compute the number density n(z) of a galaxy sample in redshift bins.

    Args:
        table: Astropy Table containing the galaxy catalog.
        solidangle: Survey solid angle in steradians.
        zlim: (z_min, z_max) tuple defining the redshift range.
        nbins: Number of redshift bin edges (produces nbins-1 bins).
        zname: Column name for redshift.

    Returns:
        nz: List of number densities (Mpc^-3, no factors of h) for each redshift bin.
        bins: Array of bin edges used.
    """
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
    """Estimate number density n(z) from the mean CiC count divided by cylinder volume.

    Alternative to getnZ when the survey solid angle is not known. Uses the identity
    n = <N_CiC> / V_cyl, where <N_CiC> is the mean counts-in-cylinders in each redshift bin.

    Args:
        table: Astropy Table containing the galaxy catalog with CiC counts.
        zlim: (z_min, z_max) tuple defining the redshift range.
        nbins: Number of redshift bin edges (produces nbins-1 bins).
        Vcyl: Cylinder volume in Mpc^3.
        zname: Column name for redshift.
        cicname: Column name for counts-in-cylinders.

    Returns:
        nz: List of estimated number densities (Mpc^-3) for each redshift bin.
        bins: Array of bin edges used.
    """
    bins = np.linspace(*zlim,num=nbins)
    nz = []
    for i in range(len(bins)-1):
        nz.append(np.average(table[(table[zname] < bins[i+1]) & (table[zname] > bins[i])][cicname])/Vcyl)
    return nz, bins


def momentSimple(order,n_elg,n_lrg,z):
    """Per-primary integrand for the raw bivariate moment <N_ELG^m * N_LRG^n>.

    Pass to makeMomentTable to compute the ordinary moments of the counts-in-cylinders
    distribution.

    Args:
        order: [elg_order, lrg_order] = [m, n].
        n_elg: ELG count for this histogram bin (integer).
        n_lrg: LRG count for this histogram bin (integer).
        z: Redshift bin index (unused; included for consistent signature).

    Returns:
        n_elg^m * n_lrg^n
    """
    n = order[0]
    m = order[1]
    return n_elg**n*n_lrg**m

def factorialProduct(N,m):
    """Compute the falling factorial (N)_m = N*(N-1)*...*(N-m+1).

    Returns 1 for m=0. Used to build factorial and Ntilde moments.

    Args:
        N: Integer count value.
        m: Order of the falling factorial (number of terms).

    Returns:
        Integer product N*(N-1)*...*(N-m+1).
    """
    product = 1
    for i in range(m):
        product = product*(N-i)
    return product

def factorialProductCombinations(N,m,nVcyl):
    """Per-primary integrand for the univariate Ntilde_m statistic.

    Computes (N)_m + sum_{l=1}^{m} C(m,l) * (-n*V)^l * (N)_{m-l}, where (N)_k is the
    falling factorial. When averaged over galaxy-centered cylinders via getMoment, this
    gives <Ntilde_m>. When averaged over randomly placed cylinders, <Ntilde_m>_randoms
    subtracts to isolate the clustering signal (see avgnpcf).

    Args:
        N: Integer count value (ELG or LRG counts in one cylinder).
        m: Moment order.
        nVcyl: Mean expected count n * V_cyl in one cylinder (number density * volume).

    Returns:
        Scalar value of the Ntilde_m integrand for this cylinder.
    """
    Ntilde = factorialProduct(N,m)
    for l in range(1,m+1):
        Ntilde += (-nVcyl)**l*comb(m,l)*factorialProduct(N,m-l)
    return Ntilde

def factorialMoment(order,n_elg,n_lrg,zbin):
    """Per-primary integrand for the bivariate factorial moment <(N_ELG)_m * (N_LRG)_n>.

    The factorial moment eliminates shot-noise terms that appear in ordinary moments,
    making it easier to relate to correlation functions. Pass to makeMomentTable to
    compute the bivariate factorial moments of the counts-in-cylinders distribution.

    Args:
        order: [elg_order, lrg_order] = [m, n].
        n_elg: ELG count for this histogram bin (integer).
        n_lrg: LRG count for this histogram bin (integer).
        zbin: Redshift bin index (unused; included for consistent signature).

    Returns:
        (n_elg)_m * (n_lrg)_n, i.e. the product of two falling factorials.
    """
    n = order[0]
    m = order[1]
    return factorialProduct(n_elg,n)*factorialProduct(n_lrg,m)

def Ntilde(order,n_elg,n_lrg,zbin,Vcyl,nz1,nz2):
    """Per-primary integrand for the bivariate Ntilde_{m,n} statistic.

    Ntilde_{m,n} is the product of two univariate Ntilde statistics, one for ELGs and one
    for LRGs. When averaged over galaxy-centered cylinders and compared to randomly placed
    cylinders, this quantity isolates the cylinder-averaged (m+1, n)-point cross-correlation
    function (see avgnpcf). Pass to makeMomentTable along with extraargs=[Vcyl, nz_elg, nz_lrg].

    Args:
        order: [elg_order, lrg_order] = [m, n].
        n_elg: ELG count for this histogram bin (integer).
        n_lrg: LRG count for this histogram bin (integer).
        zbin: Redshift bin index used to look up number densities in nz1 and nz2.
        Vcyl: Cylinder volume in Mpc^3.
        nz1: ELG number density as a list indexed by redshift bin. Must use the same
            redshift binning as the CiC histogram passed to getMoment.
        nz2: LRG number density as a list indexed by redshift bin. Same binning requirement.

    Returns:
        Scalar Ntilde_{m,n} integrand for this cylinder.
    """
    n = order[0]
    m = order[1]
    density1 = nz1[zbin]
    density2 = nz2[zbin]
    center1 = Vcyl*density1
    center2 = Vcyl*density2
    return factorialProductCombinations(n_elg,n,center1)*factorialProductCombinations(n_lrg,m,center2)

def nZintegral(order,n_elg,n_lrg,zbin,nz1,nz2):
    """Per-primary integrand for the number-density denominator factor n_ELG^m * n_LRG^n.

    Used as the momentfunction in makeMomentTable to build the denominator table for
    avgnpcf. For a table entry at order [m, n], this returns n_ELG^m * n_LRG^n at the
    redshift of that cylinder.

    Args:
        order: [elg_order, lrg_order] = [m, n].
        n_elg: ELG count bin index (unused; included for consistent signature).
        n_lrg: LRG count bin index (unused; included for consistent signature).
        zbin: Redshift bin index used to look up number densities in nz1 and nz2.
        nz1: ELG number density as a list indexed by redshift bin.
        nz2: LRG number density as a list indexed by redshift bin.

    Returns:
        nz1[zbin]^m * nz2[zbin]^n
    """
    n = order[0]
    m = order[1]
    return nz1[zbin]**n*nz2[zbin]**m

def vCylPower(order,n_elg,n_lrg,zbin,V_cyl):
    """Per-primary integrand for the cylinder-volume denominator factor V_cyl^(m+n).

    Used as the momentfunction in makeMomentTable to build the volume denominator table
    for avgnpcf. For a table entry at order [m, n], returns V_cyl^(m+n).

    Args:
        order: [elg_order, lrg_order] = [m, n].
        n_elg: ELG count bin index (unused; included for consistent signature).
        n_lrg: LRG count bin index (unused; included for consistent signature).
        zbin: Redshift bin index (unused; included for consistent signature).
        V_cyl: Cylinder volume in Mpc^3.

    Returns:
        V_cyl^(m+n)
    """
    return V_cyl**(order[0] + order[1])


def showMomentTable(table,title="",lrg = False,scientific = True,filename = "momentfigs/test"):
    """Display a moment table as a grid plot and save to PNG.

    Draws a triangular grid of cells (since moments with elg_order + lrg_order > maxorder
    are not computed) and places the moment value and standard error in each cell.

    Args:
        table: Astropy Table returned by makeMomentTable.
        title: Plot title string.
        lrg: If True, put LRG order on the x-axis and ELG order on the y-axis (i.e.,
            the table was computed for LRG-centered cylinders). If False, ELG order is
            on the x-axis.
        scientific: If True, format values in scientific notation; otherwise use fixed
            decimal with 3 significant figures.
        filename: Output path (without extension); saved as filename + ".png".
    """
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
    """Compute the element-wise ratio of two moment tables with propagated errors.

    Args:
        table1: Numerator moment table (astropy Table with "Moment" and "Standard Error").
        table2: Denominator moment table (same structure and order as table1).

    Returns:
        Astropy Table with the same orders as table1, containing table1/table2 ratios
        and propagated standard errors (assuming table1 and table2 are independent).
    """
    comp = Table()
    comp["ELG Order, LRG Order"] = table1["ELG Order, LRG Order"]
    comp["Moment"] = table1["Moment"]/table2["Moment"]
    comp["Standard Error"] = np.sqrt((table1["Standard Error"]/table2["Moment"])**2 \
                           + (table1["Moment"]/table2["Moment"]**2*table2['Standard Error'])**2)
    return comp


def avgnpcf(tab_ntilde_dat,tab_ntilde_ran,tab_nz,tab_vcyl):
    """Compute the cylinder-averaged n-point correlation function xi_bar_{m+1,n}.

    For each order [m, n] in the input tables, computes:
        xi_bar_{m+1,n} = (Ntilde_dat - Ntilde_ran) / (n_ELG^m * n_LRG^n * V_cyl^(m+n))

    where tab_ntilde_ran should be built from counts of real galaxies around randomly placed
    cylinder centers (not galaxy-centered), and tab_nz and tab_vcyl are the denominator
    factor tables built from nZintegral and vCylPower respectively.

    Args:
        tab_ntilde_dat: Ntilde moment table from galaxy-centered cylinders (data).
        tab_ntilde_ran: Ntilde moment table from randomly placed cylinders (randoms).
        tab_nz: Number-density denominator table from nZintegral.
        tab_vcyl: Volume denominator table from vCylPower.

    Returns:
        Astropy Table of cylinder-averaged correlation function values and standard errors.
    """
    tab_npcf = Table()
    tab_npcf["ELG Order, LRG Order"] = tab_ntilde_dat["ELG Order, LRG Order"]
    tab_npcf["Moment"] = (tab_ntilde_dat["Moment"]-tab_ntilde_ran["Moment"])/(tab_nz["Moment"]*tab_vcyl["Moment"])
    tab_npcf["Standard Error"] = np.sqrt((tab_ntilde_dat["Standard Error"]/(tab_nz["Moment"]*tab_vcyl["Moment"]))**2 \
                           + (tab_ntilde_ran["Standard Error"]/(tab_nz["Moment"]*tab_vcyl["Moment"]))**2 \
                                         +(tab_nz["Standard Error"]/tab_nz["Moment"])**2)
    return tab_npcf

def formatMoment(table, index, scientific=False):
    """Format a moment value and its error as a compact string for LaTeX tables.

    Args:
        table: Astropy Table with "Moment" and "Standard Error" columns.
        index: Row index of the moment to format.
        scientific: If True, use explicit notation "1.23e-04 $\\pm$ 4.5e-05".
            If False, always show 3 significant figures of the moment with the
            error in parentheses denoting uncertainty in the last displayed digit
            (e.g. 0.501 +/- 0.003 becomes "0.501 (3)"). Uses scientific notation
            internally when the moment magnitude is >= 1000 or < 0.01; otherwise
            uses decimal with trailing zeros preserved.

    Returns:
        Formatted string suitable for embedding in a LaTeX table cell.
    """
    m = table["Moment"][index]
    err = table["Standard Error"][index]

    if scientific:
        return f"{m:.2e} " + r"$\pm$" + f" {err:.1e}"

    if m == 0:
        return "0.00 (0)"

    exponent = int(np.floor(np.log10(abs(m))))

    if exponent >= 3 or exponent <= -3:
        moment_str = f"{m:.2e}"
        err_in_last = round(err / 10.0 ** (exponent - 2))
    else:
        decimals = 2 - exponent
        moment_str = f"{m:.{decimals}f}"
        err_in_last = round(err * 10.0 ** decimals)

    return moment_str + " (" + str(err_in_last) + ")"

def printMomentTableLaTeX(table,captiontxt,lrg = False,scientific = False):
    """Generate a LaTeX table string for a moment table with up to order 3 in each axis.

    Produces a triangular grid table (only cells with elg_order + lrg_order <= 3 are filled).
    Assumes the table was produced by makeMomentTable with maxorder=3, giving 9 entries.

    Args:
        table: Astropy Table returned by makeMomentTable (must have exactly 9 rows).
        captiontxt: Caption string for the LaTeX table environment.
        lrg: If True, put LRG order on the row axis and ELG order on the column axis
            (i.e., the table was computed for LRG-centered cylinders). If False, ELG
            order is on the row axis.
        scientific: Passed to formatMoment to control number formatting.

    Returns:
        LaTeX string containing the full table environment.
    """
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
    """Generate a LaTeX table string for the bivariate counts-in-cylinders distribution.

    Args:
        hist: Astropy Table of bivariate CiC probabilities (same structure as a moment table).
        captiontxt: Caption string for the LaTeX table environment.
        lrg: If True, put LRG order on the row axis. If False, ELG order is on the row axis.
        scientific: Passed to formatMoment to control number formatting.

    Returns:
        LaTeX string containing the full table environment.
    """
    #so hacky but whatever
    if lrg:
        ix = [3,6,8,0,4,7,1,5,2]
    else:
        ix = np.arange(9)
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
    
        

        