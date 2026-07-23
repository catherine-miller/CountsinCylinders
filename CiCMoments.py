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

class CiCHistnZ:
    def __init__(self,tables,elgname = "N_elgCiC",lrgname="N_lrgCiC",zname = 'Z',binmaxelg = 24,binmaxlrg = 24,
        redshiftbins = np.linspace(0.80695,0.9919,num=20), elgweights = None, lrgweights = None,
        elgweights_err = None, lrgweights_err = None):
        """Build a joint (N_ELG, N_LRG, redshift) histogram from a list of CiC tables.

        Args:
            tables: List of astropy Tables, one per subsample or mock catalog. Each table
                must contain per-primary-galaxy ELG counts, LRG counts, and redshift columns.
                Moments and their standard errors are derived from the spread across tables.
            elgname: Column name for ELG counts-in-cylinders.
            lrgname: Column name for LRG counts-in-cylinders.
            zname: Column name for redshift.
            binmaxelg: Number of ELG count bins (covers N=0 to N=binmaxelg-1). Ignored if
                elgweights is provided.
            binmaxlrg: Number of LRG count bins (covers N=0 to N=binmaxlrg-1). Ignored if
                lrgweights is provided.
            redshiftbins: Edges of the redshift bins (length = n_zbins + 1).
            elgweights: Incompleteness correction weights, one per ELG count bin. If provided,
                len(elgweights) overrides binmaxelg. Each histogram entry is multiplied by
                elgweights[i] * lrgweights[j] to correct for fiber incompleteness.
            lrgweights: Incompleteness correction weights, one per LRG count bin.
            elgweights_err: Uncertainties on elgweights, one per ELG count bin. Used to
                propagate weight errors into moment standard errors via getWeightErrorVariance.
                Defaults to zeros (no weight uncertainty).
            lrgweights_err: Uncertainties on lrgweights, one per LRG count bin.
        """
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
        histsum_raw = np.histogramdd((alltab[elgname],alltab[lrgname],alltab[zname]),bins = bins)[0]
        self.histsum_raw = histsum_raw
        self.histsum = histsum_raw * incompletenessweights[:,:,np.newaxis]
        self.zbins = bins[2]
        self.zcenters = [(self.zbins[i]+self.zbins[i+1])/2 for i in range(len(self.zbins)-1)]
        self.binmax_elg = bins[0][-1]
        self.binmax_lrg = bins[1][-1]
        self.nbins_elg = len(bins[0]) - 1
        self.nbins_lrg = len(bins[1]) - 1
        self.ntables = len(tables)
        self.weights = [len(table) for table in tables]
        self.elgweights = np.array(elgweights, dtype=float)
        self.lrgweights = np.array(lrgweights, dtype=float)
        self.elgweights_err = np.zeros(self.nbins_elg) if elgweights_err is None else np.array(elgweights_err, dtype=float)
        self.lrgweights_err = np.zeros(self.nbins_lrg) if lrgweights_err is None else np.array(lrgweights_err, dtype=float)

    def getMoment(self,hist,momentfunction,order,otherargs = []):
        """Average momentfunction over the joint (N_ELG, N_LRG, redshift) histogram.

        For each histogram cell (i, j, k), evaluates momentfunction(order, i, j, k, *otherargs)
        and returns the histogram-weighted mean. i and j are the ELG and LRG count values
        (0, 1, 2, ...) and k is the redshift bin index.

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
        momentsum = 0
        histsum = 0
        for i in range(self.nbins_elg):
            for j in range(self.nbins_lrg):
                for k in range(len(self.zbins)-1):
                    momentsum += momentfunction(order,i,j,k,*otherargs)*hist[i][j][k]
                    histsum += hist[i][j][k] #for normalization
        return momentsum/histsum

    def getWeightErrorVariance(self, momentfunction, order, moment_value, otherargs=[]):
        """Compute the moment variance contribution from uncertainties in elgweights and lrgweights.

        The moment M is a weighted average over the histogram:

            M = sum_{i,j,k} f_ijk * h_raw_ijk * w_i^ELG * w_j^LRG
                / sum_{i,j,k} h_raw_ijk * w_i^ELG * w_j^LRG

        where:
            i, j, k      -- ELG count bin, LRG count bin, redshift bin indices
            h_raw_ijk    -- self.histsum_raw[i,j,k]: raw (unweighted) count of primaries
                            with ELG count i, LRG count j, in redshift bin k
            w_i^ELG      -- self.elgweights[i]: incompleteness correction weight for ELG bin i
            w_j^LRG      -- self.lrgweights[j]: incompleteness correction weight for LRG bin j
            f_ijk        -- momentfunction(order, i, j, k): moment integrand value at this bin
            M            -- moment_value: the moment itself
            S            -- np.sum(self.histsum): total weight (normalization denominator)

        Differentiating M = T/S with respect to w_i^ELG by the quotient rule gives:

            dM/d(w_i^ELG) = (1/S) * sum_{j,k} h_raw_ijk * w_j^LRG * (f_ijk - M)

        The (f_ijk - M) centering comes directly from the quotient rule. The LRG derivative
        is symmetric: swap i<->j and ELG<->LRG throughout.

        The total variance from all weight uncertainties (assumed independent across bins
        and between ELG and LRG) is:

            var = sum_i (dM/d(w_i^ELG) * sigma_i^ELG)^2
                + sum_j (dM/d(w_j^LRG) * sigma_j^LRG)^2

        Returns zero for any bin whose weight error is zero, so this is a no-op when
        elgweights_err and lrgweights_err were not provided to __init__.

        Args:
            momentfunction: Same function passed to makeMomentTable.
            order: [elg_order, lrg_order] for this moment.
            moment_value: The already-computed moment value M, used for centering.
            otherargs: Additional arguments forwarded to momentfunction.

        Returns:
            Scalar variance (not standard error) due to weight uncertainties.
        """
        S = np.sum(self.histsum)
        variance = 0.0

        for i in range(self.nbins_elg):
            if self.elgweights_err[i] == 0:
                continue
            deriv = 0.0
            for j in range(self.nbins_lrg):
                for k in range(len(self.zbins) - 1):
                    f = momentfunction(order, i, j, k, *otherargs)
                    deriv += self.histsum_raw[i, j, k] * self.lrgweights[j] * (f - moment_value)
            deriv /= S
            variance += (deriv * self.elgweights_err[i]) ** 2

        for j in range(self.nbins_lrg):
            if self.lrgweights_err[j] == 0:
                continue
            deriv = 0.0
            for i in range(self.nbins_elg):
                for k in range(len(self.zbins) - 1):
                    f = momentfunction(order, i, j, k, *otherargs)
                    deriv += self.histsum_raw[i, j, k] * self.elgweights[i] * (f - moment_value)
            deriv /= S
            variance += (deriv * self.lrgweights_err[j]) ** 2

        return variance

    def makeMomentTable(self, maxorder,momentfunction,extraargs=[],filename = '',savesubsamples = False):
        """Compute a table of moments for all orders up to maxorder.

        Iterates over all [elg_order, lrg_order] pairs with elg_order + lrg_order <= maxorder,
        excluding [0, 0]. For each order, the moment is computed for every subsample table
        and averaged (weighted by subsample size); the standard error is std / sqrt(n_tables).

        Args:
            maxorder: Maximum total order (elg_order + lrg_order) to compute.
            momentfunction: Per-primary integrand function passed to getMoment. See getMoment
                for the required signature.
            extraargs: Additional arguments forwarded to momentfunction via getMoment.
            filename: If non-empty, saves the table to this path in ascii.ecsv format.
            savesubsamples: If True, adds a "SubsampleMoments" column containing the
                per-subsample moment values used to compute the mean and standard error.

        Returns:
            Astropy Table with columns "ELG Order, LRG Order", "Moment", "Standard Error",
            and optionally "SubsampleMoments". The standard error combines sample variance
            across subsamples and weight uncertainty from elgweights_err/lrgweights_err
            (via getWeightErrorVariance) in quadrature.
        """
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
                moment_avg = np.average(moments,weights=self.weights)
                weight_var = self.getWeightErrorVariance(momentfunction,[i,j],moment_avg,otherargs=extraargs)
                stderrlist.append(np.sqrt((np.std(moments)/np.sqrt(ntables))**2 + weight_var))
                momentlist.append(moment_avg)
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

def formatMoment(table, index, scientific = False):
    """Format a moment value and its error as a compact string for LaTeX tables.

    Args:
        table: Astropy Table with "Moment" and "Standard Error" columns.
        index: Row index of the moment to format.
        scientific: If True, use scientific notation (e.g. "1.23e-04 +/- 4.5e-05").
            If False, use compact notation where the error in the last digits is shown
            in parentheses (e.g. "0.0123 (45)").

    Returns:
        Formatted string suitable for embedding in a LaTeX table cell.
    """
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
    
        

        