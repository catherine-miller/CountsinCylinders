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


def countsAfterFiberassign(CiCtable_incomplete,targetid,CiCname):
    if targetid in CiCtable_incomplete['TARGETID']:
        counts = CiCtable_incomplete[CiCtable_incomplete['TARGETID'] == targetid][CiCname][0]
    else:
        counts = -1
    return counts

sv3elgnofiberassign['inc_counts'] = [countsAfterFiberassign(sv3elgfiberassign,ID, "N_CiC") for ID in sv3elgnofiberassign['TARGETID']]
sv3elgnofiberassign['inc_counts_sec'] = [countsAfterFiberassign(sv3elgfiberassign,ID, "N_lrgCiC") for ID in sv3elgnofiberassign['TARGETID']]


sv3lrgnofiberassign['inc_counts'] = [countsAfterFiberassign(sv3lrgfiberassign,ID, "N_CiC") for ID in sv3lrgnofiberassign['TARGETID']]
sv3lrgnofiberassign['inc_counts_sec'] = [countsAfterFiberassign(sv3lrgfiberassign,ID, "N_elgCiC") for ID in sv3lrgnofiberassign['TARGETID']]


sv3elgfiberassign_ros = [sv3elgfiberassign[(sv3elgfiberassign['rosette'] == i)] for i in np.arange(1,21)]
sv3lrgfiberassign_ros = [sv3lrgfiberassign[(sv3lrgfiberassign['rosette'] == i)] for i in np.arange(1,21)]

sv3elgfiberassign_jack = [sv3elgfiberassign[(sv3elgfiberassign['rosette'] != i)] for i in np.arange(1,21)]
sv3lrgfiberassign_jack = [sv3lrgfiberassign[(sv3lrgfiberassign['rosette'] != i)] for i in np.arange(1,21)]

def incompletenessMatrix(CiCtable_complete,CiCtable_incomplete,secondarytracerCiC,addextracount = False):
    maxcounts_primaries = np.max(CiCtable_complete["N_CiC"])
    maxcounts_secondaries = np.max(CiCtable_complete[secondarytracerCiC])
    
    incompleteness_primaries = np.zeros((maxcounts_primaries+1,maxcounts_primaries+1))
    incompleteness_secondaries = np.zeros((maxcounts_secondaries+1,maxcounts_secondaries+1))

    incompleteness_primaries_nonnorm = np.zeros((maxcounts_primaries+1,maxcounts_primaries+1))
    incompleteness_secondaries_nonnorm = np.zeros((maxcounts_secondaries+1,maxcounts_secondaries+1))
    for i in np.arange(0,maxcounts_primaries + 1):
        #make histogram
        bins = np.arange(-1,maxcounts_primaries+2)
        hist, bins = np.histogram(CiCtable_complete[CiCtable_complete['N_CiC'] == i]['inc_counts'],bins = bins)
        if addextracount:
            hist[i+1] += 1
        histnonnormalized = hist.copy()
        hist = hist/np.sum(hist)
        incompleteness_primaries[i] = hist[1:]
        #incompleteness_primaries_nonnorm[i] = histnonnormalized[1:]
    for i in np.arange(0,maxcounts_secondaries + 1):
        #make histogram
        bins = np.arange(-1,maxcounts_secondaries+2)
        hist, bins = np.histogram(CiCtable_complete[CiCtable_complete[secondarytracerCiC] == i]['inc_counts_sec'],bins=bins)
        if addextracount:
            hist[i+1] += 1
        hist = hist/np.sum(hist)
        histnonnormalized = hist.copy()
        incompleteness_secondaries[i] = hist[1:]
        #incompleteness_secondaries_nonnorm[i] = histnonnormalized[1:]
    incompleteness_primaries = incompleteness_primaries.T
    incompleteness_secondaries = incompleteness_secondaries.T

    inc_prim_inv = np.linalg.inv(incompleteness_primaries)
    inc_sec_inv = np.linalg.inv(incompleteness_secondaries)

    return inc_prim_inv, inc_sec_inv


