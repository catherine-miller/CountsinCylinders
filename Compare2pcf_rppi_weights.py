from pycorr import TwoPointCorrelationFunction
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import fitsio
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from CountsinCylinders import *

def raDecToUnitSphere(a, d):
    a = a*np.pi/180
    d = d*np.pi/180
    x = np.cos(a)*np.cos(d)
    y = np.sin(a)*np.cos(d)
    z = np.sin(d)
    return [np.array([x[i],y[i],z[i],1]) for i in range(len(a))]

def raDecToCartesian(a, d, r):
    return [dist*xyzd[:3] for dist, xyzd in zip(r, raDecToUnitSphere(a,d))]


edges = (np.array([0,1])/cosmo.h*u.Mpc,np.array([-14,14])/cosmo.h*u.Mpc)

ramin = 130
ramax = 223
decmin = -5.25
decmax = 3.75
psimin = 0.2
psimax = 1.5

#Abacus mocks

#Load in our catalogs
columns = ['RA','DEC','Z','WEIGHT']
#Abacus mocks, no downsampling, "clustering"
zlim = [0.8,1]
zlimlrg = [0.8,1]




mocknumber = 0
mockdir = "/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/"
file = mockdir+ "mock"+str(mocknumber)+"/LRG_complete_clustering.dat.fits"
sp = Table(fitsio.read(file, columns = columns))
splrg = sp[(sp["Z"] > zlimlrg[0])&(sp["Z"] < zlimlrg[1])]
splrg = splrg[angularCut(splrg,ramin,ramax,decmin,decmax)]

#combine 18 random catalogs:
rantabs = []
for i in range(18):
    file = mockdir + "mock"+str(mocknumber)+"/LRG_complete_"+str(i)+"_clustering.ran.fits"
    rantab = Table(fitsio.read(file,columns = columns))
    tabcut = rantab[(rantab["Z"] > zlimlrg[0])&(rantab["Z"] < zlimlrg[1])]
    tabcut = tabcut[angularCut(tabcut,ramin,ramax,decmin,decmax)]
    rantabs.append(tabcut)
rands_lrg = vstack(rantabs)

#ELGs
mocknumber = 0
mockdir = "/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/"
file = mockdir+ "mock"+str(mocknumber)+"/ELG_LOP_complete_clustering.dat.fits"
sp = Table(fitsio.read(file, columns = columns))
spelg = sp[(sp["Z"] > zlim[0])&(sp["Z"] < zlim[1])]
spelg = spelg[angularCut(spelg,ramin,ramax,decmin,decmax)]


#combine 18 random catalogs:
rantabs = []
for i in range(18):
    file = mockdir + "mock"+str(mocknumber)+"/ELG_LOP_complete_"+str(i)+"_clustering.ran.fits"
    rantab = Table(fitsio.read(file,columns = columns))
    tabcut = rantab[(rantab["Z"] > zlim[0])&(rantab["Z"] < zlim[1])]
    tabcut = tabcut[angularCut(tabcut,ramin,ramax,decmin,decmax)]
    rantabs.append(tabcut)
rands_elg = vstack(rantabs)

weightlrg = len(splrg)/len(rands_lrg)
weightelg = len(spelg)/len(rands_elg)


#calculate 2pcf, LRGS
lrgpos = [splrg['RA'],splrg['DEC'],cosmo.comoving_distance(splrg['Z'])/u.Mpc]
randpos = [rands_lrg['RA'],rands_lrg['DEC'],cosmo.comoving_distance(rands_lrg['Z'])/u.Mpc]
data_weights = np.repeat(1,len(splrg))
randoms_weights = np.repeat(weightlrg,len(rands_lrg))
result = TwoPointCorrelationFunction('rppi', edges, data_positions1=lrgpos, data_weights1=data_weights,
                                     randoms_positions1=randpos, randoms_weights1=randoms_weights,
                                     engine='corrfunc',position_type = 'rdd',D1D2_weight_type="product_individual",
                                    R1D2_weight_type="product_individual",R1R2_weight_type="product_individual",
                                    D1R2_weight_type="product_individual",estimator='landyszalay')
result.save("corrfunc/Abacus_LRGs_rppi_1")

#calculate 2pcf, ELGs
elgpos = [spelg['RA'],spelg['DEC'],cosmo.comoving_distance(spelg['Z'])/u.Mpc]
randpos = [rands_elg['RA'],rands_elg['DEC'],cosmo.comoving_distance(rands_elg['Z'])/u.Mpc]
data_weights = np.repeat(1,len(spelg))
randoms_weights = np.repeat(weightelg,len(rands_elg))
result = TwoPointCorrelationFunction('rppi', edges, data_positions1=elgpos, data_weights1=data_weights,
                                     randoms_positions1=randpos, randoms_weights1=randoms_weights,
                                     engine='corrfunc',position_type = 'rdd',D1D2_weight_type="product_individual",
                                    R1D2_weight_type="product_individual",R1R2_weight_type="product_individual",
                                    D1R2_weight_type="product_individual",estimator='landyszalay')
result.save("corrfunc/Abacus_ELGs_rppi_1")


#SV3
print("test")

#Load in our catalogs
columns = ['RA','DEC','Z','WEIGHT']
#Abacus mocks, no downsampling, "clustering"

file_elgS = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/ELG_HIP_S_clustering.dat.fits'
file_elgN = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/ELG_HIP_N_clustering.dat.fits'
file_lrgS = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/LRG_main_S_clustering.dat.fits'
file_lrgN = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/LRG_main_N_clustering.dat.fits'
columnnames = columns
sp1 = Table(fitsio.read(file_elgN, columns=columns))
sp2 = Table(fitsio.read(file_elgS, columns=columns))
sp3 = Table(fitsio.read(file_lrgN, columns=columns))
sp4 = Table(fitsio.read(file_lrgS, columns=columns))
spE = vstack([sp1,sp2])
spL = vstack([sp3,sp4])
spelg = spE[(spE["Z"] > zlim[0])&(spE["Z"] < zlim[1])]
spelg = spelg[rosetteOR(spelg,psimin,psimax,rosettes,"RA","DEC")]
splrg = spL[(spL["Z"] > zlimlrg[0])&(spL["Z"] < zlimlrg[1])]
splrg = splrg[rosetteOR(splrg,psimin,psimax,rosettes,"RA","DEC")]

#random catalogs:
rantabs = []
rans_elg = []
rans_lrg = []
for i in range(18):
    file_elgS = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/ELG_HIP_S_'+str(i)+'_clustering.ran.fits'
    file_elgN = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/ELG_HIP_N_'+str(i)+'_clustering.ran.fits'
    file_lrgS = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/LRG_main_S_'+str(i)+'_clustering.ran.fits'
    file_lrgN = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/LRG_main_N_'+str(i)+'_clustering.ran.fits'
    sp1 = Table(fitsio.read(file_elgN, columns=columns))
    sp2 = Table(fitsio.read(file_elgS, columns=columns))
    sp3 = Table(fitsio.read(file_lrgN, columns=columns))
    sp4 = Table(fitsio.read(file_lrgS, columns=columns))
    rans_elg.append(sp1)
    rans_elg.append(sp2)
    rans_lrg.append(sp3)
    rans_lrg.append(sp4)
spE = vstack(rans_elg)
spL = vstack(rans_lrg)
rands_elg = spE[(spE["Z"] > zlim[0])&(spE["Z"] < zlim[1])]
rands_lrg = spL[(spL["Z"] > zlimlrg[0])&(spL["Z"] < zlimlrg[1])]
rands_elg = rands_elg[rosetteOR(rands_elg,psimin,psimax,rosettes,"RA","DEC")]
rands_lrg = rands_lrg[rosetteOR(rands_lrg,psimin,psimax,rosettes,"RA","DEC")]



#weightlrg = len(splrg)/len(rands_lrg)
#weightelg = len(spelg)/len(rands_elg)

#calculate 2pcf, LRGS
lrgpos = [splrg['RA'],splrg['DEC'],cosmo.comoving_distance(splrg['Z'])/u.Mpc]
randpos = [rands_lrg['RA'],rands_lrg['DEC'],cosmo.comoving_distance(rands_lrg['Z'])/u.Mpc]
data_weights = splrg['WEIGHT']
randoms_weights = rands_lrg['WEIGHT']
result = TwoPointCorrelationFunction('rppi', edges, data_positions1=lrgpos, data_weights1=data_weights,
                                     randoms_positions1=randpos, randoms_weights1=randoms_weights,
                                     engine='corrfunc',position_type='rdd',D1D2_weight_type="product_individual",
                                    R1D2_weight_type="product_individual",R1R2_weight_type="product_individual",
                                    D1R2_weight_type="product_individual",estimator='landyszalay')
result.save("corrfunc/SV3_LRGs_rppi_weighted")


#calculate 2pcf, ELGs
elgpos = [spelg['RA'],spelg['DEC'],cosmo.comoving_distance(spelg['Z'])/u.Mpc]
randpos = [rands_elg['RA'],rands_elg['DEC'],cosmo.comoving_distance(rands_elg['Z'])/u.Mpc]
data_weights = spelg['WEIGHT']
randoms_weights = rands_elg['WEIGHT']
result = TwoPointCorrelationFunction('rppi', edges, data_positions1=elgpos, data_weights1=data_weights,
                                     randoms_positions1=randpos, randoms_weights1=randoms_weights,
                                     engine='corrfunc',position_type='rdd',D1D2_weight_type="product_individual",
                                    R1D2_weight_type="product_individual",R1R2_weight_type="product_individual",
                                    D1R2_weight_type="product_individual",estimator='landyszalay')
result.save("corrfunc/SV3_ELGs_rppi_weighted")


#bootstrap error:
for i in range(20):
    lrgros = splrg[splrg['rosette'] == i+1]
    elgros = spelg[spelg['rosette'] == i+1]
    randelgros = rands_elg[rands_elg['rosette'] == i+1]
    randlrgros = rands_lrg[rands_lrg['rosette'] == i+1]
    lrgpos = [lrgros['RA'],lrgros['DEC'],cosmo.comoving_distance(lrgros['Z'])/u.Mpc]
    randpos = [randlrgros['RA'],randlrgros['DEC'],cosmo.comoving_distance(randlrgros['Z'])/u.Mpc]
    data_weights = lrgros['WEIGHT']
    #weightlrg = len(lrgpos)/len(randpos)
    randoms_weights = randlrgros['WEIGHT']
    result = TwoPointCorrelationFunction('rppi', edges, data_positions1=lrgpos, data_weights1=data_weights,
                                         randoms_positions1=randpos, randoms_weights1=randoms_weights,
                                         engine='corrfunc',position_type='rdd',D1D2_weight_type="product_individual",
                                        R1D2_weight_type="product_individual",R1R2_weight_type="product_individual",
                                        D1R2_weight_type="product_individual",estimator='landyszalay')
    result.save("corrfunc/SV3_LRGs_rppi_ros_weighted_"+str(i))
    
    
    #calculate 2pcf, ELGs
    elgpos = [elgros['RA'],elgros['DEC'],cosmo.comoving_distance(elgros['Z'])/u.Mpc]
    randpos = [randelgros['RA'],randelgros['DEC'],cosmo.comoving_distance(randelgros['Z'])/u.Mpc]
    data_weights = elgros['WEIGHT']
    #weightelg = len(elgpos)/len(randpos)
    randoms_weights = randelgros['WEIGHT']
    result = TwoPointCorrelationFunction('rppi', edges, data_positions1=elgpos, data_weights1=data_weights,
                                         randoms_positions1=randpos, randoms_weights1=randoms_weights,
                                         engine='corrfunc',position_type='rdd',D1D2_weight_type="product_individual",
                                        R1D2_weight_type="product_individual",R1R2_weight_type="product_individual",
                                        D1R2_weight_type="product_individual",estimator='landyszalay')
    result.save("corrfunc/SV3_ELGs_rppi_ros_weighted"+str(i))


