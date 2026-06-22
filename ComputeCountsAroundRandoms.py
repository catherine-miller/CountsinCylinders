from CountsinCylinders import *
from astropy.table import Table
from astropy.table import vstack
import fitsio

#general parameters
R_CiC = 1 #radius in Mpc
L_CiC = 40 #height of cylinder in Mpc
zlim = [0.75,1]
psilim = [0.2,1.5]

#general parameters
R_CiC = 1 #radius in Mpc
L_CiC = 40 #height of cylinder in Mpc
zlim = [0.75,1]
psilim = [0.2,1.5]

R_hole = 0.03*np.pi/180 #convert from degrees to radians
lrgsuccess = 1 #0.976*0.8812216895357733
elgsuccess = 0.8062 #0.8360495775774505 #0.7171793765301621 #0.69/0.9627664301054767
#forFA Mocks, downsampled by redshift fraction but NOT using ebits
mocks = []
columns = ['RA','DEC','RSDZ','ZWARN','DESI_TARGET','MASKBITS'] #want to 
colsrans = ['RA','DEC','Z']
columnnames = columns[:3]
primarymask = makePrimariesZandAngular
ralim = [130,223]
declim  = [-5.25,3.75]
primaryargs = [*zlim,*ralim,*declim,R_CiC,L_CiC]

'''
#loadholes
holetableselg = []
holetableslrg = []
for i in range(240):
    spelg = Table.read("datafiles/forFAelg_tile"+str(i)+"_holes.csv",format = "csv")
    splrg = Table.read("datafiles/forFAlrg_tile"+str(i)+"_holes.csv",format = "csv")
    holetableselg.append(spelg)
    holetableslrg.append(splrg)
holetableelg = vstack(holetableselg)
holetablelrg = vstack(holetableslrg)
print(len(holetableelg),len(holetablelrg))



for mocknumber in range(18):
    outfilename = "forFA_countsaroundrandoms_"+str(mocknumber)+"R_"+str(R_CiC)
    mockdir = "/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/"
    file = mockdir+ "forFA"+str(mocknumber)+".fits"
    directory = "/pscratch/sd/c/crmiller/"
    elgtabs = []
    lrgtabs = []
    for i in range(1):
        elgranfile = mockdir+"/mock"+str(mocknumber)+"/ELG_LOP_complete_12_clustering.ran.fits"
        lrgranfile = mockdir+"/mock"+str(mocknumber)+"/LRG_complete_12_clustering.ran.fits"
        spranelg = Table(fitsio.read(elgranfile,columns=colsrans))
        spranlrg = Table(fitsio.read(lrgranfile,columns=colsrans))
        spranelg = spranelg[(spranelg["Z"] > zlim[0])&(spranelg["Z"] < zlim[1])&(spranelg['RA'] > ralim[0])& \
            (spranelg['RA'] < ralim[1])&(spranelg['DEC'] > declim[0])&(spranelg['DEC'] < declim[1])]
        spranlrg = spranlrg[(spranlrg["Z"] > zlim[0])&(spranlrg["Z"] < zlim[1])&(spranlrg['RA'] > ralim[0])& \
            (spranlrg['RA'] < ralim[1])&(spranlrg['DEC'] > declim[0])&(spranlrg['DEC'] < declim[1])]
        elgtabs.append(spranelg)
        lrgtabs.append(spranlrg)
    spranselg = vstack(elgtabs)
    spranselg['RSDZ']=spranselg['Z']
    spranslrg = vstack(lrgtabs)
    spranslrg['RSDZ']=spranslrg['Z']
    sp = Table(fitsio.read(file,columns=columns))
    #get lrg mask from "matched input" lrg file
    splrg = sp[(sp["DESI_TARGET"]&0x1==0x1)&(sp["ZWARN"]==0)&(sp["RSDZ"] > zlim[0])&(sp["RSDZ"] < zlim[1])]
    spelg = sp[(sp["DESI_TARGET"]&0x20==0x20)&(sp["ZWARN"]==0)&(sp["RSDZ"] > zlim[0])&(sp["RSDZ"] < zlim[1])]
    elg_downsampled = spelg[[(np.random.rand() < elgsuccess) for i in range(len(spelg))]]
    lrg_downsampled = splrg[[(np.random.rand() < lrgsuccess) for i in range(len(splrg))]]
    #we need separate ELG and LRG random catalogs
    #we count tracers of both types around each
    countsAroundRandomsHoleClean(spranslrg,elg_downsampled,lrg_downsampled,holetableelg,holetablelrg,
                      columnnames, R_CiC, L_CiC, R_hole, zlim, primarymask, primaryargs, outfilename+"lrg",directory=directory)
    countsAroundRandomsHoleClean(spranselg,elg_downsampled,lrg_downsampled,holetableelg,holetablelrg,
                      columnnames, R_CiC, L_CiC, R_hole, zlim, primarymask, primaryargs, outfilename+"elg_nodownsample",directory=directory)

    #loadholes
holetableselg = []
holetableslrg = []
for i in range(117):
    spelg = Table.read("datafiles/Holes_ELG_SV3_tile"+str(i)+"_holes.csv",format = "csv")
    splrg = Table.read("datafiles/Holes_LRG_SV3_tile"+str(i)+"_holes.csv",format = "csv")
    holetableselg.append(spelg)
    holetableslrg.append(splrg)
holetableelg = vstack(holetableselg)
holetablelrg = vstack(holetableslrg)
print(len(holetableelg),len(holetablelrg))
'''

#Counts around randoms in SV3, 7/22/25
#FINAL for paper!!
zlim = [0.75, 1]
R_hole = 0.03*np.pi/180
#elgdownsamplefrac = 0.9627664301054767
#LSS SV3 catalogs, using ELGnotqso
file_elgS = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/ELG_HIPnotqso_S_clustering.dat.fits'
file_elgN = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/ELG_HIPnotqso_N_clustering.dat.fits'
file_lrgS = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/LRG_main_S_clustering.dat.fits'
file_lrgN = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/LRG_main_N_clustering.dat.fits'
columns = ['RA','DEC','Z']
columnnames = columns
sp1 = Table(fitsio.read(file_elgN, columns=columns))
sp2 = Table(fitsio.read(file_elgS, columns=columns))
sp3 = Table(fitsio.read(file_lrgN, columns=columns))
sp4 = Table(fitsio.read(file_lrgS, columns=columns))
spE = vstack([sp1,sp2])
spL = vstack([sp3,sp4])
spelg = spE[(spE["Z"] > zlim[0])&(spE["Z"] < zlim[1])]
splrg = spL[(spL["Z"] > zlim[0])&(spL["Z"] < zlim[1])]
primarymask = makePrimariesRosettes
primaryargs = [*zlim,*psilim,R_CiC,L_CiC,'RA','DEC']
outfilename = "SV3_countsaroundrandoms_"

ranelg = []
ranlrg = []
for i in range(18):
    file_elgS = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/ELG_S_'+str(i)+'_clustering.ran.fits'
    file_elgN = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/ELG_N_'+str(i)+'_clustering.ran.fits'
    file_lrgS = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/LRG_main_S_'+str(i)+'_clustering.ran.fits'
    file_lrgN = '/global/cfs/cdirs/desi/vac/edr/lss/v2.0/LSScats/clustering/LRG_main_N_'+str(i)+'_clustering.ran.fits'
    sp1 = Table(fitsio.read(file_elgN, columns=columns))
    sp2 = Table(fitsio.read(file_elgS, columns=columns))
    sp3 = Table(fitsio.read(file_lrgN, columns=columns))
    sp4 = Table(fitsio.read(file_lrgS, columns=columns))
    ranelg.append(sp1)
    ranelg.append(sp2)
    ranlrg.append(sp3)
    ranlrg.append(sp4)
spranselg = vstack(ranelg)
spranslrg = vstack(ranlrg)


countsAroundRandomsHoleClean(spranslrg,spelg,splrg,holetableelg,holetablelrg,
                      columnnames, R_CiC, L_CiC, R_hole, zlim, primarymask, primaryargs, outfilename+"lrg", directory = "/pscratch/sd/c/crmiller/")
countsAroundRandomsHoleClean(spranselg,spelg,splrg,holetableelg,holetablelrg,
                      columnnames, R_CiC, L_CiC, R_hole, zlim, primarymask, primaryargs, outfilename+"elg", directory = "/pscratch/sd/c/crmiller/")