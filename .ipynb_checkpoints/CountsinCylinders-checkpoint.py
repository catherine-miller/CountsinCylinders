#Generalized counts in cylinders method
from astropy.table import Table
from astropy.table import vstack
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
import astropy.io.ascii
import pandas as pd
from desitarget import targetmask

rosettes = {'RA': [150.100, 179.600, 183.100, 189.900, 194.750, 210.000, 215.500, 217.800, 216.300, 219.800, 218.050, 
             242.750, 241.050, 245.880, 252.500, 269.730, 194.750, 212.800, 269.730, 236.100],
            'DEC': [2.182, 0.000, 0.000, 61.800, 28.200, 5.000, 52.500, 34.400, -0.600, -0.600, 2.430, 54.980, 43.450, 43.450,
              34.500, 66.020, 24.700, -0.600, 62.520, 43.450],
            'ID': np.array(np.arange(20)+1)}

def raDecToUnitSphere(a, d):
    a = a*np.pi/180
    d = d*np.pi/180
    x = np.cos(a)*np.cos(d)
    y = np.sin(a)*np.cos(d)
    z = np.sin(d)
    return [np.array([x[i],y[i],z[i],1]) for i in range(len(a))]

def unitSphereToCartesian(uscoord, dist):
    return [coord*d for coord, d in zip(uscoord, dist)]

def raDecToCartesian(a, d, r):
    return [dist*xyzd for dist, xyzd in zip(r, raDecToUnitSphere(a,d))]

#methods for selecting primaries
def checkZBoundary(table,zmin,zmax,cylr,cylh):
    return (table['d'] > cosmo.comoving_distance(zmin)/u.Mpc+cylh/2) \
        &(table['d'] < cosmo.comoving_distance(zmax)/u.Mpc-cylh/2)

def angularCut(table,ramin,ramax,decmin,decmax):
    return (table['DEC'] > decmin) & (table['DEC'] < decmax) & \
            (table['RA'] > ramin) & (table['RA'] < ramax)


def checkAngularSep(phi1,phi2,theta1,theta2,psimin,psimax):
    #arguments: RA1, RA2, Dec1, Dec2, min separation, max separation (in DEGREES)
    theta1 = theta1*np.pi/180
    theta2 = theta2*np.pi/180
    phi1 = phi1*np.pi/180
    phi2 = phi2*np.pi/180
    psimin = psimin*np.pi/180
    psimax = psimax*np.pi/180
    cospsi = np.sin(theta1)*np.sin(theta2)+np.cos(theta1)*np.cos(theta2)*np.cos(phi1-phi2)
    return (cospsi > np.cos(psimax))&(cospsi < np.cos(psimin))

def rosetteOR(table, psimin, psimax, rosettes, ra, dec):
    rosettemask = [checkAngularSep(rosettes['RA'][i],table[ra],rosettes['DEC'][i],table[dec],psimin,psimax)
                   for i in range(len(rosettes['RA']))]
    #return true if it belongs to any of the rosettes
    table['rosette'] = [np.array(rosettes['ID'])[r] for r in np.transpose(rosettemask)]
    #store boolean mask of all rosettes it belongs to (later we will convert this to a rosette index)
    return np.any(rosettemask,axis=0)

def UnitSphereToRaDecPosRA(UScoord):
    x,y,z = [UScoord[i][0] for i in range(len(UScoord))],[UScoord[i][1] for i in range(len(UScoord))],[UScoord[i][2] for i in range(len(UScoord))]
    d = np.arcsin(z)
    arg = x/np.cos(d)
    def ceiling(x):
        if x > 1:
            x = 1
        if x < -1:
            x = -1
        return x
    arg = [ceiling(arg[i]) for i in range(len(arg))]
    a = np.arccos(arg)*180/np.pi
    d = d*180/np.pi
    def negforarccos(a,y):
        if y < 0:
            a = - a + 360
        return a
    a = [negforarccos(a[i],y[i]) for i in range(len(a))]
    return [np.array([a[i], d[i]]) for i in range(len(a))]

def UnitSphereToRaDecPosRA(x,y,z):
    #x,y,z = [UScoord[i][0] for i in range(len(UScoord))],[UScoord[i][1] for i in range(len(UScoord))],[UScoord[i][2] for i in range(len(UScoord))]
    d = np.arcsin(z)
    arg = x/np.cos(d)
    def ceiling(x):
        if x > 1:
            x = 1
        if x < -1:
            x = -1
        return x
    arg = [ceiling(arg[i]) for i in range(len(arg))]
    a = np.arccos(arg)*180/np.pi
    d = d*180/np.pi
    def negforarccos(a,y):
        if y < 0:
            a = - a + 360
        return a
    a = [negforarccos(a[i],y[i]) for i in range(len(a))]
    return a, d

def makePrimariesRosettes(table, zmin, zmax, psimin, psimax, cylr, cylh,ra,dec):
    
    return checkZBoundary(table,zmin,zmax,cylr,cylh)&rosetteOR(table,psimin,psimax,rosettes,ra,dec)

def makePrimariesRosettesNoZBoundary(table, zmin, zmax, psimin, psimax, cylr, cylh,ra,dec):
    
    return rosetteOR(table,psimin,psimax,rosettes,ra,dec)

def makePrimariesZandAngular(table,zmin,zmax,ramin,ramax,decmin,decmax,cylr,cylh):
    return checkZBoundary(table,zmin,zmax,cylr,cylh) & angularCut(table,ramin,ramax,decmin,decmax)

def makePrimariesBox(table):
    return (table['d'] > 300)


#methods for pruning from spheres to cylinders
def checkcylinder(x1,x2,hsq,rsq):
    #arguments: [x,y,z,d] of primary, [x,y,z,d] of secondary, height of cylinder squared/4, radius of cylinder squared
    #check if two points are in cylinder with axis along line of sight
    return ((x1[3]-x2[:,3])**2 < hsq) & \
            (np.sum((x1[:3]-x2[:,:3])**2, axis = 1) - (x1[3]-x2[:,3])**2 < rsq) #check transverse separation

def tableprune(table1, table2, i, j, L_CiC, R_CiC):
    #i is the index of the primary galaxy (in table 1)
    #j is the index of the secondary galaxy (in table 2)
    if (i % 100000 == 0): print("   Galaxy "+str(i))
    return checkcylinder(table1[i]['xyzd'],
                         table2[j]['xyzd'],L_CiC**2/4,R_CiC**2)

def getNZ(table1, table2, primaryix, secondaryixlist):
    if len(ixlist) == 0:
        return 0
    else:
        return table[ixlist[0]][nz]

def countsInCylinders(spelg, splrg, columnnames, R_CiC, L_CiC, zlim, primarymask, primaryargs, outfilename,secondarymask = None,secondaryargs = None):
    #Arguments: table with elgs, table with lrgs, column names = [ra,dec,z,zwarn], function to get mask for primaries (for 
        #SV3 includes rosette limits; for mocks no such limits necessary
        
    ra,dec,z = columnnames
    elgsecondaries = spelg[(spelg[z] > zlim[0])&(spelg[z] < zlim[1])]
    lrgsecondaries = splrg[(splrg[z] > zlim[0])&(splrg[z] < zlim[1])]
    
    
    #get comoving distances of galaxies along line of sight
    elgsecondaries['d'] = cosmo.comoving_distance(elgsecondaries[z])/u.Mpc #more convenient to have unitless calculations
    lrgsecondaries['d'] = cosmo.comoving_distance(lrgsecondaries[z])/u.Mpc 

    #get Cartesian coordinates of galaxies
    elgsecondaries['xyzd'] = raDecToCartesian(elgsecondaries[ra],elgsecondaries[dec],elgsecondaries['d'])
    lrgsecondaries['xyzd'] = raDecToCartesian(lrgsecondaries[ra],lrgsecondaries[dec],lrgsecondaries['d'])
    
    elgprimaries = elgsecondaries[primarymask(elgsecondaries,*primaryargs)]
    lrgprimaries = lrgsecondaries[primarymask(lrgsecondaries,*primaryargs)]
    #primaryargs: [zmin,zmax,psimin,psimax,cylr,cylh] for SV3
    #primaryargs: [zmin,zmax,cylr,cylh] for mocks

    if secondarymask != None:
        print("applying mask to secondaries")
        elgsecondaries = elgsecondaries[secondarymask(elgsecondaries,*secondaryargs)]
        lrgsecondaries = lrgsecondaries[secondarymask(lrgsecondaries,*secondaryargs)]
    
    #make 3D maps of galaxies
    #primaries
    elgprimarymap = [coord[:3] for coord in elgprimaries['xyzd']]
    lrgprimarymap = [coord[:3] for coord in lrgprimaries['xyzd']]

    #make ckdtrees
    elgtree = cKDTree([coord[:3] for coord in elgsecondaries['xyzd']])
    lrgtree = cKDTree([coord[:3] for coord in lrgsecondaries['xyzd']])
    
    print("Searching for neighbors in spheres")
    #search for neighbors
    rsphere = np.sqrt(R_CiC**2 + (L_CiC/2)**2)
    elgprimaries['neighbors'] = elgtree.query_ball_point(elgprimarymap, rsphere)
    lrgprimaries['neighbors'] = lrgtree.query_ball_point(lrgprimarymap, rsphere)

    #search for neighbors among other class of galaxies
    elgprimaries['lrg_neighbors'] = lrgtree.query_ball_point(elgprimarymap, rsphere)
    lrgprimaries['elg_neighbors'] = elgtree.query_ball_point(lrgprimarymap, rsphere)
    
    #prune galaxies
    hsq = L_CiC**2/4
    dsq = R_CiC**2
    print("Pruning galaxies from spheres")
    print("Pruning ELG auto-correlation")
    elgprimaries['neighborspruned'] = \
    [np.array(elgprimaries[i]['neighbors'])[tableprune(elgprimaries,elgsecondaries,i,elgprimaries[i]['neighbors'],
                                                      L_CiC,R_CiC)][1:] 
                for i in range(len(elgprimaries))]
    print("Pruning LRG auto-correlation")
    lrgprimaries['neighborspruned'] = \
    [np.array(lrgprimaries[i]['neighbors'])[tableprune(lrgprimaries,lrgsecondaries,i,lrgprimaries[i]['neighbors'],
                                                      L_CiC,R_CiC)][1:]
                for i in range(len(lrgprimaries))]
    print("Pruning ELG-LRG cross-correlation")
    elgprimaries['lrg_neighborspruned'] = \
    [np.array(elgprimaries[i]['lrg_neighbors'])[tableprune(elgprimaries,lrgsecondaries,i,elgprimaries[i]['lrg_neighbors'],
                                                          L_CiC,R_CiC)]
                for i in range(len(elgprimaries))]
    print("Pruning LRG-ELG cross-correlation")
    lrgprimaries['elg_neighborspruned'] = \
    [np.array(lrgprimaries[i]['elg_neighbors'])[tableprune(lrgprimaries,elgsecondaries,i,lrgprimaries[i]['elg_neighbors'],
                                                          L_CiC,R_CiC)] 
                for i in range(len(lrgprimaries))]
    
    #count up the galaxies in cylinders
    elgprimaries['N_CiC'] = [len(n) for n in elgprimaries['neighborspruned']]
    lrgprimaries['N_CiC'] = [len(n) for n in lrgprimaries['neighborspruned']]
    elgprimaries['N_lrgCiC'] = [len(n) for n in elgprimaries['lrg_neighborspruned']]
    lrgprimaries['N_elgCiC'] = [len(n) for n in lrgprimaries['elg_neighborspruned']]
    '''
    #get N(z) for secondaries
    d, ix = lrgtree.query(elgprimarymap) #find list of nearest neighbors
    #for each elg primary, find the lrg n(z), based on "NZ" of the nearest lrg neighbor
    elgprimaries["NZ_lrg"] = [lrgsecondaries[ix[i]][nz] for i in range(len(ix))]
    d, ix = elgtree.query(lrgprimarymap)
    lrgprimaries["NZ_elg"] = [elgsecondaries[ix[i]][nz] for i in range(len(ix))]
      '''                        
    elgprimaries['x'],elgprimaries['y'],elgprimaries['z'],d = np.rollaxis(elgprimaries['xyzd'],1).tolist()
    lrgprimaries['x'],lrgprimaries['y'],lrgprimaries['z'],d = np.rollaxis(lrgprimaries['xyzd'],1).tolist()
    
    #save our results
    #elg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_lrgCiC']))
    #lrg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_elgCiC']))
    
    if ('rosette' in elgprimaries.columns):
        elgprimaries['rosette'] = [ros[0] for ros in elgprimaries['rosette']]
        lrgprimaries['rosette'] = [ros[0] for ros in lrgprimaries['rosette']]
        
        elgtowrite = elgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_lrgCiC','rosette']
        lrgtowrite = lrgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_elgCiC','rosette']
    else:
        elgtowrite = elgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_lrgCiC']
        lrgtowrite = lrgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_elgCiC']

    astropy.io.ascii.write(elgtowrite, 'datafiles/'+outfilename+'_elg.csv', overwrite=True,format='csv')
    astropy.io.ascii.write(lrgtowrite, 'datafiles/'+outfilename+'_lrg.csv', overwrite=True,format='csv')
    
    print('Saved file datafiles/'+outfilename+'_elg.csv')
    print('Saved file datafiles/'+outfilename+'_lrg.csv')

def countsInCylindersLRGonly(splrg, columnnames, R_CiC, L_CiC, zlim, primarymask, primaryargs, outfilename):
    #Argzuments: table with elgs, table with lrgs, column names = [ra,dec,z,zwarn], function to get mask for primaries (for 
        #SV3 includes rosette limits; for mocks no such limits necessary
        
    if len(columnnames) >= 4:
        ra,dec,z,zwarn = columnnames[:4]
        #make tables of secondaries
        lrgsecondaries = splrg[(splrg[zwarn]==0)&(splrg[z] > zlim[0])&(splrg[z] < zlim[1])]
    
    else:
        ra,dec,z = columnnames
        lrgsecondaries = splrg[(splrg[z] > zlim[0])&(splrg[z] < zlim[1])]
    
    #get comoving distances of galaxies along line of sight
    lrgsecondaries['d'] = cosmo.comoving_distance(lrgsecondaries[z])/u.Mpc 

    #get Cartesian coordinates of galaxies
    lrgsecondaries['xyzd'] = raDecToCartesian(lrgsecondaries[ra],lrgsecondaries[dec],lrgsecondaries['d'])
    lrgprimaries = lrgsecondaries[primarymask(lrgsecondaries,*primaryargs)]
    #primaryargs: [zmin,zmax,psimin,psimax,cylr,cylh] for SV3
    #primaryargs: [zmin,zmax,cylr,cylh] for mocks
    
    #make 3D maps of galaxies
    #primaries
    lrgprimarymap = [coord[:3] for coord in lrgprimaries['xyzd']]

    #make ckdtrees
    lrgtree = cKDTree([coord[:3] for coord in lrgsecondaries['xyzd']])
    
    print("Searching for neighbors in spheres")
    #search for neighbors
    rsphere = np.sqrt(R_CiC**2 + (L_CiC/2)**2)
    lrgprimaries['neighbors'] = lrgtree.query_ball_point(lrgprimarymap, rsphere)

    #prune galaxies
    hsq = L_CiC**2/4
    dsq = R_CiC**2

    print("Pruning LRG auto-correlation")
    lrgprimaries['neighborspruned'] = \
    [np.array(lrgprimaries[i]['neighbors'])[tableprune(lrgprimaries,lrgsecondaries,i,lrgprimaries[i]['neighbors'],
                                                      L_CiC,R_CiC)][1:]
                for i in range(len(lrgprimaries))]
    
    
    #count up the galaxies in cylinders
    lrgprimaries['N_CiC'] = [len(n) for n in lrgprimaries['neighborspruned']]
    
    lrgprimaries['x'],lrgprimaries['y'],lrgprimaries['z'],d = np.rollaxis(lrgprimaries['xyzd'],1).tolist()
    
    #save our results
    #elg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_lrgCiC']))
    #lrg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_elgCiC']))
    
    if (primarymask == makePrimariesRosettes):
        lrgprimaries['rosette'] = [ros[0] for ros in lrgprimaries['rosette']]
        
        lrgtowrite = lrgprimaries[ra,dec,z,zwarn,'x','y','z','d','N_CiC','N_elgCiC','rosette']
    else:
        lrgtowrite = lrgprimaries[ra,dec,z,'x','y','z','d','N_CiC']

    astropy.io.ascii.write(lrgtowrite, 'datafiles/'+outfilename+'_lrg.csv', overwrite=True,format='csv')
    
    print('Saved file datafiles/'+outfilename+'_lrg.csv')

def countsInCylindersCartesian(spelg, splrg, columnnames, R_CiC, L_CiC, primarymask, primaryargs, outfilename):
    #Arguments: table with elgs, table with lrgs, column names = [ra,dec,z,zwarn], function to get mask for primaries (for 
        #SV3 includes rosette limits; for mocks no such limits necessary
        
    x,y,z,d = columnnames
    elgsecondaries = spelg
    lrgsecondaries = splrg
    
    #get comoving distances of galaxies along line of sight
    #elgsecondaries['d'] = cosmo.comoving_distance(elgsecondaries[z])/u.Mpc #more convenient to have unitless calculations
    #lrgsecondaries['d'] = cosmo.comoving_distance(lrgsecondaries[z])/u.Mpc 

    #get Cartesian coordinates of galaxies
    elgsecondaries['xyzd'] = [[elgsecondaries['x'][i],elgsecondaries['y'][i],
                               elgsecondaries['z'][i],elgsecondaries['d'][i]] for i in range(len(elgsecondaries))]
    lrgsecondaries['xyzd'] = [[lrgsecondaries['x'][i],lrgsecondaries['y'][i],
                               lrgsecondaries['z'][i],lrgsecondaries['d'][i]] for i in range(len(lrgsecondaries))]
    
    elgprimaries = elgsecondaries[primarymask(elgsecondaries,*primaryargs)]
    lrgprimaries = lrgsecondaries[primarymask(lrgsecondaries,*primaryargs)]
    #primaryargs: [zmin,zmax,psimin,psimax,cylr,cylh] for SV3
    #primaryargs: [zmin,zmax,cylr,cylh] for mocks
    
    #make 3D maps of galaxies
    #primaries
    elgprimarymap = [coord[:3] for coord in elgprimaries['xyzd']]
    lrgprimarymap = [coord[:3] for coord in lrgprimaries['xyzd']]

    #make ckdtrees
    elgtree = cKDTree([coord[:3] for coord in elgsecondaries['xyzd']])
    lrgtree = cKDTree([coord[:3] for coord in lrgsecondaries['xyzd']])
    
    print("Searching for neighbors in spheres")
    #search for neighbors
    rsphere = np.sqrt(R_CiC**2 + (L_CiC/2)**2)
    elgprimaries['neighbors'] = elgtree.query_ball_point(elgprimarymap, rsphere)
    lrgprimaries['neighbors'] = lrgtree.query_ball_point(lrgprimarymap, rsphere)

    #search for neighbors among other class of galaxies
    elgprimaries['lrg_neighbors'] = lrgtree.query_ball_point(elgprimarymap, rsphere)
    lrgprimaries['elg_neighbors'] = elgtree.query_ball_point(lrgprimarymap, rsphere)
    
    #prune galaxies
    hsq = L_CiC**2/4
    dsq = R_CiC**2
    print("Pruning galaxies from spheres")
    print("Pruning ELG auto-correlation")
    elgprimaries['neighborspruned'] = \
    [np.array(elgprimaries[i]['neighbors'])[tableprune(elgprimaries,elgsecondaries,i,elgprimaries[i]['neighbors'],
                                                      L_CiC,R_CiC)][1:] 
                for i in range(len(elgprimaries))]
    print("Pruning LRG auto-correlation")
    lrgprimaries['neighborspruned'] = \
    [np.array(lrgprimaries[i]['neighbors'])[tableprune(lrgprimaries,lrgsecondaries,i,lrgprimaries[i]['neighbors'],
                                                      L_CiC,R_CiC)][1:]
                for i in range(len(lrgprimaries))]
    print("Pruning ELG-LRG cross-correlation")
    elgprimaries['lrg_neighborspruned'] = \
    [np.array(elgprimaries[i]['lrg_neighbors'])[tableprune(elgprimaries,lrgsecondaries,i,elgprimaries[i]['lrg_neighbors'],
                                                          L_CiC,R_CiC)]
                for i in range(len(elgprimaries))]
    print("Pruning LRG-ELG cross-correlation")
    lrgprimaries['elg_neighborspruned'] = \
    [np.array(lrgprimaries[i]['elg_neighbors'])[tableprune(lrgprimaries,elgsecondaries,i,lrgprimaries[i]['elg_neighbors'],
                                                          L_CiC,R_CiC)] 
                for i in range(len(lrgprimaries))]
    
    #count up the galaxies in cylinders
    elgprimaries['N_CiC'] = [len(n) for n in elgprimaries['neighborspruned']]
    lrgprimaries['N_CiC'] = [len(n) for n in lrgprimaries['neighborspruned']]
    elgprimaries['N_lrgCiC'] = [len(n) for n in elgprimaries['lrg_neighborspruned']]
    lrgprimaries['N_elgCiC'] = [len(n) for n in lrgprimaries['elg_neighborspruned']]
    
    elgprimaries['x'],elgprimaries['y'],elgprimaries['z'],d = np.rollaxis(elgprimaries['xyzd'],1).tolist()
    lrgprimaries['x'],lrgprimaries['y'],lrgprimaries['z'],d = np.rollaxis(lrgprimaries['xyzd'],1).tolist()
    
    #save our results
    #elg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_lrgCiC']))
    #lrg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_elgCiC']))
    
    if (primarymask == makePrimariesRosettes):
        elgprimaries['rosette'] = [ros[0] for ros in elgprimaries['rosette']]
        lrgprimaries['rosette'] = [ros[0] for ros in lrgprimaries['rosette']]
        
        elgtowrite = elgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_lrgCiC','rosette']
        lrgtowrite = lrgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_elgCiC','rosette']
    else:
        elgtowrite = elgprimaries['x','y','z','d','N_CiC','N_lrgCiC']
        lrgtowrite = lrgprimaries['x','y','z','d','N_CiC','N_elgCiC']

    astropy.io.ascii.write(elgtowrite, 'datafiles/'+outfilename+'_elg.csv', overwrite=True,format='csv')
    astropy.io.ascii.write(lrgtowrite, 'datafiles/'+outfilename+'_lrg.csv', overwrite=True,format='csv')
    
    print('Saved file datafiles/'+outfilename+'_elg.csv')
    print('Saved file datafiles/'+outfilename+'_lrg.csv')

def countsInCylindersHoleClean(spelg, splrg, spholeselg, spholeslrg, columnnames, R_CiC, L_CiC, R_hole, zlim, primarymask, primaryargs, outfilename):
    #Arguments: table with elgs, table with lrgs, column names = [ra,dec,z,zwarn], function to get mask for primaries (for 
        #SV3 includes rosette limits; for mocks no such limits necessary
    
    ra,dec,z = columnnames
    elgsecondaries = spelg[(spelg[z] > zlim[0])&(spelg[z] < zlim[1])]
    lrgsecondaries = splrg[(splrg[z] > zlim[0])&(splrg[z] < zlim[1])]
    
    #get comoving distances of galaxies along line of sight
    elgsecondaries['d'] = cosmo.comoving_distance(elgsecondaries[z])/u.Mpc #more convenient to have unitless calculations
    lrgsecondaries['d'] = cosmo.comoving_distance(lrgsecondaries[z])/u.Mpc 

    #get unit sphere coordinates of galaxies
    print('finding unit sphere coordinates of galaxies')
    elgsecondaries['UScoord'] = raDecToUnitSphere(elgsecondaries[ra],elgsecondaries[dec])
    lrgsecondaries['UScoord'] = raDecToUnitSphere(lrgsecondaries[ra],lrgsecondaries[dec])

    print('finding Cartesian coordinates of galaxies')
    #get Cartesian coordinates
    elgsecondaries['xyzd'] = raDecToCartesian(elgsecondaries[ra],elgsecondaries[dec],elgsecondaries['d'])
    lrgsecondaries['xyzd'] = raDecToCartesian(lrgsecondaries[ra],lrgsecondaries[dec],lrgsecondaries['d'])

    elgprimaries = elgsecondaries[primarymask(elgsecondaries,*primaryargs)]
    lrgprimaries = lrgsecondaries[primarymask(lrgsecondaries,*primaryargs)]
    #primaryargs: [zmin,zmax,psimin,psimax,cylr,cylh] for SV3
    #primaryargs: [zmin,zmax,cylr,cylh] for mocks

    ''' FOR TESTING
    spholeselg['RA'],spholeselg['DEC'] = UnitSphereToRaDecPosRA(spholeselg['x'],spholeselg['y'],spholeselg['z'])
    #plt.scatter(holetableelg['RA'],holetableelg['DEC'])
    rosettes['RAmin'] = np.array(rosettes['RA']) - 1/np.cos(np.array(rosettes['DEC']))*2
    rosettes['RAmax'] = np.array(rosettes['RA']) + 1/np.cos(np.array(rosettes['DEC']))*2
    rosettes['DECmin'] = np.array(rosettes['DEC']) - 2
    rosettes['DECmax'] = np.array(rosettes['DEC']) + 2
    
    for i in range(20):
        plt.figure()
        plt.title("Holes in Rosette "+str(i+1)+"\n Rosette Coordinates: "+str(rosettes['RA'][i])+", "+str(rosettes['DEC'][i]))
        plt.scatter(spholeselg['RA'],spholeselg['DEC'],s=1,label = "hole centers")
        plt.xlim(rosettes['RAmin'][i],rosettes['RAmax'][i])
        plt.ylim(rosettes['DECmin'][i],rosettes['DECmax'][i])
        plt.scatter(elgprimaries['RA'],elgprimaries['DEC'],s=0.5,label = "SV3 galaxies")
        plt.scatter(rosettes['RA'][i],rosettes['DEC'][i],label = 'rosette center',marker='x',color='red')
        plt.legend(loc = 'upper right')
        #plt.scatter(rantable['RA'][::100],rantable['DEC'][::100])

    FOR TESTING '''
    #make 3D maps of galaxies
    #primaries
    elgUSmap = [coord[:3] for coord in elgprimaries['UScoord']]
    lrgUSmap = [coord[:3] for coord in lrgprimaries['UScoord']]

    #hole maps
    elgholemap = [[spholeselg['x'][i],spholeselg['y'][i],spholeselg['z'][i]] for i in range(len(spholeselg))]
    lrgholemap = [[spholeslrg['x'][i],spholeslrg['y'][i],spholeslrg['z'][i]] for i in range(len(spholeslrg))]

    print('making cKDTrees')
    #make ckdtrees
    elgtree = cKDTree([coord[:3] for coord in elgsecondaries['xyzd']])
    lrgtree = cKDTree([coord[:3] for coord in lrgsecondaries['xyzd']])

    elgprimtree_US = cKDTree(elgUSmap)
    lrgprimtree_US = cKDTree(lrgUSmap)
    elgholes = cKDTree(elgholemap)
    lrgholes = cKDTree(lrgholemap)

    print("eliminating galaxies near holes")
    elginit = len(elgprimaries)
    lrginit = len(lrgprimaries)
    print("initial ELGs: "+str(elginit))
    print("initial primary LRGs: "+str(lrginit))
    elgholeneighbors = elgprimtree_US.query_ball_tree(elgholes,R_hole)
    lrgholeneighbors = lrgprimtree_US.query_ball_tree(lrgholes,R_hole)
    elgmask = [len(n) < 1 for n in elgholeneighbors]
    lrgmask = [len(n) < 1 for n in lrgholeneighbors]
    #elgeliminated = elgprimaries[[not bl for bl in elgmask]]
    #print("number of eliminated elgs",len(elgeliminated))
    #this is possibly not so efficient but whatever
    elgprimaries = elgprimaries[elgmask]
    lrgprimaries = lrgprimaries[lrgmask]

    ''' FOR TESTING
    for i in range(20):
        plt.figure()
        plt.title("Eliminated Galaxies in Rosette "+str(i+1)+"\n Rosette Coordinates: "+str(rosettes['RA'][i])+", "+str(rosettes['DEC'][i]))
        plt.scatter(spholeselg['RA'],spholeselg['DEC'],s=0.1,label = "hole centers")
        plt.xlim(rosettes['RAmin'][i],rosettes['RAmax'][i])
        plt.ylim(rosettes['DECmin'][i],rosettes['DECmax'][i])
        plt.scatter(elgprimaries['RA'],elgprimaries['DEC'],s=0.1,label = "SV3 galaxies")
        plt.scatter(rosettes['RA'][i],rosettes['DEC'][i],label = 'rosette center',marker='x',color='red')
        plt.scatter(elgeliminated['RA'],elgeliminated['DEC'],label = 'eliminated galaxies',s=4,color='red')
        plt.legend(loc = 'upper right')
        #plt.scatter(rantable['RA'][::100],rantable['DEC'][::100])
    plt.figure()
    plt.scatter(elgeliminated['RA'],elgeliminated['DEC'],label = 'eliminated galaxies',s=4,color='red')
    for i in range(20):
        print(len(elgeliminated[elgeliminated['rosette'] == i + 1]))
    FOR TESTING '''
    
    elgprimarymap = [coord[:3] for coord in elgprimaries['xyzd']]
    lrgprimarymap = [coord[:3] for coord in lrgprimaries['xyzd']]

    
    print("finished hole cleaning")
    elgfinal = len(elgprimaries)
    lrgfinal = len(lrgprimaries)
    print("ELG primaries remaining: "+str(elgfinal))
    print("LRG primaries remaining: "+str(lrgfinal))
    elgpassrate = elgfinal/elginit
    lrgpassrate = lrgfinal/lrginit
    print("Pass rate, ELGs: "+str(elgpassrate))
    print("Pass rate, LRGs: "+str(lrgpassrate))
    
    
    print("Searching for neighbors in spheres")
    #search for neighbors
    rsphere = np.sqrt(R_CiC**2 + (L_CiC/2)**2)
    elgprimaries['neighbors'] = elgtree.query_ball_point(elgprimarymap, rsphere)
    lrgprimaries['neighbors'] = lrgtree.query_ball_point(lrgprimarymap, rsphere)

    #search for neighbors among other class of galaxies
    elgprimaries['lrg_neighbors'] = lrgtree.query_ball_point(elgprimarymap, rsphere)
    lrgprimaries['elg_neighbors'] = elgtree.query_ball_point(lrgprimarymap, rsphere)
    
    #prune galaxies
    hsq = L_CiC**2/4
    dsq = R_CiC**2
    print("Pruning galaxies from spheres")
    print("Pruning ELG auto-correlation")
    elgprimaries['neighborspruned'] = \
    [np.array(elgprimaries[i]['neighbors'])[tableprune(elgprimaries,elgsecondaries,i,elgprimaries[i]['neighbors'],
                                                      L_CiC,R_CiC)][1:] 
                for i in range(len(elgprimaries))]
    print("Pruning LRG auto-correlation")
    lrgprimaries['neighborspruned'] = \
    [np.array(lrgprimaries[i]['neighbors'])[tableprune(lrgprimaries,lrgsecondaries,i,lrgprimaries[i]['neighbors'],
                                                      L_CiC,R_CiC)][1:]
                for i in range(len(lrgprimaries))]
    print("Pruning ELG-LRG cross-correlation")
    elgprimaries['lrg_neighborspruned'] = \
    [np.array(elgprimaries[i]['lrg_neighbors'])[tableprune(elgprimaries,lrgsecondaries,i,elgprimaries[i]['lrg_neighbors'],
                                                          L_CiC,R_CiC)]
                for i in range(len(elgprimaries))]
    print("Pruning LRG-ELG cross-correlation")
    lrgprimaries['elg_neighborspruned'] = \
    [np.array(lrgprimaries[i]['elg_neighbors'])[tableprune(lrgprimaries,elgsecondaries,i,lrgprimaries[i]['elg_neighbors'],
                                                          L_CiC,R_CiC)] 
                for i in range(len(lrgprimaries))]
    
    #count up the galaxies in cylinders
    elgprimaries['N_CiC'] = [len(n) for n in elgprimaries['neighborspruned']]
    lrgprimaries['N_CiC'] = [len(n) for n in lrgprimaries['neighborspruned']]
    elgprimaries['N_lrgCiC'] = [len(n) for n in elgprimaries['lrg_neighborspruned']]
    lrgprimaries['N_elgCiC'] = [len(n) for n in lrgprimaries['elg_neighborspruned']]

    '''
    #get N(z) for secondaries
    d, ix = lrgtree.query(elgprimarymap) #find list of nearest neighbors
    #for each elg primary, find the lrg n(z), based on "NZ" of the nearest lrg neighbor
    elgprimaries["NZ_lrg"] = [lrgsecondaries[ix[i]][nz] for i in range(len(ix))]
    d, ix = elgtree.query(lrgprimarymap)
    lrgprimaries["NZ_elg"] = [elgsecondaries[ix[i]][nz] for i in range(len(ix))]
    '''
    elgprimaries['x'],elgprimaries['y'],elgprimaries['z'],d = np.rollaxis(elgprimaries['xyzd'],1).tolist()
    lrgprimaries['x'],lrgprimaries['y'],lrgprimaries['z'],d = np.rollaxis(lrgprimaries['xyzd'],1).tolist()
    
    #save our results
    #elg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_lrgCiC']))
    #lrg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_elgCiC']))
    
    if ('rosette' in elgprimaries.columns):
        elgprimaries['rosette'] = [ros[0] for ros in elgprimaries['rosette']]
        lrgprimaries['rosette'] = [ros[0] for ros in lrgprimaries['rosette']]
        
        elgtowrite = elgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_lrgCiC','rosette']
        lrgtowrite = lrgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_elgCiC','rosette']
    else:
        elgtowrite = elgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_lrgCiC']
        lrgtowrite = lrgprimaries[ra,dec,z,'x','y','z','d','N_CiC','N_elgCiC']

    astropy.io.ascii.write(elgtowrite, 'datafiles/'+outfilename+'_elg.csv', overwrite=True,format='csv')
    astropy.io.ascii.write(lrgtowrite, 'datafiles/'+outfilename+'_lrg.csv', overwrite=True,format='csv')
    
    print('Saved file datafiles/'+outfilename+'_elg.csv')
    print('Saved file datafiles/'+outfilename+'_lrg.csv')

def countsAroundRandomsHoleClean(sprandoms, spelg, splrg, spholeselg, spholeslrg, columnnames, 
                                 R_CiC, L_CiC, R_hole, zlim, primarymask, primaryargs, outfilename, directory = 'datafiles'):
    #Arguments: table with elgs, table with lrgs, column names = [ra,dec,z,zwarn], function to get mask for primaries (for 
        #SV3 includes rosette limits; for mocks no such limits necessary
    print("Will try to save to location " + directory+outfilename+'.csv')
    ra,dec,z = columnnames
    randomsecondaries = sprandoms[(sprandoms[z] > zlim[0])&(sprandoms[z] < zlim[1])]
    elgsecondaries = spelg[(spelg[z] > zlim[0])&(spelg[z] < zlim[1])]
    lrgsecondaries = splrg[(splrg[z] > zlim[0])&(splrg[z] < zlim[1])]
    
    #get comoving distances of galaxies along line of sight
    randomsecondaries['d'] = cosmo.comoving_distance(randomsecondaries[z])/u.Mpc #more convenient to have unitless calculations
    elgsecondaries['d'] = cosmo.comoving_distance(elgsecondaries[z])/u.Mpc 
    lrgsecondaries['d'] = cosmo.comoving_distance(lrgsecondaries[z])/u.Mpc

    #get unit sphere coordinates of galaxies
    print('finding unit sphere coordinates of galaxies')
    randomsecondaries['UScoord'] = raDecToUnitSphere(randomsecondaries[ra],randomsecondaries[dec])
    elgsecondaries['UScoord'] = raDecToUnitSphere(elgsecondaries[ra],elgsecondaries[dec])
    lrgsecondaries['UScoord'] = raDecToUnitSphere(lrgsecondaries[ra],lrgsecondaries[dec])

    print('finding Cartesian coordinates of galaxies')
    #get Cartesian coordinates
    randomsecondaries['xyzd'] = unitSphereToCartesian(randomsecondaries['UScoord'],randomsecondaries['d'])
    elgsecondaries['xyzd'] = unitSphereToCartesian(elgsecondaries['UScoord'],elgsecondaries['d'])
    lrgsecondaries['xyzd'] = unitSphereToCartesian(lrgsecondaries['UScoord'],lrgsecondaries['d'])

    randomprimaries = randomsecondaries[primarymask(randomsecondaries,*primaryargs)]
    #primaryargs: [zmin,zmax,psimin,psimax,cylr,cylh] for SV3
    #primaryargs: [zmin,zmax,cylr,cylh] for mocks
    
    #make 3D maps of galaxies
    #primaries
    randomUSmap = [coord[:3] for coord in randomprimaries['UScoord']]

    #hole maps
    randomholemap = [[spholeslrg['x'][i],spholeslrg['y'][i],spholeslrg['z'][i]] for i in range(len(spholeslrg))] + \
            [[spholeselg['x'][i],spholeselg['y'][i],spholeselg['z'][i]] for i in range(len(spholeselg))]

    print('making cKDTrees')
    #make ckdtrees
    randomtree = cKDTree([coord[:3] for coord in randomsecondaries['xyzd']])
    elgtree = cKDTree([coord[:3] for coord in elgsecondaries['xyzd']])
    lrgtree = cKDTree([coord[:3] for coord in lrgsecondaries['xyzd']])

    randomprimtree_US = cKDTree(randomUSmap)
    randomholes = cKDTree(randomholemap)

    print("eliminating randoms near holes")
    print("initial randoms: "+str(len(randomprimaries)))
    randomholeneighbors = randomprimtree_US.query_ball_tree(randomholes,R_hole)
    randommask = [len(n) < 1 for n in randomholeneighbors]

    #this is possibly not so efficient but whatever
    randomprimaries = randomprimaries[randommask]
    randomprimarymap = [coord[:3] for coord in randomprimaries['xyzd']]

    
    
    print("Searching for neighbors in spheres")
    #search for neighbors
    rsphere = np.sqrt(R_CiC**2 + (L_CiC/2)**2)
    randomprimaries['elg_neighbors'] = elgtree.query_ball_point(randomprimarymap, rsphere)
    randomprimaries['lrg_neighbors'] = lrgtree.query_ball_point(randomprimarymap, rsphere)
    
    #prune galaxies
    hsq = L_CiC**2/4
    dsq = R_CiC**2
    print("Pruning galaxies from spheres")
    randomprimaries['elg_neighborspruned'] = \
    [np.array(randomprimaries[i]['elg_neighbors'])[tableprune(randomprimaries,elgsecondaries,i,randomprimaries[i]['elg_neighbors'],
                                                      L_CiC,R_CiC)]
                for i in range(len(randomprimaries))]
    #count up the galaxies in cylinders
    randomprimaries['N_elgCiC'] = [len(n) for n in randomprimaries['elg_neighborspruned']]
    randomprimaries['lrg_neighborspruned'] = \
    [np.array(randomprimaries[i]['lrg_neighbors'])[tableprune(randomprimaries,lrgsecondaries,i,randomprimaries[i]['lrg_neighbors'],
                                                      L_CiC,R_CiC)]
                for i in range(len(randomprimaries))]
    #count up the galaxies in cylinders
    randomprimaries['N_lrgCiC'] = [len(n) for n in randomprimaries['lrg_neighborspruned']]


    '''
    #get N(z) for secondaries
    d, ix = lrgtree.query(elgprimarymap) #find list of nearest neighbors
    #for each elg primary, find the lrg n(z), based on "NZ" of the nearest lrg neighbor
    elgprimaries["NZ_lrg"] = [lrgsecondaries[ix[i]][nz] for i in range(len(ix))]
    d, ix = elgtree.query(lrgprimarymap)
    lrgprimaries["NZ_elg"] = [elgsecondaries[ix[i]][nz] for i in range(len(ix))]
    '''
    randomprimaries['x'],randomprimaries['y'],randomprimaries['z'],d = np.rollaxis(randomprimaries['xyzd'],1).tolist()
    
    #save our results
    #elg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_lrgCiC']))
    #lrg_col = np.concatenate((columnnames,['x','y','z','d','N_CiC','N_elgCiC']))
    

    randomtowrite = randomprimaries[ra,dec,z,'x','y','z','d','N_elgCiC','N_lrgCiC']

    astropy.io.ascii.write(randomtowrite, directory+outfilename+'.csv', overwrite=True,format='csv')