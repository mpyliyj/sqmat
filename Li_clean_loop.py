# -*- coding: utf-8 -*-

import tesla
import numpy as np
import jnfdefinition as jfdf
import squarematrixdefinition as sqdf
import matplotlib.pylab as plt
import cmath


#7. Scale the one turn map in z,z* space 
def scalingmf(mf,powerindex):
	#Find a scale s so that s x**m and s is on the same scale if the term in M with maximum absolute value has power m
	#as described in "M scaling" in "Jordan Form Reformulation.one". mf is the first 4 rows of M so the scaling method is the same.
	absM = abs(mf)
	i,j = np.unravel_index(absM.argmax(), absM.shape)
	power = sum(powerindex[j])
	scalex1 = (absM.max())**(-(1./(power-1.)))
	scalem1 = 1/scalex1
	mlen = len(powerindex)
	As = np.identity(mlen)
	for i in range(mlen):
		As[i,i] = scalem1**sum(powerindex[i])
	Asm = np.identity(mlen)
	for i in range(mlen):
		Asm[i,i] = scalex1**sum(powerindex[i])
	mfs = scalem1*np.dot(mf,Asm)
	return np.array(mfs),scalex1, As,Asm

def normalcoordinate(xb,xpb,yb,ypb): 
	'''
	Calculate action w for given x,xp, and the theoretical first order tune shift in w-plane as function of x,xp
	See note "Relation to driving terms/Jordan Form Reformulation/Amplitude dependent tune", section 8, 2/19/2015
	For tune with 5 phix0 close to 2 pi, block 1, -4, 6 are used
	'''
	zxb = xb-1j*xpb
	ab0x = xb**2+xpb**2
	phib0x = cmath.phase(zxb)
	zyb = yb-1j*ypb
	ab0y = yb**2+ypb**2
	phib0y = cmath.phase(zyb)
	return ab0x,ab0y,phib0x,phib0y

def tuneshift(x,xp,y,yp,ux,uy,bKi4b4,scalex,scaley,powerindex,norder): 
	'''
	Calculate action w for given x,xp, and the theoretical first order tune shift in w-plane as function of x,xp
	See note "Relation to driving terms/Jordan Form Reformulation/Amplitude dependent tune", section 8, 2/19/2015
	For tune with 5 phix0 close to 2 pi, block 1, -4, 6 are used
	'''
	zxbar,zxbars,zybar,zybars = np.dot(bKi4b4,np.array([x,xp,y,yp]))
	zxsbar = zxbar/scalex
	zxsbars = zxbars/scalex
	zysbar = zybar/scalex
	zysbars = zybars/scalex
	Zxs = sqdf.Zcol(zxsbar,zxsbars,zysbar,zysbars,norder,powerindex) #Zxs is the zsbar,zsbars, column, here zsbar = zbar/scalex
	zxsbar = zxbar/scaley
	zxsbars = zxbars/scaley
	zysbar = zybar/scaley
	zysbars = zybars/scaley
	Zys = sqdf.Zcol(zxsbar,zxsbars,zysbar,zysbars,norder,powerindex) #Zys is the zsbar,zsbars, column, here zsbar = zbar/scaley
	wx = np.dot(ux,Zxs) #Zxs is used for wx while Zys is used for wy, separately!
	b0x = wx[0]
	dmu00x = -1j*(wx[1])/b0x
	ab0x = abs(b0x)
	phib0x = cmath.phase(b0x)
	wy = np.dot(uy,Zys)
	b0y = wy[0]
	dmu00y = -1j*(wy[1])/b0y
	ab0y = abs(b0y)
	phib0y = cmath.phase(b0y)
	return ab0x,ab0y, phib0x, phib0y


x0 = 4e-3
y0 = 4e-3
npass = 256

lattice = "nsls2sr_supercell_ch77_20150406.tslat"
ring = tesla.Ring(lattice, "RING")
norder = 7
tol = 1e-12

sexts = ring.matchElements('sl1.*')

WX1,WY1 = [],[]

for iloop,k2s in enumerate(np.linspace(0,30,1)):
    for s in sexts:
        ring[s,'K2'] = k2s


    # six phase space varialbe, 4 independent, expand to n'rd order.
    m = tesla.TPSMap(6,4,norder)
    m.c = [0, 0, 0, 0, 0, 0]
    m.m = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0],[0,0,0,0]]
    ms = ring.trackTPSMaps(m, 0, ring.elements()) #ms has the maps of all elements around the ring, m is one turn map
    if iloop == 0:
	    tw, ml = tesla.calcLinearTwiss(ring)


    #4. Construct map matrix mm
    mf,powerindex = sqdf.mfunction(m,norder) # m is one turn map
    #This program only applies when x[0]=0,xp[0]=0,y[0]=0, yp[0]=0. If not, then the map matrix mm is no longer semi-triangular
    mf[0][0] = 0
    mf[1][0] = 0
    mf[2][0] = 0
    mf[3][0] = 0
    mf = np.array(mf)

    mlen=len(powerindex)

    # sequencenumber[i1,i2,i3,i4] gives the sequence number in power index for power of x^i1*xp^i2*y^i3*yp^i4
    sequencenumber = np.zeros((norder+1,norder+1,norder+1,norder+1),'i')
    powerindex = sqdf.powerindex4(norder)
    powerindex = np.array(powerindex,'i')
    mlen = len(powerindex)

    for i in range(mlen):
	    ip = powerindex[i]
	    sequencenumber[ip[0]][ip[1]][ip[2]][ip[3]] = i

    betax0,phix0,alphax0,betay0,phiy0,alphay0 = sqdf.extracttwiss(mf)
    gammax0 = (1+alphax0**2)/betax0
    mlix = [np.cos(phix0)+alphax0*np.sin(phix0), betax0*np.sin(phix0)],[-gammax0*np.sin(phix0), np.cos(phix0)-alphax0*np.sin(phix0)]
    gammay0 = (1+alphay0**2)/betay0
    mliy = [np.cos(phiy0)+alphay0*np.sin(phiy0), betay0*np.sin(phiy0)],[-gammay0*np.sin(phiy0), np.cos(phiy0)-alphay0*np.sin(phiy0)]

    elemtx = np.zeros((4,4))
    elemtx[0:2,0:2] = mlix #Replace linear part of elegant, which is accurate only to 8 digits, 
    #by the twiss matrix obtained from the twiss parameters extract from the tpsa linear part of matrix
    elemtx[2:4,2:4] = mliy #Notice that mlix and mliy is accurate simplex to machine precision, so their determinantes are closer to zero.

    mf[:,1:5] = elemtx #Now replace the linear part of map matrix by the more accurate twiss matrix

    sqrtbetax = np.sqrt(betax0)
    sqrtbetay = np.sqrt(betay0)

    #5. Construct the BK square matrix using the first 5 rows.
    bK,bKi = sqdf.BKmatrix(betax0,phix0,alphax0,betay0,phiy0,alphay0,0,0,norder,powerindex,sequencenumber,tol)

    #6. Derive normalized map M=(BK)^(-1).mm.BK, see my notes 'Relation to Normal Form' of Wednesday, March 30, 2011 10:34 PM
    #   here #mfbk is the first 4 rows of M, it is 
    mfbk = jfdf.d3(bKi[1:5,1:5],mf,bK) 
    
    mfbk,scalemf,As,Asm = scalingmf(mfbk,powerindex)

    Ms = sqdf.squarematrix(mfbk,norder,powerindex,sequencenumber,tol) 

    # ---
    #'''
    if iloop == 0:
	    xxpyyp = np.zeros((4,npass+1))
	    xxpyyp[:,0] = np.array([x0,0,y0,0])
	    for i in xrange(1,npass+1):
		    xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])

	    Bi = np.array([[1/sqrtbetax,0,0,0],[alphax0/sqrtbetax,sqrtbetax,0,0],
			   [0,0,1/sqrtbetay,0],[0,0,alphay0/sqrtbetay,sqrtbetay]])
	    Ki = np.array([[1,-1j,0,0],[1,1j,0,0],[0,0,1,-1j],[0,0,1,1j]]) 
	    bKi4b4 = np.dot(Ki,Bi)

	    xybar = np.dot(Bi,xxpyyp)
	    xy = zip(*xybar)

    section = []
    for row in xy:
	    section.append(normalcoordinate(*row))
    st = zip(*section)

    #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
    try:
	    ux1,uxbar,Jx,scalex,Msx,As2x,Asm2x = jfdf.UMsUbarexpJ(Ms,phix0,1,powerindex,scalemf,sequencenumber[1,0,0,0],ypowerorder=norder)
	    uy1,uybar,Jy,scaley,Msy,As2y,Asm2y = jfdf.UMsUbarexpJ(Ms,phiy0,1,powerindex,scalemf,sequencenumber[0,0,1,0],ypowerorder=norder)

    except:
	    continue

    section1 = [] #section1 is without joined blocks
    xy = zip(*xxpyyp)
    for row in xy:
        row1 = row+(ux1,uy1)
	section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
				  bKi4b4,scalex,scaley,powerindex,norder))

    st1 = zip(*section1)

    print 'K2s = %.2f:'%k2s
    zmax = max(st1[0])
    zmin = min(st1[0])
    zav = np.mean(st1[0])
    zx1 = (zmax-zmin)/zav
    print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1

    zmax = max(st1[1])
    zmin = min(st1[1])
    zav = np.mean(st1[1])
    zy1 = (zmax-zmin)/zav
    print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
    print ''

    WX1.append([k2s,zx1])
    WY1.append([k2s,zy1])

WX1 = np.array(WX1)
WY1 = np.array(WY1)

plt.figure()
plt.subplot(311)
plt.plot(WX1[:,1],WY1[:,1],'o')
plt.subplot(312)
plt.plot(WX1[:,0],WX1[:,1],'-o')
plt.subplot(313)
plt.plot(WY1[:,0],WY1[:,1],'-o')
plt.savefig('WXY1.png')
plt.close()
