# -*- coding: utf-8 -*-

#import tesla
import squareMatrix_tpsa as sm
from tpsa import CTPS
import numpy as np
import jnfdefinition as jfdf
import squarematrixdefinition as sqdf
import matplotlib.pylab as plt
import cmath,copy

import apsu
apsu.ring.getMatLine()

# --- calculate chromaticity response matrix
chsext = apsu.ring.getElements('sext')[::2][-2:]
apsu.ring.getChrm(chsext)

nv,norder = 4,7
CTPS.set_dim(nv)
CTPS.set_max_degree(norder)

powerindex = sm.powerIndex(nv=nv,norder=norder)

L = [apsu.ring.bl[k].L for k in apsu.ring.klist]

tol = 1e-12
betax0,betay0 = apsu.ring.betax[0],apsu.ring.betay[0]
alphax0,alphay0 = apsu.ring.alfax[0],apsu.ring.alfay[0]
phix0,phiy0 = apsu.ring.nux*np.pi*2,apsu.ring.nuy*np.pi*2
sqrtbetax = np.sqrt(betax0)
sqrtbetay = np.sqrt(betay0)

mlen = len(powerindex)
# sequencenumber[i1,i2,i3,i4] gives the sequence number in power index for power of x^i1*xp^i2*y^i3*yp^i4
sequencenumber = np.zeros((norder+1,norder+1,norder+1,norder+1),'i')

for i in xrange(mlen):
    ipi = powerindex[i]
    sequencenumber[ipi[0]][ipi[1]][ipi[2]][ipi[3]] = i

#5. Construct the BK square matrix using the first 5 rows.
bK,bKi = sqdf.BKmatrix(betax0,phix0,alphax0,\
                       betay0,phiy0,alphay0,
                       0,0,norder,powerindex,sequencenumber,tol)

Bi = np.array([[1/sqrtbetax,0,0,0],[alphax0/sqrtbetax,sqrtbetax,0,0],
               [0,0,1/sqrtbetay,0],[0,0,alphay0/sqrtbetay,sqrtbetay]])
B = np.array([[sqrtbetax,0,0,0],[-alphax0/sqrtbetax,1/sqrtbetax,0,0],
              [0,0,sqrtbetay,0],[0,0,-alphay0/sqrtbetay,1/sqrtbetay]])
Ki = np.array([[1,-1j,0,0],[1,1j,0,0],[0,0,1,-1j],[0,0,1,1j]]) 
bKi4b4 = np.dot(Ki,Bi)


def Jxy(x,y,Bi,B,n=64):
    '''
    here Jx = sqrt(2*Jx (as SY Lee book)) 
    '''
    Jx = np.sqrt(Bi[0,0]*Bi[0,0]+Bi[1,0]*Bi[1,0])*x
    Jy = np.sqrt(Bi[2,2]*Bi[2,2]+Bi[3,2]*Bi[3,2])*y
    theta = np.linspace(0,2*np.pi,n)
    R = Jx#*(1+0.1*np.sin(3*theta-np.pi/2))
    x = R*np.cos(theta)
    xp = R*np.sin(theta)
    R = Jy
    y = R*np.cos(theta)
    yp = R*np.sin(theta)
    v = np.vstack([x,xp,y,yp])
    v = np.dot(B,v)
    return zip(v[0],v[1],v[2],v[3])

x0list = [3.0e-3,5.0e-3,7.0e-3]
y0list = [1.0e-3,2.0e-3,3.0e-3]
xylist = []
for x0,y0 in zip(x0list,y0list):
    xylist += [Jxy(x0,y0,Bi,B)]

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
	dmux = -1j*(wx[1])/b0x
	ax2 = abs(wx[2]/wx[0]-(wx[1]/wx[0])**2)
	ab0x = abs(b0x)
	phib0x = cmath.phase(b0x)
	wy = np.dot(uy,Zys)
	b0y = wy[0]
	dmuy = -1j*(wy[1])/b0y
	ay2 = abs(wy[2]/wy[0]-(wy[1]/wy[0])**2)
	ab0y = abs(b0y)
	phib0y = cmath.phase(b0y)
	return ab0x, ab0y, phib0x, phib0y, dmux, dmuy, ax2, ay2


def cirDist(k2s):
	'''
	circle distortion
	'''
	norder = 7

        # slow but general ---
	sext = apsu.ring.getElements('sext','S01A_S1')[0]
        sext.put('K2',k2s[0])
	sext = apsu.ring.getElements('sext','S01A_S2')[0]
        sext.put('K2',k2s[1])
	sext = apsu.ring.getElements('sext','S01A_S3')[0]
        sext.put('K2',k2s[2])
	sext = apsu.ring.getElements('sext','S01B_S3')[0]
        sext.put('K2',k2s[3])
	sext = apsu.ring.getElements('sext','S01B_S2')[0]
        sext.put('K2',k2s[4])
	sext = apsu.ring.getElements('sext','S01B_S1')[0]
        sext.put('K2',k2s[5])
	sext = apsu.ring.getElements('sext','S02A_S1')[0]
        sext.put('K2',k2s[6])
	sext = apsu.ring.getElements('sext','S02A_S2')[0]
        sext.put('K2',k2s[7])
	sext = apsu.ring.getElements('sext','S02A_S3')[0]
        sext.put('K2',k2s[8])
	sext = apsu.ring.getElements('sext','S02B_S3')[0]
        sext.put('K2',k2s[9])
	#sext = apsu.ring.getElements('sext','S02B_S2')[0]
        #sext.put('K2',k2s[10])
	#sext = apsu.ring.getElements('sext','S02B_S1')[0]
        #sext.put('K2',k2s[11])

        apsu.ring.cchrom1(chsext,[0.25,0.25])

        K2 = [apsu.ring.bl[k].K2 for k in apsu.ring.klist]
        
        temp = [CTPS(0,i) for i in xrange(1,nv+1)]
        for i,k in enumerate(apsu.ring.klist):
            temp = sm.lmPass(apsu.ring.mlist[i][:nv,:nv],temp)
            temp = sm.thickSextPass(L[i],K2[i],1,temp)
        temp = sm.lmPass(apsu.ring.mlist[-1][:nv,:nv],temp)
        mf = np.array([[temp[i].element(jj) for jj in xrange(mlen)] for i in xrange(nv)])
        #mf = np.array([sm.aline(temp[i],powerindex) for i in xrange(nv)])

        elemtx = mf[:,1:5]
        
        #6. Derive normalized map M=(BK)^(-1).mm.BK, see my notes 'Relation to Normal Form' of Wednesday, March 30, 2011 10:34 PM
        #   here #mfbk is the first 4 rows of M, it is 

	mfbk = jfdf.d3(bKi[1:5,1:5],mf,bK) 

	mfbk,scalemf,As,Asm = scalingmf(mfbk,powerindex)

	Ms = sqdf.squarematrix(mfbk,norder,powerindex,sequencenumber,tol)

	try:
            #print Ms[-1],Ms.shape,phix0,1,powerindex,scalemf,sequencenumber[1,0,0,0],norder
	    ux1,uxbar,Jx,scalex,Msx,As2x,Asm2x = \
		jfdf.UMsUbarexpJ(Ms,phix0,1,powerindex,scalemf,sequencenumber[1,0,0,0],ypowerorder=norder)
	    uy1,uybar,Jy,scaley,Msy,As2y,Asm2y = \
		jfdf.UMsUbarexpJ(Ms,phiy0,1,powerindex,scalemf,sequencenumber[0,0,1,0],ypowerorder=norder)
	except:
            return [1.0e8]*12

	# --- particle one by one
        zx,zy = [],[]
        for xy in xylist:
            #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
            section1 = [] #section1 is without joined blocks
            for row in xy:
                    row1 = row+(ux1,uy1)
                    section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
                                              bKi4b4,scalex,scaley,powerindex,norder))
            st1 = zip(*section1)
            #print 'K2s = %.2f:'%k2s
            zmax = max(st1[0])
            zmin = min(st1[0])
            zav = np.mean(st1[0])
            zx1 = (zmax-zmin)/zav
            #print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
            zmax = max(st1[1])
            zmin = min(st1[1])
            zav = np.mean(st1[1])
            zy1 = (zmax-zmin)/zav
            #print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
            #print ''
            zx.append(zx1)
            zy.append(zy1)

	xlist = np.linspace(-7e-3,7e-3,20) # a list of x
	ylist = np.linspace(1e-6,3.5e-3,10) # a list of y
	xyplane = [[i,j] for i in xlist for j in ylist]

	nu,da = [],[]
	for x,y in xyplane:
		t1,t2,t3,t4,t5,t6,t7,t8 = tuneshift(x,0,y,0,ux1,uy1,
						    bKi4b4,scalex,scaley,powerindex,norder)
		nu.append([t5,t6])
		da.append([t7,t8])

	da = np.array(da)
	#nux = [i[0].real/2/np.pi for i in nu]
	#nuy = [i[1].real/2/np.pi for i in nu]
	nux = [i[0].imag/2/np.pi for i in nu]
	nuy = [i[1].imag/2/np.pi for i in nu]
	
	dnuxda = np.max(np.abs(nux))
	dnuyda = np.max(np.abs(nuy))
	dax = np.max(np.abs(da[:,0]))
	day = np.max(np.abs(da[:,1]))

        k2c = [sc.K2 for sc in chsext]
	return zx+zy+[dnuxda,dnuyda,dax,day]+k2c

'''
print cirDist([2.545121e+01,-1.288647e+01,-1.383242e+01,-2.711628e+01,3.292057e+01,-4.375179e-01])
print
print cirDist([3.483584e+01,  -3.405252e+01,  -5.389466e+00,  -6.038834e+00,   2.940909e+01,  -2.685083e+01])
print
print '1.588451e+00   1.052992e+00   1.478513e-01   2.143018e-01   1.830485e+00   5.675500e-01   4.497043e-02   3.835739e-02   2.064320e-02   4.724491e-02'
'''
