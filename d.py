# -*- coding: utf-8 -*-

#import tesla
import squareMatrix_dev as sm

import pylatt as latt
import numpy as np
import jnfdefinition as jfdf
import squarematrixdefinition as sqdf
import matplotlib.pylab as plt
import cmath,copy

nv,norder = 4,7
unit,zero,one = sm.const(nv=nv)
powerindex = sm.powerIndex(nv=nv,norder=norder)

tol = 1e-12

mlen = len(powerindex)
# sequencenumber[i1,i2,i3,i4] gives the sequence number in power index for power of x^i1*xp^i2*y^i3*yp^i4
sequencenumber = np.zeros((norder+1,norder+1,norder+1,norder+1),'i')
for i in xrange(mlen):
    ipi = powerindex[i]
    sequencenumber[ipi[0]][ipi[1]][ipi[2]][ipi[3]] = i

#--- 0%
import nsls2sr_supercell_ch77_20160525 as nsls2
nsls2.ring.getMatLine()
L = [nsls2.ring.bl[k].L for k in nsls2.ring.klist]
displist = []
for m in nsls2.ring.mlist:
    displist.append(np.array(m[0:4,5])[:,0])

Ki = np.array([[1,-1j,0,0],[1,1j,0,0],[0,0,1,-1j],[0,0,1,1j]]) 
betax0,betay0 = nsls2.ring.betax[0],nsls2.ring.betay[0]
alphax0,alphay0 = nsls2.ring.alfax[0],nsls2.ring.alfay[0]
phix0,phiy0 = nsls2.ring.nux*np.pi*2,nsls2.ring.nuy*np.pi*2
sqrtbetax = np.sqrt(betax0)
sqrtbetay = np.sqrt(betay0)
bK,bKi = sqdf.BKmatrix(betax0,phix0,alphax0,\
                       betay0,phiy0,alphay0,
                       0,0,norder,powerindex,sequencenumber,tol)
Bi = np.array([[1/sqrtbetax,0,0,0],[alphax0/sqrtbetax,sqrtbetax,0,0],
               [0,0,1/sqrtbetay,0],[0,0,alphay0/sqrtbetay,sqrtbetay]])
bKi4b4 = np.dot(Ki,Bi)

#--- -1%
import n1 as nsls2n1
latt.scaleBeamline(nsls2n1.ring,delta=-0.01)
#nsls2n1.ring.update()
nsls2n1.ring.getMatLine()

#--- +1%
import p1 as nsls2p1
latt.scaleBeamline(nsls2p1.ring,delta=0.01)
#nsls2p1.ring.update()
nsls2p1.ring.getMatLine()

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
	npass = 64

	norder = 7

        # slow but general ---
        '''
	sext = nsls2.ring.getElements('sext','sh1')[0]
        sext.put('K2',k2s[0])
	sext = nsls2.ring.getElements('sext','sh3')[0]
        sext.put('K2',k2s[1])
	sext = nsls2.ring.getElements('sext','sh4')[0]
        sext.put('K2',k2s[2])
	sext = nsls2.ring.getElements('sext','sl3')[0]
        sext.put('K2',k2s[3])
	sext = nsls2.ring.getElements('sext','sl2')[0]
        sext.put('K2',k2s[4])
	sext = nsls2.ring.getElements('sext','sl1')[0]
        sext.put('K2',k2s[5])

        K2 = [nsls2.ring.bl[k].K2 for k in nsls2.ring.klist]
        '''
        # ---fast but not general, only for nsls2-II chrom+7/+7
        sm12 = [-26.21867845,30.69679902,-28.01658206]
        K2 = np.hstack((k2s[:3],sm12,k2s[3:],k2s[-1:-4:-1],sm12,k2s[-4::-1]))

        # --- on momentum 0%
        temp = copy.deepcopy(unit)
        for i,k in enumerate(nsls2.ring.klist):
            temp = sm.lmPass(nsls2.ring.mlist[i][:nv,:nv],temp)
            temp = sm.thickSextPass(L[i],K2[i],1,temp,truncate=norder)
        temp = sm.lmPass(nsls2.ring.mlist[-1][:nv,:nv],temp)
        mf = np.array([sm.aline(temp[i],powerindex) for i in xrange(nv)])

        #print mf[0],mf.shape
        elemtx = mf[:,1:5]
        
        #6. Derive normalized map M=(BK)^(-1).mm.BK, see my notes 'Relation to Normal Form' of Wednesday, March 30, 2011 10:34 PM
        #   here #mfbk is the first 4 rows of M, it is 
	mfbk = jfdf.d3(bKi[1:5,1:5],mf,bK) 

    	mfbk,scalemf,As,Asm = scalingmf(mfbk,powerindex)
        #print mfbk[0]
        
	Ms = sqdf.squarematrix(mfbk,norder,powerindex,sequencenumber,tol)        
        
	try:
            #print Ms.shape,Ms[-1],phix0,1,powerindex,scalemf,sequencenumber[1,0,0,0],norder
            ux1,uxbar,Jx,scalex,Msx,As2x,Asm2x = \
                        jfdf.UMsUbarexpJ(Ms,phix0,1,powerindex,scalemf,sequencenumber[1,0,0,0],ypowerorder=norder)
            uy1,uybar,Jy,scaley,Msy,As2y,Asm2y = \
                        jfdf.UMsUbarexpJ(Ms,phiy0,1,powerindex,scalemf,sequencenumber[0,0,1,0],ypowerorder=norder)
	except:
            #print '0 error'
            return [1e8]*22

	# --- particle 1
	x0 = 2.5e-2
	y0 = 5e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bi,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	section1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
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
	
	# --- particle 2
	x0 = 1.0e-2
	y0 = 2e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bi,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	section1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
	for row in xy:
		row1 = row+(ux1,uy1)
		section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
					  bKi4b4,scalex,scaley,powerindex,norder))
	st1 = zip(*section1)
	#print 'K2s = %.2f:'%k2s
	zmax = max(st1[0])
	zmin = min(st1[0])
	zav = np.mean(st1[0])
	zx2 = (zmax-zmin)/zav
	#print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
	zmax = max(st1[1])
	zmin = min(st1[1])
	zav = np.mean(st1[1])
	zy2 = (zmax-zmin)/zav
	#print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
	#print ''

	# --- particle 3
	x0 = 3.5e-2
	y0 = 3.0e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bi,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	section1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
	for row in xy:
		row1 = row+(ux1,uy1)
		section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
					  bKi4b4,scalex,scaley,powerindex,norder))
	st1 = zip(*section1)
	#print 'K2s = %.2f:'%k2s
	zmax = max(st1[0])
	zmin = min(st1[0])
	zav = np.mean(st1[0])
	zx3 = (zmax-zmin)/zav
	#print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
	zmax = max(st1[1])
	zmin = min(st1[1])
	zav = np.mean(st1[1])
	zy3 = (zmax-zmin)/zav
	#print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
	#print ''

        # -1%
        delta = -0.0
        K2n1 = K2/(1+delta)
        temp = copy.deepcopy(unit)
        for i,k in enumerate(nsls2n1.ring.klist):
            temp = sm.lmPass(nsls2n1.ring.mlist[i][:nv,:nv],temp)
            temp = sm.addDispersive(temp,delta,displist[i])
            temp = sm.thickSextPass(L[i],K2n1[i],1,temp,truncate=norder)
        temp = sm.lmPass(nsls2n1.ring.mlist[-1][:nv,:nv],temp)
        mf = np.array([sm.aline(temp[i],powerindex) for i in xrange(nv)])
        # ---  remove nonzero fixed point
        for i1 in xrange(len(mf)): mf[i1,0] = 0.
        elemtx = mf[:,1:5]
        
        nux,betax0n1,alphax0n1,gx = latt.matrix2twiss(elemtx,plane='x')
        nuy,betay0n1,alphay0n1,gy = latt.matrix2twiss(elemtx,plane='x')
        phix0n1,phiy0n1 = nux*np.pi*2,nuy*np.pi*2
        sqrtbetaxn1 = np.sqrt(betax0n1)
        sqrtbetayn1 = np.sqrt(betay0n1)
        bKn1,bKin1 = sqdf.BKmatrix(betax0n1,phix0n1,alphax0n1,\
                                   betay0n1,phiy0n1,alphay0n1,
                                   0,0,norder,powerindex,sequencenumber,tol)
        Bin1 = np.array([[1/sqrtbetaxn1,0,0,0],[alphax0n1/sqrtbetaxn1,sqrtbetaxn1,0,0],
                         [0,0,1/sqrtbetayn1,0],[0,0,alphay0n1/sqrtbetayn1,sqrtbetayn1]])
        bKi4b4n1 = np.dot(Ki,Bin1)

        #6. Derive normalized map M=(BK)^(-1).mm.BK, see my notes 'Relation to Normal Form' of Wednesday, March 30, 2011 10:34 PM
        #   here #mfbk is the first 4 rows of M, it is 
	mfbk = jfdf.d3(bKin1[1:5,1:5],mf,bKn1) 
	mfbk,scalemf,As,Asm = scalingmf(mfbk,powerindex)
	Ms = sqdf.squarematrix(mfbk,norder,powerindex,sequencenumber,tol)
	try:
	    ux1,uxbar,Jx,scalex,Msx,As2x,Asm2x = \
		jfdf.UMsUbarexpJ(Ms,phix0n1,1,powerindex,scalemf,sequencenumber[1,0,0,0],ypowerorder=norder)
	    uy1,uybar,Jy,scaley,Msy,As2y,Asm2y = \
		jfdf.UMsUbarexpJ(Ms,phiy0n1,1,powerindex,scalemf,sequencenumber[0,0,1,0],ypowerorder=norder)
	except:
            #print '-1 error'
            return [1e8]*22
	# --- particle 1
	x0 = 2.5e-2
	y0 = 5e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bin1,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	section1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
	for row in xy:
		row1 = row+(ux1,uy1)
		section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
					  bKi4b4n1,scalex,scaley,powerindex,norder))
	st1 = zip(*section1)
	#print 'K2s = %.2f:'%k2s
	zmax = max(st1[0])
	zmin = min(st1[0])
	zav = np.mean(st1[0])
	zx1n1 = (zmax-zmin)/zav
	#print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
	zmax = max(st1[1])
	zmin = min(st1[1])
	zav = np.mean(st1[1])
	zy1n1 = (zmax-zmin)/zav
	#print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
	#print ''
	
	# --- particle 2
	x0 = 1.0e-2
	y0 = 2e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bin1,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	section1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
	for row in xy:
		row1 = row+(ux1,uy1)
		section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
					  bKi4b4n1,scalex,scaley,powerindex,norder))
	st1 = zip(*section1)
	#print 'K2s = %.2f:'%k2s
	zmax = max(st1[0])
	zmin = min(st1[0])
	zav = np.mean(st1[0])
	zx2n1 = (zmax-zmin)/zav
	#print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
	zmax = max(st1[1])
	zmin = min(st1[1])
	zav = np.mean(st1[1])
	zy2n1 = (zmax-zmin)/zav
	#print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
	#print ''

	# --- particle 3
	x0 = 3.5e-2
	y0 = 3.0e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bin1,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	section1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
	for row in xy:
		row1 = row+(ux1,uy1)
		section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
					  bKi4b4n1,scalex,scaley,powerindex,norder))
	st1 = zip(*section1)
	#print 'K2s = %.2f:'%k2s
	zmax = max(st1[0])
	zmin = min(st1[0])
	zav = np.mean(st1[0])
	zx3n1 = (zmax-zmin)/zav
	#print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
	zmax = max(st1[1])
	zmin = min(st1[1])
	zav = np.mean(st1[1])
	zy3n1 = (zmax-zmin)/zav
	#print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
	#print ''
        # --- end of -1%
        
        # +1%
        delta = 0.01
        K2p1 = K2/(1+delta)
        temp = copy.deepcopy(unit)
        for i,k in enumerate(nsls2p1.ring.klist):
            temp = sm.lmPass(nsls2p1.ring.mlist[i][:nv,:nv],temp)
            temp = sm.addDispersive(temp,delta,displist[i])
            temp = sm.thickSextPass(L[i],K2p1[i],1,temp,truncate=norder)
        temp = sm.lmPass(nsls2p1.ring.mlist[-1][:nv,:nv],temp)
        mf = np.array([sm.aline(temp[i],powerindex) for i in xrange(nv)])
        for i1 in xrange(len(mf)): mf[i1,0] = 0.
        elemtx = mf[:,1:5]
        nux,betax0p1,alphax0p1,gx = latt.matrix2twiss(elemtx,plane='x')
        nuy,betay0p1,alphay0p1,gy = latt.matrix2twiss(elemtx,plane='x')
        phix0p1,phiy0p1 = nux*np.pi*2,nuy*np.pi*2
        sqrtbetaxp1 = np.sqrt(betax0p1)
        sqrtbetayp1 = np.sqrt(betay0p1)
        bKp1,bKip1 = sqdf.BKmatrix(betax0p1,phix0p1,alphax0p1,\
                                   betay0p1,phiy0p1,alphay0p1,
                                   0,0,norder,powerindex,sequencenumber,tol)
        Bip1 = np.array([[1/sqrtbetaxp1,0,0,0],[alphax0p1/sqrtbetaxp1,sqrtbetaxp1,0,0],
                         [0,0,1/sqrtbetayp1,0],[0,0,alphay0p1/sqrtbetayp1,sqrtbetayp1]])
        bKi4b4p1 = np.dot(Ki,Bip1)

        #6. Derive normalized map M=(BK)^(-1).mm.BK, see my notes 'Relation to Normal Form' of Wednesday, March 30, 2011 10:34 PM
        #   here #mfbk is the first 4 rows of M, it is 
	mfbk = jfdf.d3(bKip1[1:5,1:5],mf,bKp1) 
	mfbk,scalemf,As,Asm = scalingmf(mfbk,powerindex)
	Ms = sqdf.squarematrix(mfbk,norder,powerindex,sequencenumber,tol)
	try:
	    ux1,uxbar,Jx,scalex,Msx,As2x,Asm2x = \
		jfdf.UMsUbarexpJ(Ms,phix0p1,1,powerindex,scalemf,sequencenumber[1,0,0,0],ypowerorder=norder)
	    uy1,uybar,Jy,scaley,Msy,As2y,Asm2y = \
		jfdf.UMsUbarexpJ(Ms,phiy0p1,1,powerindex,scalemf,sequencenumber[0,0,1,0],ypowerorder=norder)
	except:
            #print '+1 error'
            return [1e8]*22
	# --- particle 1
	x0 = 2.5e-2
	y0 = 5e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bip1,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	sectiop1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
	for row in xy:
		row1 = row+(ux1,uy1)
		section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
					  bKi4b4p1,scalex,scaley,powerindex,norder))
	st1 = zip(*section1)
	#print 'K2s = %.2f:'%k2s
	zmax = max(st1[0])
	zmin = min(st1[0])
	zav = np.mean(st1[0])
	zx1p1 = (zmax-zmin)/zav
	#print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
	zmax = max(st1[1])
	zmin = min(st1[1])
	zav = np.mean(st1[1])
	zy1p1 = (zmax-zmin)/zav
	#print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
	#print ''
	
	# --- particle 2
	x0 = 1.0e-2
	y0 = 2e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bip1,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	section1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
	for row in xy:
		row1 = row+(ux1,uy1)
		section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
					  bKi4b4p1,scalex,scaley,powerindex,norder))
	st1 = zip(*section1)
	#print 'K2s = %.2f:'%k2s
	zmax = max(st1[0])
	zmin = min(st1[0])
	zav = np.mean(st1[0])
	zx2p1 = (zmax-zmin)/zav
	#print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
	zmax = max(st1[1])
	zmin = min(st1[1])
	zav = np.mean(st1[1])
	zy2p1 = (zmax-zmin)/zav
	#print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
	#print ''

	# --- particle 3
	x0 = 3.5e-2
	y0 = 3.0e-3
	xxpyyp = np.zeros((4,npass+1))
	xxpyyp[:,0] = np.array([x0,0,y0,0])
	for i in xrange(1,npass+1):
		xxpyyp[:,i] = np.dot(elemtx,xxpyyp[:,i-1])
	xybar = np.dot(Bip1,xxpyyp)
	xy = zip(*xybar)
	section = []
	for row in xy:
		section.append(normalcoordinate(*row))
	st = zip(*section)
        #11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay
	section1 = [] #section1 is without joined blocks
	xy = zip(*xxpyyp)
	for row in xy:
		row1 = row+(ux1,uy1)
		section1.append(tuneshift(row1[0],row1[1],row1[2],row1[3],row1[4],row1[5],
					  bKi4b4p1,scalex,scaley,powerindex,norder))
	st1 = zip(*section1)
	#print 'K2s = %.2f:'%k2s
	zmax = max(st1[0])
	zmin = min(st1[0])
	zav = np.mean(st1[0])
	zx3p1 = (zmax-zmin)/zav
	#print "for wx1 without resonance block,  (zmax-zmin)/zmean = ", zx1
	zmax = max(st1[1])
	zmin = min(st1[1])
	zav = np.mean(st1[1])
	zy3p1 = (zmax-zmin)/zav
	#print "for wy1 without resonance block,  (zmax-zmin)/zmean = ", zy1
	#print ''
        # --- end of +1%

        
	xlist = np.linspace(-3e-2,3e-2,30) #a list of theta0
	ylist = np.arange(1e-6,1.0e-2,10) #a list of theta0
	xyplane = [[i,j] for i in xlist for j in ylist]

	nu,da = [],[]
	for x,y in xyplane:
		t1,t2,t3,t4,t5,t6,t7,t8 = tuneshift(x,0,y,0,ux1,uy1,
						    bKi4b4,scalex,scaley,powerindex,norder)
		nu.append([t5,t6])
		da.append([t7,t8])

	da = np.array(da)
	nux = [i[0].real/2/np.pi for i in nu]
	nuy = [i[1].real/2/np.pi for i in nu]
	
	dnuxda = np.max(np.abs(nux))
	dnuyda = np.max(np.abs(nuy))
	dax = np.max(np.abs(da[:,0]))
	day = np.max(np.abs(da[:,1]))
	return [zx1,zy1,zx2,zy2,zx3,zy3,
                zx1n1,zy1n1,zx2n1,zy2n1,zx3n1,zy3n1,
                zx1p1,zy1p1,zx2p1,zy2p1,zx3p1,zy3p1,
                dnuxda,dnuyda,dax,day]

'''
print cirDist([2.545121e+01,-1.288647e+01,-1.383242e+01,-2.711628e+01,3.292057e+01,-4.375179e-01])
print
print cirDist([3.483584e+01,  -3.405252e+01,  -5.389466e+00,  -6.038834e+00,   2.940909e+01,  -2.685083e+01])
print
print '1.588451e+00   1.052992e+00   1.478513e-01   2.143018e-01   1.830485e+00   5.675500e-01   4.497043e-02   3.835739e-02   2.064320e-02   4.724491e-02'
'''
