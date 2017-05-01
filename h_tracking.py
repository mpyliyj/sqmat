# -*- coding: utf-8 -*-
import numpy as np
import anhe

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
    return np.dot(B,v)
    #return zip(v[0],v[1],v[2],v[3])

betax0,betay0 = anhe.ring.betax[0],anhe.ring.betay[0]
alphax0,alphay0 = anhe.ring.alfax[0],anhe.ring.alfay[0]
phix0,phiy0 = anhe.ring.nux*np.pi*2,anhe.ring.nuy*np.pi*2
sqrtbetax = np.sqrt(betax0)
sqrtbetay = np.sqrt(betay0)
NN = 64
Bi = np.array([[1/sqrtbetax,0,0,0],[alphax0/sqrtbetax,sqrtbetax,0,0],
               [0,0,1/sqrtbetay,0],[0,0,alphay0/sqrtbetay,sqrtbetay]])
B = np.array([[sqrtbetax,0,0,0],[-alphax0/sqrtbetax,1/sqrtbetax,0,0],
              [0,0,sqrtbetay,0],[0,0,-alphay0/sqrtbetay,1/sqrtbetay]])
x0list = [1.0e-2,1.5e-2,2.0e-2]
y0list = [1.0e-3,2.0e-3,3.0e-3]
xylist = []
for x0,y0 in zip(x0list,y0list):
    xylist += [Jxy(x0,y0,Bi,B,n=NN)]
xylist = np.hstack(xylist)
xy = np.zeros((6,xylist.shape[1]))
xy[:4] = xylist

def cirDist(k2s):
    '''
    circle distortion
    '''
    #norder = 7
    
    # slow but general ---
    sext = anhe.ring.getElements('sext','sh1')[0]
    sext.put('K2',k2s[0])
    sext = anhe.ring.getElements('sext','sh3')[0]
    sext.put('K2',k2s[1])
    sext = anhe.ring.getElements('sext','sh4')[0]
    sext.put('K2',k2s[2])
    sext = anhe.ring.getElements('sext','sl3')[0]
    sext.put('K2',k2s[3])
    sext = anhe.ring.getElements('sext','sl2')[0]
    sext.put('K2',k2s[4])
    sext = anhe.ring.getElements('sext','sl1')[0]
    sext.put('K2',k2s[5])

    xt = anhe.ring.eletrack(xy,sym4=1)[-1]
    xt = np.dot(Bi,xt[:4])
    cd = []
    for i in range(3):
        xx = xt[0,i*NN:(i+1)*NN]**2+xt[1,i*NN:(i+1)*NN]**2
        cd.append(np.std(xx)/np.average(xx))
        yy = xt[2,i*NN:(i+1)*NN]**2+xt[3,i*NN:(i+1)*NN]**2
        cd.append(np.std(yy)/np.average(yy))
        if np.sum(np.isnan(cd)):
            return [100.]*6
    return cd
    #print xt.shape

'''
print cirDist([2.545121e+01,-1.288647e+01,-1.383242e+01,-2.711628e+01,3.292057e+01,-4.375179e-01])
print
print cirDist([3.483584e+01,  -3.405252e+01,  -5.389466e+00,  -6.038834e+00,   2.940909e+01,  -2.685083e+01])
print
print '1.588451e+00   1.052992e+00   1.478513e-01   2.143018e-01   1.830485e+00   5.675500e-01   4.497043e-02   3.835739e-02   2.064320e-02   4.724491e-02'
'''
