# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import tesla
import matplotlib.pylab as plt
import numpy as np
import os
import sys
import commands
from numpy import array
from numpy import linalg
import time
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
t0=time.clock()
import jnfdefinition
jfdf=jnfdefinition
import squarematrixdefinition
import pickle
import scipy
from scipy import optimize

sqdf=squarematrixdefinition
global scalex,  tol


tol=1e-12

print time.clock()-t0 ,"seconds for import sqdf and jfdf"
t0=time.clock()


def sv(filename,x):
	ff=open(filename,'w')
	pickle.dump(x,ff)
	ff.close()

def rl(filename):
	ff=open(filename)
	xx=pickle.load(ff)
	ff.close()
	return xx


global Vm,U,maxchainlenposition,bKi,norder,powerindex,mlen


def wy0(x,xp,y,yp):
	#given real space x,xp calculate twiss space z=xbar-1j*pbar, then calculate normal form w
	#that is the nonlinear normalized space.
	zxbar,zxbars,zybar,zybars=np.dot(bKi[1:5,1:5], array([x,xp,y,yp]))
	zxsbar=zxbar/scalex
	zxsbars=zxbars/scalex
	zysbar=zybar/scalex
	zysbars=zybars/scalex
	Zs=sqdf.Zcol(zxsbar,zxsbars,zysbar,zysbars,norder,powerindex) #Zs is the zsbar,zsbars, column, here zsbar=zbar/scalex
	wy=np.dot(u,Zs) #w is the invariant we denoted as b0 before. Zs is scaled Z, Z is column of zbar, zbars
	return wy[0]

def wyz(zxbar,zybar):
	#given normalized space xbar,xpbar,ybar,ypbar calculate twiss space z=xbar-1j*pbar, then calculate normal form wy
	#that is the nonlinear normalized space.
	zxbars,zybars=np.conjugate(zxbar),np.conjugate(zybar)
	zxsbar=zxbar/scalex
	zxsbars=zxbars/scalex
	zysbar=zybar/scalex
	zysbars=zybars/scalex
	Zs=sqdf.Zcol(zxsbar,zxsbars,zysbar,zysbars,norder,powerindex) #Zs is the zsbar,zsbars, column, here zsbar=zbar/scalex
  	wz=np.dot(u[0],Zs) #w is the invariant we denoted as b0 before. Zs is scaled Z, Z is column of zbar, zbars
	return wz


		


def zzsexchange(norder,powerindex,sequencenumber,tol):
#zzsexchange is the map to exchange the coordinate z with zs, so its linear part shoul be:z=zs, zs=z
#  z =  0*z +   zs
#  zs=  1*z +  0*zs
	zx=np.zeros(mlen)
	zx[2]=1
	zxs=np.zeros(mlen)
	zxs[1]=1
	zy=np.zeros(mlen)
	zy[4]=1
	zys=np.zeros(mlen)
	zys[3]=1
	zzsm=[zx,zxs,zy,zys]
	zzsexchangem=sqdf.squarematrix(zzsm,norder,powerindex,sequencenumber,tol)
	return zzsexchangem







#1.read saved map matrix M
Ms10,phix0,phiy0,powerindex,norder,bK,bKi,sqrtbetax,sqrtbetay,msf,tbl,scalemf=rl("jfM.dat")
sequencenumber=np.zeros((norder+1,norder+1,norder+1,norder+1),'i')   #sequence number is given as a function of a power index 
powerindex=sqdf.powerindex4(norder)
powerindex=array(powerindex,'i')
mlen=len(powerindex)

for i in range(mlen):
	ip=powerindex[i]
	sequencenumber[ip[0]][ip[1]][ip[2]][ip[3]]=i

mlen=len(Ms10)

norder=5
plen=(norder+1)*(norder+2)*(norder+3)*(norder+4)/24
powerindex=sqdf.powerindex4(norder)
sequencenumber=np.zeros((norder+1,norder+1,norder+1,norder+1),'i')  #sequence number is given as a function of a power index
powerindex=array(powerindex,'i')
mlen=len(powerindex)
Ms10=Ms10[:mlen,:mlen]
bK=bK[:mlen,:mlen]
bKi=bKi[:mlen,:mlen]


for i in range(mlen):
	ip=powerindex[i]
	sequencenumber[ip[0]][ip[1]][ip[2]][ip[3]]=i

mlen=len(Ms10)

#3.Test program for ypowerorder=1
uy,uybar,Jy,scalexy,Ms,As2,Asm2=jfdf.UMsUbarexpJ(Ms10,phiy0,1,powerindex,scalemf,ypowerorder=5)


print "3. Check u as left eigenvectors of M: U.M=exp(i*mu)*exp(J).U:"
print "J="
jfdf.pim(Jy,len(Jy),len(Jy))
maxchainlenposition, maxchainlen, chain, chainposition=jfdf.findchainposition(Jy)
print "position of max length chain=",maxchainlenposition
print "max length chain=",maxchainlen
tmp1=np.dot(uy,Ms)
tmp2=np.exp(1j*phiy0)*np.dot(jfdf.exp(Jy,maxchainlen),uy)
tmp3=tmp1-tmp2
print "3. check uy.Ms=exp(i*mu)*exp(Jy).uy ", abs(tmp3).max()," relative error:",abs(tmp3).max()/abs(tmp1).max()



print "3. lowest order in uy[i]:\n"
for i in range(len(uy)):
	tmp=[ sum(powerindex[k])  for k,b in enumerate(abs(uy[i])) if b>1e-8]
	if tmp!=[]: print i, min(tmp)

def dmt(k,m, tol=1e-8): #Dominating terms of k'th order in u[m]
	dt=[ [i,j,abs(uy[m][i])] for i,j in enumerate(powerindex) if (j[2]+j[3]<2 and sum(j)==k and abs(uy[m][i])>tol)]
	return dt 

print "\n3. dominating terms in uy[0], their order, and size:"
cc=[ [k,sum(powerindex[k]),b,powerindex[k].tolist()]  for k,b in enumerate(abs(uy[0])) if b>1e-8]
for i in cc:
	print i



print "\n3. dominating terms of order 1 in uy[0] with y power<2:", [ [j[0],j[1].tolist(),j[2]] for j in dmt(1,0)]
print "3. dominating terms of order 2 in uy[0] with y power<2:", [ j[1].tolist() for j in dmt(2,0)]
print "3. dominating terms of order 3 in uy[0] with y power<2:",[ j[1].tolist() for j in dmt(3,0)]

print "3. Showing the lowest of power of x in every eigenvector with y power =<1:"
for k in range(len(uy)):
	tmp=[ sum(j) for i,j in enumerate(powerindex) if j[2]+j[3]<2 and abs(uy[k][i])>1e-7]
	if tmp!=[]: print "k=",k," lowest order=", min(tmp) 


#4.Test program for ypowerorder=0
ux,uxbar,Jx,scalexx,Ms,As2x,Asm2x=jfdf.UMsUbarexpJ(Ms10,phix0,1,powerindex,scalemf,ypowerorder=0)


print "\n4. Check ux as left eigenvectors of M: ux.M=exp(i*mux)*exp(Jx).ux:"
print "Jx="
jfdf.pim(Jx,len(Jx),len(Jx))
maxchainlenposition, maxchainlen, chain, chainposition=jfdf.findchainposition(Jx)
print "position of max length chain=",maxchainlenposition
print "max length chain=",maxchainlen
tmp1=np.dot(ux,Ms)
tmp2=np.exp(1j*phix0)*np.dot(jfdf.exp(Jx,maxchainlen),ux)
tmp3=tmp1-tmp2
print "4. check ux.Ms=exp(i*mux)*exp(Jx).ux ", abs(tmp3).max()," relative error:",abs(tmp3).max()/abs(tmp1).max()



print "4. lowest order in ux[i]:\n"
for i in range(len(ux)):
	tmp=[ sum(powerindex[k])  for k,b in enumerate(abs(ux[i])) if b>1e-8]
	if tmp!=[]: print i, min(tmp)

def dmt(k,m, tol=1e-8): #Dominating terms of k'th order in u[m]
	dt=[ [i,j,ux[m][i]] for i,j in enumerate(powerindex) if (j[2]+j[3]<1 and j[0]+j[1]==k and abs(ux[m][i])>tol)]
	return dt 

print "\n4. dominating terms in ux[0], their order, and size:"
cc=[ [k,sum(powerindex[k]),b,powerindex[k].tolist()]  for k,b in enumerate(abs(ux[0])) if b>1e-8 and sum(powerindex[k][2:])<1]
for i in cc:
	print i



print "\n4. dominating terms of order 1 in ux[0] with x power<1:", [ [j[0],j[1].tolist(),j[2]] for j in dmt(1,0)]
print "4. dominating terms of order 2 in ux[0] with x power<2:", [ j[1].tolist() for j in dmt(2,0)]
print "4. dominating terms of order 3 in ux[0] with x power<3:",[ j[1].tolist() for j in dmt(3,0)]

print "4. Showing the lowest of power of x in every eigenvector with y power =0:"
for k in range(len(ux)):
	tmp=[ j[0]+j[1] for i,j in enumerate(powerindex) if j[2]+j[3]<1 and abs(ux[k][i])>1e-7]
	if tmp!=[]: print "k=",k," lowest order=", min(tmp) 

#5. Plot tracking result print "\n5. Plot tracking result in x-xp plane" 
flnm=open('beamsddstail','w')
npart=1
flnm.write('%10i \n'%npart)
xmax=2e-3
ymax=2e-3
npass=8193
xxp0=[]
p_central_mev=3000
x0off=0e-7  #it is found that there is an round off error in elegant that causes residual energy delta non-zero and caused an offset for x0 and xp0
xp0off=0e-10 #which is to be removed here for Jordan form calculation. This offsets are found when we reduce the radius in w plane, the circle in the 
#zbar plane in the following calculation has a center which is shift away from origin.
x0=xmax+x0off
xp0=0+xp0off
y0=ymax
pid=1
#write into input file for elegant x0,xp0 for all particles of different theta0
#flnm.write('%10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10i \n'%(x0,xp0,0,0,0,5.870852e+03,0,pid))
flnm.write('%10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10i \n'%(x0,xp0,y0,0,0,p_central_mev/0.511,0,pid))
flnm.close()
os.system('cat beamsddshead beamsddstail >beam.sdds')
os.system('elegant  -macro=npass='+str(npass)+'   tracking.ele')
ta, xxp=commands.getstatusoutput('sddsprocess ring.w1 -filter=col,Pass,1,'+str(npass)+' -pipe=out|sddsprintout -pipe=in -col="(Cx,Cxp,Cy,Cyp)" -noLabel -noTitle ') #read output of elegant for x,xp after one turn
import StringIO
from StringIO import StringIO
xxp = StringIO(xxp) #retrieve from elegant output file as string, and change the string into a virtual file
xxp=np.loadtxt(xxp) #load the virtual file to convert it into an array of x,xp after 1 turn
xxpyyp=zip(*xxp)

ta, tune=commands.getstatusoutput("sddsexpand ring.w1 -pipe=out|sddscollapse -pipe=input,output|sddsnaff -pipe=in,out -col=Pass,Cx,Cy -term=frequencies=3 |sddsprintout -pipe=in -col") #='(xFrequency,yFrequency)' ")
print "tune x,y are:",tune

#6. Convert x,xp,y,yp to zx,zxs,zy,zys, i.e., to nomalized coordinates
from scipy import loadtxt
import StringIO
from StringIO import StringIO
ta,epu2twis=commands.getstatusoutput('sddsprintout nsls2yongjun.twi -col="(betax,alphax,psix,betay,alphay,psiy)" -noLabel -noTitle|tail -1') #read output of elegant for x,xp after one turn
epu2twis = StringIO(epu2twis) #retrieve from elegant output file as string, and change the string into a virtual file
betaxelegant,alphaxelegant,phixelegant,betayelegant,alphayelegant,phiyelegant=loadtxt(epu2twis) #load the virtual file to convert it into an array of x,xp after 1 turn
sqrtbxelegant=np.sqrt(betaxelegant)
sqrtbyelegant=np.sqrt(betayelegant)
Bielegant=array([[1/sqrtbxelegant,0,0,0],[alphaxelegant/sqrtbxelegant,sqrtbxelegant,0,0], #Bi converts x,xp,y,yp to xbar,xpbar,ybar,ypbar see SY.Lee's book, Floquet trsansform II.3 eq2.43
[0,0,1/sqrtbyelegant,0],[0,0,alphayelegant/sqrtbyelegant,sqrtbyelegant]])
Ki=array([[1,-1j,0,0],[1,1j,0,0],[0,0,1,-1j],[0,0,1,1j]]) #For Ki see Courant-Snyder variables and U transform and relation to Twiss Transform.one 12/16/2009
bKielegant=np.dot(Ki,Bielegant)
xybar=np.dot(Bielegant,xxpyyp)

import cmath

#10. study Poincare section plot using |zy|,phix,phiy

def normalcoordinate(xb,xpb,yb,ypb): #Calculate action w for given x,xp, and the theoretical first order tune shift in w-plane as function of x,xp
	#See note "Relation to driving terms/Jordan Form Reformulation/Amplitude dependent tune", section 8, 2/19/2015
	#For tune with 5 phix0 close to 2 pi, block 1, -4, 6 are used
	zxb=xb-1j*xpb
	ab0x=xb**2+xpb**2
	phib0x=cmath.phase(zxb)
	zyb=yb-1j*ypb
	ab0y=yb**2+ypb**2
	phib0y=cmath.phase(zyb)
   	return ab0x,ab0y,phib0x,  phib0y

section1=[]
xy=zip(*xybar)
for row in xy:	
	section1.append(normalcoordinate(*row))

tmp=zip(*section1)
Jylim=max(tmp[1])*1.1

fig=plt.figure(10) #phiy vz Jyb for various phiy
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section1 if abs(row[3]-4.0+i)<1e-1])
	plt.plot(stnarrow[2],stnarrow[1],'.') #phix vz Jyb
	plt.axis([-4,4,0.5*Jylim,Jylim])
	plt.xlabel('phix')
	plt.ylabel('|Jy|')
	plt.title('phiy='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.10 |Jy| vz. phi_x for different phi_y', fontsize = 18)

fig=plt.figure(11) #phiy vz Jyb for various phiy
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section1 if abs(row[2]-4.0+i)<1e-1])
	plt.plot(stnarrow[3],stnarrow[1],'.') #phix vz Jyb
	plt.axis([-4,4,0.5*Jylim,Jylim])
	plt.xlabel('phiy')
	plt.ylabel('|Jy|')
	plt.title('phix='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.11 |Jy| vz. phi_y for different phi_x', fontsize = 18)

st=zip(*section1)
sv('junk',st)

#11. FFT
import numpy.fft as fft

ybar=xybar[2]
n=len(ybar)
fy=fft.fft(ybar)
ff=np.arange(n)/(1.0*n)
plt.figure(12)
plt.plot(ff,np.real(fy))
plt.plot(ff,np.imag(fy))
plt.xlabel('nuy')
plt.ylabel('ay(f)')
plt.savefig('junk12.png')

xbar=xybar[0]
xpbar=xybar[1]
n=len(xbar)
fx=fft.fft(xbar)
plt.figure(13)
plt.plot(ff,np.real(fx))
plt.plot(ff,np.imag(fx))
plt.xlabel('nux')
plt.ylabel('ax(f)')
plt.savefig('junk13.png')

print "\n x spectrum"
findex=np.argsort(abs(fx))[-4:]
print "peak frequency of xbar is at frequency=",ff[findex]
print " with phase-amplitude=",fx[findex]
print "\n y spectrum"
findex=np.argsort(abs(fy))[-4:]
print "peak frequency of ybar is at frequency=",ff[findex]
print " with phase-amplitude=",fy[findex]

ypbar=xybar[3]
n=len(ypbar)
fyp=fft.fft(ypbar)
print "\n yp spectrum"
findex=np.argsort(abs(fyp))[-4:]
print "peak frequency of ybar is at frequency=",ff[findex]
print " with phase-amplitude=",fyp[findex]

plt.figure(13)
plt.plot(ff,np.real(fy))
plt.plot(ff,np.imag(fy))
plt.xlabel('nuy')
plt.ylabel('ay(f)')
plt.axis([ff[findex][-1]*0.99,ff[findex][-1]*1.01,-max(abs(fy)),max(abs(fy))])

#14. Showing beat in ybar^2+ypbar^2 
plt.figure(14)
plt.plot(st[1])
plt.xlabel('turn number')
plt.ylabel('ybar^2+ypbar^2')


#15. Frequencies in ybar^2+ypbar^2'
plt.figure(15)
jy2=fft.fft(st[1])
plt.plot(ff,np.real(jy2))
plt.plot(ff,np.imag(jy2))
plt.plot(ff,1e-5*np.imag(fy),'r.')
plt.axis([0,1,-4e-6,1e-5])
plt.title('Fig.15 Frequencies in ybar^2+ypbar^2')
plt.savefig('junk15.png')

tmp=np.argsort(abs(jy2))
print "\nmax of amplitude and frequency of Jy^2 are,", jy2[tmp[-3:]], ff[tmp[-3:]]

#16. There is beat also in ybar^2:
plt.figure(16)
y2=ybar*ybar
plt.plot(y2)
plt.xlabel('turn number')
plt.ylabel('ybar^2')
plt.title('Fig.16 ybar^2 vz turn #')

#17. Spectrum of zbar=ybar-1j*ypbar
#Found that  change ybar-1j*ypbar to ybar-2j*ypbar creates a 1-Qy tune, there is no 3Qy.
plt.figure(17)
zybar=ybar-1j*ypbar
fzybar=fft.fft(zybar)
plt.plot(ff,np.real(fzybar))
plt.plot(ff,np.imag(fzybar))
#plt.plot(ff,1e-5*np.imag(fzybar),'r.')
#plt.axis([0,1,-4e-6,1e-5])
plt.title('Fig.17 Frequencies in zybar')
plt.savefig('junk17.png')
print "\n zybar spectrum"
findex=np.argsort(abs(fzybar))[-4:]
print "peak frequency of zybar is at frequency=",ff[findex]
print " with phase-amplitude=",fzybar[findex]
tmp1=array([ [i,a]    for i,a in enumerate(fzybar) if 0.1 <ff[i]<0.2]) #Find another peak sufficiently far away from the first peak so it is not the shoulder of the first peak.
tmp2=zip(*tmp1)
findex=np.argsort(abs(array(tmp2[1])))[-20:]
tmp3=array(map(int,tmp2[0]))
tmp4=tmp3[findex]
print tmp4
print ff[tmp4]
print array(tmp2[1])[findex]
print '2nd peak in fzybar="',array(tmp2[1])[findex][-1], " at tune=",ff[tmp4[-1]], " position in fzybar=", tmp4[-1]

#18. Plot trajectory in zbar plane
r=np.sqrt(ybar*ybar+ypbar*ypbar)
phi=np.arange(0,2*np.pi, 2*np.pi/30)
rmax=max(r)
cmax=array([ array([rmax*np.cos(row),rmax*np.sin(row)]) for row in phi])
cmax=zip(*cmax)
plt.figure(18)
plt.plot(cmax[0],cmax[1],'bo')
rmin=min(r)
cmin=array([ array([rmin*np.cos(row),rmin*np.sin(row)]) for row in phi])
cmin=zip(*cmin)
plt.plot(cmin[0],cmin[1],'go')
plt.axes().set_aspect('equal')
tmp1=array(cmax[0])*array(cmax[0])+array(cmax[1])*array(cmax[1])
tmp2=array(cmin[0])*array(cmin[0])+array(cmin[1])*array(cmin[1])
plt.plot(ybar,ypbar,'r.')
plt.savefig('junk18.png')

#19. Spectrum of zxbar=xbar-1j*xpbar
#Found that  change ybar-1j*ypbar to ybar-2j*ypbar creates a 1-Qy tune, there is no 3Qy.
plt.figure(19)
zxbar=xbar-1j*xpbar
fzxbar=fft.fft(zxbar)
plt.plot(ff,np.real(fzxbar))
plt.plot(ff,np.imag(fzxbar))
plt.xlabel('nux')
plt.ylabel('fzxbar')
plt.title('Fig.19 Frequencies in zxbar')
plt.savefig('junk19.png')
print "\n zxbar spectrum"
findex=np.argsort(abs(fzxbar))[-4:]
print "peak frequency of zxbar is at frequency=",ff[findex]
print " with phase-amplitude=",fzxbar[findex]

#20. Find zxbar spectrum's 2nd peak by looking for delta fzxbar
dfzxbar=array([  (abs(fzxbar[i+1])-abs(fzxbar[i]))/abs(fzxbar[i]) for i in range(len(fzxbar)-1)])
idfzxbar=np.argsort(dfzxbar)
print "peak tune:", ff[idfzxbar[-10:]], " index=", idfzxbar[-10:]
print "amplitude:", abs(fzxbar[idfzxbar[-10:]])
tmp1=np.argsort(abs(fzxbar)[idfzxbar[-10:]])
peaktuneindex=idfzxbar[-10:][tmp1]
peaktuneindex2=[ i-3+np.argmax(abs(fzxbar[i-3:i+3])) for i in peaktuneindex]
print "peak tune:", ff[peaktuneindex2]
print "peaks amplitude:", abs(fzxbar)[peaktuneindex2]

