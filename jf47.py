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


def checkjnf(u,J,scale,nx,phix0,ny,phiy0,Ms,info,tol1,tol2):
	print "\n",info['sectionname'],". Check u as left eigenvectors of M: U.M=exp(i*mu)*exp(J).U:"
	print info['Jname'],"="
	jfdf.pim(J,len(J),len(J))
	maxchainlenposition, maxchainlen, chain, chainposition=jfdf.findchainposition(J)
	print "position of max length chain=",maxchainlenposition
	print "max length chain=",maxchainlen
	tmp1=np.dot(u,Ms)
	tmp2=np.exp(1j*(nx*phix0+ny*phiy0))*np.dot(jfdf.exp(J,maxchainlen),u)
	tmp3=tmp1-tmp2
	print info['sectionname'],". check ",info['uname'],".",info['Msname'],"=exp(i*",info['muname'],")*exp(",info['Jname'],").",info['uname'], ",", abs(tmp3).max()," relative error:",abs(tmp3).max()/abs(tmp1).max()


	print info['sectionname'],". lowest order in ",info['uname'],"[i]:\n"
	for i in range(len(u)):
		tmp=[ sum(powerindex[k])  for k,b in enumerate(abs(u[i])) if b>1e-8]
		if tmp!=[]: print i, min(tmp)

	def dmt(k,m, tol=tol2): #Dominating terms of k'th order in uy[m]
		dt=[ [i,j,abs(u[m][i])] for i,j in enumerate(powerindex) if (sum(j)==k and abs(u[m][i])>tol2)]
		return dt 

	print "\n",info['sectionname'], ". dominating terms in ",info['uname'],"[0], their order, and size:"
	cc=[ [k,sum(powerindex[k]),b,powerindex[k].tolist()]  for k,b in enumerate(abs(u[0])) if b>tol2]
	cci=np.argsort([ i[2] for i in cc])
	ccs=[ cc[i] for i in cci]
	print "\n", info["sectionname"],". 20 lowest order terms:"
	for i in cc[:20]:
		print i

	print "\n", info["sectionname"],". 20 dominating terms:"
	for i in ccs[-20:]:
		print i


	print "\n",info['sectionname'],". dominating terms of order 1 in ",info['uname'],"[0] :", [ [j[0],j[1].tolist(),j[2]] for j in dmt(1,0)]
	print info['sectionname'],". dominating terms of order 2 in ",info['uname'],"[0]:", [ [j[1].tolist(),j[2]] for j in dmt(2,0)]
	print info['sectionname'],". dominating terms of order 3 in ",info['uname'],"[0]:",[ [j[1].tolist(),j[2]] for j in dmt(3,0)]

	print info['sectionname'],". Showing the lowest of power of x,y in every eigenvector :"
	for k in range(len(u)):
		tmp=[ sum(j) for i,j in enumerate(powerindex) if  abs(u[k][i])>tol2]
		if tmp!=[]: print "k=",k," lowest order=", min(tmp)
	return




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

norder=7
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


#3.Check Jordan block phiy0
uy,uybar,Jy,scaley,Ms,As2,Asm2=jfdf.UMsUbarexpJ(Ms10,phiy0,1,powerindex,scalemf,sequencenumber[0,0,1,0],ypowerorder=7)
checkjnf(uy,Jy,scaley,0,phix0,1,phiy0,Ms,{'sectionname':'3','Jname':'Jy','uname':'uy','Msname':'Msy','muname':'muy'},1e-8,1e-5)

#4. Check block phix0

print "\n6. Check ux as left eigenvectors of M: ux.M=exp(i*mux)*exp(Jx).ux:"
ux,uxbar,Jx,scalex,Ms,As2x,Asm2x=jfdf.UMsUbarexpJ(Ms10,phix0,1,powerindex,scalemf,sequencenumber[1,0,0,0],ypowerorder=7)
checkjnf(ux,Jx,scalex,1,phix0,0,phiy0,Ms,{'sectionname':'4','Jname':'Jx','uname':'ux','Msname':'Ms','muname':'mux'},1e-8,1e-5)

#5. Plot tracking result 
print "\n5. Tracking x,y motion"
flnm=open('beamsddstail','w')
npart=1
flnm.write('%10i \n'%npart)
xmax=15e-3
ymax=3e-3
npass=513 #8193
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
commands.getstatusoutput('sddsprocess ring.w1 -filter=col,Pass,1,'+str(npass)+' -pipe=out|sddsprintout -pipe=in -col="(Cx,Cxp,Cy,Cyp)" -noLabel -noTitle > xxpyyp.txt')
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
ta,nsls2twis=commands.getstatusoutput('sddsprintout nsls2yongjun.twi -col="(betax,alphax,psix,betay,alphay,psiy)" -noLabel -noTitle|tail -1') #read output of elegant for x,xp after one turn
nsls2twis = StringIO(nsls2twis) #retrieve from elegant output file as string, and change the string into a virtual file
betaxelegant,alphaxelegant,phixelegant,betayelegant,alphayelegant,phiyelegant=loadtxt(nsls2twis) #load the virtual file to convert it into an array of x,xp after 1 turn
sqrtbxelegant=np.sqrt(betaxelegant)
sqrtbyelegant=np.sqrt(betayelegant)
Bielegant=array([[1/sqrtbxelegant,0,0,0],[alphaxelegant/sqrtbxelegant,sqrtbxelegant,0,0], #Bi converts x,xp,y,yp to xbar,xpbar,ybar,ypbar see SY.Lee's book, Floquet trsansform II.3 eq2.43
[0,0,1/sqrtbyelegant,0],[0,0,alphayelegant/sqrtbyelegant,sqrtbyelegant]])
Ki=array([[1,-1j,0,0],[1,1j,0,0],[0,0,1,-1j],[0,0,1,1j]]) #For Ki see Courant-Snyder variables and U transform and relation to Twiss Transform.one 12/16/2009
bKielegant=np.dot(Ki,Bielegant)
xybar=np.dot(Bielegant,xxpyyp)

import cmath


#9. study Poincare section plot using |zy|,phix,phiy
print "\n9. study Poincare section plot using |zy|,phix,phiy"

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

section=[]
xy=zip(*xybar)
for row in xy:
	section.append(normalcoordinate(*row))

tmp=zip(*section)
Jylim=max(tmp[1])*1.1

fig=plt.figure(91) #phiy vz Jyb for various phiy
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section if abs(row[3]-4.0+i)<1e-1])
	plt.plot(stnarrow[2],stnarrow[1],'.') #phix vz Jyb
	plt.axis([-4,4,0,Jylim])
	plt.xlabel('phix')
	plt.ylabel('|Jy|')
	plt.title('phiy='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.10 |Jy| vz. phi_x for different phi_y', fontsize = 18)

fig=plt.figure(92) #phiy vz Jyb for various phiy
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section if abs(row[2]-4.0+i)<1e-1])
	plt.plot(stnarrow[3],stnarrow[1],'.') #phix vz Jyb
	plt.axis([-4,4,0.,Jylim])
	plt.xlabel('phiy')
	plt.ylabel('|Jy|')
	plt.title('phix='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.11 |Jy| vz. phi_y for different phi_x', fontsize = 18)

st=zip(*section)

zmax=max(st[1])
zmin=min(st[1])
zav=np.mean(st[1])
print " for Jy,  (zmax-zmin)/zmean=",(zmax-zmin)/zav

Jxlim=max(tmp[0])*1.1

fig=plt.figure(93) #phiy vz Jxb for various phiy
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section if abs(row[3]-4.0+i)<1e-1])
	plt.plot(stnarrow[2],stnarrow[0],'.') #phix vz Jyb
	plt.axis([-4,4,0,Jxlim])
	plt.xlabel('phix')
	plt.ylabel('|Jx|')
	plt.title('phiy='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.93 |Jx| vz. phi_x for different phi_y', fontsize = 18)

fig=plt.figure(94) #phiy vz Jyb for various phiy
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section if abs(row[2]-4.0+i)<1e-1])
	plt.plot(stnarrow[3],stnarrow[0],'.') #phix vz Jyb
	plt.axis([-4,4,0.,Jxlim])
	plt.xlabel('phiy')
	plt.ylabel('|Jx|')
	plt.title('phix='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.94 |Jx| vz. phi_y for different phi_x', fontsize = 18)

st=zip(*section)

zmax=max(st[0])
zmin=min(st[0])
zav=np.mean(st[0])
print " for Jx,  (zmax-zmin)/zmean=",(zmax-zmin)/zav

#10. Study the change of wy after one turn for a fixed x,xp and fixed initial wy.
def tuneshift(x,xp,y,yp,ux,uy): #Calculate action w for given x,xp, and the theoretical first order tune shift in w-plane as function of x,xp
	#See note "Relation to driving terms/Jordan Form Reformulation/Amplitude dependent tune", section 8, 2/19/2015
	#For tune with 5 phix0 close to 2 pi, block 1, -4, 6 are used
	zxbar,zxbars,zybar,zybars=np.dot(bKielegant, array([x,xp,y,yp]))
	zxsbar=zxbar/scalex
	zxsbars=zxbars/scalex
	zysbar=zybar/scalex
	zysbars=zybars/scalex
	Zxs=sqdf.Zcol(zxsbar,zxsbars,zysbar,zysbars,norder,powerindex) #Zxs is the zsbar,zsbars, column, here zsbar=zbar/scalex
	zxsbar=zxbar/scaley
	zxsbars=zxbars/scaley
	zysbar=zybar/scaley
	zysbars=zybars/scaley
	Zys=sqdf.Zcol(zxsbar,zxsbars,zysbar,zysbars,norder,powerindex) #Zys is the zsbar,zsbars, column, here zsbar=zbar/scaley
	wx=np.dot(ux,Zxs) #Zxs is used for wx while Zys is used for wy, separately!
	b0x=wx[0]
	dmu00x=-1j*(wx[1])/b0x
	ab0x=abs(b0x)
	phib0x=cmath.phase(b0x)
	wy=np.dot(uy,Zys)
	b0y=wy[0]
	dmu00y=-1j*(wy[1])/b0y
	ab0y=abs(b0y)
	phib0y=cmath.phase(b0y)
   	return ab0x,ab0y, phib0x, phib0y
#   	return np.real(b0x), np.imag(b0x)

#11. Prepare data for |wx|,thetax,|wy|,thetay to plot Poincare section of thetax,|wy|,thetay

section1=[] #section1 is without joined blocks
xy=zip(*xxpyyp)
for row in xy:
	row1=row+(ux,uy)
	section1.append(tuneshift(*row1))

st1=zip(*section1)

zmax=max(st1[1])
zmin=min(st1[1])
zav=np.mean(st1[1])
print " for wy1 without resonance block,  (zmax-zmin)/zmean=",(zmax-zmin)/zav

sv('junk',[st,st1])

#12. Poincare section for wy,thetax and thetay
fig=plt.figure(121) #thetax vz |wy| for various thetay
zmax=max(st1[1])
zmin=min(st1[1])
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section1 if abs(row[3]-4.0+i)<5e-1])
	plt.plot(stnarrow[2],stnarrow[1],'.') #phix vz Jyb
	plt.axis([-4,4,0,1.1*zmax])
	plt.xlabel('thetax')
	plt.ylabel('|wy|')
	plt.title('thetay='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.121 |wy| vz. theta_x for different theta_y', fontsize = 18)

fig=plt.figure(122) #thetay vz |wy| for various thetax
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section1 if abs(row[2]-4.0+i)<5e-1])
	plt.plot(stnarrow[3],stnarrow[1],'.') #phix vz Jyb
	plt.axis([-4,4,0,1.1*zmax])
	plt.xlabel('thetay')
	plt.ylabel('|wy|')
	plt.title('thetax='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.122 |wy| vz. theta_y for different theta_x', fontsize = 18)

#13. Poincare section for wx,thetax and thetay
fig=plt.figure(131) #thetax vz |wy| for various thetay
zmax=max(st1[0])
zmin=min(st1[0])
zav=np.mean(st1[0])
print " for wx1 without resonance block,  (zmax-zmin)/zmean=",(zmax-zmin)/zav
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section1 if abs(row[3]-4.0+i)<5e-1])
	plt.plot(stnarrow[2],stnarrow[0],'.') #thetax vz |wx|
	plt.axis([-4,4,0,1.1*zmax])
	plt.xlabel('thetax')
	plt.ylabel('|wx|')
	plt.title('thetay='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.131 |wx| vz. theta_x for different theta_y', fontsize = 18)
plt.savefig('junk131.png')

fig=plt.figure(132) #thetay vz |wy| for various thetax
for i in range(1,8):
	plt.subplot(2,4, i)
	stnarrow=zip(*[ row for row in section1 if abs(row[2]-4.0+i)<5e-1])
	plt.plot(stnarrow[3],stnarrow[0],'.') #phix vz Jyb
	plt.axis([-4,4,0,1.1*zmax])
	plt.xlabel('thetay')
	plt.ylabel('|wx|')
	plt.title('thetax='+str(-4+i))

txt = fig.text(0.452, 0.95, 'Fig.132 |wx| vz. theta_y for different theta_x', fontsize = 18)
plt.savefig('junk132.png')


#22. Find resonances 
qx=phix0/2/np.pi
qy=phiy0/2/np.pi

import math
nux,ix=math.modf(qx)
nuy,iy=math.modf(qy)
tbl=[]
resorder=8
for nx in range(-resorder,resorder+1):
	for ny in range(-resorder,resorder+1):
		fp,ip=math.modf(nx*nux+ny*nuy)
		tbl.append([nx,ny,fp,ip])

tbl=array(tbl)
ttbl=np.transpose(tbl)
idx=np.argsort(abs(ttbl)[2])
tbl1=tbl[idx]
print "first few resonances:",tbl1[:7]
sys.exit(0)

