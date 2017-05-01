# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import tesla
import matplotlib.pylab as plt
import numpy as np
import os, sys
import commands
from numpy import array
from numpy import linalg
import time
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle


global Vm,U,maxchainlenposition,bKi,norder,powerindex

def sv(filename,x):
	ff=open(filename,'w')
	pickle.dump(x,ff)
	ff.close()

def rl(filename):
	ff=open(filename)
	xx=pickle.load(ff)
	ff.close()
	return xx

t0=time.clock()
import jnfdefinition
jfdf=jnfdefinition
import squarematrixdefinition
sqdf=squarematrixdefinition
print time.clock()-t0 ,"seconds for import sqdf and jfdf"
t0=time.clock()

#1.Extract linear transposrt matrix from elegant to be used to replace linear part of the tpsa becasue it has small difference .
elematrix="nsls2yongjun.sdds"
import StringIO
from StringIO import StringIO
import scipy
from scipy import loadtxt
ta, elemtx=commands.getstatusoutput('sddsprintout nsls2yongjun.sdds -col="(R11,R12,R13,R14,R21,R22,R23,R24,R31,R32,R33,R34,R41,R42,R43,R44)"|tail -2 ') #read output of elegant for x,xp after one turn
elemtx = StringIO(elemtx) #retrieve from elegant output file as string, and change the string into a virtual file
elemtx = loadtxt(elemtx) #load the virtual file to convert it into an array of x,xp after 1 turn
elemtx=elemtx.reshape(4,4)

ta,elems=commands.getstatusoutput('sddsprocess nsls2yongjun.sdds -match=col,ElementType=KSEXT* -filter=col,ElementOccurence,1,1 -pipe=out|\
sddsprintout -pipe=in -col="(R11,R12,R13,R14,R21,R22,R23,R24,R31,R32,R33,R34,R41,R42,R43,R44)" -nolabel -noTitle')
elems = StringIO(elems) #retrieve from elegant output file as string, and change the string into a virtual file
elems = loadtxt(elems) #load the virtual file to convert it into an array of x,xp after 1 turn
elems = elems.reshape(9,4,4)

#2. Extract m from tpsa as one turn map, ms is the maps from origin to all sextupoles
#ring = tesla.Ring("fodo_02.tslat", "RING")

lattice="nsls2sr_supercell_ch77_20150406.tslat"
#lattice="junk.tslat"
ring=tesla.Ring(lattice, "RING")
norder=9
tol=1e-12
# six phase space varialbe, 4 independent, expand to n'rd order.
m = tesla.TPSMap(6,4,norder)
m.c=[0, 0, 0, 0, 0, 0]
m.m = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0],[0,0,0,0]]
ms=ring.trackTPSMaps(m, 0, ring.elements()) #ms has the maps of all elements around the ring, m is one turn map
tw, ml = tesla.calcLinearTwiss(ring)

#convert m into a list vss, so that every row of vss is same as a row you get when you print m, to be used to check results
vss=[]
for i in range(6):
	pv, vs, od = m.dump(i)
	vss.append(vs)

for i in range(5,len(vss[0])):
    vss[4].append('ns')

vss=zip(*vss)


#import sys
#sys.exit(0)

#ml = tesla.linearMatrices(ring)
#tw = tesla.computeTwiss(ml)
#print ms
#print m
#3.collect element names, twiss parameters such as betax vs. s for all sextupoles into a table
sname=[ ring.elementName(i) for i in range(ring.elements()) if ring.elementType(i)=='SEXTUPOLE']
stype=[ ring.elementType(i) for i in range(ring.elements()) if ring.elementType(i)=='SEXTUPOLE']
#tw1=[ [i,{'alphax': tw['alphax'][i+1], 'alphay': tw['alphay'][i+1], 'gammay': tw['gammay'][i+1], 'gammax': tw['gammax'][i+1], 's': tw['s'][i+1], 'phiy': tw['psiy'][i+1], 'phix': tw['psix'][i+1], 'betay': tw['betay'][i+1], 'betax': tw['betax'][i+1]}]  for i in range(ring.elements()) if ring.elementType(i)=='SEXTUPOLE' ]
sextupoles =[ i for i in range(ring.elements()) if ring.elementType(i)=='SEXTUPOLE'] #sequence of all sextupoles in the ring
twi=[ dict({'alphax': tw['alphax'][i+1], 'alphay': tw['alphay'][i+1], 'gammay': tw['gammay'][i+1], 'gammax': tw['gammax'][i+1], 's': tw['s'][i+1], 'phiy': tw['psiy'][i+1], 'phix': tw['psix'][i+1], 'betay': tw['betay'][i+1], 'betax': tw['betax'][i+1]}) for i in sextupoles]
#tbl=zip(*[tmp, tw2])

#Use regular expression to read sextupole data from the file "nsls2_cell_id_test.tslat"
import StringIO
from StringIO import StringIO
import scipy
from scipy import loadtxt
import commands
import re



tmp, ss=commands.getstatusoutput('grep Sextupole '+lattice)
ss = StringIO(ss) #retrieve from elegant output file as string, and change the string into a virtual file
ss=ss.read() #read strings defining setupole strengths and length
ss=ss.split("\n") # change newlines into commas. So between any two commas is a string defining one sextupole.
ss=[ i.split(":") for i in ss] #split each sentence into two string using ":" as separator, thus between any two commas are a pair of string.
ss=dict(ss)  #change the pairs of strings into dictionary




st=[]
sw=[]
for i in sname:
	tmpt = re.match(r".*K2 *=([^,]+);+", ss[i]) #use regular expression to read each line given by dictionary defining a sextupole and recognize the sextupole strength as a string
	tmpw = re.match(r".*L *=([^,]+),.+", ss[i]) #use regular expression to read each line and recognize the sextupole strength as a string
	print i, tmpt.group(1)
	st.append(float(tmpt.group(1))) #convert the string into float number for sextupole strength
	sw.append(float(tmpw.group(1))) #convert the string into float number for sextupole width

tbl=[ [sname[i],stype[i],twi[i],st[i],sw[i]] for i in range(len(st))] #pack all data about sextupole into a table used same as in aps1map.py


#sys.exit(0)



print time.clock()-t0 ,"seconds for getting m from tpsa"
t0=time.clock()

#4. Construct map matrix mm
mf,powerindex=sqdf.mfunction(m,norder) # m is one turn map
#This program only applies when x[0]=0,xp[0]=0,y[0]=0, yp[0]=0. If not, then the map matrix mm is no longer semi-triangular
mf[0][0]=0
mf[1][0]=0
mf[2][0]=0
mf[3][0]=0
mf=array(mf)



mlen=len(powerindex)

#sequencenumber[i1,i2,i3,i4] gives the sequence number in power index for power of x^i1*xp^i2*y^i3*yp^i4
sequencenumber=np.zeros((norder+1,norder+1,norder+1,norder+1),'i')
powerindex=sqdf.powerindex4(norder)
powerindex=array(powerindex,'i')
mlen=len(powerindex)

for i in range(mlen):
	ip=powerindex[i]
	sequencenumber[ip[0]][ip[1]][ip[2]][ip[3]]=i


#Extract twiss parameters from tpsa linear part, then use these twiss parameters to reconstruct linear part of the matrix to be accurately simplectic

betax0,phix0,alphax0,betay0,phiy0,alphay0=sqdf.extracttwiss(mf)
gammax0=(1+alphax0**2)/betax0
mlix=[np.cos(phix0)+alphax0*np.sin(phix0), betax0*np.sin(phix0)],[-gammax0*np.sin(phix0), np.cos(phix0)-alphax0*np.sin(phix0)]
gammay0=(1+alphay0**2)/betay0
mliy=[np.cos(phiy0)+alphay0*np.sin(phiy0), betay0*np.sin(phiy0)],[-gammay0*np.sin(phiy0), np.cos(phiy0)-alphay0*np.sin(phiy0)]

print "det(mlix)=", np.linalg.det(mlix) #Showing the new linear part has determinant=1
print "elemtx[:2,:2]=", np.linalg.det(elemtx[:2,:2])
print "det(mliy)=", np.linalg.det(mliy)
print "elemtx[2:4,2:4]=", np.linalg.det(elemtx[2:4,2:4])
elemtx[:2,:2]=mlix #Replace linear part of elegant, which is accurate only to 8 digits, by the twiss matrix obtained from the twiss parameters extract from the tpsa linear part of matrix
elemtx[2:4,2:4]=mliy #Notice that mlix and mliy is accurate simplex to machine precision, so their determinantes are closer to zero.

mf[:,1:5]=elemtx #Now replace the linear part of map matrix by the more accurate twiss matrix

sqrtbetax=np.sqrt(betax0)
sqrtbetay=np.sqrt(betay0)
print time.clock()-t0 ,"seconds for map matrix"
t0=time.clock()


#5. Construct the BK square matrix using the first 5 rows.
bK,bKi=sqdf.BKmatrix(betax0,phix0,alphax0,betay0,phiy0,alphay0, 0,0,norder,powerindex,sequencenumber,tol)
print time.clock()-t0 ,"seconds for bK,bKi"
t0=time.clock()




#6. Derive normalized map M=(BK)^(-1).mm.BK, see my notes 'Relation to Normal Form' of Wednesday, March 30, 2011 10:34 PM
#   here #mfbk is the first 4 rows of M, it is 
mfbk=jfdf.d3(bKi[1:5,1:5],mf,bK) 



#7. Scale the one turn map in z,z* space 
def scalingmf(mf,powerindex):
        #Find a scale s so that s x**m and s is on the same scale if the term in M with maximum absolute value has power m
	#as described in "M scaling" in "Jordan Form Reformulation.one". mf is the first 4 rows of M so the scaling method is the same.
	absM=abs(mf)
	i,j=np.unravel_index(absM.argmax(), absM.shape)
	power=sum(powerindex[j])
	scalex1=(absM.max())**(-(1./(power-1.)))
	scalem1=1/scalex1
	mlen=len(powerindex)
#	mflen=len(mf)
	print "scalemf=", scalex1
	As=np.identity(mlen)
	for i in range(mlen):
		As[i,i]=scalem1**sum(powerindex[i])
	Asm=np.identity(mlen)
	for i in range(mlen):
		Asm[i,i]=scalex1**sum(powerindex[i])
        mfs=scalem1*np.dot(mf,Asm)
        return array(mfs),scalex1, As,Asm

mftmp1=array(mfbk).copy()
mfbk,scalemf,As,Asm=scalingmf(mfbk,powerindex)


#Ms =As.M.Asm is the scaled M 
Ms=sqdf.squarematrix(mfbk,norder,powerindex,sequencenumber,tol) 

#M=jfdf.d3(Asm,Ms,As)


print time.clock()-t0 ,"seconds for bKi.mm.bK"
t0=time.clock()

#7.Extract maps for sextupoles around the ring
msf=[]
for i in sextupoles:
	mftmp,powerindex=sqdf.mfunction(ms[i],norder)
	msf.append(mftmp)

#Replace the linear part of tpsa by linear part of elegant to be self consistent.
#msf1=[]
#for ims,msfi in enumerate(msf):
#	msfi=array(msfi)
#	msfi[:,1:5]=elems[ims]
#	msf1.append(msfi)


#8.Found that only msf[0], i.e., the first sextupole, has no nonlinear terms, 
#so we have to append zeroes to make it in the same form as other sextupoles
tmp1=np.zeros([4,len(msf[1][0])-5])
msf[0]=np.hstack([msf[0],tmp1])

msf=array(msf)
#print "test changes due to linear part replacement at 267'th sextupole:"
#jfdf.prm(msf[269][:,1:7],4,6)
#jfdf.prm(msf1[269][:,1:7],4,6)
print "test deteminantes of linear part of square matrix Ms:"
print "det(Ms[1:3,1:3])=", np.linalg.det(Ms[1:3,1:3])
print "det(Ms[3:5,3:5])=", np.linalg.det(Ms[3:5,3:5])
#9. Save map matrix Ms to be used by aps2jnf.py
for j in range(len(Ms)):
	for i in range(0,len(Ms)):
		if j<i: Ms[i,j]=0

sv("jfM.dat",[Ms,phix0,phiy0,powerindex,norder,bK,bKi,sqrtbetax,sqrtbetay,msf,tbl,scalemf])


mn=Ms
#Check if mn is indeed an upper triagular matrix
tmp=[ [abs(mn[i][j]) for j in range(0,len(mn)) if j<i] for i in range(0,len(mn))]
tmp1=[ item for sublist in tmp for item in sublist]
print "maximum of lower triangle is:", max(tmp1)
tmp=[ [abs(mn[i][j]) for j in range(0,len(mn)) if j>i] for i in range(0,len(mn))]
tmp1=[ item for sublist in tmp for item in sublist]
print "maximum of upper triangle is:", max(tmp1)
tmp=[ [abs(mn[i][j]) for j in range(0,len(mn)) if j==i] for i in range(0,len(mn))]
tmp1=[ item for sublist in tmp for item in sublist]
print "maximum of diagonal is:", max(tmp1)



