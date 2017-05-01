# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
#This file has to be downloaded to local computer to be able use scatter3D plot, togehter with the data used in the file.

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

import pickle
import scipy
from scipy import optimize


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
st=rl('junk')
st1=rl('junk1')
st2=rl('junk2')
#plt.figure(1)
#plt.plot(st[3],st[1],'.') #phiy vz Jy
#plt.axes().set_aspect('equal')
#plt.figure(2)
#plt.plot(st[2],st[1],'.')#phix vz. Jx

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

zmax=max(st[1])
zmin=min(st[1])
print " for Jy,  (zmax-zmin)/zmin=",(zmax-zmin)/zmin

ax.scatter(st[2], st[3], st[1], c='r', marker='o')

ax.set_xlabel('phix')
ax.set_ylabel('phiy')
ax.set_zlabel('Jy')
ax.set_xlim3d(-4,4)
ax.set_ylim3d(-4,4)
ax.set_zlim3d(zmin*0.,zmax*1.1)


fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

zmax=max(st1[1])
zmin=min(st1[1])
print " for wy1,  (zmax-zmin)/zmin=",(zmax-zmin)/zmin

ax.scatter(st1[2], st1[3], st1[1], c='r', marker='o')

ax.set_xlabel('thetax')
ax.set_ylabel('thetay')
ax.set_zlabel('wy1')
ax.set_xlim3d(-4,4)
ax.set_ylim3d(-4,4)
ax.set_zlim3d(zmin*0.,zmax*1.1)

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

zmax=max(st2[1])
zmin=min(st2[1])
print " for wy2, (zmax-zmin)/zmin=",(zmax-zmin)/zmin

ax.scatter(st2[2], st2[3], st2[1], c='r', marker='o')

ax.set_xlabel('thetax')
ax.set_ylabel('thetay')
ax.set_zlabel('wy2')
ax.set_xlim3d(-4,4)
ax.set_ylim3d(-4,4)
ax.set_zlim3d(zmin*0.,zmax*1.1)

plt.show()
