# b = paper.cirDist()[0]
# a = paper.cirDist()[0]
import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20

ii = 1
X1 = np.reshape(b[2],(100,100))[::ii]
Y1 = np.reshape(b[3],(100,100))[::ii]
WY1 = np.reshape(b[1],(100,100))[::ii]

X = np.reshape(a[2],(100,100))[::ii]
Y = np.reshape(a[3],(100,100))[::ii]
WY = np.reshape(a[1],(100,100))[::ii]

for i in xrange(100/ii):
    idx = np.argsort(X[i])
    X[i] = X[i][idx]
    Y[i] = Y[i][idx] 
    WY[i] = WY[i][idx] 

for i in xrange(100/ii):
    idx = np.argsort(Y[:,i])
    X[:,i] = X[:,i][idx]
    Y[:,i] = Y[:,i][idx] 
    WY[:,i] = WY[:,i][idx] 

for i in xrange(100/ii):
    idx = np.argsort(X1[i])
    X1[i] = X1[i][idx]
    Y1[i] = Y1[i][idx] 
    WY1[i] = WY1[i][idx] 

for i in xrange(100/ii):
    idx = np.argsort(Y1[:,i])
    X1[:,i] = X1[:,i][idx]
    Y1[:,i] = Y1[:,i][idx] 
    WY1[:,i] = WY1[:,i][idx] 

fig = plt.figure(figsize=(6,12))
ax = fig.add_subplot(211, projection='3d')
#ax.scatter(X,Y,WY/np.average(WY),marker='.',c='r')
ax.plot_wireframe(X, Y, WY/np.average(WY),color='r',rstride=5, cstride=5)
ax.set_zlim3d(0.6, 1.5)
ax.set_xticks([-3.14,0,3.14])
ax.set_yticks([-3.14,0,3.14])
ax.set_xlabel(r'$\theta_x (rad)$',fontsize=18)
ax.set_ylabel(r'$\theta_y (rad)$',fontsize=18)
ax.text(-2,3,1.4,r'$r_{y0}/\bar{r}_{y0}$',fontsize=25)
#ax.set_zlabel(r'$\frac{r_{y0}}{\bar{r_{y0}}}$',fontsize=18)

ax = fig.add_subplot(212, projection='3d')
#ax.scatter(X1,Y1,WY1/np.average(WY1),marker='.',c='b')
ax.plot_wireframe(X1, Y1, WY1/np.average(WY1),color='b',rstride=5, cstride=5)
ax.set_zlim3d(0.6, 1.5)
ax.set_xticks([-3,-1.5,0,1.5,3])
ax.set_yticks([-3,-1.5,0,1.5,3])
ax.set_xlabel(r'$\theta_x (rad)$',fontsize=18)
ax.set_ylabel(r'$\theta_y (rad)$',fontsize=18)
#ax.set_zlabel(r'$\frac{r_{y0}}{\bar{r_{y0}}}$',fontsize=18)
ax.text(-2,3,1.4,r'$r_{y0}/\bar{r}_{y0}$',fontsize=25)
#plt.savefig('flat_plane.eps')
#plt.savefig('flat_plane.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X1, Y1, WY1/np.average(WY1)-0.5,color='r',rstride=5, cstride=5,label='solution A')
ax.plot_wireframe(X, Y, WY/np.average(WY)+0.5,color='b',rstride=5, cstride=5,label='solution B')
#ax.set_zlim3d(0.6, 1.5)
#ax.set_xticks([-3,-1.5,0,1.5,3])
#ax.set_yticks([-3,-1.5,0,1.5,3])
plt.xticks([-3.14,3.14],[r'$-\pi$',r'$\pi$'])
plt.yticks([-3.14,3.14],[r'$-\pi$',r'$\pi$'])
ax.set_zticks([])
ax.set_xlabel(r'$\theta_x (rad)$',fontsize=18)
ax.set_ylabel(r'$\theta_y (rad)$',fontsize=18)
#ax.set_zlabel(r'$\frac{r_{y0}}{\bar{r_{y0}}}$',fontsize=18)
ax.text(-2,3,2.0,r'$r_{y}/\bar{r}_{y}$',fontsize=25)
plt.legend()
plt.savefig('flat_plane.eps')
plt.savefig('flat_plane.png')
plt.show()
