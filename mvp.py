import numpy as np
import math
import copy
import itertools

class mvp(object):
    '''
    multi-variables polynomial class
    use:   mvp(index, value)
    index: variable index matrix
    value: coefficient vector
    '''

    def __init__(self, index0, value0):
        '''
        initial mvp class with index matrix (2d) and coefficient vector (1d)
        '''
        index0,value0 = np.array(index0,dtype=int),np.array(value0,dtype=float)
        if index0.shape[0] != len(value0):
            raise RuntimeError('unmatched indices and values length')
        self.index = index0
        self.value = value0


    def __repr__(self):
        '''
        print polynomial on screen
        '''
        s = '\n'
        formt = self.index.shape[1]*'%2d ' + '%15.6g\n'
        for i in range(self.index.shape[0]):
            temp = np.append(self.index[i,:], self.value[i])
            s += formt % tuple(temp)
        s += '\n'
        #s += '----------\n# of terms: %d\n'%(self.index.shape[0])
        return s

    
    def simplify(self):
        '''
        Combine like terms in the polynomial to simplify
        use: a.simplify(),  where a is a mvp
        notice: no return value
        '''
        # combine like terms
        m,n = self.index.shape
        flag = np.ones(m,dtype=int)
        for i in range(m)[-1:0:-1]:
            if not any(flag[:i]):
                break
            if flag[i] == 0:
                continue
            d = np.sum(self.index[i,:]==self.index[0:i,:],axis=1)
            d, = np.nonzero(d==n)
            if len(d) >= 1:
                self.value[i] += sum(self.value[d])
                self.value[d] = 0
                flag[d] = 0
        # remove terms with coefficient zeros 
        con = self.value!=0
        self.index = self.index[con]
        self.value = self.value[con]        


    def __mul__(self, b):
        '''
        multipole self with another mvp
        use: A * B
        '''
        na, nc = self.index.shape
        nb, nd = b.index.shape
        if nc != nd:
            raise RuntimeError('unmatched dimensions')
        C = np.zeros((na*nb, nc), int)
        D = np.zeros(na*nb)
        ci = 0
        for i in xrange(na):
            for j in xrange(nb):
                C[ci] = self.index[i] + b.index[j]
                D[ci] = self.value[i] * b.value[j]
                ci += 1
        f = mvp(C, D)
        return f


    def __pow__(self, n):
        '''
        calculator self n-th power series
        use: A ** n
        notice: n is non-negative integer, code don't check it
        '''
        n = int(n)
        nb  = self.index.shape[1]
        Bi = np.zeros((1, nb), int)
        Bv = np.array([1.])
        B = mvp(Bi, Bv)
        if n != 0:
            for i in xrange(n):
                B *= self
        return B

    def square(self,truncate=5):
        '''
        fast square
        '''
        i1,v1 = [],[]
        for i,v in zip(self.index,self.value):
            i1.append(i*2)
            v1.append(v*v)
        n = len(self.value)
        i2,v2 = [],[]
        for k,l in itertools.combinations(range(n), 2):
            i2.append(self.index[k]+self.index[l])
            v2.append(self.value[k]*self.value[l]*2)
        t = mvp(np.append(i1,i2,axis=0),np.append(v1,v2))
        t.truncate(order=truncate)
        return t

    def __add__(self, b):
        '''
        add self with another mvp
        use: a + b
        '''
        Ci = np.append(self.index, b.index, axis=0)
        Cv = np.append(self.value, b.value)
        C = mvp(Ci, Cv)
        return C


    def __sub__(self, b):
        '''
        subtract self by another mvp
        use: a - b
        '''
        Ci = np.append(self.index, b.index, axis=0)
        Cv = np.append(self.value, -b.value)
        C = mvp(Ci, Cv)
        return C


    def truncate(self,order=2):
        '''
        truncate at the given order, 2 by default
        '''
        n = np.sum(self.index,axis=1)<=order
        self.index = self.index[n]
        self.value = self.value[n]


    def derivative(self, n):
        '''
        calculate self partial derivative on the n-th element
        use: a.derivative(n)
        notice: n is the index element to be derivated
        '''
        na = self.index.shape[1]
        if n > na or n < 1:
            raise RuntimeError('variable index out of range!')
        # remove the terms without the cooresponding element
        di = np.copy(self.index)
        dv = np.copy(self.value)
        con = di[:, n-1] > 0
        di = np.compress(con, di, axis=0)
        dv = np.compress(con, dv)
        # derivative of the left terms 
        dv *= di[:, n-1]
        di[:, n-1] -= 1
        d = mvp(di, dv)
        return d


    def integral(self, n):
        '''
        calculate integral of self on the n-th element
        use: a.integral(n)
        '''
        na = self.index.shape[1]
        if n > na or n < 1:
            print 'Error: variable index is out of range!'
            return
        di = np.copy(self.index)
        dv = np.copy(self.value)
        di[:, n-1] += 1
        dv /= di[:, n-1]
        d = mvp(di, dv)
        return d


    def pb(self, b, n = 1):
        '''
        calculate the Possion bracket of self with another mvp up to n times
        use: a.pb(b, n)
        notice: default n is 1, that is [a, b]. n is a positive integer
                a.pb(b, n) = [a, [a, ...[a, b]]]
        '''
        pab = []
        nd = self.index.shape[1]/2
        for i in range(n):
            if i == 0:
                x = b
            else:
                x = pab[i-1]
            # zero
            Ci = np.zeros((1, self.index.shape[1]), int)
            Cv = np.zeros(1)
            C = mvp(Ci, Cv)
            for ni in range(nd):
                Cp = self.derivative(2*ni+1)*x.derivative(2*ni+2)-\
                     self.derivative(2*ni+2)*x.derivative(2*ni+1)
                C += Cp
            pab.append(C)
        return pab


    def lexp(self, b, n=5):
        '''
        calculate the Lie exponential map of self on another mvp with truncation
        of Possion bracket at n times according definition
        use: a.lexp(b, n)
        notice: default n is 5, that is exp(:a:)b
        '''
        PAB = self.pb(b, n)
        C = b.copy()
        for i in range(n):
            PAB[i].value /= math.factorial(i + 1)
            C += PAB[i]
        return C


    def exp(self, b, nd=5):
        '''
        calculate the Lie exponential map of self on another mvp with truncation
        of Possion bracket at n times with exp(:a:) penetration into each variable 
        use: a.exp(b, n)
        notice: default n is 5, that is exp(:a:)b truncated at 5th Possion bracket.
                efficient and accurate!
        '''
        n = b.index.shape[1]
        # LE exp(:self:) on single variable
        LE = []
        # xb single variable base
        xi = np.identity(n, int)
        xv = np.ones(n)
        xb = []
        for i in range(n):
            xb.append(mvp(xi[i:i+1,:], xv[i:i+1]))
        # exp(:self:) on each variable
        for i in range(n):
            if sum(b.index[:,i]) > 0:
                LE.append(self.lexp(xb[i], nd))
            else:
                LE.append([])
        C = self.const(0)
        for i in range(b.index.shape[0]):
            Ct = self.const(1)
            for j in range(n):
                if b.index[i, j] > 0:
                    Ct *= LE[j] ** b.index[i,j]
            Ct.value *= b.value[i]
            C += Ct
            C.simplify()
        return C


    def chop(self, eps = 1.0e-8):
        '''
        Chop the terms with coefficient abs value smaller than eps
        use: a.chop(eps)
        notice: default eps is 1.0e-8
        '''
        b = self.copy()
        cond = np.abs(b.value) > eps
        b.index = np.compress(cond, b.index, axis=0)
        b.value = np.compress(cond, b.value)
        return b


    def pickWithIndex(self,index):
        '''
        return the first cofficients of given index
        '''
        try:
            n = self.index.shape[1]
        except:
            print('empty mvp')
            return 0.
        if len(index) > n:
            newindex = np.array(index[:n])
        elif len(index) < n:
            newindex = np.append(index,np.zeros(n-len(index),dtype=int))
        else:
            newindex = np.array(index)
        for i,idx in enumerate(self.index):
            if not any(newindex-idx):
                return self.value[i]
        return 0.
            

    def pick(self, n):
        '''
        Pick n-th order homogeneous terms
        '''
        b = self.copy()
        cond = np.sum(b.index, axis=1) == n
        b.index = b.index[cond]
        b.value = b.value[cond]
        #b.index = np.compress(cond, b.index, axis=0)
        #b.value = np.compress(cond, b.value)
        return b


    def copy(self):
        '''
        copy self to a new mvp
        use: a.copy()
        '''
        return mvp(np.copy(self.index), np.copy(self.value))


    def bar(self):
        '''
        plot self coefficent in a bar chart
        use: a.bar()
        '''
        i = np.arange(self.index.shape[0])
        plt.bar(i, self.value)
        xl = []
        for j in i:
            xl.append(str(self.index[j])[1:-1:2])
        plt.xticks(i+0.5, xl, rotation=90)
        plt.show()


    def one_like(self):
        return mvp(np.zeros((1,self.index.shape[1])),[1])


    def zero_like(self):
        return mvp(np.zeros((1,self.index.shape[1])),[0])

        
    def const(self, c):
        '''
        creat constant mvp with same dimension as self and coefficient c
        use: a.const(c)
        notice c must be a number, code will not check it!
        '''
        n = self.index.shape[1]
        i = np.zeros((1, n), int)
        v = c*np.ones(1)
        return mvp(i, v)


    def save(self, fid, cform='%15.6e'):
        '''
        save mvp to file with specified format on coefficient
        use: a.save('file_name')
        '''
        f = open(fid, 'w')
        n1, n2 = self.index.shape
        form = n2 * '%3d' + cform + '\n'
        for i in range(n1):
            tmp = form % tuple(np.append(self.index[i], self.value[i]))
            f.write(tmp)
        f.close()
 

    def eval(self, x):
        '''
        evaluate mvp with x, x is a 1d array (list) data
        use: a.eval(x)
        '''
        n1, n2 = self.index.shape
        if n2 != len(x):
            print 'Error: dimension are not matched'
            return
        x0 = 0
        for i in range(n1):
            x1 = self.value[i]
            for j in range(n2):
                x1 *= x[j] ** self.index[i, j]
            x0 += x1
        return x0


    def decomp(self, nd = 0):
        '''
        decompose a mvp into a list of monomial mvps to ceertain
        order by using BCH 
        use: a.decomp(nd=1)
        nd: order of BCH, default is zero 
        notice: unfinished yet, so far only work for 0 and 1 order
        '''
        n = len(self.value)
        alist = [mvp(self.index[i:i+1], self.value[i:i+1])
                 for i in range(n)]
        if nd == 0:
            return alist
        else:
            s = self.const(0)
            for i in range(n-1):
                for j in range(i+1, n):
                    tp = alist[i].pb(alist[j])[0]
                    s.index = np.append(s.index, tp.index, axis = 0)
                    s.value = np.append(s.value, tp.value)
            s.simplify()
            blist = [mvp(s.index[i:i+1], -s.value[i:i+1]/2)
                     for i in range(len(s.value))]
            return alist + blist


    def monomial(self, x):
        '''
        calculate the map for a Hamiltionian of monomial to act on
        a vector
        use: self.monomial(x)
        notice: ref: A. Chao's note on Lie algebra (9-128)
        '''
        if len(self.value) > 1:
            print 'Error: mvp is not a monomial!'
            return
        xn = []
        for i in range(0, self.index.shape[1], 2):
            a = self.value[0]
            for j in range(self.index.shape[1]):
                if j != i and j != i+1:
                     a *= x[j]**self.index[0, j] 
            if self.index[0, i] ==  self.index[0, i+1]:
                if  self.index[0, i] != 0:
                    tp = np.exp(a * self.index[0, i]
                                * (x[i]*x[i+1])**(self.index[0, i]-1))
                else:
                    tp = 1
                xn.append(x[i] /tp)
                xn.append(x[i+1] * tp)
            else:
                if self.index[0,i] != 0 and self.index[0,i+1] != 0:
                    tp = 1. + a * (self.index[0, i] - self.index[0,i+1]) * \
                        x[i]**(self.index[0,i]-1) * x[i+1]**(self.index[0,i+1]-1)
                    # in python 2.7, int/int = int, here *1.0 is used to floatize 
                    qp =  tp**(self.index[0,i+1]*1./(self.index[0,i+1] - self.index[0,i]))
                    pp = tp / qp
                    xn.append(x[i] * qp)
                    xn.append(x[i+1] * pp)
                elif self.index[0,i] == 0 and self.index[0,i+1] != 0:
                    xn.append(x[i] - a * self.index[0,i+1] * x[i+1]**(self.index[0,i+1]-1))
                    xn.append(x[i+1])
                elif self.index[0,i] != 0 and self.index[0,i+1] == 0:
                    xn.append(x[i])
                    xn.append(x[i+1] + a * self.index[0,i] * x[i]**(self.index[0,i]-1))
        return xn
