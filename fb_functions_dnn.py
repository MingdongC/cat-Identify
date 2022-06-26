import numpy as np

#前向传播sigmoid函数
def sigmoid_forward(z):

    A = 1 / (1 + np.exp(-z))
    cache = z

    assert(A.shape == z.shape)

    return A,cache

#前向传播relu函数
def relu_forward(z):

    A = np.maximum(0,z)
    cache = z

    assert(A.shape == z.shape)

    return A,cache

#后向传播sigmoid函数
def relu_backward(dA,Z):

    dz = np.array(dA,copy = True)
    dz[Z<=0] = 0

    assert(dz.shape == dA.shape)

    return dz                               #dz是成本函数对z的偏导数

#后向传播relu函数
def sigmoid_backward(dA,Z):

    s = 1 / (1 + np.exp(-Z))
    dz = dA*s*(1-s)

    assert(dz.shape == dA.shape)

    return dz                               #dz是成本函数对z的偏导数

def g_softsign_forward(z):

    A = ((0.5*z) / (1 + np.abs(z))) + 0.5
    cache = z

    assert (A.shape == z.shape)
    return A,cache

def g_softsign_backward(dA,z):

    def func1(z):
        y = 0.5 / (((1 + z) * (1 + z)))
        return y

    def func2(z):
        y = -0.5 / (((1 + z) * (1 + z)))
        return  y

    dz = np.piecewise(z,[z>0,z<=0],[func1,func2])

    assert (dz.shape == dA.shape)

    return dz