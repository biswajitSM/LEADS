import math
import warnings
import numpy as np
from numpy import zeros

# try:
#     import cupy as cp
# except ImportError:
#     cupy = None

# np = np if cp is None else cp

# if np is not cp:
#     warnings.warn("could not import cupy... falling back to numpy & cpu.")

def iterative_deconv(data,kernel,iteration,rule):
    # if np is not np:
    data = np.asarray(data)
    kernel = np.asarray(kernel)

    if data.ndim > 2:
        data_de = np.zeros((data.shape[0], data.shape[1],data.shape[2]), dtype = 'float32')
        for i in range(0, data.shape[0]):
            data_de[i, :, ] = (deblur_core(data[i, :,:], kernel, iteration, rule)).real
    else:
        data_de = (deblur_core(data, kernel, iteration, rule)).real

    if np is not np:
        data_de = np.asnumpy(data_de)

    return data_de

def deblur_core(data, kernel, iteration, rule):

    #data = cp.asnumpy(data)
    kernel = np.array(kernel)
    kernel = kernel / sum(sum(kernel))
    kernel_initial = kernel
    [dx,dy] = data.shape

    B = math.floor(min(dx,dy)/6)
    data = np.pad(data, [int(B),int(B)], 'edge')
    yk = data
    xk = zeros((data.shape[0], data.shape[1]), dtype = 'float32')
    vk = zeros((data.shape[0], data.shape[1]), dtype = 'float32')
    otf = psf2otf(kernel_initial, data.shape)

    if rule == 2: 
    #LandWeber deconv
        t = 1
        gamma1 = 1
        for i in range(0,iteration):

            if i == 0:
                xk_update = data

                xk = data + t*np.fft.ifftn(np.conj(otf)) * (np.fft.fftn(data) - (otf *np.fft.fftn(data)))
            else:
                gamma2 = 1/2*(4 * gamma1*gamma1 + gamma1**4)**(1/2) - gamma1**2
                beta = -gamma2 *(1 - 1 / gamma1)
                yk_update = xk + beta * (xk - xk_update)
                yk = yk_update + t * np.fft.ifftn(np.conj(otf) * (np.fft.fftn(data) - (otf * np.fft.fftn(yk_update))))
                yk = np.maximum(yk, 1e-6, dtype = 'float32')
                gamma1 = gamma2
                xk_update = xk
                xk = yk

    elif rule == 1:
    #Richardson-Lucy deconv

        for iter in range(0, iteration):

            xk_update = xk
            rliter1 = rliter(yk, data, otf)

            xk = yk * ((np.fft.ifftn(np.conj(otf) * rliter1)).real) / ( (np.fft.ifftn(np.fft.fftn(np.ones(data.shape)) * otf)).real)

            xk = np.maximum(xk, 1e-6, dtype = 'float32')

            vk_update = vk

            vk =np.maximum(xk - yk, 1e-6 , dtype = 'float32')

            if iter == 0:
                alpha = 0
                yk = xk
                yk = np.maximum(yk, 1e-6,dtype = 'float32')
                yk = np.array(yk)

            else:

                alpha = sum(sum(vk_update * vk))/(sum(sum(vk_update * vk_update)) + math.e)
                alpha = np.maximum(np.minimum(alpha, 1), 1e-6, dtype = 'float32')
               # start = time.clock()
                yk = xk + alpha * (xk - xk_update)
                yk = np.maximum(yk, 1e-6, dtype = 'float32')
                yk[np.isnan(yk)] = 1e-6
                #end = time.clock()
               # print(start, end)
                #K=np.isnan(yk)

    yk[yk < 0] = 0
    yk = np.array(yk, dtype = 'float32')
    data_decon = yk[B + 0:yk.shape[0] - B, B + 0: yk.shape[1] - B]

    return data_decon

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = np.array(outSize - psfSize)
    psf = np.pad(psf, ((0, int(padSize[0])), (0, int(padSize[1]))), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf

def rliter(yk,data,otf):
    rliter = np.fft.fftn(data / np.maximum(np.fft.ifftn(otf * np.fft.fftn(yk)), 1e-6))
    return rliter

